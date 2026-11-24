/// the metal kernels, compiled once at load with fast math off.
///
/// weights stay as the checkpoint's raw bf16 bits and are widened to 32-bit inside each kernel,
/// which is exact, so the math and every cache are 32-bit while weight reads cost 16 bits.
///
/// what holds where:
/// - these kernels: every output value is computed in a fixed order that depends only on the
///   row's own data and context, never on how many rows share the call or where the row sits.
///   most values come from one thread; the reductions split into runs of 64 (rms norm, softmax,
///   the value mix) add their partial sums in a fixed order too. a decoder step uses only these,
///   so a row's logits are bit-identical at any batch width.
/// - apple's matrix kernel (`MPSMatrixMultiplication`, used by the prefill and the encoder for
///   matrix products of `wideRows` rows or more, and for square prefill attention): its order is
///   apple's and not documented. measured on an m4 against a 6,400-row run, a row's bits do not
///   depend on the row count or the row's position once the count is past a boundary that grows
///   with the inner width: 10 rows at 1,024 and 2,048, 30 at 6,144, 40 at 4,096, 90 at 4,320 and
///   75 at 7,680. below it apple's kernel takes another path with other rounding. so those passes
///   are a pure function of the piece, and two pieces read in one product stay bit-identical to
///   being read alone only when each is past the boundary by itself.
enum Kernels {
    static let source = """
    #include <metal_stdlib>
    using namespace metal;

    static inline float4 widen4(packed_ushort4 v) { return as_type<float4>(uint4(ushort4(v)) << 16); }
    static inline float widen(ushort v) { return as_type<float>(uint(v) << 16); }
    static inline float sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }

    // erf to 32-bit accuracy, since metal has none built in. chebyshev fits made for this project:
    // erf(z) / z over z squared below 1, and erfc(z) * exp(z squared) * z over 1 / z above it.
    constant float erf_near[10] = {
        0.9754769206047058f, -0.14226120710372925f, 0.010035581886768341f, -0.0005768764531239867f,
        2.7419931939220987e-05f, -1.1043175618397072e-06f, 3.848875707035404e-08f, -1.180858300031673e-09f,
        3.2334145777523915e-11f, -7.990898524476431e-13f };
    constant float erf_far[15] = {
        0.49471360445022583f, -0.06627963483333588f, -0.0022764645982533693f, 0.0017172450898215175f,
        -0.00032014495809562504f, 2.6743226044345647e-05f, 3.9686956370132975e-06f, -2.2059100501792273e-06f,
        5.188227873986762e-07f, -6.999918866767985e-08f, -9.071270401506126e-10f, 3.819515370651061e-09f,
        -1.3416439070823571e-09f, 2.9478922081160874e-10f, -3.786764998792158e-11f };

    static inline float erf_exact(float z) {
        float x = fabs(z);
        float r;
        if (x <= 1.0f) {
            float s = 2.0f * x * x - 1.0f;
            float b1 = 0.0f, b2 = 0.0f;
            for (int k = 9; k >= 1; k--) { float b = erf_near[k] + 2.0f * s * b1 - b2; b2 = b1; b1 = b; }
            r = x * (erf_near[0] + s * b1 - b2);
        } else if (x < 6.0f) {
            float hi = x * x;
            float lo = fma(x, x, -hi);
            float decay = exp(-hi) * (1.0f - lo);
            float w = 1.0f / x;
            float s = (w - 0.16666666667f) * 2.4f - 1.0f;
            float b1 = 0.0f, b2 = 0.0f;
            for (int k = 14; k >= 1; k--) { float b = erf_far[k] + 2.0f * s * b1 - b2; b2 = b1; b1 = b; }
            r = 1.0f - decay * ((erf_far[0] + s * b1 - b2) * w);
        } else {
            r = 1.0f;
        }
        return z < 0.0f ? -r : r;
    }

    struct CountOnly { uint count; };

    struct GemmParams { uint rows; uint cols; uint count; uint hasBias; uint xStride; uint yStride; };

    // y[n][r] = sum over c of w[r][c] * x[n][c], plus bias[r]. one thread per output value.
    kernel void gemm(const device ushort *W [[buffer(0)]],
                     const device float *X [[buffer(1)]],
                     device float *Y [[buffer(2)]],
                     const device ushort *B [[buffer(3)]],
                     constant GemmParams &p [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
        uint r = gid.x;
        uint n = gid.y;
        if (r >= p.rows || n >= p.count) return;
        const device ushort *w = W + (ulong)r * p.cols;
        const device float *x = X + (ulong)n * p.xStride;
        const device packed_ushort4 *w4 = (const device packed_ushort4 *)w;
        const device packed_float4 *x4 = (const device packed_float4 *)x;
        uint quads = p.cols >> 2;
        uint pairs = quads & ~1u;
        // two running sums keep the gpu from waiting on one add chain. the order is still fixed.
        float4 even = float4(0.0f);
        float4 odd = float4(0.0f);
        for (uint c = 0; c < pairs; c += 2) {
            even += widen4(w4[c]) * float4(x4[c]);
            odd += widen4(w4[c + 1]) * float4(x4[c + 1]);
        }
        if (pairs < quads) even += widen4(w4[pairs]) * float4(x4[pairs]);
        float s = sum4(even + odd);
        for (uint c = quads << 2; c < p.cols; c++) s += widen(w[c]) * x[c];
        if (p.hasBias != 0) s += widen(B[r]);
        Y[(ulong)n * p.yStride + r] = s;
    }

    // the same product for three rows per thread: the weight row is read once and used three
    // times, which about doubles a wide batch. each row keeps its own two running sums and adds
    // in exactly the order `gemm` does, so every value is bit-identical to it at any batch width.
    // for weights without a bias whose width is a multiple of 8, which is every decoder matrix.
    kernel void gemm3(const device ushort *W [[buffer(0)]],
                      const device float *X [[buffer(1)]],
                      device float *Y [[buffer(2)]],
                      constant GemmParams &p [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
        uint r = gid.x;
        uint n = gid.y * 3;
        if (r >= p.rows || n >= p.count) return;
        const device packed_ushort4 *w4 = (const device packed_ushort4 *)(W + (ulong)r * p.cols);
        // a row past the end reads the last real row and is never written
        uint last = p.count - 1;
        const device packed_float4 *x0 = (const device packed_float4 *)(X + (ulong)n * p.xStride);
        const device packed_float4 *x1 = (const device packed_float4 *)(X + (ulong)min(n + 1, last) * p.xStride);
        const device packed_float4 *x2 = (const device packed_float4 *)(X + (ulong)min(n + 2, last) * p.xStride);
        float4 even0 = float4(0.0f), odd0 = float4(0.0f);
        float4 even1 = float4(0.0f), odd1 = float4(0.0f);
        float4 even2 = float4(0.0f), odd2 = float4(0.0f);
        uint quads = p.cols >> 2;
        for (uint c = 0; c < quads; c += 2) {
            float4 w0 = widen4(w4[c]);
            float4 w1 = widen4(w4[c + 1]);
            even0 += w0 * float4(x0[c]); odd0 += w1 * float4(x0[c + 1]);
            even1 += w0 * float4(x1[c]); odd1 += w1 * float4(x1[c + 1]);
            even2 += w0 * float4(x2[c]); odd2 += w1 * float4(x2[c + 1]);
        }
        Y[(ulong)n * p.yStride + r] = sum4(even0 + odd0);
        if (n + 1 < p.count) Y[(ulong)(n + 1) * p.yStride + r] = sum4(even1 + odd1);
        if (n + 2 < p.count) Y[(ulong)(n + 2) * p.yStride + r] = sum4(even2 + odd2);
    }

    kernel void widen(const device ushort *W [[buffer(0)]],
                      device float *Y [[buffer(1)]],
                      constant CountOnly &p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
        if (gid >= p.count) return;
        ((device packed_float4 *)Y)[gid] = widen4(((const device packed_ushort4 *)W)[gid]);
    }

    struct BiasParams { uint rows; uint count; uint stride; };

    kernel void add_bias(device float *Y [[buffer(0)]],
                         const device ushort *B [[buffer(1)]],
                         constant BiasParams &p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.rows || gid.y >= p.count) return;
        Y[(ulong)gid.y * p.stride + gid.x] += widen(B[gid.x]);
    }

    struct NormParams { uint dim; uint count; float eps; uint hasBias; };

    kernel void layer_norm(const device float *X [[buffer(0)]],
                           device float *Y [[buffer(1)]],
                           const device ushort *G [[buffer(2)]],
                           const device ushort *B [[buffer(3)]],
                           constant NormParams &p [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
        if (gid >= p.count) return;
        const device float *x = X + (ulong)gid * p.dim;
        device float *y = Y + (ulong)gid * p.dim;
        const device packed_float4 *x4 = (const device packed_float4 *)x;
        uint quads = p.dim >> 2;
        float4 acc = float4(0.0f);
        for (uint i = 0; i < quads; i++) acc += float4(x4[i]);
        float mean = sum4(acc) / float(p.dim);
        float4 var = float4(0.0f);
        for (uint i = 0; i < quads; i++) { float4 d = float4(x4[i]) - mean; var += d * d; }
        float inv = 1.0f / sqrt(sum4(var) / float(p.dim) + p.eps);
        for (uint i = 0; i < p.dim; i++) y[i] = (x[i] - mean) * inv * widen(G[i]) + widen(B[i]);
    }

    // rms norm in two parallel stages: sums of squares over runs of 64 values, then every output
    // quad adds those partial sums in the same fixed order. a single thread looping over the whole
    // row is several times slower on the gpu.
    kernel void rms_partial(const device float *X [[buffer(0)]],
                            device float *P [[buffer(1)]],
                            constant NormParams &p [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]]) {
        uint runs = p.dim >> 6;
        if (gid.x >= runs || gid.y >= p.count) return;
        const device packed_float4 *x4 = (const device packed_float4 *)(X + (ulong)gid.y * p.dim + (gid.x << 6));
        float4 acc = float4(0.0f);
        for (uint i = 0; i < 16; i++) { float4 v = float4(x4[i]); acc += v * v; }
        P[(ulong)gid.y * runs + gid.x] = sum4(acc);
    }

    kernel void rms_scale(const device float *X [[buffer(0)]],
                          device float *Y [[buffer(1)]],
                          const device ushort *G [[buffer(2)]],
                          const device float *P [[buffer(3)]],
                          constant NormParams &p [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
        uint runs = p.dim >> 6;
        if (gid.x >= (p.dim >> 2) || gid.y >= p.count) return;
        const device float *partial = P + (ulong)gid.y * runs;
        float total = 0.0f;
        for (uint i = 0; i < runs; i++) total += partial[i];
        float inv = 1.0f / sqrt(total / float(p.dim) + p.eps);
        float4 x = float4(((const device packed_float4 *)(X + (ulong)gid.y * p.dim))[gid.x]);
        float4 g = widen4(((const device packed_ushort4 *)G)[gid.x]);
        ((device packed_float4 *)(Y + (ulong)gid.y * p.dim))[gid.x] = g * (x * inv);
    }

    struct RopeParams { uint heads; uint headDim; uint count; float eps; };

    // per head: rms norm over the head, then the rotary position turn. writes row n to out[target[n]].
    kernel void head_norm_rope(const device float *X [[buffer(0)]],
                               device float *Y [[buffer(1)]],
                               const device ushort *G [[buffer(2)]],
                               const device float *COS [[buffer(3)]],
                               const device float *SIN [[buffer(4)]],
                               const device int *POSITION [[buffer(5)]],
                               const device int *TARGET [[buffer(6)]],
                               constant RopeParams &p [[buffer(7)]],
                               uint2 gid [[thread_position_in_grid]]) {
        uint h = gid.x;
        uint n = gid.y;
        if (h >= p.heads || n >= p.count) return;
        uint width = p.heads * p.headDim;
        const device float *x = X + (ulong)n * width + h * p.headDim;
        device float *y = Y + (ulong)TARGET[n] * width + h * p.headDim;
        const device packed_float4 *x4 = (const device packed_float4 *)x;
        float4 acc = float4(0.0f);
        for (uint i = 0; i < (p.headDim >> 2); i++) { float4 v = float4(x4[i]); acc += v * v; }
        float inv = 1.0f / sqrt(sum4(acc) / float(p.headDim) + p.eps);
        uint half_dim = p.headDim >> 1;
        const device float *c = COS + (ulong)POSITION[n] * half_dim;
        const device float *s = SIN + (ulong)POSITION[n] * half_dim;
        for (uint i = 0; i < half_dim; i++) {
            float a = widen(G[i]) * (x[i] * inv);
            float b = widen(G[i + half_dim]) * (x[i + half_dim] * inv);
            y[i] = a * c[i] - b * s[i];
            y[i + half_dim] = b * c[i] + a * s[i];
        }
    }

    struct RowParams { uint dim; uint count; };

    kernel void scatter_rows(const device float *X [[buffer(0)]],
                             device float *Y [[buffer(1)]],
                             const device int *TARGET [[buffer(2)]],
                             constant RowParams &p [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.dim || gid.y >= p.count) return;
        Y[(ulong)TARGET[gid.y] * p.dim + gid.x] = X[(ulong)gid.y * p.dim + gid.x];
    }

    kernel void embed(const device ushort *E [[buffer(0)]],
                      device float *Y [[buffer(1)]],
                      const device int *TOKEN [[buffer(2)]],
                      constant RowParams &p [[buffer(3)]],
                      uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.dim || gid.y >= p.count) return;
        Y[(ulong)gid.y * p.dim + gid.x] = widen(E[(ulong)TOKEN[gid.y] * p.dim + gid.x]);
    }

    // context of one query row: a run of consecutive slots, then a list of slots from the ancestry table.
    struct Context { uint runStart; uint runCount; uint listOffset; uint listCount; };
    struct AttentionParams { uint heads; uint keyHeads; uint headDim; uint rows; uint maxContext; float scale; };

    static inline uint slot_of(Context c, const device int *LIST, uint s) {
        return s < c.runCount ? c.runStart + s : uint(LIST[c.listOffset + (s - c.runCount)]);
    }

    kernel void attention_scores(const device float *Q [[buffer(0)]],
                                 const device float *K [[buffer(1)]],
                                 const device Context *CONTEXT [[buffer(2)]],
                                 const device int *LIST [[buffer(3)]],
                                 device float *SCORES [[buffer(4)]],
                                 constant AttentionParams &p [[buffer(5)]],
                                 uint3 gid [[thread_position_in_grid]]) {
        uint s = gid.x;
        uint h = gid.y;
        uint n = gid.z;
        if (h >= p.heads || n >= p.rows) return;
        Context c = CONTEXT[n];
        if (s >= c.runCount + c.listCount) return;
        uint slot = slot_of(c, LIST, s);
        uint keyHead = h / (p.heads / p.keyHeads);
        const device packed_float4 *q = (const device packed_float4 *)(Q + ((ulong)n * p.heads + h) * p.headDim);
        const device packed_float4 *k = (const device packed_float4 *)(K + ((ulong)slot * p.keyHeads + keyHead) * p.headDim);
        float4 acc = float4(0.0f);
        for (uint i = 0; i < (p.headDim >> 2); i++) acc += float4(q[i]) * float4(k[i]);
        SCORES[((ulong)n * p.heads + h) * p.maxContext + s] = sum4(acc) * p.scale;
    }

    // softmax and the value mix, split into runs of 64 context slots so no thread walks the whole
    // context alone. every stage adds in a fixed order that depends only on the row's own context,
    // so a row's result is the same at any batch width.
    kernel void attention_run_max(const device Context *CONTEXT [[buffer(2)]],
                                  const device float *SCORES [[buffer(4)]],
                                  constant AttentionParams &p [[buffer(5)]],
                                  device float *RUNS [[buffer(6)]],
                                  uint3 gid [[thread_position_in_grid]]) {
        uint run = gid.x;
        uint h = gid.y;
        uint n = gid.z;
        if (h >= p.heads || n >= p.rows) return;
        Context c = CONTEXT[n];
        uint count = c.runCount + c.listCount;
        uint first = run << 6;
        if (first >= count) return;
        uint last = min(first + 64, count);
        const device float *row = SCORES + ((ulong)n * p.heads + h) * p.maxContext;
        float top = row[first];
        for (uint s = first + 1; s < last; s++) top = max(top, row[s]);
        RUNS[((ulong)n * p.heads + h) * (p.maxContext >> 6) + run] = top;
    }

    kernel void attention_exp(const device Context *CONTEXT [[buffer(2)]],
                              device float *SCORES [[buffer(4)]],
                              constant AttentionParams &p [[buffer(5)]],
                              const device float *RUNS [[buffer(6)]],
                              uint3 gid [[thread_position_in_grid]]) {
        uint s = gid.x;
        uint h = gid.y;
        uint n = gid.z;
        if (h >= p.heads || n >= p.rows) return;
        Context c = CONTEXT[n];
        uint count = c.runCount + c.listCount;
        if (s >= count) return;
        const device float *runs = RUNS + ((ulong)n * p.heads + h) * (p.maxContext >> 6);
        uint runCount = (count + 63) >> 6;
        float top = runs[0];
        for (uint i = 1; i < runCount; i++) top = max(top, runs[i]);
        device float *row = SCORES + ((ulong)n * p.heads + h) * p.maxContext;
        row[s] = exp(row[s] - top);
    }

    // the three kernels below finish a causal softmax whose scores and value mix were done by
    // apple's matrix kernel over the whole square. entries past a row's own context become zero
    // weights, so the matrix product only sees what the row may attend to.
    kernel void attention_exp_causal(const device Context *CONTEXT [[buffer(2)]],
                                     device float *SCORES [[buffer(4)]],
                                     constant AttentionParams &p [[buffer(5)]],
                                     const device float *RUNS [[buffer(6)]],
                                     uint3 gid [[thread_position_in_grid]]) {
        uint s = gid.x;
        uint h = gid.y;
        uint n = gid.z;
        if (s >= p.maxContext || h >= p.heads || n >= p.rows) return;
        Context c = CONTEXT[n];
        uint count = c.runCount + c.listCount;
        device float *row = SCORES + ((ulong)n * p.heads + h) * p.maxContext;
        if (s >= count) { row[s] = 0.0f; return; }
        const device float *runs = RUNS + ((ulong)n * p.heads + h) * (p.maxContext >> 6);
        uint runCount = (count + 63) >> 6;
        float top = runs[0];
        for (uint i = 1; i < runCount; i++) top = max(top, runs[i]);
        row[s] = exp(row[s] - top);
    }

    kernel void attention_run_total(const device float *SCORES [[buffer(4)]],
                                    constant AttentionParams &p [[buffer(5)]],
                                    device float *RUNS [[buffer(6)]],
                                    uint3 gid [[thread_position_in_grid]]) {
        uint run = gid.x;
        uint h = gid.y;
        uint n = gid.z;
        if (run >= (p.maxContext >> 6) || h >= p.heads || n >= p.rows) return;
        const device packed_float4 *row = (const device packed_float4 *)(SCORES + ((ulong)n * p.heads + h) * p.maxContext + (run << 6));
        float4 acc = float4(0.0f);
        for (uint i = 0; i < 16; i++) acc += float4(row[i]);
        RUNS[((ulong)n * p.heads + h) * (p.maxContext >> 6) + run] = sum4(acc);
    }

    kernel void attention_divide(constant AttentionParams &p [[buffer(5)]],
                                 const device float *RUNS [[buffer(6)]],
                                 device float *Y [[buffer(7)]],
                                 uint3 gid [[thread_position_in_grid]]) {
        uint d = gid.x;
        uint h = gid.y;
        uint n = gid.z;
        if (d >= (p.headDim >> 2) || h >= p.heads || n >= p.rows) return;
        const device float *runs = RUNS + ((ulong)n * p.heads + h) * (p.maxContext >> 6);
        float total = 0.0f;
        for (uint i = 0; i < (p.maxContext >> 6); i++) total += runs[i];
        device packed_float4 *y = (device packed_float4 *)(Y + ((ulong)n * p.heads + h) * p.headDim);
        y[d] = float4(y[d]) / total;
    }

    // per run of 64 slots: the weighted sum of values for one quad of the head, and in the last
    // lane group the run's total weight.
    kernel void attention_partial(const device float *V [[buffer(1)]],
                                  const device Context *CONTEXT [[buffer(2)]],
                                  const device int *LIST [[buffer(3)]],
                                  const device float *SCORES [[buffer(4)]],
                                  constant AttentionParams &p [[buffer(5)]],
                                  device float *PARTIAL [[buffer(6)]],
                                  uint3 gid [[thread_position_in_grid]]) {
        uint quads = p.headDim >> 2;
        uint d = gid.x % (quads + 1);
        uint run = gid.x / (quads + 1);
        uint h = gid.y;
        uint n = gid.z;
        if (h >= p.heads || n >= p.rows) return;
        Context c = CONTEXT[n];
        uint count = c.runCount + c.listCount;
        uint first = run << 6;
        if (first >= count) return;
        uint last = min(first + 64, count);
        const device float *row = SCORES + ((ulong)n * p.heads + h) * p.maxContext;
        device packed_float4 *out = (device packed_float4 *)(PARTIAL
            + (((ulong)n * p.heads + h) * (p.maxContext >> 6) + run) * (p.headDim + 4));
        if (d == quads) {
            float4 total = float4(0.0f);
            for (uint s = first; s < last; s++) total.x += row[s];
            out[quads] = total;
            return;
        }
        uint keyHead = h / (p.heads / p.keyHeads);
        float4 acc = float4(0.0f);
        for (uint s = first; s < last; s++) {
            uint slot = slot_of(c, LIST, s);
            const device packed_float4 *v = (const device packed_float4 *)(V + ((ulong)slot * p.keyHeads + keyHead) * p.headDim);
            acc += row[s] * float4(v[d]);
        }
        out[d] = acc;
    }

    kernel void attention_combine(const device Context *CONTEXT [[buffer(2)]],
                                  constant AttentionParams &p [[buffer(5)]],
                                  const device float *PARTIAL [[buffer(6)]],
                                  device float *Y [[buffer(7)]],
                                  uint3 gid [[thread_position_in_grid]]) {
        uint quads = p.headDim >> 2;
        uint d = gid.x;
        uint h = gid.y;
        uint n = gid.z;
        if (d >= quads || h >= p.heads || n >= p.rows) return;
        Context c = CONTEXT[n];
        uint runCount = (c.runCount + c.listCount + 63) >> 6;
        const device packed_float4 *runs = (const device packed_float4 *)(PARTIAL
            + ((ulong)n * p.heads + h) * (p.maxContext >> 6) * (p.headDim + 4));
        float4 acc = float4(0.0f);
        float total = 0.0f;
        for (uint i = 0; i < runCount; i++) {
            acc += float4(runs[i * (quads + 1) + d]);
            total += float4(runs[i * (quads + 1) + quads]).x;
        }
        ((device packed_float4 *)(Y + ((ulong)n * p.heads + h) * p.headDim))[d] = acc / total;
    }

    struct CountParams { uint count; };

    kernel void add(const device float *A [[buffer(0)]],
                    const device float *B [[buffer(1)]],
                    device float *Y [[buffer(2)]],
                    constant CountParams &p [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
        if (gid >= p.count) return;
        Y[gid] = A[gid] + B[gid];
    }

    kernel void silu_mul(const device float *G [[buffer(0)]],
                         const device float *U [[buffer(1)]],
                         device float *Y [[buffer(2)]],
                         constant CountParams &p [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
        if (gid >= p.count) return;
        float g = G[gid];
        Y[gid] = (g * (1.0f / (1.0f + exp(-g)))) * U[gid];
    }

    kernel void gelu(device float *X [[buffer(0)]],
                     constant CountParams &p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
        if (gid >= p.count) return;
        float x = X[gid];
        X[gid] = x * 0.5f * (1.0f + erf_exact(x * 0.70710678118f));
    }

    struct PatchParams { uint chunks; uint height; uint width; uint channels; uint outHeight; uint outWidth; };

    // 3x3 patches at stride 2 with one cell of zero padding, laid out to match the checkpoint's
    // [out][in][ky][kx] weights so the convolution becomes one gemm.
    kernel void patches(const device float *X [[buffer(0)]],
                        device float *Y [[buffer(1)]],
                        constant PatchParams &p [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]]) {
        uint col = gid.x;
        uint cell = gid.y;
        uint chunk = gid.z;
        uint cols = p.channels * 9;
        if (col >= cols || cell >= p.outHeight * p.outWidth || chunk >= p.chunks) return;
        uint channel = col / 9;
        uint ky = (col % 9) / 3;
        uint kx = col % 3;
        int iy = int(cell / p.outWidth) * 2 - 1 + int(ky);
        int ix = int(cell % p.outWidth) * 2 - 1 + int(kx);
        float value = 0.0f;
        if (iy >= 0 && iy < int(p.height) && ix >= 0 && ix < int(p.width)) {
            value = X[(((ulong)chunk * p.height + uint(iy)) * p.width + uint(ix)) * p.channels + channel];
        }
        Y[((ulong)chunk * p.outHeight * p.outWidth + cell) * cols + col] = value;
    }

    // [chunk][frequency][time][channel] to [chunk][time][channel * frequencies + frequency]
    kernel void flatten_frequency(const device float *X [[buffer(0)]],
                                  device float *Y [[buffer(1)]],
                                  constant PatchParams &p [[buffer(2)]],
                                  uint3 gid [[thread_position_in_grid]]) {
        uint feature = gid.x;
        uint time = gid.y;
        uint chunk = gid.z;
        uint features = p.channels * p.height;
        if (feature >= features || time >= p.width || chunk >= p.chunks) return;
        uint channel = feature / p.height;
        uint frequency = feature % p.height;
        Y[((ulong)chunk * p.width + time) * features + feature] =
            X[(((ulong)chunk * p.height + frequency) * p.width + time) * p.channels + channel];
    }

    struct PositionParams { uint dim; uint count; uint period; };

    kernel void add_positions(device float *X [[buffer(0)]],
                              const device float *POSITIONS [[buffer(1)]],
                              constant PositionParams &p [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= p.dim || gid.y >= p.count) return;
        X[(ulong)gid.y * p.dim + gid.x] += POSITIONS[(gid.y % p.period) * p.dim + gid.x];
    }
    """
}
