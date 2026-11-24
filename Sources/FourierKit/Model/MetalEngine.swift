import Foundation
import Metal
import MetalPerformanceShaders

/// the sizes read from the checkpoint's config.json.
struct ModelShape: Sendable {
    var melBins = 128
    var chunkFrames = 100
    var tokensPerChunk = 13
    var windowTokens = 104
    var convChannels = 480
    var encoderLayers = 24
    var encoderWidth = 1024
    var encoderHeads = 16
    var encoderInner = 4096
    var audioWidth = 2048

    var decoderLayers = 28
    var width = 2048
    var heads = 16
    var keyHeads = 8
    var headDim = 128
    var inner = 6144
    var vocabulary = 151_936
    var normEps: Float = 1e-6
    var ropeBase: Double = 1_000_000
    var audioTokenID: Int32 = 151_676

    var keyWidth: Int { keyHeads * headDim }

    static func load(configJSON: URL) throws -> ModelShape {
        let data = try Data(contentsOf: configJSON)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let audio = root["audio_config"] as? [String: Any],
              let text = root["text_config"] as? [String: Any]
        else { throw EngineFailure.badConfig("config.json has no audio_config or text_config") }
        var shape = ModelShape()
        func int(_ source: [String: Any], _ key: String, _ fallback: Int) -> Int { (source[key] as? NSNumber)?.intValue ?? fallback }
        shape.melBins = int(audio, "num_mel_bins", shape.melBins)
        let window = int(audio, "n_window", 50)
        shape.chunkFrames = window * 2
        shape.tokensPerChunk = int(audio, "max_position_embeddings", shape.tokensPerChunk)
        shape.windowTokens = shape.tokensPerChunk * (int(audio, "n_window_infer", 800) / shape.chunkFrames)
        shape.convChannels = int(audio, "downsample_hidden_size", shape.convChannels)
        shape.encoderLayers = int(audio, "encoder_layers", shape.encoderLayers)
        shape.encoderWidth = int(audio, "d_model", shape.encoderWidth)
        shape.encoderHeads = int(audio, "encoder_attention_heads", shape.encoderHeads)
        shape.encoderInner = int(audio, "encoder_ffn_dim", shape.encoderInner)
        shape.audioWidth = int(audio, "output_dim", shape.audioWidth)
        shape.decoderLayers = int(text, "num_hidden_layers", shape.decoderLayers)
        shape.width = int(text, "hidden_size", shape.width)
        shape.heads = int(text, "num_attention_heads", shape.heads)
        shape.keyHeads = int(text, "num_key_value_heads", shape.keyHeads)
        shape.headDim = int(text, "head_dim", shape.headDim)
        shape.inner = int(text, "intermediate_size", shape.inner)
        shape.vocabulary = int(text, "vocab_size", shape.vocabulary)
        if let eps = text["rms_norm_eps"] as? NSNumber { shape.normEps = eps.floatValue }
        if let rope = text["rope_parameters"] as? [String: Any], let base = rope["rope_theta"] as? NSNumber {
            shape.ropeBase = base.doubleValue
        }
        if let token = root["audio_token_id"] as? NSNumber { shape.audioTokenID = token.int32Value }
        guard shape.audioWidth == shape.width else {
            throw EngineFailure.badConfig("the audio projector width does not match the decoder width")
        }
        // what the kernels assume about the sizes: whole quads, whole runs of 64, grouped heads
        guard shape.heads * shape.headDim == shape.width else {
            throw EngineFailure.badConfig("heads × head_dim (\(shape.heads) × \(shape.headDim)) is not hidden_size \(shape.width)")
        }
        guard shape.keyHeads > 0, shape.heads % shape.keyHeads == 0 else {
            throw EngineFailure.badConfig("\(shape.heads) attention heads do not split into \(shape.keyHeads) key heads")
        }
        guard shape.width % 64 == 0, shape.headDim % 8 == 0, shape.inner % 8 == 0 else {
            throw EngineFailure.badConfig("hidden_size must be a multiple of 64, head_dim and intermediate_size of 8")
        }
        guard shape.encoderHeads > 0, shape.encoderWidth % shape.encoderHeads == 0, (shape.encoderWidth / shape.encoderHeads) % 4 == 0,
              shape.encoderWidth % 4 == 0, shape.encoderInner % 4 == 0 else {
            throw EngineFailure.badConfig("the audio encoder's d_model does not split into \(shape.encoderHeads) heads of whole quads")
        }
        return shape
    }

    /// every tensor the engine reads, with the shape the config implies
    func expectedTensors() -> [String: [Int]] {
        var heights = melBins
        for _ in 0..<3 { heights = (heights - 1) / 2 + 1 }
        var expected: [String: [Int]] = [:]
        let conv = "model.audio_tower."
        expected[conv + "conv2d1.weight"] = [convChannels, 1, 3, 3]
        expected[conv + "conv2d2.weight"] = [convChannels, convChannels, 3, 3]
        expected[conv + "conv2d3.weight"] = [convChannels, convChannels, 3, 3]
        for name in ["conv2d1", "conv2d2", "conv2d3"] { expected[conv + name + ".bias"] = [convChannels] }
        expected[conv + "conv_out.weight"] = [encoderWidth, convChannels * heights]
        for layer in 0..<encoderLayers {
            let p = conv + "layers.\(layer)."
            for name in ["self_attn_layer_norm", "final_layer_norm"] {
                expected[p + name + ".weight"] = [encoderWidth]
                expected[p + name + ".bias"] = [encoderWidth]
            }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                expected[p + "self_attn." + name + ".weight"] = [encoderWidth, encoderWidth]
                expected[p + "self_attn." + name + ".bias"] = [encoderWidth]
            }
            expected[p + "fc1.weight"] = [encoderInner, encoderWidth]
            expected[p + "fc1.bias"] = [encoderInner]
            expected[p + "fc2.weight"] = [encoderWidth, encoderInner]
            expected[p + "fc2.bias"] = [encoderWidth]
        }
        expected[conv + "ln_post.weight"] = [encoderWidth]
        expected[conv + "ln_post.bias"] = [encoderWidth]
        expected["model.multi_modal_projector.linear_1.weight"] = [encoderWidth, encoderWidth]
        expected["model.multi_modal_projector.linear_1.bias"] = [encoderWidth]
        expected["model.multi_modal_projector.linear_2.weight"] = [audioWidth, encoderWidth]
        expected["model.multi_modal_projector.linear_2.bias"] = [audioWidth]
        let text = "model.language_model."
        expected[text + "embed_tokens.weight"] = [vocabulary, width]
        expected[text + "norm.weight"] = [width]
        for layer in 0..<decoderLayers {
            let p = text + "layers.\(layer)."
            expected[p + "input_layernorm.weight"] = [width]
            expected[p + "post_attention_layernorm.weight"] = [width]
            expected[p + "self_attn.q_proj.weight"] = [heads * headDim, width]
            expected[p + "self_attn.k_proj.weight"] = [keyWidth, width]
            expected[p + "self_attn.v_proj.weight"] = [keyWidth, width]
            expected[p + "self_attn.o_proj.weight"] = [width, heads * headDim]
            expected[p + "self_attn.q_norm.weight"] = [headDim]
            expected[p + "self_attn.k_norm.weight"] = [headDim]
            expected[p + "mlp.gate_proj.weight"] = [inner, width]
            expected[p + "mlp.up_proj.weight"] = [inner, width]
            expected[p + "mlp.down_proj.weight"] = [width, inner]
        }
        return expected
    }

    /// checks every tensor the engine reads against the config: present, bf16, the right shape and
    /// exactly two bytes per value. the kernels take their sizes from the config alone, so a
    /// mismatch would read past a tensor or misread it rather than fail.
    func validate(_ weights: SafeTensors) throws {
        for (name, shape) in expectedTensors().sorted(by: { $0.key < $1.key }) {
            guard let tensor = weights.tensors[name] else {
                throw EngineFailure.badConfig("the weights have no tensor named \(name)")
            }
            guard tensor.dtype == "BF16" else {
                throw EngineFailure.badConfig("\(name) is \(tensor.dtype.lowercased()), and the engine reads raw bf16")
            }
            guard tensor.shape == shape else {
                throw EngineFailure.badConfig("\(name) has shape \(tensor.shape), and config.json implies \(shape)")
            }
            guard tensor.byteCount == shape.reduce(1, *) * 2 else {
                throw EngineFailure.badConfig("\(name) holds \(tensor.byteCount) bytes, not the \(shape.reduce(1, *) * 2) its shape needs")
            }
        }
    }
}

enum EngineFailure: Error, CustomStringConvertible {
    case noDevice
    case badConfig(String)
    case gpu(String)
    case outOfSlots
    case tooLong(String)
    /// the device cannot hold something the engine needs, such as one tensor in one buffer
    case deviceLimit(String)

    var description: String {
        switch self {
        #if os(macOS)
        case .noDevice: "this mac has no metal device"
        #else
        case .noDevice: "this device has no metal gpu"
        #endif
        case .badConfig(let why): "the model config is not usable: \(why)"
        case .gpu(let why): "a gpu call failed: \(why)"
        case .outOfSlots: "the decoder cache is full"
        case .tooLong(let why): why
        case .deviceLimit(let why): why
        }
    }
}

/// a tensor's place on the gpu: the metal buffer that wraps it and its byte offset inside that buffer.
struct WeightRef {
    let buffer: MTLBuffer
    let offset: Int
}

/// the gpu engine: the checkpoint mapped into metal buffers, the kernels, and the passes built from them.
///
/// one owner at a time. the scheduler above it is the only caller, and it never overlaps two calls.
final class MetalEngine: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let shape: ModelShape
    let deviceProfile: DeviceProfile
    let limits: EngineLimits
    let weights: SafeTensors
    /// one buffer over the whole mapping when the device allows it, else page-aligned windows
    let weightBuffers: [MTLBuffer]
    private let weightRefs: [String: WeightRef]
    let maxPositions = 4096

    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var scratch: [String: MTLBuffer] = [:]
    private var multiplies: [String: MPSMatrixMultiplication] = [:]
    /// from this many rows up, a per-piece multiply (prefill, encoder) goes to apple's kernel on
    /// 32-bit copies of the weights made on the fly. below it, and always in a decoder step, the
    /// bandwidth kernel reads the raw 16-bit weights.
    var wideRows = 32
    /// from this many prompt rows up, prefill attention uses the matrix kernel over the whole square
    var squareRows = 64

    let ropeCos: MTLBuffer
    let ropeSin: MTLBuffer
    let encoderPositions: MTLBuffer

    // decoder cache arena, [layer][slot][keyHeads * headDim]
    let slotCapacity: Int
    let keyArena: MTLBuffer
    let valueArena: MTLBuffer
    var freeSlots: [Range<Int>]

    /// gpu seconds per kernel, collected only while this is not nil. for measuring, not for shipping paths.
    var profile: [String: Double]?

    /// seconds the gpu spent inside calls, for the busy share in live stats
    private(set) var gpuSeconds: Double = 0
    /// the gpu's own clock for the last `run`: first kernel start to last kernel end
    private(set) var lastGPUTime: Double = 0

    /// - parameter settings: the generation defaults, which size the decoder cache
    /// - parameter slotCapacity: overrides the cache size derived from the device
    /// - parameter weightWindow: wraps the weights in windows of at most this many bytes even when one
    ///   buffer would do, to check the windowed path on a mac
    init(modelDirectory: URL, settings: GenerationSettings, slotCapacity: Int? = nil, weightWindow: Int? = nil) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineFailure.noDevice
        }
        self.device = device
        self.queue = queue
        let shape = try ModelShape.load(configJSON: modelDirectory.appendingPathComponent("config.json"))
        self.shape = shape
        weights = try SafeTensors(url: modelDirectory.appendingPathComponent("model.safetensors"))
        for (name, tensor) in weights.tensors where tensor.dtype != "BF16" {
            throw EngineFailure.badConfig("\(name) is \(tensor.dtype.lowercased()), and the engine reads raw bf16")
        }
        try shape.validate(weights)
        deviceProfile = DeviceProfile.measure(device: device)
        limits = try EngineLimits(
            profile: deviceProfile, shape: shape, mappedLength: weights.mappedLength, weightBytes: weights.fileLength, settings: settings)
        let window = weightWindow ?? (limits.weightsWindowed ? device.maxBufferLength : nil)
        (weightBuffers, weightRefs) = try Self.wrap(weights, device: device, window: window)

        let options = MTLCompileOptions()
        options.mathMode = .safe
        options.languageVersion = .version3_2
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Kernels.source, options: options)
        } catch {
            throw EngineFailure.gpu("the kernels did not compile: \(error)")
        }
        for name in library.functionNames {
            guard let function = library.makeFunction(name: name) else { continue }
            function.label = name
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = function
            descriptor.label = name
            pipelines[name] = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
        }

        // rotary tables, in 32-bit float like the reference computes them
        let half = shape.headDim / 2
        var cosines = [Float](repeating: 0, count: maxPositions * half)
        var sines = [Float](repeating: 0, count: maxPositions * half)
        let frequencies: [Float] = (0..<half).map { 1.0 / powf(Float(shape.ropeBase), Float(2 * $0) / Float(shape.headDim)) }
        for position in 0..<maxPositions {
            for index in 0..<half {
                let angle = Float(position) * frequencies[index]
                cosines[position * half + index] = cosf(angle)
                sines[position * half + index] = sinf(angle)
            }
        }
        ropeCos = try Self.buffer(device, cosines)
        ropeSin = try Self.buffer(device, sines)

        // sinusoid positions for the encoder, one row per token of a 1-second chunk
        let width = shape.encoderWidth
        let increment = Float(log(10_000.0) / Double(width / 2 - 1))
        var positions = [Float](repeating: 0, count: shape.tokensPerChunk * width)
        for position in 0..<shape.tokensPerChunk {
            for index in 0..<(width / 2) {
                let scaled = Float(position) * expf(-increment * Float(index))
                positions[position * width + index] = sinf(scaled)
                positions[position * width + width / 2 + index] = cosf(scaled)
            }
        }
        encoderPositions = try Self.buffer(device, positions)

        let slotCapacity = slotCapacity ?? limits.slotCapacity
        self.slotCapacity = slotCapacity
        let arenaBytes = shape.decoderLayers * slotCapacity * shape.keyWidth * 4
        guard let keys = device.makeBuffer(length: arenaBytes, options: .storageModeShared),
              let values = device.makeBuffer(length: arenaBytes, options: .storageModeShared)
        else { throw EngineFailure.gpu("the decoder cache could not be allocated") }
        keyArena = keys
        valueArena = values
        freeSlots = [0..<slotCapacity]
    }

    private static func buffer(_ device: MTLDevice, _ values: [Float]) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared) else {
            throw EngineFailure.gpu("a table could not be allocated")
        }
        return buffer
    }

    var weightBytes: Int { weights.fileLength }

    // MARK: scratch

    /// a reusable buffer of at least `bytes`. contents are whatever the last pass left.
    func scratch(_ name: String, bytes: Int) throws -> MTLBuffer {
        if let existing = scratch[name], existing.length >= bytes { return existing }
        let length = max(bytes, 16)
        guard let made = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw EngineFailure.gpu("a \(length) byte work buffer could not be allocated")
        }
        scratch[name] = made
        return made
    }

    /// the buffer for a matrix's 32-bit copy, at least `bytes` long (64 mb covers every matrix
    /// but the output head, which never runs on apple's kernel)
    fileprivate func wideScratch(bytes: Int) throws -> MTLBuffer {
        try scratch("wide.weights", bytes: max(bytes, 64 << 20))
    }

    func releaseScratch(prefix: String) {
        scratch = scratch.filter { !$0.key.hasPrefix(prefix) }
    }

    func weight(_ name: String) throws -> WeightRef {
        guard let ref = weightRefs[name] else { throw SafeTensors.Failure.missing(name) }
        return ref
    }

    /// wraps the mapped checkpoint without copying. when it fits one buffer, that buffer starts at
    /// the file's first byte and every tensor keeps its file offset. otherwise each window starts on
    /// the page at or before its first tensor and holds whole tensors only.
    /// - parameter window: nil for one buffer, else the most bytes one window may span
    private static func wrap(_ weights: SafeTensors, device: MTLDevice, window: Int?) throws -> ([MTLBuffer], [String: WeightRef]) {
        let length = weights.mappedLength
        let page = Int(getpagesize())
        var windows: [(start: Int, end: Int)] = []
        var assigned: [(name: String, offset: Int, window: Int)] = []
        if let window {
            let limit = window / page * page
            for tensor in weights.tensors.values.sorted(by: { $0.offset < $1.offset }) {
                let start = tensor.offset / page * page
                let end = min(length, (tensor.offset + tensor.byteCount + page - 1) / page * page)
                guard end - start <= limit else {
                    throw EngineFailure.deviceLimit(
                        "the weight \(tensor.name) is \(tensor.byteCount) bytes, more than one metal buffer can hold on this device (\(limit) bytes)")
                }
                if let last = windows.last, end - last.start <= limit {
                    windows[windows.count - 1].end = max(last.end, end)
                } else {
                    windows.append((start, end))
                }
                assigned.append((tensor.name, tensor.offset, windows.count - 1))
            }
        } else {
            windows = [(0, length)]
            assigned = weights.tensors.values.map { ($0.name, $0.offset, 0) }
        }
        let buffers = try windows.map { window in
            guard let buffer = device.makeBuffer(
                bytesNoCopy: weights.base + window.start, length: window.end - window.start,
                options: .storageModeShared, deallocator: nil
            ) else { throw EngineFailure.gpu("the weights could not be wrapped in a metal buffer") }
            return buffer
        }
        var refs: [String: WeightRef] = [:]
        for entry in assigned {
            refs[entry.name] = WeightRef(buffer: buffers[entry.window], offset: entry.offset - windows[entry.window].start)
        }
        return (buffers, refs)
    }

    // MARK: running

    /// encodes one command buffer and suspends until the gpu has finished it.
    func run(_ body: (Pass) throws -> Void) async throws {
        let commandBuffer = try encode(body)
        let started = DispatchTime.now().uptimeNanoseconds
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            commandBuffer.addCompletedHandler { _ in continuation.resume() }
            commandBuffer.commit()
        }
        gpuSeconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        lastGPUTime = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        if let error = commandBuffer.error { throw EngineFailure.gpu(error.localizedDescription) }
    }

    /// commits several command buffers in order and suspends until the last one is done.
    /// long passes are cut into pieces this way so no single buffer runs long enough to time out.
    func runAll(_ bodies: [(Pass) throws -> Void]) async throws {
        var buffers: [MTLCommandBuffer] = []
        for body in bodies { buffers.append(try encode(body)) }
        guard let last = buffers.last else { return }
        let started = DispatchTime.now().uptimeNanoseconds
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            last.addCompletedHandler { _ in continuation.resume() }
            for buffer in buffers { buffer.commit() }
        }
        gpuSeconds += Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        lastGPUTime = last.gpuEndTime - buffers[0].gpuStartTime
        for buffer in buffers {
            if let error = buffer.error { throw EngineFailure.gpu(error.localizedDescription) }
        }
    }

    private func encode(_ body: (Pass) throws -> Void) throws -> MTLCommandBuffer {
        guard let commandBuffer = queue.makeCommandBuffer() else { throw EngineFailure.gpu("no command buffer") }
        let pass = Pass(engine: self, commandBuffer: commandBuffer)
        defer { pass.close() }
        try body(pass)
        return pass.commandBuffer
    }

    /// apple's matrix multiply for one shape, made once. it is about four times faster than the
    /// bandwidth kernel when many rows share a call, which is the prefill and the encoder.
    fileprivate func multiply(rows: Int, cols: Int, count: Int) -> MPSMatrixMultiplication {
        let key = "\(count)x\(rows)x\(cols)"
        if let made = multiplies[key] { return made }
        let made = MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: true,
            resultRows: count, resultColumns: rows, interiorColumns: cols, alpha: 1, beta: 0
        )
        if multiplies.count > 256 { multiplies.removeAll() }
        multiplies[key] = made
        return made
    }

    /// apple's matrix multiply for the two products of square attention: scores (transposed right
    /// side, scaled) and the value mix (plain).
    fileprivate func product(resultRows: Int, resultColumns: Int, interior: Int, transposeRight: Bool, alpha: Double) -> MPSMatrixMultiplication {
        let key = "p\(resultRows)x\(resultColumns)x\(interior)\(transposeRight ? "t" : "n")"
        if let made = multiplies[key] { return made }
        let made = MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: transposeRight,
            resultRows: resultRows, resultColumns: resultColumns, interiorColumns: interior, alpha: alpha, beta: 0
        )
        if multiplies.count > 256 { multiplies.removeAll() }
        multiplies[key] = made
        return made
    }

    fileprivate func pipeline(_ name: String) -> MTLComputePipelineState {
        pipelines[name]!
    }
}

/// a place in a buffer
struct Span {
    var buffer: MTLBuffer
    var offset: Int = 0

    init(_ buffer: MTLBuffer, _ offset: Int = 0) {
        self.buffer = buffer
        self.offset = offset
    }
}

/// one command buffer and the kernel calls that go into it, in order.
final class Pass {
    let engine: MetalEngine
    private(set) var commandBuffer: MTLCommandBuffer
    private var open: MTLComputeCommandEncoder?

    init(engine: MetalEngine, commandBuffer: MTLCommandBuffer) {
        self.engine = engine
        self.commandBuffer = commandBuffer
    }

    private var encoder: MTLComputeCommandEncoder {
        if let open { return open }
        let made = commandBuffer.makeComputeCommandEncoder()!
        open = made
        return made
    }

    func close() {
        open?.endEncoding()
        open = nil
    }

    /// in profile mode every kernel call runs in a command buffer of its own, so the gpu's own
    /// clock says where a pass spends its time. off unless the engine's profile is switched on.
    private func mark(_ name: String) {
        guard engine.profile != nil else { return }
        close()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        engine.profile![name, default: 0] += commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        commandBuffer = engine.queue.makeCommandBuffer()!
    }

    private func use(_ name: String) -> MTLComputePipelineState {
        let pipeline = engine.pipeline(name)
        encoder.setComputePipelineState(pipeline)
        return pipeline
    }

    private func dispatch(_ pipeline: MTLComputePipelineState, _ width: Int, _ height: Int = 1, _ depth: Int = 1) {
        guard width > 0, height > 0, depth > 0 else { return }
        encoder.dispatchThreads(
            MTLSize(width: width, height: height, depth: depth),
            threadsPerThreadgroup: MTLSize(width: min(pipeline.threadExecutionWidth, width), height: 1, depth: 1)
        )
        mark(pipeline.label ?? "kernel")
    }

    private func bind(_ span: Span, _ index: Int) {
        encoder.setBuffer(span.buffer, offset: span.offset, index: index)
    }

    private func weight(_ ref: WeightRef, _ index: Int) {
        encoder.setBuffer(ref.buffer, offset: ref.offset, index: index)
    }

    private func params<T>(_ value: T, _ index: Int) {
        withUnsafeBytes(of: value) { encoder.setBytes($0.baseAddress!, length: $0.count, index: index) }
    }

    /// y[n] = w x[n] + bias for `count` rows
    ///
    /// which kernel runs depends only on the shape, `count` and `matrix`, never on what memory is
    /// free: a failed allocation throws rather than fall back to a kernel with other rounding.
    /// - parameter matrix: allows apple's matrix kernel. only for per-piece passes (prefill, encoder):
    ///   its bits for a row depend on the row count, so a decoder step must never use it.
    func gemm(weight w: WeightRef, bias: WeightRef? = nil, rows: Int, cols: Int, x: Span, xStride: Int? = nil, y: Span, yStride: Int? = nil,
              count: Int, matrix: Bool = false) throws {
        guard count > 0 else { return }
        if matrix, count >= engine.wideRows, cols >= 16 {
            let wide = try engine.wideScratch(bytes: rows * cols * 4)
            var pipeline = use("widen")
            weight(w, 0)
            bind(Span(wide), 1)
            params(UInt32(rows * cols / 4), 3)
            dispatch(pipeline, rows * cols / 4)
            close()
            let left = MPSMatrix(buffer: x.buffer, offset: x.offset, descriptor: MPSMatrixDescriptor(
                rows: count, columns: cols, rowBytes: (xStride ?? cols) * 4, dataType: .float32))
            let right = MPSMatrix(buffer: wide, descriptor: MPSMatrixDescriptor(
                rows: rows, columns: cols, rowBytes: cols * 4, dataType: .float32))
            let result = MPSMatrix(buffer: y.buffer, offset: y.offset, descriptor: MPSMatrixDescriptor(
                rows: count, columns: rows, rowBytes: (yStride ?? rows) * 4, dataType: .float32))
            engine.multiply(rows: rows, cols: cols, count: count)
                .encode(commandBuffer: commandBuffer, leftMatrix: left, rightMatrix: right, resultMatrix: result)
            mark("wide multiply")
            if let bias {
                pipeline = use("add_bias")
                bind(y, 0)
                weight(bias, 1)
                params((UInt32(rows), UInt32(count), UInt32(yStride ?? rows)), 3)
                dispatch(pipeline, rows, count)
            }
            return
        }
        if bias == nil, cols % 8 == 0, count >= 2 {
            // three rows per thread share each weight read. bit-identical to the kernel below.
            let pipeline = use("gemm3")
            weight(w, 0)
            bind(x, 1)
            bind(y, 2)
            params((UInt32(rows), UInt32(cols), UInt32(count), UInt32(0), UInt32(xStride ?? cols), UInt32(yStride ?? rows)), 4)
            encoder.dispatchThreads(
                MTLSize(width: rows, height: (count + 2) / 3, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(64, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            )
            mark(rows > 100_000 ? "gemm, output head" : "gemm")
            return
        }
        let pipeline = use("gemm")
        weight(w, 0)
        bind(x, 1)
        bind(y, 2)
        weight(bias ?? w, 3)
        params((UInt32(rows), UInt32(cols), UInt32(count), UInt32(bias == nil ? 0 : 1), UInt32(xStride ?? cols), UInt32(yStride ?? rows)), 4)
        encoder.dispatchThreads(
            MTLSize(width: rows, height: count, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
        mark(rows > 100_000 ? "gemm, output head" : "gemm")
    }

    func layerNorm(x: Span, y: Span, gain: WeightRef, bias: WeightRef, dim: Int, count: Int) {
        let pipeline = use("layer_norm")
        bind(x, 0)
        bind(y, 1)
        weight(gain, 2)
        weight(bias, 3)
        params((UInt32(dim), UInt32(count), Float(1e-5), UInt32(1)), 4)
        dispatch(pipeline, count)
    }

    func rmsNorm(x: Span, y: Span, gain: WeightRef, dim: Int, count: Int, eps: Float) throws {
        precondition(dim % 64 == 0)
        let partial = try engine.scratch("norm.partial", bytes: count * (dim / 64) * 4)
        let shared = (UInt32(dim), UInt32(count), eps, UInt32(0))
        var pipeline = use("rms_partial")
        bind(x, 0)
        bind(Span(partial), 1)
        params(shared, 4)
        dispatch(pipeline, dim / 64, count)
        pipeline = use("rms_scale")
        bind(x, 0)
        bind(y, 1)
        weight(gain, 2)
        bind(Span(partial), 3)
        params(shared, 4)
        dispatch(pipeline, dim / 4, count)
    }

    func headNormRope(x: Span, y: Span, gain: WeightRef, positions: Span, targets: Span, heads: Int, headDim: Int, count: Int, eps: Float) {
        let pipeline = use("head_norm_rope")
        bind(x, 0)
        bind(y, 1)
        weight(gain, 2)
        bind(Span(engine.ropeCos), 3)
        bind(Span(engine.ropeSin), 4)
        bind(positions, 5)
        bind(targets, 6)
        params((UInt32(heads), UInt32(headDim), UInt32(count), eps), 7)
        dispatch(pipeline, heads, count)
    }

    func scatterRows(x: Span, y: Span, targets: Span, dim: Int, count: Int) {
        let pipeline = use("scatter_rows")
        bind(x, 0)
        bind(y, 1)
        bind(targets, 2)
        params((UInt32(dim), UInt32(count)), 3)
        dispatch(pipeline, dim, count)
    }

    func embed(table: WeightRef, y: Span, tokens: Span, dim: Int, count: Int) {
        let pipeline = use("embed")
        weight(table, 0)
        bind(y, 1)
        bind(tokens, 2)
        params((UInt32(dim), UInt32(count)), 3)
        dispatch(pipeline, dim, count)
    }

    /// softmax attention of each query row over its own context, written as one mixed value row per query.
    /// `maxContext` must be a multiple of 64: the softmax and the mix work in runs of 64 slots.
    func attention(q: Span, keys: Span, values: Span, contexts: Span, list: Span, scores: Span, y: Span,
                   heads: Int, keyHeads: Int, headDim: Int, rows: Int, maxContext: Int) throws {
        precondition(maxContext % 64 == 0)
        let scale = 1.0 / Float(headDim).squareRoot()
        let shared = (UInt32(heads), UInt32(keyHeads), UInt32(headDim), UInt32(rows), UInt32(maxContext), scale)

        var pipeline = use("attention_scores")
        bind(q, 0)
        bind(keys, 1)
        bind(contexts, 2)
        bind(list, 3)
        bind(scores, 4)
        params(shared, 5)
        dispatch(pipeline, maxContext, heads, rows)

        let runs = maxContext / 64
        let runMax = try engine.scratch("attention.max", bytes: rows * heads * runs * 4)
        let partial = try engine.scratch("attention.partial", bytes: rows * heads * runs * (headDim + 4) * 4)

        pipeline = use("attention_run_max")
        bind(contexts, 2)
        bind(scores, 4)
        params(shared, 5)
        bind(Span(runMax), 6)
        dispatch(pipeline, runs, heads, rows)

        pipeline = use("attention_exp")
        bind(contexts, 2)
        bind(scores, 4)
        params(shared, 5)
        bind(Span(runMax), 6)
        dispatch(pipeline, maxContext, heads, rows)

        pipeline = use("attention_partial")
        bind(values, 1)
        bind(contexts, 2)
        bind(list, 3)
        bind(scores, 4)
        params(shared, 5)
        bind(Span(partial), 6)
        dispatch(pipeline, runs * (headDim / 4 + 1), heads, rows)

        pipeline = use("attention_combine")
        bind(contexts, 2)
        params(shared, 5)
        bind(Span(partial), 6)
        bind(y, 7)
        dispatch(pipeline, headDim / 4, heads, rows)
    }

    /// causal attention for a prefill, where row t attends to the first t + 1 rows of the same
    /// pass. both products run on apple's matrix kernel over the whole square, one head at a time,
    /// which is several times faster than the per-slot kernels once there are hundreds of rows.
    /// - parameter firstSlot: the cache slot of row 0. rows are in consecutive slots.
    func attentionSquare(q: Span, keys: Span, values: Span, contexts: Span, scores: Span, y: Span,
                         heads: Int, keyHeads: Int, headDim: Int, rows: Int, maxContext: Int, firstSlot: Int) throws {
        precondition(maxContext % 64 == 0 && rows <= maxContext)
        let scale = 1.0 / Float(headDim).squareRoot()
        let shared = (UInt32(heads), UInt32(keyHeads), UInt32(headDim), UInt32(rows), UInt32(maxContext), scale)
        let runs = maxContext / 64
        let runBuffer = try engine.scratch("attention.max", bytes: rows * heads * runs * 4)
        let keyWidth = keyHeads * headDim
        let group = heads / keyHeads
        func matrix(_ span: Span, extra: Int, rows: Int, columns: Int, stride: Int) -> MPSMatrix {
            MPSMatrix(buffer: span.buffer, offset: span.offset + extra * 4,
                      descriptor: MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: stride * 4, dataType: .float32))
        }

        close()
        let scoreProduct = engine.product(resultRows: rows, resultColumns: rows, interior: headDim, transposeRight: true, alpha: Double(scale))
        for head in 0..<heads {
            scoreProduct.encode(
                commandBuffer: commandBuffer,
                leftMatrix: matrix(q, extra: head * headDim, rows: rows, columns: headDim, stride: heads * headDim),
                rightMatrix: matrix(keys, extra: firstSlot * keyWidth + head / group * headDim, rows: rows, columns: headDim, stride: keyWidth),
                resultMatrix: matrix(scores, extra: head * maxContext, rows: rows, columns: rows, stride: heads * maxContext)
            )
        }
        mark("square scores")

        var pipeline = use("attention_run_max")
        bind(contexts, 2)
        bind(scores, 4)
        params(shared, 5)
        bind(Span(runBuffer), 6)
        dispatch(pipeline, runs, heads, rows)

        pipeline = use("attention_exp_causal")
        bind(contexts, 2)
        bind(scores, 4)
        params(shared, 5)
        bind(Span(runBuffer), 6)
        dispatch(pipeline, maxContext, heads, rows)

        pipeline = use("attention_run_total")
        bind(scores, 4)
        params(shared, 5)
        bind(Span(runBuffer), 6)
        dispatch(pipeline, runs, heads, rows)

        close()
        let mixProduct = engine.product(resultRows: rows, resultColumns: headDim, interior: rows, transposeRight: false, alpha: 1)
        for head in 0..<heads {
            mixProduct.encode(
                commandBuffer: commandBuffer,
                leftMatrix: matrix(scores, extra: head * maxContext, rows: rows, columns: rows, stride: heads * maxContext),
                rightMatrix: matrix(values, extra: firstSlot * keyWidth + head / group * headDim, rows: rows, columns: headDim, stride: keyWidth),
                resultMatrix: matrix(y, extra: head * headDim, rows: rows, columns: headDim, stride: heads * headDim)
            )
        }
        mark("square mix")

        pipeline = use("attention_divide")
        params(shared, 5)
        bind(Span(runBuffer), 6)
        bind(y, 7)
        dispatch(pipeline, headDim / 4, heads, rows)
    }

    func add(_ a: Span, _ b: Span, into y: Span, count: Int) {
        let pipeline = use("add")
        bind(a, 0)
        bind(b, 1)
        bind(y, 2)
        params(UInt32(count), 3)
        dispatch(pipeline, count)
    }

    func siluMul(gate: Span, up: Span, into y: Span, count: Int) {
        let pipeline = use("silu_mul")
        bind(gate, 0)
        bind(up, 1)
        bind(y, 2)
        params(UInt32(count), 3)
        dispatch(pipeline, count)
    }

    func gelu(_ x: Span, count: Int) {
        let pipeline = use("gelu")
        bind(x, 0)
        params(UInt32(count), 3)
        dispatch(pipeline, count)
    }

    func patches(x: Span, y: Span, chunks: Int, height: Int, width: Int, channels: Int) {
        let pipeline = use("patches")
        bind(x, 0)
        bind(y, 1)
        let outHeight = (height - 1) / 2 + 1
        let outWidth = (width - 1) / 2 + 1
        params((UInt32(chunks), UInt32(height), UInt32(width), UInt32(channels), UInt32(outHeight), UInt32(outWidth)), 2)
        dispatch(pipeline, channels * 9, outHeight * outWidth, chunks)
    }

    func flattenFrequency(x: Span, y: Span, chunks: Int, height: Int, width: Int, channels: Int) {
        let pipeline = use("flatten_frequency")
        bind(x, 0)
        bind(y, 1)
        params((UInt32(chunks), UInt32(height), UInt32(width), UInt32(channels), UInt32(0), UInt32(0)), 2)
        dispatch(pipeline, channels * height, width, chunks)
    }

    func addPositions(x: Span, positions: Span, dim: Int, count: Int, period: Int) {
        let pipeline = use("add_positions")
        bind(x, 0)
        bind(positions, 1)
        params((UInt32(dim), UInt32(count), UInt32(period)), 2)
        dispatch(pipeline, dim, count)
    }
}
