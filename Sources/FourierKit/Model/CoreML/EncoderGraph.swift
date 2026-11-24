import Foundation

/// the conv front, 24 encoder layers and the projector for one 8-second window, as an ml program.
///
/// inputs: `mel` [8 chunks][1][128 bins][100 frames] and `mask` [1][1][104], an additive attention mask
/// that is 0 for valid tokens and very negative for padded ones. output: `audio` [104][2048].
/// io stays float32 and the graph casts to float16 inside, so one caller serves both precisions.
struct EncoderGraph {
    static let formatVersion = 3
    static let chunks = 8
    static let bins = 128
    static let chunkFrames = 100
    static let tokensPerChunk = 13
    static let windowTokens = chunks * tokensPerChunk
    static let maskValue: Float = -10_000

    /// how the token stream is laid out between layers
    enum Layout: String, Sendable {
        /// [token][feature] with linear ops, the plain transcription of the reference
        case rows
        /// [1][feature][1][token] with 1x1 convolutions, the layout apple recommends for transformers on the neural engine
        case channels
    }

    enum Failure: Error, CustomStringConvertible {
        case weights(String)
        var description: String {
            switch self {
            case .weights(let why): "the encoder weights are not what core ml conversion expects: \(why)"
            }
        }
    }

    let weights: SafeTensors
    let compute: MILType
    let layout: Layout
    /// diagnostic: end after this many layers and return the raw [104][1024] stream, to find where precision drifts
    var layerLimit: Int? = nil

    /// streams the weights into the package's blob and writes the program. returns what float16 lost.
    func write(to package: MLPackageWriter) throws -> Float16Loss {
        try package.prepare()
        let blobs = try WeightBlobWriter(url: package.weightsURL)
        let b = MILBuilder(compute: compute, blobs: blobs)

        let tower = "model.audio_tower."
        let channels = try weights.tensor(tower + "conv2d1.weight").shape[0]
        let width = try weights.tensor(tower + "conv_out.weight").shape[0]
        let heads = 16
        let headDim = width / heads
        var layers = 0
        while weights.has(tower + "layers.\(layers).fc1.weight") { layers += 1 }
        let audioWidth = try weights.tensor("model.multi_modal_projector.linear_2.weight").shape[0]
        let tokens = Self.windowTokens

        let melShape = [Self.chunks, 1, Self.bins, Self.chunkFrames]
        let inputs = [
            MLPackageWriter.Feature(name: "mel", type: .float32, shape: melShape),
            MLPackageWriter.Feature(name: "mask", type: .float32, shape: [1, 1, tokens]),
        ]
        var mel = MILValue(name: "mel", type: .float32, shape: melShape)
        var mask = MILValue(name: "mask", type: .float32, shape: [1, 1, tokens])
        if compute != .float32 {
            mel = b.cast(mel, to: compute, name: "mel_cast")
            mask = b.cast(mask, to: compute, name: "mask_cast")
        }

        // conv front: every 1-second chunk is a batch item, so chunks never mix
        var x = mel
        for index in 1...3 {
            let w = try param(b, tower + "conv2d\(index).weight")
            let bias = try param(b, tower + "conv2d\(index).bias")
            x = b.conv(x, weight: w, bias: bias, stride: 2, pad: 1, name: "conv\(index)")
            if layerLimit == index - 7 { return try finishEarly(b, blobs, package, inputs, x) }
            x = gelu(b, x, name: "conv\(index)_gelu")
            if layerLimit == index - 4 { return try finishEarly(b, blobs, package, inputs, x) }
        }
        let freq = x.shape[2], time = x.shape[3]
        precondition(time == Self.tokensPerChunk)
        let positionTable = Self.positions(width: width)
        switch layout {
        case .rows:
            // [chunk][channel][freq][time] to [chunk * time][channel * freq]
            x = b.reshape(x, [Self.chunks, channels * freq, time])
            x = b.transpose(x, [0, 2, 1])
            x = b.reshape(x, [tokens, channels * freq])
            x = b.linear(x, weight: try param(b, tower + "conv_out.weight"), bias: nil, name: "conv_out")
            let positions = try b.tensor("positions", shape: [tokens, width], checkpoint: false) { out in
                for row in 0..<tokens {
                    for column in 0..<width { out[row * width + column] = positionTable[(row % Self.tokensPerChunk) * width + column] }
                }
            }
            x = b.add(x, positions, name: "embedded")
        case .channels:
            // [chunk][channel * freq][1][time], projected, then chunks joined along the token axis
            x = b.reshape(x, [Self.chunks, channels * freq, 1, time])
            x = b.conv(x, weight: try param(b, tower + "conv_out.weight", pointwise: true), bias: nil, stride: 1, pad: 0, name: "conv_out")
            x = b.transpose(x, [1, 0, 2, 3])
            x = b.reshape(x, [1, width, 1, tokens])
            let positions = try b.tensor("positions", shape: [1, width, 1, tokens], checkpoint: false) { out in
                for column in 0..<width {
                    for row in 0..<tokens { out[column * tokens + row] = positionTable[(row % Self.tokensPerChunk) * width + column] }
                }
            }
            x = b.add(x, positions, name: "embedded")
        }
        let featureAxis = layout == .rows ? -1 : 1

        if let layerLimit { layers = min(layers, layerLimit) }
        for layer in 0..<layers {
            let p = tower + "layers.\(layer)."
            let n = "l\(layer)_"
            var h = b.layerNorm(x, axis: featureAxis, gamma: try param(b, p + "self_attn_layer_norm.weight"),
                                beta: try param(b, p + "self_attn_layer_norm.bias"), epsilon: 1e-5, name: n + "ln1")
            let scale = b.scalar(1.0 / Float(headDim).squareRoot(), name: "scale")
            let q0 = try dense(b, h, p + "self_attn.q_proj", name: n + "q")
            let k0 = try dense(b, h, p + "self_attn.k_proj", name: n + "k")
            let v0 = try dense(b, h, p + "self_attn.v_proj", name: n + "v")
            var mixed: MILValue
            switch layout {
            case .rows:
                func split(_ y: MILValue) -> MILValue {
                    b.transpose(b.reshape(y, [tokens, heads, headDim]), [1, 0, 2], name: y.name + "_heads")
                }
                // the 1/8 scale is a power of two, so applying it to q instead of the scores changes no bits in float32
                let q = b.mul(split(q0), scale, name: n + "q_scaled")
                var scores = b.matmul(q, split(k0), transposeY: true, name: n + "scores")
                scores = b.add(scores, mask, name: n + "masked")
                let attention = b.softmax(scores, axis: -1, name: n + "softmax")
                mixed = b.matmul(attention, split(v0), name: n + "mixed")
                mixed = b.reshape(b.transpose(mixed, [1, 0, 2]), [tokens, width], name: n + "merged")
            case .channels:
                // [head][dim][token]: scores come out as [head][query][key] with keys on the last axis
                let q = b.mul(b.reshape(q0, [heads, headDim, tokens]), scale, name: n + "q_scaled")
                let k = b.reshape(k0, [heads, headDim, tokens])
                let v = b.reshape(v0, [heads, headDim, tokens])
                var scores = b.matmul(q, k, transposeX: true, name: n + "scores")
                scores = b.add(scores, mask, name: n + "masked")
                let attention = b.softmax(scores, axis: -1, name: n + "softmax")
                mixed = b.matmul(v, attention, transposeY: true, name: n + "mixed")
                mixed = b.reshape(mixed, [1, width, 1, tokens], name: n + "merged")
            }
            let attended = try dense(b, mixed, p + "self_attn.out_proj", name: n + "out")
            x = b.add(x, attended, name: n + "residual1")
            h = b.layerNorm(x, axis: featureAxis, gamma: try param(b, p + "final_layer_norm.weight"),
                            beta: try param(b, p + "final_layer_norm.bias"), epsilon: 1e-5, name: n + "ln2")
            var f = try dense(b, h, p + "fc1", name: n + "fc1")
            f = gelu(b, f, name: n + "fc1_gelu")
            f = try dense(b, f, p + "fc2", name: n + "fc2")
            x = b.add(x, f, name: n + "residual2")
        }

        if layerLimit != nil {
            if layout == .channels { x = b.transpose(b.reshape(x, [width, tokens]), [1, 0], name: "tokens_first") }
            return try finishEarly(b, blobs, package, inputs, x)
        }
        x = b.layerNorm(x, axis: featureAxis, gamma: try param(b, tower + "ln_post.weight"),
                        beta: try param(b, tower + "ln_post.bias"), epsilon: 1e-5, name: "ln_post")
        let projector = "model.multi_modal_projector."
        x = try dense(b, x, projector + "linear_1", name: "linear_1")
        x = gelu(b, x, name: "linear_1_gelu")
        x = try dense(b, x, projector + "linear_2", name: "linear_2")
        if layout == .channels {
            x = b.transpose(b.reshape(x, [audioWidth, tokens]), [1, 0], name: "tokens_first")
        }
        let outputName = "audio"
        x = compute == .float32
            ? b.op("identity", ["x": x], shape: x.shape, name: outputName)
            : b.cast(x, to: .float32, name: outputName)
        precondition(x.name == outputName && x.shape == [tokens, audioWidth])

        try blobs.finish()
        try package.finish(
            inputs: inputs,
            outputs: [MLPackageWriter.Feature(name: outputName, type: .float32, shape: x.shape)],
            operations: b.operations,
            metadata: ["fourierkit.encoder.format": "\(Self.formatVersion)", "fourierkit.encoder.compute": compute.castName,
                       "fourierkit.encoder.layout": layout.rawValue]
        )
        return b.loss
    }

    /// gelu spelled out with the erf op. the neural engine runs the gelu op as an approximation in both modes,
    /// off by up to 7e-3 after the first convolution, while this form stays within float16 rounding.
    private func gelu(_ b: MILBuilder, _ x: MILValue, name: String) -> MILValue {
        let e = b.op("erf", ["x": b.mul(x, b.scalar(Float(0.5.squareRoot()), name: "rsqrt2"))], shape: x.shape, name: name + "_erf")
        let half = b.mul(x, b.scalar(0.5, name: "half"))
        return b.add(half, b.mul(half, e), name: name)
    }

    private func finishEarly(_ b: MILBuilder, _ blobs: WeightBlobWriter, _ package: MLPackageWriter,
                             _ inputs: [MLPackageWriter.Feature], _ x: MILValue) throws -> Float16Loss {
        let y = compute == .float32 ? b.op("identity", ["x": x], shape: x.shape, name: "audio") : b.cast(x, to: .float32, name: "audio")
        try blobs.finish()
        try package.finish(inputs: inputs, outputs: [MLPackageWriter.Feature(name: "audio", type: .float32, shape: y.shape)],
                           operations: b.operations, metadata: [:])
        return b.loss
    }

    /// a checkpoint linear layer: a linear op on rows, a 1x1 convolution on channels
    private func dense(_ b: MILBuilder, _ x: MILValue, _ prefix: String, name: String) throws -> MILValue {
        let bias = try param(b, prefix + ".bias")
        switch layout {
        case .rows:
            return b.linear(x, weight: try param(b, prefix + ".weight"), bias: bias, name: name)
        case .channels:
            return b.conv(x, weight: try param(b, prefix + ".weight", pointwise: true), bias: bias, stride: 1, pad: 0, name: name)
        }
    }

    /// one checkpoint tensor, widened from bf16 exactly, then stored at the graph's precision.
    /// `pointwise` turns an [out][in] matrix into [out][in][1][1] convolution weights, the same bytes.
    private func param(_ b: MILBuilder, _ name: String, pointwise: Bool = false) throws -> MILValue {
        let tensor = try weights.tensor(name)
        let source = weights.pointer(tensor)
        let label = name.replacingOccurrences(of: "model.audio_tower.", with: "")
            .replacingOccurrences(of: "model.multi_modal_projector.", with: "projector.")
        return try b.tensor(label, shape: pointwise ? tensor.shape + [1, 1] : tensor.shape) { out in
            switch tensor.dtype {
            case "BF16":
                for i in 0..<tensor.count {
                    out[i] = Float(bitPattern: UInt32(source.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)) << 16)
                }
            case "F32":
                for i in 0..<tensor.count { out[i] = source.loadUnaligned(fromByteOffset: i * 4, as: Float.self) }
            default:
                throw Failure.weights("\(name) is \(tensor.dtype)")
            }
        }
    }

    /// whisper-style sinusoids, computed in float32 the way the reference computes them
    static func positions(width: Int) -> [Float] {
        let half = width / 2
        let increment = Float(log(10_000.0) / Double(half - 1))
        var table = [Float](repeating: 0, count: tokensPerChunk * width)
        for position in 0..<tokensPerChunk {
            for index in 0..<half {
                let scaled = Float(position) * expf(-increment * Float(index))
                table[position * width + index] = sinf(scaled)
                table[position * width + half + index] = cosf(scaled)
            }
        }
        return table
    }
}
