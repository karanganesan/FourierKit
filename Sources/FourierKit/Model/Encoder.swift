import Foundation
import Metal

/// the audio tokens of one piece, ready for the decoder: [count][width] 32-bit floats.
struct AudioEmbedding: @unchecked Sendable {
    let buffer: MTLBuffer
    let count: Int
    let width: Int

    func values() -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: Float.self, capacity: count * width), count: count * width))
    }
}

extension MetalEngine {
    /// wraps audio tokens another chip produced so the decoder can read them
    func audioEmbedding(from values: [Float]) throws -> AudioEmbedding {
        let width = shape.audioWidth
        guard !values.isEmpty, values.count % width == 0,
              let buffer = device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared)
        else { throw EngineFailure.gpu("the audio tokens could not be allocated") }
        return AudioEmbedding(buffer: buffer, count: values.count / width, width: width)
    }

    /// how many audio tokens a clip with this many valid mel frames becomes
    func audioTokenCount(frames: Int) -> Int {
        let rest = frames % shape.chunkFrames
        var tail = rest
        if tail > 0 { for _ in 0..<3 { tail = (tail - 1) / 2 + 1 } }
        return frames / shape.chunkFrames * shape.tokensPerChunk + tail
    }

    /// conv front, 24 layers and the projector for one clip.
    /// - parameter mel: [bin][paddedFrame], padded with zeros to a whole number of 1-second chunks
    func encode(mel: [Float], frames: Int, paddedFrames: Int) async throws -> AudioEmbedding {
        let bins = shape.melBins
        let chunkFrames = shape.chunkFrames
        precondition(paddedFrames % chunkFrames == 0 && mel.count == bins * paddedFrames)
        let chunks = paddedFrames / chunkFrames
        let channels = shape.convChannels
        let width = shape.encoderWidth
        let perChunk = shape.tokensPerChunk
        let tokens = audioTokenCount(frames: frames)
        guard tokens > 0 else { throw EngineFailure.tooLong("the audio is empty") }

        // sizes after each stride-2 convolution: frequency 128, 64, 32, 16 and time 100, 50, 25, 13
        var heights = [bins]
        var widths = [chunkFrames]
        for _ in 0..<3 {
            heights.append((heights.last! - 1) / 2 + 1)
            widths.append((widths.last! - 1) / 2 + 1)
        }
        let cells = (0...3).map { heights[$0] * widths[$0] }
        let flat = channels * heights[3]
        let group = limits.convGroup

        let x = try scratch("enc.x", bytes: chunks * perChunk * width * 4)
        let input = try scratch("enc.in", bytes: group * cells[0] * 4)
        let patch1 = try scratch("enc.p1", bytes: group * cells[1] * 9 * 4)
        let act1 = try scratch("enc.a1", bytes: group * cells[1] * channels * 4)
        let patch2 = try scratch("enc.p2", bytes: group * cells[2] * channels * 9 * 4)
        let act2 = try scratch("enc.a2", bytes: group * cells[2] * channels * 4)
        let patch3 = try scratch("enc.p3", bytes: group * cells[3] * channels * 9 * 4)
        let act3 = try scratch("enc.a3", bytes: group * cells[3] * channels * 4)
        let flattened = try scratch("enc.flat", bytes: group * perChunk * flat * 4)

        let conv = "model.audio_tower."
        let w1 = try weight(conv + "conv2d1.weight"), b1 = try weight(conv + "conv2d1.bias")
        let w2 = try weight(conv + "conv2d2.weight"), b2 = try weight(conv + "conv2d2.bias")
        let w3 = try weight(conv + "conv2d3.weight"), b3 = try weight(conv + "conv2d3.bias")
        let wOut = try weight(conv + "conv_out.weight")

        var start = 0
        while start < chunks {
            let n = min(group, chunks - start)
            // the gpu has finished the last group by now, so the input buffer is free to refill
            let destination = input.contents().bindMemory(to: Float.self, capacity: group * cells[0])
            mel.withUnsafeBufferPointer { source in
                for chunk in 0..<n {
                    for bin in 0..<bins {
                        let from = source.baseAddress! + bin * paddedFrames + (start + chunk) * chunkFrames
                        (destination + (chunk * bins + bin) * chunkFrames).update(from: from, count: chunkFrames)
                    }
                }
            }
            let first = start
            try await run { pass in
                pass.patches(x: Span(input), y: Span(patch1), chunks: n, height: heights[0], width: widths[0], channels: 1)
                pass.gemm(weight: w1, bias: b1, rows: channels, cols: 9, x: Span(patch1), y: Span(act1), count: n * cells[1])
                pass.gelu(Span(act1), count: n * cells[1] * channels)
                pass.patches(x: Span(act1), y: Span(patch2), chunks: n, height: heights[1], width: widths[1], channels: channels)
                pass.gemm(weight: w2, bias: b2, rows: channels, cols: channels * 9, x: Span(patch2), y: Span(act2), count: n * cells[2])
                pass.gelu(Span(act2), count: n * cells[2] * channels)
                pass.patches(x: Span(act2), y: Span(patch3), chunks: n, height: heights[2], width: widths[2], channels: channels)
                pass.gemm(weight: w3, bias: b3, rows: channels, cols: channels * 9, x: Span(patch3), y: Span(act3), count: n * cells[3])
                pass.gelu(Span(act3), count: n * cells[3] * channels)
                pass.flattenFrequency(x: Span(act3), y: Span(flattened), chunks: n, height: heights[3], width: widths[3], channels: channels)
                pass.gemm(weight: wOut, rows: width, cols: flat, x: Span(flattened), y: Span(x, first * perChunk * width * 4), count: n * perChunk)
            }
            start += n
        }

        // only the last chunk can be partial, so the valid tokens are the first `tokens` rows
        let window = shape.windowTokens
        let stride = (window + 63) / 64 * 64
        let heads = shape.encoderHeads
        let headDim = width / heads
        let contexts = try scratch("enc.ctx", bytes: tokens * 16)
        let contextPointer = contexts.contents().bindMemory(to: UInt32.self, capacity: tokens * 4)
        for token in 0..<tokens {
            let windowStart = token / window * window
            contextPointer[token * 4] = UInt32(windowStart)
            contextPointer[token * 4 + 1] = UInt32(min(window, tokens - windowStart))
            contextPointer[token * 4 + 2] = 0
            contextPointer[token * 4 + 3] = 0
        }
        let h = try scratch("enc.h", bytes: tokens * width * 4)
        let q = try scratch("enc.q", bytes: tokens * width * 4)
        let k = try scratch("enc.k", bytes: tokens * width * 4)
        let v = try scratch("enc.v", bytes: tokens * width * 4)
        let mixed = try scratch("enc.mix", bytes: tokens * width * 4)
        let o = try scratch("enc.o", bytes: tokens * width * 4)
        let inner = try scratch("enc.inner", bytes: tokens * shape.encoderInner * 4)
        let scores = try scratch("enc.scores", bytes: tokens * heads * stride * 4)
        let output = try device.makeBuffer(length: tokens * shape.audioWidth * 4, options: .storageModeShared)
            ?? { throw EngineFailure.gpu("the audio tokens could not be allocated") }()

        var bodies: [(Pass) throws -> Void] = []
        let positions = encoderPositions
        bodies.append { pass in
            pass.addPositions(x: Span(x), positions: Span(positions), dim: width, count: tokens, period: perChunk)
        }
        for layer in 0..<shape.encoderLayers {
            let p = "model.audio_tower.layers.\(layer)."
            let ln1 = (try weight(p + "self_attn_layer_norm.weight"), try weight(p + "self_attn_layer_norm.bias"))
            let ln2 = (try weight(p + "final_layer_norm.weight"), try weight(p + "final_layer_norm.bias"))
            let wq = (try weight(p + "self_attn.q_proj.weight"), try weight(p + "self_attn.q_proj.bias"))
            let wk = (try weight(p + "self_attn.k_proj.weight"), try weight(p + "self_attn.k_proj.bias"))
            let wv = (try weight(p + "self_attn.v_proj.weight"), try weight(p + "self_attn.v_proj.bias"))
            let wo = (try weight(p + "self_attn.out_proj.weight"), try weight(p + "self_attn.out_proj.bias"))
            let fc1 = (try weight(p + "fc1.weight"), try weight(p + "fc1.bias"))
            let fc2 = (try weight(p + "fc2.weight"), try weight(p + "fc2.bias"))
            let innerWidth = shape.encoderInner
            bodies.append { pass in
                pass.layerNorm(x: Span(x), y: Span(h), gain: ln1.0, bias: ln1.1, dim: width, count: tokens)
                pass.gemm(weight: wq.0, bias: wq.1, rows: width, cols: width, x: Span(h), y: Span(q), count: tokens)
                pass.gemm(weight: wk.0, bias: wk.1, rows: width, cols: width, x: Span(h), y: Span(k), count: tokens)
                pass.gemm(weight: wv.0, bias: wv.1, rows: width, cols: width, x: Span(h), y: Span(v), count: tokens)
                pass.attention(q: Span(q), keys: Span(k), values: Span(v), contexts: Span(contexts), list: Span(contexts),
                               scores: Span(scores), y: Span(mixed), heads: heads, keyHeads: heads, headDim: headDim,
                               rows: tokens, maxContext: stride)
                pass.gemm(weight: wo.0, bias: wo.1, rows: width, cols: width, x: Span(mixed), y: Span(o), count: tokens)
                pass.add(Span(x), Span(o), into: Span(x), count: tokens * width)
                pass.layerNorm(x: Span(x), y: Span(h), gain: ln2.0, bias: ln2.1, dim: width, count: tokens)
                pass.gemm(weight: fc1.0, bias: fc1.1, rows: innerWidth, cols: width, x: Span(h), y: Span(inner), count: tokens)
                pass.gelu(Span(inner), count: tokens * innerWidth)
                pass.gemm(weight: fc2.0, bias: fc2.1, rows: width, cols: innerWidth, x: Span(inner), y: Span(o), count: tokens)
                pass.add(Span(x), Span(o), into: Span(x), count: tokens * width)
            }
        }
        let post = (try weight("model.audio_tower.ln_post.weight"), try weight("model.audio_tower.ln_post.bias"))
        let p1 = (try weight("model.multi_modal_projector.linear_1.weight"), try weight("model.multi_modal_projector.linear_1.bias"))
        let p2 = (try weight("model.multi_modal_projector.linear_2.weight"), try weight("model.multi_modal_projector.linear_2.bias"))
        let audioWidth = shape.audioWidth
        bodies.append { pass in
            pass.layerNorm(x: Span(x), y: Span(h), gain: post.0, bias: post.1, dim: width, count: tokens)
            pass.gemm(weight: p1.0, bias: p1.1, rows: width, cols: width, x: Span(h), y: Span(o), count: tokens)
            pass.gelu(Span(o), count: tokens * width)
            pass.gemm(weight: p2.0, bias: p2.1, rows: audioWidth, cols: width, x: Span(o), y: Span(output), count: tokens)
        }
        try await runAll(bodies)
        return AudioEmbedding(buffer: output, count: tokens, width: audioWidth)
    }
}
