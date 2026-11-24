import Foundation
import Metal

/// one audio piece inside the decoder: its run of cache slots and which slots make up each beam.
///
/// beam history is never copied. the prompt is one shared run of slots, every step writes each
/// row to a slot of its own, and a beam is just the list of slots it came through.
final class DecoderPiece: @unchecked Sendable {
    let slots: Range<Int>
    let promptLength: Int
    let beams: Int
    /// per running beam, the step slots it came through, oldest first
    fileprivate(set) var chains: [[Int32]]
    fileprivate(set) var steps = 0

    fileprivate init(slots: Range<Int>, promptLength: Int, beams: Int) {
        self.slots = slots
        self.promptLength = promptLength
        self.beams = beams
        chains = Array(repeating: [], count: beams)
    }

    /// cache slots in use: the prompt plus every step row so far
    var usedSlots: Int { promptLength + steps * beams }

    /// after beam search picks its survivors: new beam i continues the row `parents[i]`.
    func reorder(parents: [Int]) {
        chains = parents.map { chains[$0] }
    }
}

/// rows of next-token logits in a gpu work buffer. valid until the next call of the same kind.
struct LogitRows: @unchecked Sendable {
    let values: UnsafeBufferPointer<Float>
}

extension MetalEngine {
    private func takeSlots(_ count: Int) throws -> Range<Int> {
        guard let index = freeSlots.firstIndex(where: { $0.count >= count }) else { throw EngineFailure.outOfSlots }
        let free = freeSlots[index]
        let taken = free.lowerBound..<(free.lowerBound + count)
        if taken.upperBound == free.upperBound {
            freeSlots.remove(at: index)
        } else {
            freeSlots[index] = taken.upperBound..<free.upperBound
        }
        return taken
    }

    /// hands a finished piece's slots back, so the next piece can take them at once.
    func release(_ piece: DecoderPiece) {
        freeSlots.append(piece.slots)
        freeSlots.sort { $0.lowerBound < $1.lowerBound }
        var merged: [Range<Int>] = []
        for range in freeSlots {
            if let last = merged.last, last.upperBound == range.lowerBound {
                merged[merged.count - 1] = last.lowerBound..<range.upperBound
            } else {
                merged.append(range)
            }
        }
        freeSlots = merged
    }

    /// free cache slots, for the scheduler's admission check
    var largestFreeRun: Int { freeSlots.map(\.count).max() ?? 0 }

    func slotsNeeded(promptLength: Int, beams: Int, maxNewTokens: Int) -> Int {
        promptLength + beams * maxNewTokens
    }

    private struct LayerOffsets {
        var norm1 = 0, norm2 = 0, q = 0, k = 0, v = 0, o = 0, qNorm = 0, kNorm = 0, gate = 0, up = 0, down = 0
    }

    private func layerOffsets(_ layer: Int) throws -> LayerOffsets {
        let p = "model.language_model.layers.\(layer)."
        return LayerOffsets(
            norm1: try offset(p + "input_layernorm.weight"),
            norm2: try offset(p + "post_attention_layernorm.weight"),
            q: try offset(p + "self_attn.q_proj.weight"),
            k: try offset(p + "self_attn.k_proj.weight"),
            v: try offset(p + "self_attn.v_proj.weight"),
            o: try offset(p + "self_attn.o_proj.weight"),
            qNorm: try offset(p + "self_attn.q_norm.weight"),
            kNorm: try offset(p + "self_attn.k_norm.weight"),
            gate: try offset(p + "mlp.gate_proj.weight"),
            up: try offset(p + "mlp.up_proj.weight"),
            down: try offset(p + "mlp.down_proj.weight")
        )
    }

    /// the 28 layers over `rows` rows whose embeddings are already in the `x` work buffer.
    /// each row writes its key and value to its own slot and attends over its own context.
    private func layerBodies(prefix: String, rows: Int, maxContext: Int, squareFrom firstSlot: Int? = nil) throws -> [(Pass) throws -> Void] {
        let width = shape.width
        let keyWidth = shape.keyWidth
        let inner = shape.inner
        let heads = shape.heads
        let keyHeads = shape.keyHeads
        let headDim = shape.headDim
        let eps = shape.normEps
        let x = try scratch(prefix + "x", bytes: rows * width * 4)
        let h = try scratch(prefix + "h", bytes: rows * width * 4)
        let qRaw = try scratch(prefix + "qraw", bytes: rows * width * 4)
        let q = try scratch(prefix + "q", bytes: rows * width * 4)
        let kRaw = try scratch(prefix + "kraw", bytes: rows * keyWidth * 4)
        let vRaw = try scratch(prefix + "vraw", bytes: rows * keyWidth * 4)
        let mixed = try scratch(prefix + "mix", bytes: rows * width * 4)
        let o = try scratch(prefix + "o", bytes: rows * width * 4)
        let gate = try scratch(prefix + "gate", bytes: rows * inner * 4)
        let up = try scratch(prefix + "up", bytes: rows * inner * 4)
        let scores = try scratch(prefix + "scores", bytes: rows * heads * maxContext * 4)
        let positions = try scratch(prefix + "pos", bytes: rows * 4)
        let targets = try scratch(prefix + "slot", bytes: rows * 4)
        let identity = try scratch(prefix + "row", bytes: rows * 4)
        let contexts = try scratch(prefix + "ctx", bytes: rows * 16)
        let list = try scratch(prefix + "list", bytes: 16)
        let layerBytes = slotCapacity * keyWidth * 4
        let keys = keyArena
        let values = valueArena

        var bodies: [(Pass) throws -> Void] = []
        for layer in 0..<shape.decoderLayers {
            let w = try layerOffsets(layer)
            let keySpan = Span(keys, layer * layerBytes)
            let valueSpan = Span(values, layer * layerBytes)
            bodies.append { pass in
                pass.rmsNorm(x: Span(x), y: Span(h), gain: w.norm1, dim: width, count: rows, eps: eps)
                pass.gemm(weight: w.q, rows: width, cols: width, x: Span(h), y: Span(qRaw), count: rows)
                pass.gemm(weight: w.k, rows: keyWidth, cols: width, x: Span(h), y: Span(kRaw), count: rows)
                pass.gemm(weight: w.v, rows: keyWidth, cols: width, x: Span(h), y: Span(vRaw), count: rows)
                pass.headNormRope(x: Span(qRaw), y: Span(q), gain: w.qNorm, positions: Span(positions), targets: Span(identity),
                                  heads: heads, headDim: headDim, count: rows, eps: eps)
                pass.headNormRope(x: Span(kRaw), y: keySpan, gain: w.kNorm, positions: Span(positions), targets: Span(targets),
                                  heads: keyHeads, headDim: headDim, count: rows, eps: eps)
                pass.scatterRows(x: Span(vRaw), y: valueSpan, targets: Span(targets), dim: keyWidth, count: rows)
                if let firstSlot {
                    pass.attentionSquare(q: Span(q), keys: keySpan, values: valueSpan, contexts: Span(contexts),
                                         scores: Span(scores), y: Span(mixed), heads: heads, keyHeads: keyHeads, headDim: headDim,
                                         rows: rows, maxContext: maxContext, firstSlot: firstSlot)
                } else {
                    pass.attention(q: Span(q), keys: keySpan, values: valueSpan, contexts: Span(contexts), list: Span(list),
                                   scores: Span(scores), y: Span(mixed), heads: heads, keyHeads: keyHeads, headDim: headDim,
                                   rows: rows, maxContext: maxContext)
                }
                pass.gemm(weight: w.o, rows: width, cols: width, x: Span(mixed), y: Span(o), count: rows)
                pass.add(Span(x), Span(o), into: Span(x), count: rows * width)
                pass.rmsNorm(x: Span(x), y: Span(h), gain: w.norm2, dim: width, count: rows, eps: eps)
                pass.gemm(weight: w.gate, rows: inner, cols: width, x: Span(h), y: Span(gate), count: rows)
                pass.gemm(weight: w.up, rows: inner, cols: width, x: Span(h), y: Span(up), count: rows)
                pass.siluMul(gate: Span(gate), up: Span(up), into: Span(gate), count: rows * inner)
                pass.gemm(weight: w.down, rows: width, cols: inner, x: Span(gate), y: Span(o), count: rows)
                pass.add(Span(x), Span(o), into: Span(x), count: rows * width)
            }
        }
        return bodies
    }

    /// the output head for `count` rows of the `x` work buffer starting at `firstRow`.
    private func headBody(prefix: String, firstRow: Int, count: Int) throws -> (Pass) throws -> Void {
        let width = shape.width
        let vocabulary = shape.vocabulary
        let eps = shape.normEps
        let x = try scratch(prefix + "x", bytes: 16)
        let h = try scratch(prefix + "h", bytes: 16)
        let logits = try scratch(prefix + "logits", bytes: count * vocabulary * 4)
        let norm = try offset("model.language_model.norm.weight")
        let table = try offset("model.language_model.embed_tokens.weight")
        return { pass in
            pass.rmsNorm(x: Span(x, firstRow * width * 4), y: Span(h), gain: norm, dim: width, count: count, eps: eps)
            pass.gemm(weight: table, rows: vocabulary, cols: width, x: Span(h), y: Span(logits), count: count)
        }
    }

    /// reads the prompt and all audio tokens once. every beam of the piece then shares that cache.
    /// - returns: the piece, and the next-token logits after the prompt (one row of `vocabulary` floats,
    ///   valid until the next prefill)
    func prefill(prompt: [Int32], audio: AudioEmbedding, beams: Int, maxNewTokens: Int) async throws -> (DecoderPiece, LogitRows) {
        let rows = prompt.count
        guard rows + maxNewTokens <= maxPositions else {
            throw EngineFailure.tooLong("a piece of \(rows) prompt tokens is too long for one pass")
        }
        guard let audioStart = prompt.firstIndex(of: shape.audioTokenID),
              prompt[audioStart..<min(prompt.count, audioStart + audio.count)].allSatisfy({ $0 == shape.audioTokenID }),
              prompt.filter({ $0 == shape.audioTokenID }).count == audio.count
        else { throw EngineFailure.badConfig("the prompt's audio placeholders do not match the audio tokens") }

        let piece = DecoderPiece(
            slots: try takeSlots(slotsNeeded(promptLength: rows, beams: beams, maxNewTokens: maxNewTokens)),
            promptLength: rows, beams: beams
        )
        let prefix = "pre."
        let width = shape.width
        do {
            // from a few dozen rows up, the square products go to apple's matrix kernel
            let square = rows >= squareRows ? piece.slots.lowerBound : nil
            var bodies = try layerBodies(prefix: prefix, rows: rows, maxContext: (rows + 63) / 64 * 64, squareFrom: square)
            let x = try scratch(prefix + "x", bytes: 16)
            let tokens = try scratch(prefix + "tok", bytes: rows * 4)
            let audioRows = try scratch(prefix + "arow", bytes: audio.count * 4)
            fill(tokens, prompt)
            fill(audioRows, (0..<audio.count).map { Int32(audioStart + $0) })
            fill(try scratch(prefix + "pos", bytes: 16), (0..<rows).map { Int32($0) })
            fill(try scratch(prefix + "row", bytes: 16), (0..<rows).map { Int32($0) })
            fill(try scratch(prefix + "slot", bytes: 16), (0..<rows).map { Int32(piece.slots.lowerBound + $0) })
            let contexts = try scratch(prefix + "ctx", bytes: 16).contents().bindMemory(to: UInt32.self, capacity: rows * 4)
            for row in 0..<rows {
                // causal: row t sees slots 0...t of this piece
                contexts[row * 4] = UInt32(piece.slots.lowerBound)
                contexts[row * 4 + 1] = UInt32(row + 1)
                contexts[row * 4 + 2] = 0
                contexts[row * 4 + 3] = 0
            }
            let table = try offset("model.language_model.embed_tokens.weight")
            let audioCount = audio.count
            let audioBuffer = audio.buffer
            bodies.insert({ pass in
                pass.embed(table: table, y: Span(x), tokens: Span(tokens), dim: width, count: rows)
                pass.scatterRows(x: Span(audioBuffer), y: Span(x), targets: Span(audioRows), dim: width, count: audioCount)
            }, at: 0)
            bodies.append(try headBody(prefix: prefix, firstRow: rows - 1, count: 1))
            try await runAll(bodies)
        } catch {
            release(piece)
            throw error
        }
        let logits = try scratch(prefix + "logits", bytes: 16)
        let pointer = logits.contents().bindMemory(to: Float.self, capacity: shape.vocabulary)
        return (piece, LogitRows(values: UnsafeBufferPointer(start: pointer, count: shape.vocabulary)))
    }

    /// one decoder step for every running beam of every active piece, as one wide batch.
    /// - parameter inputs: per piece, the newest token of each of its beams
    /// - returns: one row of logits per input row, in order, valid until the next step
    func step(_ inputs: [(piece: DecoderPiece, tokens: [Int32])]) async throws -> LogitRows {
        let rows = inputs.reduce(0) { $0 + $1.tokens.count }
        precondition(rows > 0)
        let prefix = "step."
        var maxList = 1
        var maxContext = 1
        for input in inputs {
            precondition(input.tokens.count == input.piece.beams)
            guard input.piece.usedSlots + input.piece.beams <= input.piece.slots.count else { throw EngineFailure.outOfSlots }
            maxList = max(maxList, input.piece.steps + 1)
            maxContext = max(maxContext, input.piece.promptLength + input.piece.steps + 1)
        }
        // round up so the work buffers are not rebuilt every step
        maxList = (maxList + 63) / 64 * 64
        maxContext = (maxContext + 255) / 256 * 256
        // sized before the layers are built, since they hold on to the buffer they are given
        let listBuffer = try scratch(prefix + "list", bytes: rows * maxList * 4)
        var bodies = try layerBodies(prefix: prefix, rows: rows, maxContext: maxContext)

        let width = shape.width
        let x = try scratch(prefix + "x", bytes: 16)
        let tokens = try scratch(prefix + "tok", bytes: rows * 4).contents().bindMemory(to: Int32.self, capacity: rows)
        let positions = try scratch(prefix + "pos", bytes: 16).contents().bindMemory(to: Int32.self, capacity: rows)
        let identity = try scratch(prefix + "row", bytes: 16).contents().bindMemory(to: Int32.self, capacity: rows)
        let targets = try scratch(prefix + "slot", bytes: 16).contents().bindMemory(to: Int32.self, capacity: rows)
        let contexts = try scratch(prefix + "ctx", bytes: 16).contents().bindMemory(to: UInt32.self, capacity: rows * 4)
        let list = listBuffer.contents().bindMemory(to: Int32.self, capacity: rows * maxList)

        var row = 0
        for input in inputs {
            let piece = input.piece
            for beam in 0..<piece.beams {
                let slot = Int32(piece.slots.lowerBound + piece.promptLength + piece.steps * piece.beams + beam)
                piece.chains[beam].append(slot)
                tokens[row] = input.tokens[beam]
                positions[row] = Int32(piece.promptLength + piece.steps)
                identity[row] = Int32(row)
                targets[row] = slot
                contexts[row * 4] = UInt32(piece.slots.lowerBound)
                contexts[row * 4 + 1] = UInt32(piece.promptLength)
                contexts[row * 4 + 2] = UInt32(row * maxList)
                contexts[row * 4 + 3] = UInt32(piece.chains[beam].count)
                piece.chains[beam].withUnsafeBufferPointer { chain in
                    (list + row * maxList).update(from: chain.baseAddress!, count: chain.count)
                }
                row += 1
            }
            piece.steps += 1
        }

        let table = try offset("model.language_model.embed_tokens.weight")
        let tokenBuffer = try scratch(prefix + "tok", bytes: 16)
        let head = try headBody(prefix: prefix, firstRow: 0, count: rows)
        try await run { pass in
            pass.embed(table: table, y: Span(x), tokens: Span(tokenBuffer), dim: width, count: rows)
            for body in bodies { try body(pass) }
            try head(pass)
        }
        bodies.removeAll()
        let logits = try scratch(prefix + "logits", bytes: 16)
        let pointer = logits.contents().bindMemory(to: Float.self, capacity: rows * shape.vocabulary)
        return LogitRows(values: UnsafeBufferPointer(start: pointer, count: rows * shape.vocabulary))
    }

    private func fill(_ buffer: MTLBuffer, _ values: [Int32]) {
        values.withUnsafeBufferPointer { source in
            buffer.contents().copyMemory(from: source.baseAddress!, byteCount: values.count * 4)
        }
    }
}
