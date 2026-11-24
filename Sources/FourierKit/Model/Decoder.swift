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

    private struct LayerWeights {
        var norm1, norm2, q, k, v, o, qNorm, kNorm, gate, up, down: WeightRef
    }

    private func layerWeights(_ layer: Int) throws -> LayerWeights {
        let p = "model.language_model.layers.\(layer)."
        return LayerWeights(
            norm1: try weight(p + "input_layernorm.weight"),
            norm2: try weight(p + "post_attention_layernorm.weight"),
            q: try weight(p + "self_attn.q_proj.weight"),
            k: try weight(p + "self_attn.k_proj.weight"),
            v: try weight(p + "self_attn.v_proj.weight"),
            o: try weight(p + "self_attn.o_proj.weight"),
            qNorm: try weight(p + "self_attn.q_norm.weight"),
            kNorm: try weight(p + "self_attn.k_norm.weight"),
            gate: try weight(p + "mlp.gate_proj.weight"),
            up: try weight(p + "mlp.up_proj.weight"),
            down: try weight(p + "mlp.down_proj.weight")
        )
    }

    /// a run of rows that attend among themselves: one piece of a prefill, or a whole step batch
    private struct Segment {
        var rowStart: Int
        var rows: Int
        var maxContext: Int
        /// the cache slot of the run's first row when its attention runs as square matrix products
        var squareSlot: Int?
    }

    /// the 28 layers over `rows` rows whose embeddings are already in the `x` work buffer.
    /// each row writes its key and value to its own slot and attends over its own context.
    /// matrix products run over all rows at once, attention once per segment.
    /// - parameter matrix: a per-piece pass (prefill) that may use apple's matrix kernel; never a step
    private func layerBodies(prefix: String, rows: Int, segments: [Segment], matrix: Bool) throws -> [(Pass) throws -> Void] {
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
        // segments run one after another, so they take turns with one scores buffer
        let scores = try scratch(prefix + "scores", bytes: segments.map { $0.rows * heads * $0.maxContext * 4 }.max() ?? 16)
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
            let w = try layerWeights(layer)
            let keySpan = Span(keys, layer * layerBytes)
            let valueSpan = Span(values, layer * layerBytes)
            bodies.append { pass in
                try pass.rmsNorm(x: Span(x), y: Span(h), gain: w.norm1, dim: width, count: rows, eps: eps)
                try pass.gemm(weight: w.q, rows: width, cols: width, x: Span(h), y: Span(qRaw), count: rows, matrix: matrix)
                try pass.gemm(weight: w.k, rows: keyWidth, cols: width, x: Span(h), y: Span(kRaw), count: rows, matrix: matrix)
                try pass.gemm(weight: w.v, rows: keyWidth, cols: width, x: Span(h), y: Span(vRaw), count: rows, matrix: matrix)
                pass.headNormRope(x: Span(qRaw), y: Span(q), gain: w.qNorm, positions: Span(positions), targets: Span(identity),
                                  heads: heads, headDim: headDim, count: rows, eps: eps)
                pass.headNormRope(x: Span(kRaw), y: keySpan, gain: w.kNorm, positions: Span(positions), targets: Span(targets),
                                  heads: keyHeads, headDim: headDim, count: rows, eps: eps)
                pass.scatterRows(x: Span(vRaw), y: valueSpan, targets: Span(targets), dim: keyWidth, count: rows)
                for segment in segments {
                    let rowBytes = segment.rowStart * width * 4
                    if let firstSlot = segment.squareSlot {
                        try pass.attentionSquare(q: Span(q, rowBytes), keys: keySpan, values: valueSpan, contexts: Span(contexts, segment.rowStart * 16),
                                                 scores: Span(scores), y: Span(mixed, rowBytes), heads: heads, keyHeads: keyHeads, headDim: headDim,
                                                 rows: segment.rows, maxContext: segment.maxContext, firstSlot: firstSlot)
                    } else {
                        try pass.attention(q: Span(q, rowBytes), keys: keySpan, values: valueSpan, contexts: Span(contexts, segment.rowStart * 16),
                                           list: Span(list), scores: Span(scores), y: Span(mixed, rowBytes), heads: heads, keyHeads: keyHeads,
                                           headDim: headDim, rows: segment.rows, maxContext: segment.maxContext)
                    }
                }
                try pass.gemm(weight: w.o, rows: width, cols: width, x: Span(mixed), y: Span(o), count: rows, matrix: matrix)
                pass.add(Span(x), Span(o), into: Span(x), count: rows * width)
                try pass.rmsNorm(x: Span(x), y: Span(h), gain: w.norm2, dim: width, count: rows, eps: eps)
                try pass.gemm(weight: w.gate, rows: inner, cols: width, x: Span(h), y: Span(gate), count: rows, matrix: matrix)
                try pass.gemm(weight: w.up, rows: inner, cols: width, x: Span(h), y: Span(up), count: rows, matrix: matrix)
                pass.siluMul(gate: Span(gate), up: Span(up), into: Span(gate), count: rows * inner)
                try pass.gemm(weight: w.down, rows: width, cols: inner, x: Span(gate), y: Span(o), count: rows, matrix: matrix)
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
        let norm = try weight("model.language_model.norm.weight")
        let table = try weight("model.language_model.embed_tokens.weight")
        return { pass in
            try pass.rmsNorm(x: Span(x, firstRow * width * 4), y: Span(h), gain: norm, dim: width, count: count, eps: eps)
            try pass.gemm(weight: table, rows: vocabulary, cols: width, x: Span(h), y: Span(logits), count: count)
        }
    }

    /// reads the prompt and all audio tokens once. every beam of the piece then shares that cache.
    /// - returns: the piece, and the next-token logits after the prompt (one row of `vocabulary` floats,
    ///   valid until the next prefill)
    func prefill(prompt: [Int32], audio: AudioEmbedding, beams: Int, maxNewTokens: Int) async throws -> (DecoderPiece, LogitRows) {
        try await prefillTogether([PrefillRequest(prompt: prompt, audio: audio, beams: beams, maxNewTokens: maxNewTokens)])[0]
    }

    /// one piece for `prefill(pieces:)`
    struct PrefillRequest {
        var prompt: [Int32]
        var audio: AudioEmbedding
        var beams: Int
        var maxNewTokens: Int
    }

    /// whether a piece of this many prompt rows can share a prefill pass with others and keep the
    /// bits it gets alone: its matrix products must already run on apple's kernel by themselves
    /// (`wideRows`) and past the row count where that kernel changes paths for every inner width
    /// the decoder has (30 rows at 6,144, measured; 10 at 2,048).
    func canShareFill(rows: Int) -> Bool { rows >= max(wideRows, 30) }

    /// reads several pieces in one pass: every matrix product runs once over all their rows, so each
    /// matrix is widened once, while attention stays within each piece. every piece's cache rows and
    /// logits are bit-identical to reading it alone, because a piece is only stacked with others when
    /// `canShareFill` holds for it; a piece that is too short is read alone first.
    /// - returns: per piece, in order, the piece and its next-token logits (valid until the next prefill)
    func prefill(pieces requests: [PrefillRequest]) async throws -> [(DecoderPiece, LogitRows)] {
        var results: [(DecoderPiece, LogitRows)?] = Array(repeating: nil, count: requests.count)
        var shared: [Int] = []
        var finished: [DecoderPiece] = []
        let vocabulary = shape.vocabulary
        do {
            // logits of a lone pass live in the buffer the next pass reuses, so they are copied out
            let lone = try scratch("pre.lone.logits", bytes: requests.count * vocabulary * 4)
                .contents().bindMemory(to: Float.self, capacity: requests.count * vocabulary)
            for (index, request) in requests.enumerated() {
                if requests.count > 1, canShareFill(rows: request.prompt.count) { shared.append(index); continue }
                let (piece, logits) = try await prefillTogether([request])[0]
                finished.append(piece)
                let copy = lone + index * vocabulary
                copy.update(from: logits.values.baseAddress!, count: vocabulary)
                results[index] = (piece, LogitRows(values: UnsafeBufferPointer(start: copy, count: vocabulary)))
            }
            if !shared.isEmpty {
                let together = try await prefillTogether(shared.map { requests[$0] })
                for (position, index) in shared.enumerated() { results[index] = together[position] }
            }
        } catch {
            for piece in finished { release(piece) }
            throw error
        }
        return results.map { $0! }
    }

    private func prefillTogether(_ requests: [PrefillRequest]) async throws -> [(DecoderPiece, LogitRows)] {
        var audioStarts: [Int] = []
        for request in requests {
            let prompt = request.prompt, audio = request.audio
            guard prompt.count + request.maxNewTokens <= maxPositions else {
                throw EngineFailure.tooLong("a piece of \(prompt.count) prompt tokens is too long for one pass")
            }
            guard let audioStart = prompt.firstIndex(of: shape.audioTokenID),
                  prompt[audioStart..<min(prompt.count, audioStart + audio.count)].allSatisfy({ $0 == shape.audioTokenID }),
                  prompt.filter({ $0 == shape.audioTokenID }).count == audio.count
            else { throw EngineFailure.badConfig("the prompt's audio placeholders do not match the audio tokens") }
            audioStarts.append(audioStart)
        }

        var pieces: [DecoderPiece] = []
        let prefix = "pre."
        let width = shape.width
        let vocabulary = shape.vocabulary
        do {
            var segments: [Segment] = []
            var rowStart = 0
            for request in requests {
                let rows = request.prompt.count
                let piece = DecoderPiece(
                    slots: try takeSlots(slotsNeeded(promptLength: rows, beams: request.beams, maxNewTokens: request.maxNewTokens)),
                    promptLength: rows, beams: request.beams
                )
                pieces.append(piece)
                // from a few dozen rows up, the square products go to apple's matrix kernel
                segments.append(Segment(rowStart: rowStart, rows: rows, maxContext: (rows + 63) / 64 * 64,
                                        squareSlot: rows >= squareRows ? piece.slots.lowerBound : nil))
                rowStart += rows
            }
            let rows = rowStart
            var bodies = try layerBodies(prefix: prefix, rows: rows, segments: segments, matrix: true)
            let x = try scratch(prefix + "x", bytes: 16)
            let tokens = try scratch(prefix + "tok", bytes: rows * 4)
            let audioRows = try scratch(prefix + "arow", bytes: requests.reduce(0) { $0 + $1.audio.count } * 4)
            let h = try scratch(prefix + "h", bytes: 16)
            let logits = try scratch(prefix + "logits", bytes: requests.count * vocabulary * 4)
            var positions: [Int32] = [], slots: [Int32] = [], promptTokens: [Int32] = [], audioTargets: [Int32] = []
            let contexts = try scratch(prefix + "ctx", bytes: 16).contents().bindMemory(to: UInt32.self, capacity: rows * 4)
            for (index, segment) in segments.enumerated() {
                let piece = pieces[index]
                promptTokens += requests[index].prompt
                positions += (0..<segment.rows).map { Int32($0) }
                slots += (0..<segment.rows).map { Int32(piece.slots.lowerBound + $0) }
                audioTargets += (0..<requests[index].audio.count).map { Int32(segment.rowStart + audioStarts[index] + $0) }
                for row in 0..<segment.rows {
                    // causal: row t sees slots 0...t of this piece
                    let at = (segment.rowStart + row) * 4
                    contexts[at] = UInt32(piece.slots.lowerBound)
                    contexts[at + 1] = UInt32(row + 1)
                    contexts[at + 2] = 0
                    contexts[at + 3] = 0
                }
            }
            fill(tokens, promptTokens)
            fill(audioRows, audioTargets)
            fill(try scratch(prefix + "pos", bytes: 16), positions)
            fill(try scratch(prefix + "row", bytes: 16), (0..<rows).map { Int32($0) })
            fill(try scratch(prefix + "slot", bytes: 16), slots)
            let table = try weight("model.language_model.embed_tokens.weight")
            let norm = try weight("model.language_model.norm.weight")
            let audios = requests.map { (buffer: $0.audio.buffer, count: $0.audio.count) }
            let eps = shape.normEps
            bodies.insert({ pass in
                pass.embed(table: table, y: Span(x), tokens: Span(tokens), dim: width, count: rows)
                var done = 0
                for audio in audios {
                    pass.scatterRows(x: Span(audio.buffer), y: Span(x), targets: Span(audioRows, done * 4), dim: width, count: audio.count)
                    done += audio.count
                }
            }, at: 0)
            // the output head on each piece's last row, one row at a time as when read alone
            bodies.append { pass in
                for (index, segment) in segments.enumerated() {
                    let last = segment.rowStart + segment.rows - 1
                    try pass.rmsNorm(x: Span(x, last * width * 4), y: Span(h), gain: norm, dim: width, count: 1, eps: eps)
                    try pass.gemm(weight: table, rows: vocabulary, cols: width, x: Span(h), y: Span(logits, index * vocabulary * 4), count: 1)
                }
            }
            try await runAll(bodies)
            let pointer = logits.contents().bindMemory(to: Float.self, capacity: requests.count * vocabulary)
            return pieces.enumerated().map { index, piece in
                (piece, LogitRows(values: UnsafeBufferPointer(start: pointer + index * vocabulary, count: vocabulary)))
            }
        } catch {
            for piece in pieces { release(piece) }
            throw error
        }
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
        var bodies = try layerBodies(prefix: prefix, rows: rows, segments: [Segment(rowStart: 0, rows: rows, maxContext: maxContext, squareSlot: nil)], matrix: false)

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

        let table = try weight("model.language_model.embed_tokens.weight")
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
