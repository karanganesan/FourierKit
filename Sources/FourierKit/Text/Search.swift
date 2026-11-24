import Accelerate
import Foundation

/// the decoding settings of `generation_config.json` that the search reproduces.
struct GenerationSettings: Sendable {
    var numBeams: Int = 3
    var maxNewTokens: Int = 512
    var noRepeatNgramSize: Int = 8
    var lengthPenalty: Float = 1.0
    var eosTokenIDs: [Int32] = [151643, 151645]
    var padTokenID: Int32 = 151645
    /// single-token bans (`bad_words_ids`). an eos id in here is ignored, as transformers filters it out.
    var bannedTokens: [Int32] = []

    enum LoadError: Error, CustomStringConvertible {
        case malformed(String)
        case unsupported(String)

        var description: String {
            switch self {
            case .malformed(let what): "malformed generation_config.json: \(what)"
            case .unsupported(let what): "unsupported generation setting: \(what)"
            }
        }
    }

    /// reads the file, and throws for any setting the search does not reproduce (sampling,
    /// multi-token bans, repetition penalty, early stopping, ...) rather than silently differ.
    static func load(generationConfigJSON: URL) throws -> GenerationSettings {
        let root = try JSONValue.parse(contentsOf: generationConfigJSON)
        guard root.members != nil else { throw LoadError.malformed("not an object") }
        var s = GenerationSettings()

        func int(_ key: String) throws -> Int? {
            guard let v = root[key], !v.isNull else { return nil }
            guard let n = v.number, let i = Int(exactly: n) else { throw LoadError.malformed(key) }
            return i
        }
        func number(_ key: String) throws -> Double? {
            guard let v = root[key], !v.isNull else { return nil }
            guard let n = v.number else { throw LoadError.malformed(key) }
            return n
        }

        if root["do_sample"]?.bool == true { throw LoadError.unsupported("do_sample") }
        s.numBeams = try int("num_beams") ?? 1
        guard let maxNew = try int("max_new_tokens") else { throw LoadError.unsupported("no max_new_tokens") }
        s.maxNewTokens = maxNew
        s.noRepeatNgramSize = try int("no_repeat_ngram_size") ?? 0
        s.lengthPenalty = Float(try number("length_penalty") ?? 1.0)
        guard let eos = root["eos_token_id"]?.integers, !eos.isEmpty else { throw LoadError.malformed("eos_token_id") }
        s.eosTokenIDs = eos.map { Int32($0) }
        if let pad = try int("pad_token_id") { s.padTokenID = Int32(pad) } else { s.padTokenID = s.eosTokenIDs[0] }

        if let bad = root["bad_words_ids"], !bad.isNull {
            guard let list = bad.array else { throw LoadError.malformed("bad_words_ids") }
            var banned: [Int32] = []
            banned.reserveCapacity(list.count)
            for entry in list {
                guard let ids = entry.array?.compactMap({ $0.number }), ids.count == (entry.array?.count ?? -1),
                      !ids.isEmpty else { throw LoadError.malformed("bad_words_ids entry") }
                guard ids.count == 1 else { throw LoadError.unsupported("multi-token bad_words_ids entry") }
                guard let id = Int32(exactly: ids[0]), id >= 0 else { throw LoadError.malformed("bad_words_ids id") }
                banned.append(id)
            }
            s.bannedTokens = banned
        }

        if let rp = try number("repetition_penalty"), rp != 1.0 { throw LoadError.unsupported("repetition_penalty") }
        if let early = root["early_stopping"], !early.isNull, early.bool != false {
            throw LoadError.unsupported("early_stopping")
        }
        if let n = try int("num_return_sequences"), n != 1 { throw LoadError.unsupported("num_return_sequences") }
        if let n = try int("min_length"), n > 0 { throw LoadError.unsupported("min_length") }
        if let n = try int("min_new_tokens"), n > 0 { throw LoadError.unsupported("min_new_tokens") }
        if let n = try int("encoder_no_repeat_ngram_size"), n > 0 { throw LoadError.unsupported("encoder_no_repeat_ngram_size") }
        if let n = try int("num_beam_groups"), n > 1 { throw LoadError.unsupported("num_beam_groups") }
        for key in ["sequence_bias", "suppress_tokens", "begin_suppress_tokens", "forced_bos_token_id",
                    "forced_eos_token_id", "exponential_decay_length_penalty", "constraints", "force_words_ids"] {
            if let v = root[key], !v.isNull { throw LoadError.unsupported(key) }
        }
        if root["renormalize_logits"]?.bool == true { throw LoadError.unsupported("renormalize_logits") }
        if root["remove_invalid_values"]?.bool == true { throw LoadError.unsupported("remove_invalid_values") }
        if let g = try number("guidance_scale"), g != 1 { throw LoadError.unsupported("guidance_scale") }
        guard s.numBeams >= 1, s.maxNewTokens >= 1, s.noRepeatNgramSize >= 0 else {
            throw LoadError.malformed("num_beams, max_new_tokens or no_repeat_ngram_size out of range")
        }
        return s
    }
}

/// one entry of the top candidates of a step: accumulated score, the running beam it extends and
/// the token it adds.
struct Candidate: Sendable {
    var score: Float
    var beam: Int
    var token: Int32
}

/// the logits processors of the generation config, applied to one row in transformers' order:
/// no-repeat-ngram (over the whole sequence, prompt included) sets -inf, then the ban adds a
/// bias of 0 or -inf.
private struct RowProcessors: Sendable {
    let vocabularySize: Int
    let ngramSize: Int
    /// 0 or -inf per token; empty when nothing is banned.
    let banBias: [Float]

    init(settings: GenerationSettings, vocabularySize: Int) {
        self.vocabularySize = vocabularySize
        self.ngramSize = settings.noRepeatNgramSize
        var bias: [Float] = []
        let eos = Set(settings.eosTokenIDs)
        let bans = settings.bannedTokens.filter { !eos.contains($0) }
        if !bans.isEmpty {
            bias = [Float](repeating: 0, count: vocabularySize)
            for t in bans {
                precondition(t >= 0 && Int(t) < vocabularySize, "banned token \(t) is outside the vocabulary")
                bias[Int(t)] = -.infinity
            }
        }
        self.banBias = bias
    }

    /// sets -inf on every token that would complete an n-gram already in `sequence`.
    func banRepeats(_ sequence: [Int32], _ row: UnsafeMutablePointer<Float>) {
        let n = ngramSize
        let length = sequence.count
        guard n > 0, length >= n else { return }
        sequence.withUnsafeBufferPointer { seq in
            let prefixStart = length - n + 1
            if n == 1 {
                for t in seq where t >= 0 && Int(t) < vocabularySize { row[Int(t)] = -.infinity }
                return
            }
            let last = seq[length - 1]
            // windows j..<j+n lie fully inside the sequence; the last prefix token sits at j+n-2
            var j = 0
            while j + n <= length {
                if seq[j + n - 2] == last {
                    var same = true
                    var k = 0
                    while k < n - 2 {
                        if seq[j + k] != seq[prefixStart + k] { same = false; break }
                        k += 1
                    }
                    if same {
                        let t = seq[j + n - 1]
                        if t >= 0 && Int(t) < vocabularySize { row[Int(t)] = -.infinity }
                    }
                }
                j += 1
            }
        }
    }

    func applyBans(_ row: UnsafeMutablePointer<Float>) {
        guard !banBias.isEmpty else { return }
        banBias.withUnsafeBufferPointer { b in
            vDSP_vadd(row, 1, b.baseAddress!, 1, row, 1, vDSP_Length(vocabularySize))
        }
    }
}

/// torch's cpu float32 log-softmax, reproduced bit for bit: `(x - max) - log(sum)`, where the sum
/// of `exp(x - max)` runs in four float lanes over the row (the tail added into the first lanes)
/// and the lanes fold as `(l0 + l2) + (l1 + l3)`; exp and log are sleef's, see `TorchFloat`.
///
/// the beam step needs `((logprob + ban bias) + beam score)`; one fused pass writes that with the
/// same roundings as torch's separate ops.
private enum RowMath {
    /// writes `((((x - max) - logSum) + bias) + score)` and returns true when torch's log-softmax
    /// would have made the row nan (a nan, a +inf, or an all -inf row).
    static func logSoftmaxPlus(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ count: Int,
                               bias: UnsafePointer<Float>?, score: Float) -> Bool {
        var maxValue: Float = 0
        vDSP_maxv(x, 1, &maxValue, vDSP_Length(count))
        let sum = laneSumOfExp(x, maxValue, count)
        // any nan in the row, or a non-finite max, leaves a nan in the sum
        if sum.isNaN || !maxValue.isFinite {
            for i in 0..<count { out[i] = .nan }
            return true
        }
        let logSum = TorchFloat.log(sum)
        let m = SIMD4<Float>(repeating: maxValue)
        let l = SIMD4<Float>(repeating: logSum)
        let sc = SIMD4<Float>(repeating: score)
        let src = UnsafeRawPointer(x)
        let dst = UnsafeMutableRawPointer(out)
        var i = 0
        if let bias {
            let b = UnsafeRawPointer(bias)
            while i + 4 <= count {
                let v = src.loadUnaligned(fromByteOffset: i * 4, as: SIMD4<Float>.self)
                let bv = b.loadUnaligned(fromByteOffset: i * 4, as: SIMD4<Float>.self)
                dst.storeBytes(of: (((v - m) - l) + bv) + sc, toByteOffset: i * 4, as: SIMD4<Float>.self)
                i += 4
            }
            while i < count {
                out[i] = (((x[i] - maxValue) - logSum) + bias[i]) + score
                i += 1
            }
        } else {
            while i + 4 <= count {
                let v = src.loadUnaligned(fromByteOffset: i * 4, as: SIMD4<Float>.self)
                dst.storeBytes(of: ((v - m) - l) + sc, toByteOffset: i * 4, as: SIMD4<Float>.self)
                i += 4
            }
            while i < count {
                out[i] = ((x[i] - maxValue) - logSum) + score
                i += 1
            }
        }
        return false
    }

    /// `((x + bias) + score)` for rows that are already log-probabilities.
    static func plus(_ x: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ count: Int,
                     bias: UnsafePointer<Float>?, score: Float) {
        for i in 0..<count {
            let v = bias.map { x[i] + $0[i] } ?? x[i]
            out[i] = v + score
        }
    }

    private static func laneSumOfExp(_ x: UnsafePointer<Float>, _ maxValue: Float, _ count: Int) -> Float {
        let m = SIMD4<Float>(repeating: maxValue)
        if count < 4 {
            var lanes = SIMD4<Float>(repeating: 0)
            for i in 0..<count { lanes[i] = x[i] }
            let e = TorchFloat.exp(lanes - m)
            var s = e[0]
            for i in 1..<count { s += e[i] }
            return s
        }
        let raw = UnsafeRawPointer(x)
        @inline(__always) func load(_ at: Int) -> SIMD4<Float> {
            raw.loadUnaligned(fromByteOffset: at * 4, as: SIMD4<Float>.self) - m
        }
        var acc = TorchFloat.exp(load(0))
        var i = 4
        let whole = count - count % 4
        // four independent exps per pass hide the polynomial's latency; the adds keep torch's order
        while i + 16 <= whole {
            let e0 = TorchFloat.exp(load(i))
            let e1 = TorchFloat.exp(load(i + 4))
            let e2 = TorchFloat.exp(load(i + 8))
            let e3 = TorchFloat.exp(load(i + 12))
            acc += e0
            acc += e1
            acc += e2
            acc += e3
            i += 16
        }
        while i < whole {
            acc += TorchFloat.exp(load(i))
            i += 4
        }
        if i < count {
            var lanes = SIMD4<Float>(repeating: 0)
            for k in 0..<(count - i) { lanes[k] = x[i + k] }
            let e = TorchFloat.exp(lanes - m)
            for k in 0..<(count - i) { acc[k] += e[k] }
        }
        return (acc[0] + acc[2]) + (acc[1] + acc[3])
    }
}

/// raw pointers handed to the parallel row workers; each worker touches only its own rows.
private struct RowJob: @unchecked Sendable {
    let input: UnsafePointer<Float>
    let output: UnsafeMutablePointer<Float>
    let bias: UnsafePointer<Float>?
    let nanFlags: UnsafeMutablePointer<Bool>
}

private func containsNaN(_ x: UnsafePointer<Float>, _ count: Int) -> Bool {
    var total: Float = 0
    vDSP_sve(x, 1, &total, vDSP_Length(count))
    if !total.isNaN { return false }
    for i in 0..<count where x[i].isNaN { return true }
    return false
}

/// `torch.topk(x, k)` on a long row: the heap pass of libc++'s partial_sort, skipping blocks whose
/// maximum cannot enter the heap (an entry only enters when strictly better than the heap top).
private func heapTopK(_ x: UnsafePointer<Float>, _ n: Int, _ k: Int, _ heap: inout [TopK.Entry],
                      allowSkipping: Bool) -> [TopK.Entry] {
    precondition(k * 64 <= n)
    heap.removeAll(keepingCapacity: true)
    for i in 0..<k { heap.append(TopK.Entry(value: x[i], index: Int32(i))) }
    let block = 512
    heap.withUnsafeMutableBufferPointer { h in
        TopK.makeHeap(h, k)
        var i = k
        while i < n {
            let end = min(n, i + block)
            if allowSkipping {
                var blockMax: Float = 0
                vDSP_maxv(x + i, 1, &blockMax, vDSP_Length(end - i))
                if !(blockMax > h[0].value) && !h[0].value.isNaN {
                    i = end
                    continue
                }
            }
            while i < end {
                let e = TopK.Entry(value: x[i], index: Int32(i))
                if TopK.better(e, h[0]) {
                    h[0] = e
                    TopK.siftDown(h, k, 0)
                }
                i += 1
            }
        }
        TopK.sortHeap(h, k)
    }
    return Array(heap[0..<k])
}

/// the top `k` of the whole buffer in torch's order.
private func topK(_ x: UnsafePointer<Float>, _ n: Int, _ k: Int, _ heap: inout [TopK.Entry],
                  mayContainNaN: Bool) -> [TopK.Entry] {
    if TopK.usesHeap(n: n, k: k) {
        return heapTopK(x, n, k, &heap, allowSkipping: !(mayContainNaN && containsNaN(x, n)))
    }
    var entries = (0..<n).map { TopK.Entry(value: x[$0], index: Int32($0)) }
    return TopK.select(&entries, k: k)
}

/// transformers' beam search (`_beam_search`, generation defaults: no sampling, early_stopping
/// false), as a cpu state machine fed with the model's raw logits.
///
/// per step: float32 log-softmax per row, the processors, plus the running beam score, then the
/// top `max(2, 1 + eos count) * numBeams` over all beams together. candidates that hit eos or the
/// length limit are pushed down by -1e9 before the next running beams are picked, and the
/// `numBeams` best of them may enter the finished set with the length penalty. the loop ends when
/// the best running beam cannot beat the worst finished one, or every candidate hit a stop.
struct BeamSearch: Sendable {
    private struct Hypothesis: Sendable {
        var generated: [Int32]
        var score: Float
        var finished: Bool
    }

    let settings: GenerationSettings
    let vocabularySize: Int
    let recordCandidates: Int

    private let prompt: [Int32]
    private let maxLength: Int
    private let beamsToKeep: Int
    private let processors: RowProcessors
    private let eos: Set<Int32>

    /// full sequence (prompt + generated) of each running beam, in transformers' row order.
    private var running: [[Int32]]
    private var runningScores: [Float]
    private var finished: [Hypothesis]
    private var earlyStopUnsatisfied = true
    private(set) var isFinished = false
    private(set) var step = 0

    // scratch reused across steps
    private var accumulated: [Float]
    private var heap: [TopK.Entry] = []

    init(prompt: [Int32], settings: GenerationSettings, vocabularySize: Int, recordCandidates: Int = 0) {
        precondition(settings.numBeams >= 1 && settings.maxNewTokens >= 1 && vocabularySize >= 1)
        self.settings = settings
        self.vocabularySize = vocabularySize
        self.recordCandidates = recordCandidates
        self.prompt = prompt
        self.maxLength = prompt.count + settings.maxNewTokens
        self.beamsToKeep = max(2, 1 + settings.eosTokenIDs.count) * settings.numBeams
        precondition(beamsToKeep <= settings.numBeams * vocabularySize, "vocabulary too small for the beam width")
        self.processors = RowProcessors(settings: settings, vocabularySize: vocabularySize)
        self.eos = Set(settings.eosTokenIDs)
        let k = settings.numBeams
        self.running = Array(repeating: prompt, count: k)
        self.runningScores = [0] + Array(repeating: -1e9, count: k - 1)
        self.finished = Array(repeating: Hypothesis(generated: [], score: -1e9, finished: false), count: k)
        self.accumulated = [Float](repeating: 0, count: k * vocabularySize)
    }

    /// the token each running beam feeds the model next, `numBeams` entries. after the first
    /// `advance` these are the newest tokens; before it, the prompt's last token.
    var nextInputs: [Int32] {
        running.map { $0.last ?? settings.padTokenID }
    }

    /// feed one row of raw float32 logits per running beam (`numBeams * vocabularySize`, row-major).
    /// the first call takes the prefill logits repeated on every row. returns, for each new running
    /// beam, the beam it continues (what transformers passes to reorder_cache), and the first
    /// `min(recordCandidates, max(2, 1 + eos count) * numBeams)` candidates of the step in order.
    mutating func advance(logits: UnsafeBufferPointer<Float>) -> (parents: [Int], candidates: [Candidate]) {
        advance(rows: logits, normalized: false)
    }

    /// the same step for rows that are already float32 log-softmaxed (for example on the gpu).
    mutating func advance(logProbabilities: UnsafeBufferPointer<Float>) -> (parents: [Int], candidates: [Candidate]) {
        advance(rows: logProbabilities, normalized: true)
    }

    private mutating func advance(rows input: UnsafeBufferPointer<Float>, normalized: Bool)
        -> (parents: [Int], candidates: [Candidate]) {
        precondition(!isFinished, "advance after the search finished")
        let k = settings.numBeams
        let v = vocabularySize
        precondition(input.count == k * v, "expected \(k) rows of \(v) logits")
        let curLen = prompt.count + step

        // b. log-probs, processors, plus the running score
        let processors = self.processors
        let running = self.running
        let runningScores = self.runningScores
        // rows are independent, so they run in parallel. the n-gram bans go in last: torch sets
        // them to -inf before the ban bias and the beam score, which leave -inf unchanged.
        var nanRows = [Bool](repeating: false, count: k)
        accumulated.withUnsafeMutableBufferPointer { acc in
            processors.banBias.withUnsafeBufferPointer { biasBuffer in
                nanRows.withUnsafeMutableBufferPointer { nanFlags in
                    let shared = RowJob(input: input.baseAddress!, output: acc.baseAddress!,
                                        bias: biasBuffer.isEmpty ? nil : biasBuffer.baseAddress,
                                        nanFlags: nanFlags.baseAddress!)
                    let work: @Sendable (Int) -> Void = { b in
                        let row = shared.output + b * v
                        if normalized {
                            RowMath.plus(shared.input + b * v, row, v, bias: shared.bias, score: runningScores[b])
                            shared.nanFlags[b] = true
                        } else {
                            shared.nanFlags[b] = RowMath.logSoftmaxPlus(shared.input + b * v, row, v,
                                                                        bias: shared.bias, score: runningScores[b])
                        }
                        processors.banRepeats(running[b], row)
                    }
                    if k > 1 && v >= 4096 {
                        DispatchQueue.concurrentPerform(iterations: k, execute: work)
                    } else {
                        for b in 0..<k { work(b) }
                    }
                }
            }
        }
        let mayContainNaN = nanRows.contains(true) || runningScores.contains { $0.isNaN }

        // c. top candidates over beams x vocabulary
        let top = accumulated.withUnsafeBufferPointer { acc in
            topK(acc.baseAddress!, k * v, beamsToKeep, &heap, mayContainNaN: mayContainNaN)
        }
        let candidateScores = top.map(\.value)
        let candidateBeams = top.map { Int($0.index) / v }
        let candidateTokens = top.map { Int32(Int($0.index) % v) }

        // d. stopping criteria per candidate
        let atMaxLength = curLen + 1 >= maxLength
        let hits = candidateTokens.map { atMaxLength || eos.contains($0) }

        // e. next running beams: finished candidates pushed down by -1e9
        let masked = zip(candidateScores, hits).map { $1 ? $0 + Float(-1e9) : $0 }
        let picks = TopK.select(masked, k: k)
        let nextRunning = picks.map { p -> [Int32] in
            let c = Int(p.index)
            var seq = running[candidateBeams[c]]
            seq.append(candidateTokens[c])
            return seq
        }
        let parents = picks.map { candidateBeams[Int($0.index)] }
        let nextScores = picks.map(\.value)

        // f. finished set
        let length = curLen + 1 - prompt.count
        let penalty = Float(Foundation.pow(Double(length), Double(settings.lengthPenalty)))
        var merged: [Float] = finished.map(\.score)
        var mergedHyps = finished
        for c in 0..<beamsToKeep {
            let justFinished = hits[c] && c < k
            var s = candidateScores[c] / penalty
            // early_stopping=false never marks the set as full, so that term adds -0
            if !earlyStopUnsatisfied { s += Float(-1e9) }
            if !justFinished { s += Float(-1e9) }
            merged.append(s)
            var generated = Array(running[candidateBeams[c]][prompt.count...])
            generated.append(candidateTokens[c])
            mergedHyps.append(Hypothesis(generated: generated, score: s, finished: justFinished))
        }
        let kept = TopK.select(merged, k: k)
        finished = kept.map { mergedHyps[Int($0.index)] }

        // g. advance and check whether the search can still improve
        self.running = nextRunning
        self.runningScores = nextScores
        step += 1
        let hypotheticalLength = curLen + 1 - prompt.count
        let hypPenalty = Float(Foundation.pow(Double(hypotheticalLength), Double(settings.lengthPenalty)))
        let bestPossible = nextScores[0] / hypPenalty
        let worstScore = finished.map(\.score).min()!
        var anyImprovable = false
        for h in finished {
            let worst: Float = h.finished ? worstScore : -1e9
            if bestPossible > worst { anyImprovable = true }
        }
        earlyStopUnsatisfied = earlyStopUnsatisfied && anyImprovable
        let allHit = hits.allSatisfy { $0 }
        isFinished = !earlyStopUnsatisfied || allHit

        let recorded = min(recordCandidates, beamsToKeep)
        let candidates = (0..<recorded).map {
            Candidate(score: candidateScores[$0], beam: candidateBeams[$0], token: candidateTokens[$0])
        }
        return (parents, candidates)
    }

    /// the returned sequence: generated ids only (prompt stripped, eos kept), and its score.
    /// transformers crops the best finished entry to the tokens it actually generated, so an
    /// entry that never finished gives no tokens.
    func result() -> (tokens: [Int32], score: Float) {
        let best = finished[0]
        return (best.generated, best.score)
    }
}

/// transformers' greedy search (`_sample` without sampling), used when `numBeams == 1`. the
/// processors act on the raw logits (no log-softmax) and the pick is the first maximum.
struct GreedySearch: Sendable {
    let settings: GenerationSettings
    let vocabularySize: Int
    let recordCandidates: Int

    private let promptLength: Int
    private let maxLength: Int
    private let processors: RowProcessors
    private let eos: Set<Int32>
    private var sequence: [Int32]
    private var row: [Float]
    private var heap: [TopK.Entry] = []
    private(set) var isFinished = false
    private(set) var step = 0

    init(prompt: [Int32], settings: GenerationSettings, vocabularySize: Int, recordCandidates: Int = 0) {
        precondition(settings.maxNewTokens >= 1 && vocabularySize >= 1)
        self.settings = settings
        self.vocabularySize = vocabularySize
        self.recordCandidates = recordCandidates
        self.promptLength = prompt.count
        self.maxLength = prompt.count + settings.maxNewTokens
        self.processors = RowProcessors(settings: settings, vocabularySize: vocabularySize)
        self.eos = Set(settings.eosTokenIDs)
        self.sequence = prompt
        self.row = [Float](repeating: 0, count: vocabularySize)
    }

    /// one entry: the token to feed next (the newest token after the first advance).
    var nextInputs: [Int32] {
        [sequence.last ?? settings.padTokenID]
    }

    /// feed one row of raw logits. `parents` is always `[0]`; `candidates` holds the top
    /// `recordCandidates` processed logits (diagnostics; the pick itself is the first maximum).
    mutating func advance(logits: UnsafeBufferPointer<Float>) -> (parents: [Int], candidates: [Candidate]) {
        precondition(!isFinished, "advance after the search finished")
        let v = vocabularySize
        precondition(logits.count == v, "expected one row of \(v) logits")
        let processors = self.processors
        let sequence = self.sequence
        let token: Int32 = row.withUnsafeMutableBufferPointer { r in
            let p = r.baseAddress!
            p.update(from: logits.baseAddress!, count: v)
            processors.banRepeats(sequence, p)
            processors.applyBans(p)
            return Int32(Self.firstMax(p, v))
        }
        var candidates: [Candidate] = []
        if recordCandidates > 0 {
            let top = row.withUnsafeBufferPointer {
                topK($0.baseAddress!, v, min(recordCandidates, v), &heap, mayContainNaN: true)
            }
            candidates = top.map { Candidate(score: $0.value, beam: 0, token: $0.index) }
        }
        self.sequence.append(token)
        step += 1
        if self.sequence.count >= maxLength || eos.contains(token) { isFinished = true }
        return ([0], candidates)
    }

    /// `torch.argmax`: the first index of the maximum, a nan counting as the maximum.
    private static func firstMax(_ x: UnsafePointer<Float>, _ n: Int) -> Int {
        if containsNaN(x, n) {
            for i in 0..<n where x[i].isNaN { return i }
        }
        // vDSP_maxvi reports the lowest index among equal maxima
        var best: Float = 0
        var at: vDSP_Length = 0
        vDSP_maxvi(x, 1, &best, &at, vDSP_Length(n))
        return Int(at)
    }

    /// the generated ids (prompt stripped, eos kept).
    func result() -> [Int32] {
        Array(sequence[promptLength...])
    }
}
