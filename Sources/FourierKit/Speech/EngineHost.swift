import Foundation

/// one piece of audio to run through the model in a single pass.
struct PieceRequest: Sendable {
    enum Priority: Int, Sendable, Comparable {
        case file = 0
        case live = 1
        static func < (a: Priority, b: Priority) -> Bool { a.rawValue < b.rawValue }
    }

    /// 16 khz mono, short enough for one pass
    var samples: [Float]
    var language: String? = "English"
    var context: String? = nil
    var settings: GenerationSettings
    /// tokens already settled, read with the prompt so only what follows is generated
    var prefix: [Int32] = []
    var priority: Priority = .file
    /// how many ranked candidates to keep per step, for the golden test
    var recordCandidates: Int = 0
    /// called after every decoder step of a greedy piece with the tokens generated so far, so live
    /// text can grow token by token instead of arriving when the pass ends. called on the host.
    var onTokens: (@Sendable ([Int32]) -> Void)? = nil
}

struct PieceTimings: Sendable {
    var queued: Double = 0
    var mel: Double = 0
    var encode: Double = 0
    var prefill: Double = 0
    var decode: Double = 0
    /// from the request arriving to the first decoder result
    var firstToken: Double = 0
    var total: Double = 0
}

struct PieceResult: Sendable {
    /// generated tokens, without the prompt or the prefix
    var tokens: [Int32]
    /// the model's exact text for prefix + generated, after the reference cleanup
    var text: String
    var reachedStop: Bool
    var score: Float
    var promptLength: Int
    var audioTokens: Int
    var steps: Int
    var timings: PieceTimings
    var candidates: [[Candidate]]
    /// widest batch this piece shared a step with, in rows
    var widestBatch: Int
}

struct HostSnapshot: Sendable {
    var loaded = false
    var activePieces = 0
    var queuedPieces = 0
    var batchRows = 0
    var maxRows = 0
    var stepsPerSecond: Double = 0
    var gpuBusy: Double = 0
    var cpuBusy: Double = 0
    var weightBytes: UInt64 = 0
    var cacheBytes: UInt64 = 0
    var totalSteps = 0
    var device: DeviceReport?
}

enum HostFailure: Error, Sendable {
    case notLoaded
    case cancelled
    case engine(String)
}

/// the one owner of the engine in this process.
///
/// every piece from every job lands here. pieces share one wide decoder batch: each step runs all
/// running beams of all active pieces in one gpu call, a finished piece frees its slots at once,
/// and the next piece is read in between steps. live pieces go first.
actor EngineHost {
    static let shared = EngineHost()

    private var engine: MetalEngine?
    private(set) var tokenizer: Tokenizer?
    private(set) var defaults = GenerationSettings()
    private(set) var modelDirectory: URL?

    /// which chip runs the conv front, encoder and projector
    enum EncoderChip: String, Sendable { case gpu, npu }
    private(set) var encoderChip: EncoderChip = .gpu
    private var neural: NeuralEncoder?

    /// moves the encoder to another chip. the neural engine model is written and compiled on
    /// first use, then cached beside the weights.
    func setEncoderChip(_ chip: EncoderChip) async throws {
        guard let engine, let modelDirectory else { throw HostFailure.notLoaded }
        if chip == .npu, neural == nil {
            neural = try await NeuralEncoder(
                weights: engine.weights, cacheDirectory: modelDirectory.appendingPathComponent("coreml", isDirectory: true)
            )
        }
        encoderChip = chip
    }

    /// what core ml planned for the neural engine encoder, if it is loaded
    var neuralPlan: PlacementReport? { neural?.plan }

    /// the widest batch allowed, in rows: the engine's derived width, narrowed under pressure.
    var maxRows = EngineLimits.rowCap
    private var underPressure = false

    private struct Pending {
        var id: Int
        var request: PieceRequest
        var arrived: Double
        var continuation: CheckedContinuation<PieceResult, Error>
    }

    private enum Search {
        case beam(BeamSearch)
        case greedy(GreedySearch)
    }

    private final class Active {
        var id: Int
        var request: PieceRequest
        var piece: DecoderPiece
        var search: Search
        var prompt: [Int32]
        var audioTokens: Int
        var timings: PieceTimings
        var arrived: Double
        var decodeStarted: Double
        var candidates: [[Candidate]] = []
        var widest = 0
        var continuation: CheckedContinuation<PieceResult, Error>

        init(id: Int, request: PieceRequest, piece: DecoderPiece, search: Search, prompt: [Int32], audioTokens: Int,
             timings: PieceTimings, arrived: Double, decodeStarted: Double, continuation: CheckedContinuation<PieceResult, Error>) {
            self.id = id; self.request = request; self.piece = piece; self.search = search; self.prompt = prompt
            self.audioTokens = audioTokens; self.timings = timings; self.arrived = arrived
            self.decodeStarted = decodeStarted; self.continuation = continuation
        }
    }

    private var pending: [Pending] = []
    private var active: [Active] = []
    private var cancelled: Set<Int> = []
    private var nextID = 0
    private var pumping = false

    private var totalSteps = 0
    private var recentSteps: [(time: Double, gpu: Double, cpu: Double)] = []
    private var cpuSeconds: Double = 0

    static func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

    var isLoaded: Bool { engine != nil }

    /// maps the weights, compiles the kernels and loads the tokenizer. does nothing if already loaded.
    func load(modelDirectory: URL) throws {
        if engine != nil, self.modelDirectory == modelDirectory { return }
        // the generation defaults come first, since they size the decoder cache
        let settings = try GenerationSettings.load(generationConfigJSON: modelDirectory.appendingPathComponent("generation_config.json"))
        let engine = try MetalEngine(modelDirectory: modelDirectory, settings: settings)
        tokenizer = try Tokenizer(contentsOf: modelDirectory.appendingPathComponent("tokenizer.json"))
        defaults = settings
        self.engine = engine
        self.modelDirectory = modelDirectory
        EngineLimits.current.update { $0 = engine.limits }
        maxRows = engine.limits.maxRows(underPressure: underPressure)
    }

    /// lets go of the weights and caches. waits for nothing: callers cool only when idle.
    func unload() {
        guard pending.isEmpty, active.isEmpty else { return }
        engine = nil
        neural = nil
        encoderChip = .gpu
        tokenizer = nil
        modelDirectory = nil
    }

    var isIdle: Bool { pending.isEmpty && active.isEmpty }

    func setMaxRows(_ rows: Int) { maxRows = max(1, rows) }

    /// thermal, power or memory pressure narrows the batch to the width the engine derived for it.
    func setUnderPressure(_ pressured: Bool) {
        underPressure = pressured
        if let engine { maxRows = engine.limits.maxRows(underPressure: pressured) }
    }

    func snapshot() -> HostSnapshot {
        var snapshot = HostSnapshot()
        guard let engine else { return snapshot }
        snapshot.loaded = true
        snapshot.activePieces = active.count
        snapshot.queuedPieces = pending.count
        snapshot.batchRows = active.reduce(0) { $0 + $1.piece.beams }
        snapshot.maxRows = maxRows
        snapshot.weightBytes = UInt64(engine.weightBytes)
        let perSlot = engine.shape.decoderLayers * engine.shape.keyWidth * 4 * 2
        snapshot.cacheBytes = UInt64(active.reduce(0) { $0 + $1.piece.usedSlots } * perSlot)
        snapshot.totalSteps = totalSteps
        snapshot.device = DeviceReport(engine.deviceProfile, engine.limits, weightBuffers: engine.weightBuffers.count, slotCapacity: engine.slotCapacity)
        let time = Self.now()
        let recent = recentSteps.filter { time - $0.time < 2 }
        if let first = recent.first, let last = recent.last, recent.count > 1, last.time > first.time {
            let span = last.time - first.time
            snapshot.stepsPerSecond = Double(recent.count - 1) / span
            snapshot.gpuBusy = min(1, (last.gpu - first.gpu) / span)
            snapshot.cpuBusy = min(1, (last.cpu - first.cpu) / span)
        }
        return snapshot
    }

    /// runs one piece and returns when it is done. safe to call from many tasks at once.
    func transcribe(_ request: PieceRequest) async throws -> PieceResult {
        guard engine != nil else { throw HostFailure.notLoaded }
        let id = nextID
        nextID += 1
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                pending.append(Pending(id: id, request: request, arrived: Self.now(), continuation: continuation))
                if !pumping {
                    pumping = true
                    Task { await self.pump() }
                }
            }
        } onCancel: {
            Task { await self.cancel(id) }
        }
    }

    private func cancel(_ id: Int) {
        if let index = pending.firstIndex(where: { $0.id == id }) {
            pending.remove(at: index).continuation.resume(throwing: CancellationError())
        } else if pumping {
            // running, or being read in right now: it is dropped at the next step boundary
            cancelled.insert(id)
        }
    }

    // MARK: the loop

    private func pump() async {
        while !pending.isEmpty || !active.isEmpty {
            await admit()
            dropCancelled()
            if !active.isEmpty { await step() }
        }
        cancelled.removeAll()
        pumping = false
    }

    /// reads in waiting pieces while there is room. with pieces already running it takes one per
    /// step, so a long prefill never holds live text back for long.
    private func admit() async {
        guard let engine, let tokenizer else {
            for item in pending { item.continuation.resume(throwing: HostFailure.notLoaded) }
            pending.removeAll()
            return
        }
        var admitted = 0
        while !pending.isEmpty {
            // live first, then oldest
            let index = pending.indices.max { a, b in
                (pending[a].request.priority, -pending[a].arrived) < (pending[b].request.priority, -pending[b].arrived)
            }!
            let rows = active.reduce(0) { $0 + $1.piece.beams }
            let wanted = pending[index].request.settings.numBeams
            if !active.isEmpty, rows + wanted > maxRows { break }
            if !active.isEmpty, admitted >= 1 { break }
            let item = pending.remove(at: index)
            do {
                if let started = try await start(item, engine: engine, tokenizer: tokenizer) {
                    active.append(started)
                }
            } catch EngineFailure.outOfSlots where !active.isEmpty {
                // no room in the cache yet. it goes back in line until a running piece ends.
                pending.append(item)
                break
            } catch {
                item.continuation.resume(throwing: error)
            }
            admitted += 1
        }
    }

    private func start(_ item: Pending, engine: MetalEngine, tokenizer: Tokenizer) async throws -> Active? {
        var timings = PieceTimings()
        var clock = Self.now()
        timings.queued = clock - item.arrived
        let request = item.request
        let mel = LogMel.features(samples: request.samples)
        timings.mel = Self.now() - clock
        cpuSeconds += timings.mel

        clock = Self.now()
        let audio: AudioEmbedding
        if encoderChip == .npu, let neural {
            audio = try engine.audioEmbedding(from: try await neural.encode(mel: mel.data, frames: mel.frames, paddedFrames: mel.paddedFrames))
        } else {
            audio = try await engine.encode(mel: mel.data, frames: mel.frames, paddedFrames: mel.paddedFrames)
        }
        timings.encode = Self.now() - clock

        clock = Self.now()
        let base = Prompt.transcription(audioTokens: audio.count, language: request.language, context: request.context, tokenizer: tokenizer)
        let prompt = base + request.prefix
        let settings = request.settings
        let (piece, firstRow) = try await engine.prefill(prompt: prompt, audio: audio, beams: settings.numBeams, maxNewTokens: settings.maxNewTokens)
        let logits = firstRow.values
        timings.prefill = Self.now() - clock

        let vocabulary = engine.shape.vocabulary
        clock = Self.now()
        var search: Search
        var candidates: [[Candidate]] = []
        if settings.numBeams > 1 {
            var beam = BeamSearch(prompt: prompt, settings: settings, vocabularySize: vocabulary, recordCandidates: request.recordCandidates)
            // transformers expands the prompt to one identical row per beam for the first step
            var rows = [Float](repeating: 0, count: settings.numBeams * vocabulary)
            rows.withUnsafeMutableBufferPointer { target in
                for beamIndex in 0..<settings.numBeams {
                    (target.baseAddress! + beamIndex * vocabulary).update(from: logits.baseAddress!, count: vocabulary)
                }
            }
            let outcome = rows.withUnsafeBufferPointer { beam.advance(logits: $0) }
            piece.reorder(parents: outcome.parents)
            if request.recordCandidates > 0 { candidates.append(outcome.candidates) }
            search = .beam(beam)
        } else {
            var greedy = GreedySearch(prompt: prompt, settings: settings, vocabularySize: vocabulary, recordCandidates: request.recordCandidates)
            let outcome = greedy.advance(logits: logits)
            if request.recordCandidates > 0 { candidates.append(outcome.candidates) }
            search = .greedy(greedy)
            request.onTokens?(greedy.result())
        }
        cpuSeconds += Self.now() - clock
        timings.firstToken = Self.now() - item.arrived

        let started = Active(
            id: item.id, request: request, piece: piece, search: search, prompt: prompt, audioTokens: audio.count,
            timings: timings, arrived: item.arrived, decodeStarted: Self.now(), continuation: item.continuation
        )
        started.candidates = candidates
        if isFinished(started) {
            finish(started, engine: engine, tokenizer: tokenizer)
            return nil
        }
        return started
    }

    private func isFinished(_ item: Active) -> Bool {
        switch item.search {
        case .beam(let beam): beam.isFinished
        case .greedy(let greedy): greedy.isFinished
        }
    }

    private func dropCancelled() {
        guard !cancelled.isEmpty, let engine else { return }
        defer { cancelled.removeAll() }
        for item in active where cancelled.contains(item.id) {
            engine.release(item.piece)
            item.continuation.resume(throwing: CancellationError())
        }
        active.removeAll { cancelled.contains($0.id) }
    }

    private func step() async {
        guard let engine, let tokenizer else { return }
        let inputs: [(piece: DecoderPiece, tokens: [Int32])] = active.map { item in
            switch item.search {
            case .beam(let beam): (item.piece, beam.nextInputs)
            case .greedy(let greedy): (item.piece, greedy.nextInputs)
            }
        }
        let rows = inputs.reduce(0) { $0 + $1.tokens.count }
        let logits: UnsafeBufferPointer<Float>
        do {
            logits = try await engine.step(inputs).values
        } catch {
            // a failed gpu call fails every piece that shared it. the layer above reruns them.
            for item in active {
                engine.release(item.piece)
                item.continuation.resume(throwing: error)
            }
            active.removeAll()
            return
        }
        let clock = Self.now()
        let vocabulary = engine.shape.vocabulary
        var row = 0
        var finished: [Int] = []
        for (index, item) in active.enumerated() {
            let count = item.piece.beams
            let slice = UnsafeBufferPointer(rebasing: logits[(row * vocabulary)..<((row + count) * vocabulary)])
            switch item.search {
            case .beam(var beam):
                let outcome = beam.advance(logits: slice)
                item.piece.reorder(parents: outcome.parents)
                if item.request.recordCandidates > 0 { item.candidates.append(outcome.candidates) }
                item.search = .beam(beam)
            case .greedy(var greedy):
                let outcome = greedy.advance(logits: slice)
                if item.request.recordCandidates > 0 { item.candidates.append(outcome.candidates) }
                item.search = .greedy(greedy)
                item.request.onTokens?(greedy.result())
            }
            item.widest = max(item.widest, rows)
            if isFinished(item) { finished.append(index) }
            row += count
        }
        cpuSeconds += Self.now() - clock
        totalSteps += 1
        recentSteps.append((Self.now(), engine.gpuSeconds, cpuSeconds))
        if recentSteps.count > 256 { recentSteps.removeFirst(recentSteps.count - 256) }
        for index in finished.reversed() {
            finish(active.remove(at: index), engine: engine, tokenizer: tokenizer)
        }
    }

    private func finish(_ item: Active, engine: MetalEngine, tokenizer: Tokenizer) {
        let tokens: [Int32]
        let score: Float
        switch item.search {
        case .beam(let beam):
            (tokens, score) = beam.result()
        case .greedy(let greedy):
            tokens = greedy.result()
            score = 0
        }
        let steps = item.piece.steps + 1
        engine.release(item.piece)
        var timings = item.timings
        timings.decode = Self.now() - item.decodeStarted
        timings.total = Self.now() - item.arrived
        let decoded = tokenizer.decode(item.prompt + tokens, skipSpecialTokens: true)
        let result = PieceResult(
            tokens: tokens,
            text: TranscriptText.parse(decoded).transcription,
            reachedStop: tokens.last.map { item.request.settings.eosTokenIDs.contains($0) } ?? false,
            score: score,
            promptLength: item.prompt.count,
            audioTokens: item.audioTokens,
            steps: steps,
            timings: timings,
            candidates: item.candidates,
            widestBatch: max(item.widest, item.piece.beams)
        )
        item.continuation.resume(returning: result)
    }
}
