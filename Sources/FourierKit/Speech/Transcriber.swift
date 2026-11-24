import Foundation

/// settings, language and priority for a run of pieces.
struct DecodePlan: Sendable {
    var settings: GenerationSettings
    /// the resolved name, e.g. "English"
    var language: String
    var context: String?
    var priority: PieceRequest.Priority

    /// the checkpoint's settings with only what the caller set changed. the defaults change nothing.
    static func make(_ decoding: DecodingOptions, priority: PieceRequest.Priority) async throws -> DecodePlan {
        guard let language = Prompt.languageName(for: decoding.language) else {
            throw FourierError.unsupportedLanguage(decoding.language)
        }
        var settings = await EngineHost.shared.defaults
        switch decoding.search {
        case .automatic: break
        case .greedy: settings.numBeams = 1
        case .beam(let count):
            guard count >= 1, count <= 12 else { throw FourierError.invalidOptions("beam count must be 1 to 12") }
            settings.numBeams = count
        }
        if let maxTokens = decoding.maxTokens {
            guard maxTokens >= 1 else { throw FourierError.invalidOptions("max tokens must be at least 1") }
            settings.maxNewTokens = maxTokens
        }
        if let guardSize = decoding.repeatGuard { settings.noRepeatNgramSize = max(0, guardSize) }
        return DecodePlan(settings: settings, language: language, context: decoding.context, priority: priority)
    }
}

enum Transcriber {
    static let onePassLimit = 120 * 16_000

    struct Outcome: Sendable {
        /// sample range in the audio given to `decode`
        var range: Range<Int>
        var result: PieceResult
        /// when the piece was sent, for time to first text
        var sent: Double
        var splitDepth: Int
    }

    /// up to 120 s is one piece; longer audio is cut at quiet points.
    static func ranges(samples: [Float], chunking: Chunking) -> [Range<Int>] {
        guard samples.count > onePassLimit else { return [0..<samples.count] }
        let seconds: Double = switch chunking {
        case .automatic: EngineLimits.automaticPieceSeconds
        case .maxPieceLength(let length): min(120, max(1, length.inSeconds))
        }
        return AudioSplitter.pieces(samples: samples, maxSeconds: seconds)
    }

    /// runs every range at once (the host batches them) and returns the outcomes in audio order.
    /// a piece that hits the token limit is split at its quietest point and decoded again.
    static func decode(samples: [Float], ranges: [Range<Int>], plan: DecodePlan,
                       done: @escaping @Sendable (Int) -> Void = { _ in }) async throws -> [Outcome] {
        try await withThrowingTaskGroup(of: (Int, [Outcome]).self) { group in
            for (index, range) in ranges.enumerated() {
                group.addTask {
                    (index, try await decodeRange(samples: samples, range: range, plan: plan, depth: 0, done: done))
                }
            }
            var results = [[Outcome]](repeating: [], count: ranges.count)
            for try await (index, outcomes) in group { results[index] = outcomes }
            return results.flatMap { $0 }
        }
    }

    private static func decodeRange(samples: [Float], range: Range<Int>, plan: DecodePlan, depth: Int,
                                    done: @escaping @Sendable (Int) -> Void) async throws -> [Outcome] {
        try Task.checkCancellation()
        let request = PieceRequest(
            samples: Array(samples[range]), language: plan.language, context: plan.context,
            settings: plan.settings, priority: plan.priority)
        let sent = now()
        let result = try await returningOnCancel { try await EngineHost.shared.transcribe(request) }
        await Runtime.shared.recordPiece(
            start: sent, end: now(), audioSeconds: Double(range.count) / 16_000,
            tokens: result.tokens.count, firstText: nil)
        // a piece that never reached a stop token was cut by the limit, so it runs again in halves
        if !result.reachedStop, depth < 6, range.count >= 2 * 16_000,
           let (first, second) = AudioSplitter.split(samples: samples, range: range) {
            async let a = decodeRange(samples: samples, range: first, plan: plan, depth: depth + 1, done: done)
            async let b = decodeRange(samples: samples, range: second, plan: plan, depth: depth + 1, done: done)
            return try await a + b
        }
        done(range.count)
        return [Outcome(range: range, result: result, sent: sent, splitDepth: depth)]
    }

    static func join(_ texts: [String]) -> String {
        texts.filter { !$0.isEmpty }.joined(separator: " ")
    }

    /// - parameter live: the audio arrived in real time, so wall-clock time says nothing about the
    ///   engine. speed is then the audio that was decoded over the time the engine spent on it.
    static func transcript(outcomes: [Outcome], offset: Int = 0, audioSamples: Int, plan: DecodePlan, started: Double, live: Bool = false) -> Transcript {
        let elapsed = now() - started
        var metrics = TranscriptMetrics()
        metrics.elapsed = .seconds(elapsed)
        metrics.realtimeSpeed = elapsed > 0 ? Double(audioSamples) / 16_000 / elapsed : 0
        metrics.placement = Runtime.shared.placement.value
        metrics.precision = Runtime.shared.precision.value
        let eos = Set(plan.settings.eosTokenIDs)
        var first: Double?
        for outcome in outcomes {
            let result = outcome.result
            metrics.tokens += result.tokens.filter { !eos.contains($0) }.count
            metrics.decoderSteps += result.steps
            metrics.widestBatch = max(metrics.widestBatch, result.widestBatch)
            metrics.melTime += .seconds(result.timings.mel)
            metrics.encodeTime += .seconds(result.timings.encode)
            metrics.prefillTime += .seconds(result.timings.prefill)
            metrics.decodeTime += .seconds(result.timings.decode)
            if outcome.splitDepth > 0 { metrics.tokenLimitSplits += 1 }
            let firstText = outcome.sent - started + result.timings.firstToken
            first = min(first ?? firstText, firstText)
        }
        metrics.timeToFirstText = first.map { .seconds(max(0, $0)) }
        if live {
            let decoded = Double(outcomes.reduce(0) { $0 + $1.range.count }) / 16_000
            let working = outcomes.reduce(0.0) {
                $0 + $1.result.timings.mel + $1.result.timings.encode + $1.result.timings.prefill + $1.result.timings.decode
            }
            metrics.realtimeSpeed = working > 0 ? decoded / working : 0
        }
        let pieces = outcomes.map { outcome in
            TranscriptPiece(
                text: outcome.result.text,
                samples: (outcome.range.lowerBound + offset)..<(outcome.range.upperBound + offset),
                reachedStop: outcome.result.reachedStop)
        }
        return Transcript(
            text: join(pieces.map(\.text)), language: plan.language, pieces: pieces,
            duration: .seconds(Double(audioSamples) / 16_000), metrics: metrics)
    }
}

/// the audio a transcription reads.
enum TranscriptionInput: Sendable {
    case file(URL)
    case samples([Float], sampleRate: Double)
}

/// one transcription with its own progress, result and cancel.
public final class TranscriptionJob: Sendable {
    public let id: UUID
    private let task: Task<Transcript, any Error>
    private let progressFeed: Broadcaster<Double>

    /// fraction of the audio done, 0...1, newest value only. ends when the job ends.
    public var progress: AsyncStream<Double> { progressFeed.stream() }

    /// the latest fraction done
    public var fraction: Double { progressFeed.latest ?? 0 }

    /// waits for the transcript. cancelling the waiting task does not cancel the job; `cancel()` does.
    public var result: Transcript {
        get async throws {
            do {
                return try await task.value
            } catch {
                throw FourierError.wrap(error)
            }
        }
    }

    /// stops at the next piece boundary; `result` then throws `FourierError.cancelled`.
    public func cancel() {
        task.cancel()
    }

    init(input: TranscriptionInput, options: FourierOptions) {
        let id = UUID()
        let feed = Broadcaster<Double>(initial: 0)
        self.id = id
        progressFeed = feed
        task = Task.detached(priority: .userInitiated) {
            let runtime = Runtime.shared
            let job = await runtime.register(.transcribe, phase: .preparing)
            do {
                let transcript = try await Self.run(input: input, options: options, job: job, feed: feed)
                await runtime.end(job)
                feed.finish(with: 1)
                return transcript
            } catch {
                let wrapped = FourierError.wrap(error)
                await runtime.end(job, error: wrapped)
                feed.finish()
                throw wrapped
            }
        }
    }

    static func read(_ input: TranscriptionInput) async throws -> [Float] {
        switch input {
        case .file(let url): try await offload { try AudioFileReader.read(url) }
        case .samples(let samples, let rate):
            rate == 16_000 ? samples : await offload { Resampler.resample(samples, from: rate) }
        }
    }

    private static func run(input: TranscriptionInput, options: FourierOptions, job: UUID, feed: Broadcaster<Double>) async throws -> Transcript {
        let runtime = Runtime.shared
        let started = now()
        // the audio is read while the engine gets ready
        async let audio = read(input)
        try await runtime.ensureReady(options)
        await runtime.update(job, phase: .reading)
        let samples = try await audio
        let plan = try await DecodePlan.make(options.decoding, priority: .file)
        await runtime.update(job, phase: .running, progress: .some(0), audio: .seconds(Double(samples.count) / 16_000))
        guard !samples.isEmpty else {
            return Transcriber.transcript(outcomes: [], audioSamples: 0, plan: plan, started: started)
        }

        let ranges = Transcriber.ranges(samples: samples, chunking: options.chunking)
        let total = samples.count
        let doneCount = Locked(0)
        let outcomes = try await Transcriber.decode(samples: samples, ranges: ranges, plan: plan) { count in
            let fraction = Double(doneCount.update { $0 += count; return $0 }) / Double(total)
            feed.send(min(1, fraction))
            Task { await runtime.update(job, progress: .some(min(1, fraction))) }
        }
        let transcript = Transcriber.transcript(outcomes: outcomes, audioSamples: total, plan: plan, started: started)
        await runtime.recordLive(timeToFirstText: transcript.metrics.timeToFirstText)
        return transcript
    }
}
