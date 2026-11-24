import Foundation

/// a running listen(). read `events` for live text; `finish()` returns the exact transcript.
public final class LiveSession: Sendable {
    public let id: UUID
    /// one reader at a time. cancelling the reading task does not stop the session; only finish() or cancel() do.
    public let events: LiveEvents
    private let engine: LiveEngine
    private let channel: EventChannel

    init(id: UUID, engine: LiveEngine, channel: EventChannel) {
        self.id = id
        self.engine = engine
        self.channel = channel
        events = LiveEvents(channel: channel)
    }

    /// stops the input, waits for the last phrase's exact pass, ends the event stream and returns the final text.
    /// calling it again returns the same transcript.
    @concurrent
    public func finish() async throws -> Transcript {
        try await engine.finish()
    }

    /// stops the input, discards everything and ends the event stream.
    public func cancel() {
        channel.finish()
        Task.detached(priority: .userInitiated) { [engine] in await engine.cancel() }
    }

    /// the newest update, nil before any text
    public var latestUpdate: LiveUpdate? {
        get async { await engine.latest }
    }
}

/// live events with the plan's stream rules: levels and updates keep only the newest, the rest queue in order.
public struct LiveEvents: AsyncSequence, Sendable {
    public typealias Element = LiveEvent
    let channel: EventChannel

    public struct AsyncIterator: AsyncIteratorProtocol {
        let channel: EventChannel
        public mutating func next() async throws -> LiveEvent? {
            try await channel.next()
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(channel: channel)
    }
}

final class EventChannel: Sendable {
    private struct State: Sendable {
        var queue: [LiveEvent] = []
        var waiter: CheckedContinuation<LiveEvent?, any Error>?
        var finished = false
        var failure: FourierError?
        /// settled text of the last update handed to a reader, for `newlySettled`
        var delivered = ""
    }

    private let state = Locked(State())

    func send(_ event: LiveEvent) {
        let ready: (CheckedContinuation<LiveEvent?, any Error>, LiveEvent)? = state.update { state in
            guard !state.finished else { return nil }
            if let waiter = state.waiter {
                state.waiter = nil
                return (waiter, Self.prepare(event, &state))
            }
            // the newest level and the newest update replace older ones, and move to the back
            switch event {
            case .level: state.queue.removeAll { if case .level = $0 { true } else { false } }
            case .update: state.queue.removeAll { if case .update = $0 { true } else { false } }
            default: break
            }
            state.queue.append(event)
            return nil
        }
        if let (waiter, event) = ready { waiter.resume(returning: event) }
    }

    private static func prepare(_ event: LiveEvent, _ state: inout State) -> LiveEvent {
        guard case .update(var update) = event else { return event }
        update.newlySettled = update.settled.hasPrefix(state.delivered)
            ? String(update.settled.dropFirst(state.delivered.count)) : update.settled
        state.delivered = update.settled
        return .update(update)
    }

    func next() async throws -> LiveEvent? {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<LiveEvent?, any Error>) in
                enum Outcome { case event(LiveEvent), end, fail(FourierError), wait }
                let (outcome, displaced): (Outcome, CheckedContinuation<LiveEvent?, any Error>?) = state.update { state in
                    if !state.queue.isEmpty {
                        let event = state.queue.removeFirst()
                        return (.event(Self.prepare(event, &state)), nil)
                    }
                    if state.finished { return (state.failure.map { .fail($0) } ?? .end, nil) }
                    if Task.isCancelled { return (.end, nil) }
                    let displaced = state.waiter
                    state.waiter = continuation
                    return (.wait, displaced)
                }
                displaced?.resume(returning: nil)
                switch outcome {
                case .event(let event): continuation.resume(returning: event)
                case .end: continuation.resume(returning: nil)
                case .fail(let error): continuation.resume(throwing: error)
                case .wait: break
                }
            }
        } onCancel: {
            let waiter = state.update { state -> CheckedContinuation<LiveEvent?, any Error>? in
                defer { state.waiter = nil }
                return state.waiter
            }
            waiter?.resume(returning: nil)
        }
    }

    /// ends the stream after the events already queued.
    func finish(throwing error: FourierError? = nil) {
        let waiter = state.update { state -> CheckedContinuation<LiveEvent?, any Error>? in
            guard !state.finished else { return nil }
            state.finished = true
            state.failure = error
            defer { state.waiter = nil }
            return state.waiter
        }
        if let error { waiter?.resume(throwing: error) } else { waiter?.resume(returning: nil) }
    }
}

/// the brain of one listen(): audio in, phrases, draft and closing passes, exact passes, events out.
///
/// a phrase opens when speech starts (with pre-roll) and closes on a pause or at the length limit.
/// while it is open, a draft pass runs over its audio every update interval with the phrase's settled
/// tokens as prefix, so only the tail is generated. tokens that agree across `settleAfter` passes settle.
/// a closing pass settles the rest, and an exact pass with the reference settings makes the final text.
actor LiveEngine {
    let id: UUID
    private let options: ListenOptions
    private let channel: EventChannel
    private let usesMicrophone: Bool
    private let runtime = Runtime.shared
    private let host = EngineHost.shared
    private let tokenizer: Tokenizer
    private let draftPlan: DecodePlan
    private let finalPlan: DecodePlan
    private let inputDevice: String?

    private var input: AudioInput?
    private var ingestTask: Task<Void, Never>?
    private var eventTask: Task<Void, Never>?
    private var worker: Task<Void, Never>?

    // audio. `audio[0]` is absolute position `base`.
    private var audio: [Float] = []
    private var base = 0
    private var heard = 0
    private var lastChunkTime = now()
    private var detector: any SpeechDetecting
    private var noise: (any NoiseProcessor)?
    private let gain: Float?
    private let preRollSamples: Int
    private let maxPhraseSamples: Int
    private var lastJobReport = 0.0

    // phrases
    private struct Pass {
        var tokens: [Int32]
        /// wall time the newest audio in the pass was heard
        var heardAt: Double
    }

    private struct Open {
        var id: Int
        var start: Int
        var settled: [Int32] = []
        var settledText = ""
        var history: [Pass] = []
        var draft: [Int32] = []
        var draftText = ""
        /// end of the audio the latest pass read
        var covered: Int
    }

    private struct Closing {
        var phrase: Open
        var end: Int
        var samples: [Float]
    }

    private var open: Open?
    private var closing: [Closing] = []
    private var closed: [LivePhrase] = []
    private var lastClosedEnd = 0
    private var nextPhraseID = 0
    private var exact: [(start: Int, task: Task<[Transcriber.Outcome], any Error>)] = []
    /// the session's settled text; it is only ever appended to
    private var ledger = ""
    private var startsWithSpace: [Int32: Bool] = [:]

    // scheduling
    private var passCount = 0
    /// the draft pass whose tokens may still arrive, 0 when none is running
    private var activePass = 0
    private var passing = false
    private var lastPassStart = 0.0
    private var passTime: Double?
    private var interval: Double
    private let baseInterval: Double
    private var warnedBehind = -Double.infinity
    private var stopping = false
    private var stopped = false
    private var finishTask: Task<Transcript, any Error>?
    private let startedAt = now()
    private var speechStartedAt: Double?
    private var firstTextReported = false
    private(set) var latest: LiveUpdate?

    init(id: UUID, options: ListenOptions, inputDevice: String?, channel: EventChannel, usesMicrophone: Bool,
         tokenizer: Tokenizer, draftPlan: DecodePlan, finalPlan: DecodePlan) {
        self.id = id
        self.options = options
        self.inputDevice = inputDevice
        self.channel = channel
        self.usesMicrophone = usesMicrophone
        self.tokenizer = tokenizer
        self.draftPlan = draftPlan
        self.finalPlan = finalPlan
        var detector: any SpeechDetecting = options.speechDetector ?? EnergySpeechDetector()
        detector.configure(pauseToEndPhrase: options.pauseToEndPhrase)
        self.detector = detector
        if case .custom(let processor) = options.noise { noise = processor }
        if case .fixed(let decibels) = options.gain { gain = powf(10, decibels / 20) } else { gain = nil }
        preRollSamples = max(0, Int(options.preRoll.inSeconds * 16_000))
        maxPhraseSamples = max(16_000, min(120 * 16_000, Int(options.maxPhraseLength.inSeconds * 16_000)))
        switch options.updateInterval {
        case .automatic: baseInterval = 0.3
        case .fixed(let duration): baseInterval = max(0.05, duration.inSeconds)
        }
        interval = baseInterval
    }

    // MARK: start and stop

    func start(source: any AudioSource) async throws {
        let voice: Bool = switch options.echoCancellation {
        case .on: true
        case .off: false
        case .automatic: if case .automatic = options.noise { true } else { false }
        }
        let automaticGain = if case .automatic = options.gain { true } else { false }
        let request = AudioSourceRequest(purpose: .listen, voiceProcessing: voice, automaticGain: automaticGain,
                                         inputDevice: inputDevice, preRoll: options.preRoll)
        let input = try await source.start(request)
        self.input = input
        if usesMicrophone { await runtime.setCapture(.on, inputUID: .some(inputDevice)) }
        channel.send(.started)

        eventTask = Task.detached(priority: .utility) { [weak self] in
            for await event in input.events {
                await self?.inputEvent(event)
            }
        }
        ingestTask = Task.detached(priority: .userInitiated) { [weak self] in
            for await chunk in input.samples {
                guard let self else { return }
                await self.ingest(chunk)
            }
            await self?.inputEnded()
        }
        worker = Task.detached(priority: .userInitiated) { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                if await !self.work() {
                    try? await Task.sleep(for: .milliseconds(15))
                }
            }
        }
    }

    func finish() async throws -> Transcript {
        if stopped && finishTask == nil { throw FourierError.sessionEnded }
        if let finishTask { return try await finishTask.value }
        let task = Task { try await self.finishWork() }
        finishTask = task
        do {
            return try await task.value
        } catch {
            throw FourierError.wrap(error)
        }
    }

    private func finishWork() async throws -> Transcript {
        stopping = true
        await runtime.update(id, phase: .finishing)
        if let input { await input.stop() }
        await ingestTask?.value
        // the last closing pass settles the last phrase before the final update
        while !closing.isEmpty || passing {
            try await Task.sleep(for: .milliseconds(10))
        }
        worker?.cancel()
        eventTask?.cancel()
        emitUpdate()
        do {
            let transcript = try await finalTranscript()
            stopped = true
            channel.finish()
            if usesMicrophone { await runtime.setCapture(.off) }
            await runtime.setSpeech(.silent)
            await runtime.end(id)
            if options.keepWarm == .never { await runtime.coolIfIdle() }
            return transcript
        } catch {
            let wrapped = FourierError.wrap(error)
            stopped = true
            channel.finish(throwing: wrapped)
            if usesMicrophone { await runtime.setCapture(.off) }
            await runtime.end(id, error: wrapped)
            throw wrapped
        }
    }

    private func finalTranscript() async throws -> Transcript {
        switch options.finalPass {
        case .perPhrase:
            var outcomes: [Transcriber.Outcome] = []
            for (start, task) in exact {
                for var outcome in try await task.value {
                    outcome.range = (outcome.range.lowerBound + start)..<(outcome.range.upperBound + start)
                    outcomes.append(outcome)
                }
            }
            return Transcriber.transcript(outcomes: outcomes, audioSamples: heard, plan: finalPlan, started: startedAt, live: true)
        case .fullAudio:
            // `audio` was never trimmed in this mode, so it holds everything from position 0
            let samples = audio
            let ranges = Transcriber.ranges(samples: samples, chunking: .automatic)
            let outcomes = samples.isEmpty ? [] : try await Transcriber.decode(samples: samples, ranges: ranges, plan: finalPlan)
            return Transcriber.transcript(outcomes: outcomes, audioSamples: heard, plan: finalPlan, started: startedAt, live: true)
        }
    }

    func cancel() async {
        guard !stopped else { return }
        stopped = true
        stopping = true
        channel.finish()
        worker?.cancel()
        for (_, task) in exact { task.cancel() }
        if let input { await input.stop() }
        ingestTask?.cancel()
        eventTask?.cancel()
        if usesMicrophone { await runtime.setCapture(.off) }
        await runtime.setSpeech(.silent)
        await runtime.end(id, error: .cancelled)
        if options.keepWarm == .never { await runtime.coolIfIdle() }
    }

    // MARK: audio in

    private func ingest(_ raw: [Float]) async {
        guard !stopped else { return }
        var chunk = raw
        if let gain { for index in chunk.indices { chunk[index] *= gain } }
        if var processor = noise {
            let cleaned = processor.process(chunk)
            noise = processor
            if cleaned.count == chunk.count { chunk = cleaned }
        }
        let level = meterLevel(chunk)
        channel.send(.level(level))
        if usesMicrophone { runtime.levels.send(level) }

        audio.append(contentsOf: chunk)
        heard += chunk.count
        lastChunkTime = now()

        for boundary in detector.feed(chunk) {
            switch boundary {
            case .started(let at): await openPhrase(at: at)
            case .ended: await closePhrase(at: heard)
            }
        }
        if let open, heard - open.start >= maxPhraseSamples {
            await forceClose()
        }
        trim()
        if now() - lastJobReport > 0.25 {
            lastJobReport = now()
            await runtime.update(id, audio: .seconds(Double(heard) / 16_000))
        }
    }

    private func inputEnded() async {
        if let boundary = detector.flush(), case .ended = boundary {
            await closePhrase(at: heard)
        }
        if open != nil { await closePhrase(at: heard) }
        if !stopping { channel.send(.warning(.inputEnded)) }
    }

    private func inputEvent(_ event: AudioInputEvent) async {
        switch event {
        case .capture(let capture):
            if usesMicrophone { await runtime.setCapture(capture) }
            if capture == .interrupted { channel.send(.warning(.inputInterrupted)) }
            if case .failed = capture { channel.send(.warning(.inputInterrupted)) }
        case .inputChanged(let lost):
            channel.send(.warning(.inputChanged(audioMayBeLost: lost)))
        case .deviceUnavailable(let uid):
            channel.send(.warning(.inputDeviceUnavailable(uid)))
        case .voiceProcessingUnavailable(let reason):
            channel.send(.warning(.voiceProcessingUnavailable(reason)))
        case .samplesDropped(let count):
            channel.send(.warning(.audioDropped(.seconds(Double(count) / 16_000))))
        }
    }

    /// drops audio no phrase can need any more. full-audio finals keep everything.
    private func trim() {
        guard options.finalPass == .perPhrase else { return }
        let keep = min(open?.start ?? heard, heard - preRollSamples - 16_000)
        let drop = keep - base
        guard drop > 10 * 16_000 else { return }
        audio.removeFirst(drop)
        base += drop
    }

    private func slice(_ range: Range<Int>) -> [Float] {
        let lower = max(range.lowerBound, base) - base
        let upper = min(range.upperBound, heard) - base
        guard upper > lower else { return [] }
        return Array(audio[lower..<upper])
    }

    private func wallTime(of position: Int) -> Double {
        lastChunkTime - Double(heard - position) / 16_000
    }

    // MARK: phrases

    private func openPhrase(at position: Int, preRoll: Bool = true) async {
        guard open == nil, !stopped else { return }
        let start = max(preRoll ? position - preRollSamples : position, lastClosedEnd, base, 0)
        open = Open(id: nextPhraseID, start: start, covered: start)
        nextPhraseID += 1
        speechStartedAt = wallTime(of: position)
        channel.send(.speechStarted)
        await runtime.setSpeech(.speaking)
    }

    private func closePhrase(at position: Int) async {
        guard let phrase = open else { return }
        let end = min(heard, max(position, phrase.start + 1))
        let samples = slice(phrase.start..<end)
        open = nil
        lastClosedEnd = end
        closing.append(Closing(phrase: phrase, end: end, samples: samples))
        if options.finalPass == .perPhrase, !samples.isEmpty {
            let plan = finalPlan
            let task = Task.detached(priority: .userInitiated) {
                try await Transcriber.decode(samples: samples, ranges: [0..<samples.count], plan: plan)
            }
            exact.append((phrase.start, task))
        }
        await runtime.setSpeech(.silent)
    }

    /// cuts a phrase that reached the length limit at the quietest moment of its last 2 s, and carries on.
    private func forceClose() async {
        guard let phrase = open else { return }
        let windowStart = max(phrase.start + 16_000, heard - 2 * 16_000)
        let cut = quietest(in: windowStart..<heard) ?? heard
        await closePhrase(at: cut)
        channel.send(.warning(.phraseTooLong))
        await openPhrase(at: cut, preRoll: false)
    }

    private func quietest(in range: Range<Int>) -> Int? {
        let samples = slice(range)
        let frame = 160, region = 20
        let frames = samples.count / frame
        guard frames > region else { return nil }
        var energy = [Double](repeating: 0, count: frames)
        for f in 0..<frames {
            var sum = 0.0
            for i in (f * frame)..<(f * frame + frame) { sum += Double(samples[i] * samples[i]) }
            energy[f] = sum
        }
        var best = 0, bestEnergy = Double.infinity, running = energy[0..<region].reduce(0, +)
        for f in 0...(frames - region) {
            if f > 0 { running += energy[f + region - 1] - energy[f - 1] }
            if running <= bestEnergy { bestEnergy = running; best = f }
        }
        return range.lowerBound + best * frame + region * frame / 2
    }

    // MARK: passes

    /// one step of the worker: a closing pass first, else a draft if one is due. false when idle.
    private func work() async -> Bool {
        if let next = closing.first {
            await runClosing(next)
            closing.removeFirst()
            return true
        }
        guard !stopping || open != nil, let phrase = open, !passing else { return false }
        let due = now() - lastPassStart >= interval
        let enough = heard - phrase.start >= 4_800 && heard - phrase.covered >= 1_600
        guard due, enough else { return false }
        await runDraft(phrase)
        return true
    }

    private func runDraft(_ phrase: Open) async {
        passing = true
        defer { passing = false }
        let end = heard
        let heardAt = lastChunkTime
        let started = now()
        lastPassStart = started
        var request = PieceRequest(
            samples: slice(phrase.start..<end), language: draftPlan.language, context: draftPlan.context,
            settings: draftPlan.settings, prefix: phrase.settled, priority: .live)
        // tokens are shown as they are decoded, so text grows through a pass instead of landing at its end
        passCount += 1
        let pass = passCount
        activePass = pass
        defer { if activePass == pass { activePass = 0 } }
        if options.drafts {
            let phraseID = phrase.id
            request.onTokens = { [weak self] tokens in
                Task { await self?.partial(tokens, pass: pass, phrase: phraseID) }
            }
        }
        let result: PieceResult
        do {
            result = try await host.transcribe(request)
        } catch {
            return
        }
        activePass = 0
        adapt(passTime: now() - started)
        await runtime.recordLive(draftDelay: .seconds(now() - heardAt))
        // the phrase may have closed while the pass ran; its closing pass takes over
        guard var current = open, current.id == phrase.id, current.settled == phrase.settled else { return }

        let full = phrase.settled + strip(result.tokens)
        current.history.append(Pass(tokens: full, heardAt: heardAt))
        if current.history.count > options.settleAfter { current.history.removeFirst(current.history.count - options.settleAfter) }
        current.covered = end
        if current.history.count >= max(1, options.settleAfter) {
            let agreed = settlePoint(current.history.map(\.tokens), settled: current.settled.count)
            if agreed > current.settled.count {
                settle(&current, tokens: Array(full[current.settled.count..<agreed]))
                await runtime.recordLive(settleDelay: .seconds(now() - current.history[0].heardAt))
            }
        }
        current.draft = Array(full[current.settled.count...])
        current.draftText = decode(current.draft, trimLeading: current.settledText.isEmpty)
        open = current
        emitUpdate()
    }

    /// tokens from a draft pass that is still running. the draft only ever grows from these: a
    /// partial that is not yet longer than what is on screen changes nothing, so text extends at
    /// its tail token by token, and corrections to earlier words land when the pass ends.
    private func partial(_ tokens: [Int32], pass: Int, phrase: Int) {
        guard activePass == pass, var current = open, current.id == phrase else { return }
        let fresh = strip(tokens)
        guard fresh.count > current.draft.count else { return }
        current.draft = fresh
        current.draftText = decode(fresh, trimLeading: current.settledText.isEmpty)
        open = current
        emitUpdate()
    }

    private func runClosing(_ item: Closing) async {
        passing = true
        defer { passing = false }
        var phrase = item.phrase
        if !item.samples.isEmpty, !stopped {
            let request = PieceRequest(
                samples: item.samples, language: draftPlan.language, context: draftPlan.context,
                settings: draftPlan.settings, prefix: phrase.settled, priority: .live)
            if let result = try? await host.transcribe(request) {
                let tail = strip(result.tokens)
                if !tail.isEmpty { settle(&phrase, tokens: tail) }
            }
        }
        let done = LivePhrase(
            id: phrase.id, settled: phrase.settledText, draft: "",
            start: .seconds(Double(phrase.start) / 16_000), end: .seconds(Double(item.end) / 16_000), isClosed: true)
        closed.append(done)
        channel.send(.phraseEnded(done))
        emitUpdate()
    }

    /// how many tokens all recent passes agree on, kept off the newest pass's last token (the audio edge)
    /// and pulled back to a word start, so settled text ends on a whole word.
    private func settlePoint(_ passes: [[Int32]], settled: Int) -> Int {
        guard let newest = passes.last else { return settled }
        var agreed = newest.count
        for pass in passes.dropLast() {
            var index = 0
            while index < min(agreed, pass.count), pass[index] == newest[index] { index += 1 }
            agreed = min(agreed, index)
        }
        agreed = min(agreed, newest.count - 1)
        while agreed > settled, !beginsWord(newest[agreed]) { agreed -= 1 }
        return max(settled, agreed)
    }

    /// appends to a phrase's settled tokens and text, and to the session's settled text. nothing is ever taken back.
    private func settle(_ phrase: inout Open, tokens: [Int32]) {
        guard !tokens.isEmpty else { return }
        let piece = decode(tokens, trimLeading: phrase.settledText.isEmpty)
        let wasEmpty = phrase.settledText.isEmpty
        phrase.settled += tokens
        phrase.settledText += piece
        guard !piece.isEmpty else { return }
        let before = ledger
        ledger += (wasEmpty && !ledger.isEmpty) ? " " + piece : piece
        precondition(ledger.hasPrefix(before), "settled text must only grow")
    }

    private func adapt(passTime time: Double) {
        passTime = passTime.map { $0 * 0.7 + time * 0.3 } ?? time
        guard let passTime else { return }
        switch options.updateInterval {
        case .automatic:
            // the next pass waits a little longer than a pass takes, so delay cannot pile up
            interval = max(baseInterval, passTime * 1.15)
        case .fixed:
            interval = max(baseInterval, passTime)
        }
        if interval > baseInterval * 2, now() - warnedBehind > 10 {
            warnedBehind = now()
            channel.send(.warning(.fellBehind(updateInterval: .milliseconds(Int(interval * 1000)))))
        }
    }

    private func emitUpdate() {
        var phrases = closed
        for item in closing where !phrases.contains(where: { $0.id == item.phrase.id }) {
            phrases.append(LivePhrase(
                id: item.phrase.id, settled: item.phrase.settledText, draft: "",
                start: .seconds(Double(item.phrase.start) / 16_000), end: .seconds(Double(item.end) / 16_000), isClosed: false))
        }
        var draft = ""
        if let open {
            let shownDraft = options.drafts ? open.draftText : ""
            phrases.append(LivePhrase(
                id: open.id, settled: open.settledText, draft: shownDraft,
                start: .seconds(Double(open.start) / 16_000), end: .seconds(Double(open.covered) / 16_000), isClosed: false))
            draft = shownDraft
            if !draft.isEmpty, open.settledText.isEmpty, !ledger.isEmpty { draft = " " + draft }
        }
        let covered = open?.covered ?? heard
        let update = LiveUpdate(
            settled: ledger, draft: draft, text: ledger + draft, newlySettled: "",
            phrases: phrases, audioTime: .seconds(Double(heard) / 16_000),
            delay: .seconds(Double(max(0, heard - covered)) / 16_000))
        latest = update
        channel.send(.update(update))
        if !firstTextReported, !update.text.isEmpty, let speechStartedAt {
            firstTextReported = true
            let delay = Duration.seconds(max(0, now() - speechStartedAt))
            Task { await Runtime.shared.recordLive(timeToFirstText: delay) }
        }
    }

    // MARK: tokens

    private func strip(_ tokens: [Int32]) -> [Int32] {
        let eos = Set(draftPlan.settings.eosTokenIDs)
        var tokens = tokens
        while let last = tokens.last, eos.contains(last) { tokens.removeLast() }
        return tokens
    }

    private func decode(_ tokens: [Int32], trimLeading: Bool) -> String {
        let text = tokenizer.decode(tokens, skipSpecialTokens: true)
        return trimLeading ? String(text.drop { $0.isWhitespace }) : text
    }

    private func beginsWord(_ token: Int32) -> Bool {
        if let known = startsWithSpace[token] { return known }
        let text = tokenizer.decode([token], skipSpecialTokens: true)
        let value = text.first.map { $0.isWhitespace || $0.isPunctuation } ?? false
        startsWithSpace[token] = value
        return value
    }
}
