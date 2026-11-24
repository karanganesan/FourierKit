import Foundation

/// raw capture with a live level and, while it runs, draft text. `stop()` hands back the audio,
/// untouched, for `transcribe(_:)`.
public final class Recorder: Sendable {
    public let id: UUID
    private let input: AudioInput
    private let levelFeed: Broadcaster<Float>
    private let store: Locked<Store>
    private let reader: Task<Void, Never>
    private let finished: Locked<Recording?>
    private let usesMicrophone: Bool
    private let draftFeed: Broadcaster<TranscriptionUpdate>
    private let drafter: Task<Void, Never>?

    private struct Store: Sendable {
        var samples: [Float] = []
        var dropped = 0
    }

    /// input loudness for a meter, 0 (silence) to 1 (full scale), newest value only. ends when recording stops.
    public var levels: AsyncStream<Float> { levelFeed.stream() }

    /// the newest level
    public var level: Float { levelFeed.latest ?? 0 }

    /// audio captured so far
    public var duration: Duration { .seconds(Double(store.update { $0.samples.count }) / 16_000) }

    /// what has been heard so far, as text, newest value only. ends when recording stops.
    ///
    /// these are quick drafts over the raw audio. `settled` is text for audio already cut off at a
    /// quiet point and does not change while recording; `draft` is the newest stretch and can.
    /// neither is the transcript: that comes from `transcribe(_:)` on the recording, in one exact
    /// pass, and may word things differently. empty if `recordDrafts` is off or the model is not
    /// on this device yet.
    public var drafts: AsyncStream<TranscriptionUpdate> { draftFeed.stream() }

    /// the newest draft, if any text has arrived
    public var latestDraft: TranscriptionUpdate? { draftFeed.latest }

    public let startedAt: Date

    init(input: AudioInput, usesMicrophone: Bool, job: UUID, drafts: FourierOptions? = nil) {
        id = job
        self.input = input
        self.usesMicrophone = usesMicrophone
        startedAt = Date()
        let levelFeed = Broadcaster<Float>(initial: 0)
        let store = Locked(Store())
        self.levelFeed = levelFeed
        self.store = store
        finished = Locked(nil)
        let draftFeed = Broadcaster<TranscriptionUpdate>()
        self.draftFeed = draftFeed
        drafter = drafts.map { options in
            Task.detached(priority: .userInitiated) {
                await Recorder.draft(store: store, feed: draftFeed, options: options)
            }
        }
        let runtime = Runtime.shared
        reader = Task.detached(priority: .userInitiated) {
            let events = Task.detached(priority: .utility) {
                for await event in input.events {
                    switch event {
                    case .capture(let capture) where usesMicrophone:
                        await runtime.setCapture(capture)
                    case .samplesDropped(let count):
                        store.update { $0.dropped += count }
                    default:
                        break
                    }
                }
            }
            var lastReport = now()
            for await chunk in input.samples {
                let count = store.update { store -> Int in
                    store.samples.append(contentsOf: chunk)
                    return store.samples.count
                }
                let level = meterLevel(chunk)
                levelFeed.send(level)
                if usesMicrophone { runtime.levels.send(level) }
                if now() - lastReport > 0.25 {
                    lastReport = now()
                    await runtime.update(job, audio: .seconds(Double(count) / 16_000))
                }
            }
            events.cancel()
            levelFeed.finish(with: 0)
            if usesMicrophone { runtime.levels.send(0) }
        }
    }

    /// stops capture and returns everything recorded. calling it again returns the same recording.
    @concurrent
    public func stop() async -> Recording {
        if let done = finished.value { return done }
        await input.stop()
        await reader.value
        drafter?.cancel()
        draftFeed.finish()
        let recording = finished.update { finished -> Recording in
            if let finished { return finished }
            let (samples, dropped) = store.update { ($0.samples, $0.dropped) }
            let recording = Recording(samples: samples, startedAt: startedAt, droppedSamples: dropped)
            finished = recording
            return recording
        }
        if usesMicrophone { await Runtime.shared.setCapture(.off) }
        await Runtime.shared.end(id)
        return recording
    }

    /// stops capture and throws the audio away.
    @concurrent
    public func cancel() async {
        _ = await stop()
        store.update { $0.samples = [] }
    }

    // MARK: drafts while recording

    /// the draft loop: one quick greedy pass at a time over the audio since the last cut. when that
    /// stretch grows past 20 s it is cut at a quiet point, the part before the cut gets one last
    /// pass and is set aside, and drafting goes on from the cut, so a pass never grows with the
    /// length of the recording.
    private static func draft(store: Locked<Store>, feed: Broadcaster<TranscriptionUpdate>, options: FourierOptions) async {
        guard (try? await Runtime.shared.ensureReady(options)) != nil else { return }
        var decoding = options.decoding
        decoding.search = .greedy
        guard let plan = try? await DecodePlan.make(decoding, priority: .live),
              let tokenizer = await EngineHost.shared.tokenizer
        else { return }
        let eos = Set(plan.settings.eosTokenIDs)
        @Sendable func text(_ tokens: [Int32]) -> String {
            tokenizer.decode(tokens.filter { !eos.contains($0) }, skipSpecialTokens: true)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        func pass(_ samples: [Float], shown: Locked<Int>? = nil, settled: String = "") async -> String? {
            var building = PieceRequest(
                samples: samples, language: plan.language, context: plan.context, settings: plan.settings, priority: .live)
            if let shown {
                // the tail grows token by token, and only ever grows inside one pass
                building.onTokens = { tokens in
                    let partial = text(tokens)
                    let longer = shown.update { count -> Bool in
                        guard partial.count > count else { return false }
                        count = partial.count
                        return true
                    }
                    if longer { feed.send(TranscriptionUpdate(settled: settled, draft: partial)) }
                }
            }
            let request = building
            guard let result = try? await EngineHost.shared.transcribe(request) else { return nil }
            return text(result.tokens)
        }

        var settled: [String] = []
        var cut = 0
        var covered = 0
        var interval = 0.5
        var lastPass = now() - interval
        let shown = Locked(0)
        while !Task.isCancelled {
            let count = store.update { $0.samples.count }
            let due = now() - lastPass >= interval
            // at least half a second to work with, and something new since the last pass
            guard due, count - cut >= 8_000, count - covered >= 3_200 else {
                try? await Task.sleep(for: .milliseconds(40))
                continue
            }
            var tail = store.update { Array($0.samples[cut..<min(count, $0.samples.count)]) }
            if tail.count > 20 * 16_000 {
                let first = AudioSplitter.pieces(samples: tail, maxSeconds: 16)[0]
                if first.count < tail.count, let done = await pass(Array(tail[first])) {
                    if !done.isEmpty { settled.append(done) }
                    cut += first.count
                    tail = Array(tail[first.upperBound...])
                    shown.update { $0 = 0 }
                    feed.send(TranscriptionUpdate(settled: settled.joined(separator: " "), draft: ""))
                }
            }
            guard !Task.isCancelled else { return }
            lastPass = now()
            let joined = settled.joined(separator: " ")
            if let draft = await pass(tail, shown: shown, settled: joined), !Task.isCancelled {
                shown.update { $0 = draft.count }
                feed.send(TranscriptionUpdate(settled: joined, draft: draft))
            }
            covered = count
            // as with listen, the rate follows what a pass costs, so drafts never pile up
            interval = max(0.5, (now() - lastPass) * 1.15)
        }
    }
}
