import Foundation

/// raw capture with a live level. `stop()` hands back the audio, untouched, for `transcribe(_:)`.
public final class Recorder: Sendable {
    public let id: UUID
    private let input: AudioInput
    private let levelFeed: Broadcaster<Float>
    private let store: Locked<Store>
    private let reader: Task<Void, Never>
    private let finished: Locked<Recording?>
    private let usesMicrophone: Bool

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

    public let startedAt: Date

    init(input: AudioInput, usesMicrophone: Bool, job: UUID) {
        id = job
        self.input = input
        self.usesMicrophone = usesMicrophone
        startedAt = Date()
        let levelFeed = Broadcaster<Float>(initial: 0)
        let store = Locked(Store())
        self.levelFeed = levelFeed
        self.store = store
        finished = Locked(nil)
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
}
