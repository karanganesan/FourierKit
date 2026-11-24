import Foundation

/// turns stage reports into one bar that only moves forward, eased and capped to one update per frame.
///
/// real stages follow their reported fraction. estimated stages (compile, load, warm-up) follow past
/// timings on a curve that slows toward, and holds below, the stage's end. 1.0 only comes from `stop(success: true)`.
final class PrepareTracker: @unchecked Sendable {
    let feed: Broadcaster<PrepareProgress>
    /// told about every eased value, off the main thread
    var onChange: (@Sendable (PrepareProgress) -> Void)? {
        get { lock.withLock { _onChange } }
        set { lock.withLock { _onChange = newValue } }
    }

    private let lock = NSLock()
    private var _onChange: (@Sendable (PrepareProgress) -> Void)?
    private var _stage: PrepareProgress.Stage = .check
    private var stageFraction: Double = 0
    private var estimateStart: Double?
    private var estimateExpected: Double = 1
    private var displayed: Double = 0
    private var bytesDone: Int64 = 0
    private var bytesTotal: Int64 = 0
    private var rate: Double = 0
    private var rateSince: Double = 0
    private var lastBytes: (time: Double, bytes: Int64)?
    private var ticker: Task<Void, Never>?
    private var stopped = false
    private let timings = PrepareTimings.load()

    init() {
        feed = Broadcaster(initial: PrepareProgress(fraction: 0, stage: .check), interval: .milliseconds(16))
    }

    var stage: PrepareProgress.Stage { lock.withLock { _stage } }

    var current: PrepareProgress { lock.withLock { progressLocked() } }

    func start() {
        let task = Task.detached(priority: .utility) { [weak self] in
            while !Task.isCancelled {
                guard let self, let progress = self.tick() else { return }
                self.feed.send(progress)
                self.onChange?(progress)
                try? await Task.sleep(for: .milliseconds(16))
            }
        }
        lock.withLock { ticker = task }
    }

    /// a report from the model store.
    func store(_ update: ModelStoreProgress) {
        lock.withLock {
            if update.stage != _stage {
                _stage = update.stage
                estimateStart = nil
                if update.stage == .download { lastBytes = nil; rate = 0 }
            }
            stageFraction = min(1, max(0, update.fraction))
            if update.bytesTotal > 0 {
                bytesDone = update.bytesDone
                bytesTotal = update.bytesTotal
            }
            if update.stage == .download {
                let time = now()
                if let last = lastBytes, time - last.time > 0.25 {
                    let instant = Double(update.bytesDone - last.bytes) / (time - last.time)
                    if instant >= 0 {
                        if rate == 0 { rateSince = time }
                        rate = rate == 0 ? instant : rate * 0.9 + instant * 0.1
                    }
                    lastBytes = (time, update.bytesDone)
                } else if lastBytes == nil {
                    lastBytes = (time, update.bytesDone)
                }
            }
        }
    }

    /// a stage with no progress of its own, expected to take about `expected` seconds.
    func estimate(_ stage: PrepareProgress.Stage, expected: Double) {
        lock.withLock {
            _stage = stage
            stageFraction = 0
            estimateStart = now()
            estimateExpected = max(0.05, expected)
        }
    }

    func complete(_ stage: PrepareProgress.Stage) {
        lock.withLock {
            guard stage == _stage else { return }
            estimateStart = nil
            stageFraction = 1
        }
    }

    func stop(success: Bool) {
        let final: PrepareProgress = lock.withLock {
            stopped = true
            ticker?.cancel()
            if success {
                _stage = .ready
                displayed = 1
            }
            return progressLocked()
        }
        onChange?(final)
        feed.finish(with: final)
    }

    // MARK: inside the lock

    private func tick() -> PrepareProgress? {
        lock.withLock {
            guard !stopped else { return nil }
            let target = targetLocked()
            if target > displayed {
                let step = (target - displayed) * 0.25
                displayed = step < 0.0005 ? target : displayed + step
            }
            displayed = min(displayed, 0.999)
            return progressLocked()
        }
    }

    private func targetLocked() -> Double {
        let fraction: Double
        if let estimateStart {
            // slows toward the end and holds below it until the stage really finishes
            fraction = (1 - exp(-(now() - estimateStart) / estimateExpected)) * 0.95
        } else {
            fraction = stageFraction
        }
        return _stage.start + _stage.share * min(fraction, 1)
    }

    private func progressLocked() -> PrepareProgress {
        var remaining: Duration?
        if _stage == .download, rate > 0, now() - rateSince > 2, bytesTotal > 0 {
            let download = Double(max(0, bytesTotal - bytesDone)) / rate
            // hashing reads about 2 gb/s, then the engine loads and warms up
            let after = Double(bytesTotal) / 2e9 + timings.compile + timings.warmUp
            remaining = .seconds(download + after)
        }
        return PrepareProgress(fraction: displayed, stage: _stage, bytesDone: bytesDone, bytesTotal: bytesTotal, timeRemaining: remaining)
    }
}
