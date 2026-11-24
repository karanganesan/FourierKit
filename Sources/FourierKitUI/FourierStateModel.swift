import FourierKit
import Foundation
import Observation

/// one point of the speed chart.
public struct FourierMetricSample: Sendable, Hashable, Identifiable {
    public var id: Int
    public var date: Date
    public var realtimeSpeed: Double
    public var tokensPerSecond: Double
}

/// the sdk's state, metrics, prepare progress and microphone on the main actor: the single source the kit's views read.
///
/// feed it from a view with `.task { await model.observe() }`. any number of views can observe at once;
/// the streams run once and stop when the last observer goes away. values arrive already coalesced
/// off the main actor, so the main actor only stores the newest one.
@Observable
@MainActor
public final class FourierStateModel {
    /// the sdk this model reads. every `Fourier` shares one engine, so swapping it keeps the state.
    public var speech: Fourier

    /// the newest snapshot from `speech.states`
    public private(set) var state: FourierState
    /// the newest prepare progress, nil before any prepare was seen
    public private(set) var prepare: PrepareProgress?
    /// permission, capture and devices
    public private(set) var microphone: MicrophoneState
    /// the microphone's input level while record() or listen() runs, 0...1
    public private(set) var level: Float = 0
    /// the newest metrics, filled only while `observeMetrics()` runs
    public private(set) var metrics = FourierMetrics()
    /// the last 30 s of speed, four points a second, filled only while `observeMetrics()` runs
    public private(set) var speedHistory: [FourierMetricSample] = []
    /// chip, memory, heat and power, refreshed once a second while `observeMetrics()` runs
    public private(set) var machine = FourierMachine()
    /// the error from the last `prepare()` call, nil once it succeeds
    public private(set) var prepareError: FourierError?
    /// true while a `prepare()` call from this model runs
    public private(set) var isPreparing = false

    /// the engine is loaded and can start at once
    public var isReady: Bool { state.engine == .warm || state.engine == .busy }

    /// how many points `speedHistory` keeps
    public var historyLength = 120

    @ObservationIgnored private var observers = 0
    @ObservationIgnored private var feeds: [Task<Void, Never>] = []
    @ObservationIgnored private var metricObservers = 0
    @ObservationIgnored private var metricFeeds: [Task<Void, Never>] = []
    @ObservationIgnored private var prepareFeed: Task<Void, Never>?
    @ObservationIgnored private var sampleID = 0

    public init(_ speech: Fourier = Fourier()) {
        self.speech = speech
        state = FourierState()
        microphone = MicrophoneState()
    }

    // MARK: feeds

    /// reads states, prepare progress, microphone states and levels until the calling task is cancelled.
    public func observe() async {
        observers += 1
        if feeds.isEmpty { feeds = startFeeds() }
        await Self.waitForCancellation()
        observers -= 1
        if observers == 0 {
            feeds.forEach { $0.cancel() }
            feeds = []
            prepareFeed?.cancel()
            prepareFeed = nil
        }
    }

    /// reads live metrics until the calling task is cancelled. the sdk collects them only while someone reads.
    public func observeMetrics() async {
        metricObservers += 1
        if metricFeeds.isEmpty { metricFeeds = startMetricFeeds() }
        await Self.waitForCancellation()
        metricObservers -= 1
        if metricObservers == 0 {
            metricFeeds.forEach { $0.cancel() }
            metricFeeds = []
        }
    }

    private func startFeeds() -> [Task<Void, Never>] {
        let speech = speech
        watchPrepare()
        let states = Task { [weak self] in
            for await state in speech.states {
                guard let self else { return }
                self.receive(state)
            }
        }
        let microphones = Task { [weak self] in
            for await microphone in speech.microphoneStates {
                guard let self else { return }
                if self.microphone != microphone { self.microphone = microphone }
            }
        }
        let levels = Task { [weak self] in
            for await level in speech.microphoneLevels {
                guard let self else { return }
                if self.level != level { self.level = level }
            }
        }
        return [states, microphones, levels]
    }

    private func startMetricFeeds() -> [Task<Void, Never>] {
        let speech = speech
        let metrics = Task { [weak self] in
            for await metrics in speech.metrics {
                guard let self else { return }
                self.receive(metrics)
            }
        }
        let machine = Task { [weak self] in
            while !Task.isCancelled {
                let machine = await Task.detached(priority: .utility) { FourierMachine.read() }.value
                guard let self else { return }
                if self.machine != machine { self.machine = machine }
                try? await Task.sleep(for: .seconds(1))
            }
        }
        return [metrics, machine]
    }

    /// follows one prepare run. the sdk ends the stream when a run ends, so a new reader starts when the engine goes cold again.
    private func watchPrepare() {
        guard prepareFeed == nil else { return }
        let speech = speech
        prepareFeed = Task { [weak self] in
            for await progress in speech.prepareProgress {
                guard let self else { return }
                if self.prepare != progress { self.prepare = progress }
            }
            self?.prepareFeed = nil
        }
    }

    private func receive(_ new: FourierState) {
        let wasLoaded = state.engine == .warm || state.engine == .busy
        if state != new { state = new }
        if let progress = new.prepare, prepare != progress { prepare = progress }
        if wasLoaded, new.engine == .cold || new.engine == .preparing || new.engine == .loading, observers > 0 {
            watchPrepare()
        }
        if microphone != new.microphone { microphone = new.microphone }
    }

    private func receive(_ new: FourierMetrics) {
        if metrics != new { metrics = new }
        sampleID += 1
        speedHistory.append(FourierMetricSample(
            id: sampleID, date: Date(), realtimeSpeed: new.realtimeSpeed, tokensPerSecond: new.tokensPerSecond))
        if speedHistory.count > historyLength { speedHistory.removeFirst(speedHistory.count - historyLength) }
    }

    private static func waitForCancellation() async {
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(3600))
        }
    }

    // MARK: actions

    /// downloads, verifies, loads and warms the model. errors land in `prepareError` and `state.error`.
    public func prepare() async {
        guard !isPreparing else { return }
        isPreparing = true
        prepareError = nil
        if observers > 0 { watchPrepare() }
        do {
            try await speech.prepare()
        } catch let error as FourierError {
            if error != .cancelled { prepareError = error }
        } catch {
            prepareError = .engineFailure(String(describing: error))
        }
        isPreparing = false
    }

    /// shows the system prompt only if the user was never asked.
    @discardableResult
    public func requestMicrophone() async -> MicrophoneState.Permission {
        let permission = await speech.requestMicrophone()
        microphone = await speech.microphone
        return permission
    }

    /// opens privacy & security > microphone in system settings.
    public func openMicrophoneSettings() {
        speech.openMicrophoneSettings()
    }

    /// reads the microphone again, e.g. when the app becomes active. never prompts.
    public func refreshMicrophone() async {
        let latest = await speech.microphone
        if microphone != latest { microphone = latest }
    }
}
