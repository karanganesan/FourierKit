import Foundation

/// the process-wide coordinator above the engine: prepare, keep-warm, pressure, jobs, state and metrics.
/// every `Fourier` shares it, so there is one engine and one source of truth per process.
actor Runtime {
    static let shared = Runtime()

    let host = EngineHost.shared

    // published, readable without hopping onto the actor
    nonisolated let states: Broadcaster<FourierState>
    nonisolated let microphoneStates: Broadcaster<MicrophoneState>
    nonisolated let levels: Broadcaster<Float>
    nonisolated let metricsFeed: Broadcaster<FourierMetrics>
    nonisolated let prepareFeed = PrepareFeed()

    private var state = FourierState()

    // engine
    private var warm = false
    private var cooling = false
    private var coolWaiters: [CheckedContinuation<Void, Never>] = []
    /// what the loaded engine was prepared for
    private var loadedKey: PrepareKey?
    private var run: PrepareRun?
    private var keepWarm: KeepWarm = .automatic
    private var lastUse = now()
    private var idleCheck: Task<Void, Never>?
    static let idleTimeout: Double = 5 * 60

    // pressure
    private var memoryWarning = false
    private var monitorsStarted = false
    private var memorySource: (any DispatchSourceMemoryPressure)?
    private var observers: [any NSObjectProtocol] = []
    private var deviceWatch: Task<Void, Never>?
    private var activeInputUID: String?

    // metrics
    private var metricsTask: Task<Void, Never>?
    private var finished: [(start: Double, end: Double, audio: Double, tokens: Int)] = []
    private var lastTimeToFirstText: Duration?
    private var lastDraftDelay: Duration?
    private var lastSettleDelay: Duration?

    static let shippedPlacement: [Stage: Chip] = [.mel: .cpu, .encoder: .gpu, .decoder: .gpu, .search: .cpu]
    static let shippedPrecision: [Stage: Precision] = [.mel: .float32, .encoder: .bfloat16, .decoder: .bfloat16, .search: .float32]
    /// chips each stage can run on today; the encoder also has a core ml neural engine path
    static let shippedChips: [Stage: Set<Chip>] = [.mel: [.cpu], .encoder: [.gpu, .npu], .decoder: [.gpu], .search: [.cpu]]

    /// the shipped chips this device has: no neural engine pin where core ml lists none, as in the simulator
    nonisolated static var availableChips: [Stage: Set<Chip>] {
        DeviceProfile.hasNeuralEngine ? shippedChips : shippedChips.mapValues { $0.subtracting([.npu]) }
    }

    /// what is actually loaded, readable from any thread
    nonisolated let placement = Locked(Runtime.shippedPlacement)
    nonisolated let precision = Locked(Runtime.shippedPrecision)

    private init() {
        var initial = FourierState()
        initial.placement = Self.shippedPlacement
        initial.precision = Self.shippedPrecision
        initial.microphone = Self.readMicrophone(device: nil, capture: .off)
        state = initial
        states = Broadcaster(initial: initial)
        microphoneStates = Broadcaster(initial: initial.microphone)
        levels = Broadcaster(initial: 0)
        metricsFeed = Broadcaster(interval: .milliseconds(16)) { _ in
            Task { await Runtime.shared.metricsReadersChanged() }
        }
        Task { await Runtime.shared.startMonitors() }
    }

    // MARK: state

    var snapshot: FourierState { state }

    private func publish() {
        refreshEngineState()
        states.send(state)
    }

    private func refreshEngineState() {
        let engine: EngineState
        if let run {
            engine = run.tracker.stage.order >= PrepareProgress.Stage.compile.order ? .loading : .preparing
        } else if cooling {
            engine = .cooling
        } else if warm {
            engine = state.jobs.contains { $0.phase == .running || $0.phase == .finishing } ? .busy : .warm
        } else if state.error != nil, state.engine == .failed {
            engine = .failed
        } else {
            engine = .cold
        }
        state.engine = engine
    }

    // MARK: prepare

    /// returns when the engine is ready for these options.
    ///
    /// it joins a run for the same options; a run for other options, or one its last waiter
    /// cancelled, is waited out and then this one starts. a cool in progress is waited out too.
    func ensureReady(_ options: FourierOptions) async throws {
        keepWarm = options.keepWarm
        let key = PrepareKey(options)
        while true {
            try Task.checkCancellation()
            if cooling {
                await withCheckedContinuation { coolWaiters.append($0) }
                continue
            }
            if let run {
                if run.key == key {
                    do {
                        try await run.join()
                        return
                    } catch is PrepareRun.Abandoned {}
                }
                try await run.waitOut()
                continue
            }
            if warm, loadedKey == key {
                touch()
                return
            }
            do {
                try await startRun(options, key: key).join()
                return
            } catch is PrepareRun.Abandoned {
                continue
            }
        }
    }

    private func startRun(_ options: FourierOptions, key: PrepareKey) -> PrepareRun {
        let tracker = PrepareTracker()
        let run = PrepareRun(tracker: tracker, key: key)
        self.run = run
        state.error = nil
        state.prepare = tracker.current
        publish()
        prepareFeed.begin(run)
        tracker.onChange = { progress in
            Task { await Runtime.shared.prepareChanged(run, progress) }
        }
        run.task = Task.detached(priority: .utility) {
            do {
                try await Runtime.shared.performPrepare(options, key: key, tracker: tracker)
                await Runtime.shared.prepareEnded(run, error: nil)
            } catch {
                await Runtime.shared.prepareEnded(run, error: FourierError.wrap(error))
            }
        }
        tracker.start()
        return run
    }

    private func prepareChanged(_ run: PrepareRun, _ progress: PrepareProgress) {
        guard self.run === run else { return }
        state.prepare = progress
        publish()
    }

    private func prepareEnded(_ run: PrepareRun, error: FourierError?) {
        if self.run === run { self.run = nil }
        state.prepare = nil
        if let error, error != .cancelled {
            state.error = error
            state.engine = .failed
        }
        run.tracker.stop(success: error == nil)
        prepareFeed.end(run, ready: error == nil)
        run.complete(error)
        publish()
        // a failed or cancelled run may still have loaded an engine; the idle check lets it go
        touch()
    }

    nonisolated func checkPins(_ compute: ComputeOptions) throws {
        for (stage, chip) in compute.chips where !(Self.availableChips[stage] ?? []).contains(chip) {
            throw FourierError.unsupportedPin(stage: stage, chip: chip)
        }
        let encoderChip = Self.encoderChip(for: compute)
        for (stage, precision) in compute.precision where precision != .automatic {
            let available = stage == .encoder ? Self.precision(ofEncoderOn: encoderChip) : Self.shippedPrecision[stage]
            if available != precision { throw FourierError.unsupportedPrecision(stage: stage, precision: precision) }
        }
    }

    /// the gpu unless a pin asks for the neural engine. `compute.policy` is not read: every policy
    /// resolves to the fastest exact placement. a policy may only take stages where the neural engine
    /// passes the golden test, and its 16-bit encoder does not (same tokens on the golden clips, many
    /// times the safety margin). only an explicit pin, which is outside the exactness claim, moves it.
    nonisolated static func encoderChip(for compute: ComputeOptions) -> Chip {
        compute.chips[.encoder] ?? .gpu
    }

    nonisolated static func precision(ofEncoderOn chip: Chip) -> Precision {
        // core ml runs the neural engine encoder in 16-bit floats; the metal path reads bf16 as stored
        chip == .npu ? .float16 : .bfloat16
    }

    nonisolated func modelRequest(_ options: FourierOptions) throws -> ModelRequest {
        let model = options.model
        let pinned = model.repository == ModelLocation.repository && model.revision == ModelLocation.revision
        guard let sha = model.sha256 ?? (pinned ? ModelLocation.weightsSHA256 : nil) else {
            throw FourierError.unverifiedRevision(model.revision)
        }
        return ModelRequest(
            repository: model.repository, revision: model.revision, directory: model.resolvedDirectory,
            files: ModelLocation.files, weightsFile: "model.safetensors", weightsSHA256: sha,
            weightsBytes: pinned && model.sha256 == nil ? ModelLocation.weightsBytes : nil, offline: model.offline)
    }

    private func performPrepare(_ options: FourierOptions, key: PrepareKey, tracker: PrepareTracker) async throws {
        try checkPins(options.compute)
        let request = try modelRequest(options)
        let directory = try await options.modelStore.prepare(request) { update in
            tracker.store(update)
        }
        try Task.checkCancellation()

        // compile and load happen together inside the engine's init, so their shares follow past timings
        let timings = PrepareTimings.load()
        tracker.estimate(.compile, expected: timings.compile)
        let started = now()
        let target = directory.standardizedFileURL
        let loaded = await host.isLoaded
        if loadedKey?.directory != target || !loaded {
            do {
                try await host.load(modelDirectory: target)
            } catch HostFailure.busy {
                throw FourierError.engineFailure("another model is still running. wait for its work to end or cancel it before switching models")
            } catch {
                throw FourierError.modelInvalid(String(describing: error))
            }
            // the engine is loaded from here on, so a later failure or cancel still lets it cool
            warm = true
            cooling = false
            loadedKey = nil
        }
        let wanted = Self.encoderChip(for: options.compute)
        let current: Chip = await host.encoderChip == .npu ? .npu : .gpu
        if wanted != current {
            do {
                try await host.setEncoderChip(wanted == .npu ? .npu : .gpu)
            } catch where options.compute.chips[.encoder] == nil {
                // unreachable today: without a pin the encoder stays on the gpu
            } catch {
                throw FourierError.engineFailure("the neural engine encoder could not load: \(error)")
            }
        }
        await refreshPlacement()
        let loadTime = now() - started
        tracker.complete(.compile)
        tracker.estimate(.load, expected: 0.2)
        tracker.complete(.load)
        try Task.checkCancellation()

        tracker.estimate(.warmUp, expected: timings.warmUp)
        let warmStarted = now()
        var settings = await host.defaults
        settings.maxNewTokens = 4
        _ = try await host.transcribe(PieceRequest(samples: [Float](repeating: 0, count: 16_000), settings: settings, priority: .live))
        tracker.complete(.warmUp)
        PrepareTimings(compile: max(0.2, loadTime), warmUp: max(0.1, now() - warmStarted)).save()
        warm = true
        cooling = false
        loadedKey = key
        applyPressure()
    }

    /// reports the chips that are really loaded.
    private func refreshPlacement() async {
        let encoder: Chip = await host.encoderChip == .npu ? .npu : .gpu
        var chips = Self.shippedPlacement
        chips[.encoder] = encoder
        var precisions = Self.shippedPrecision
        precisions[.encoder] = Self.precision(ofEncoderOn: encoder)
        placement.update { $0 = chips }
        precision.update { $0 = precisions }
        state.placement = chips
        state.precision = precisions
        publish()
    }

    /// true when the default store has already verified this model, so a background warm-up costs no download.
    nonisolated func modelLooksPresent(_ options: FourierOptions) -> Bool {
        guard let request = try? modelRequest(options) else { return false }
        return HuggingFaceModelStore.isVerified(request.directory.appendingPathComponent(request.weightsFile), sha256: request.weightsSHA256)
    }

    // MARK: keep warm

    func touch() {
        lastUse = now()
        scheduleIdleCheck()
    }

    func setKeepWarm(_ policy: KeepWarm) { keepWarm = policy }

    private func scheduleIdleCheck() {
        idleCheck?.cancel()
        guard warm else { return }
        switch keepWarm {
        case .always:
            return
        case .never:
            idleCheck = Task { await self.coolIfIdle() }
        case .automatic:
            idleCheck = Task {
                try? await Task.sleep(for: .seconds(Self.idleTimeout))
                guard !Task.isCancelled else { return }
                await self.coolIfIdle(after: Self.idleTimeout)
            }
        }
    }

    func coolIfIdle(after idle: Double = 0) async {
        guard warm, run == nil, !cooling, state.jobs.isEmpty, now() - lastUse >= idle - 0.5 else { return }
        if await cool() { return }
        // a piece left over from an ended job is still on the engine: look again once it is done
        guard warm, run == nil, state.jobs.isEmpty else { return }
        idleCheck?.cancel()
        idleCheck = Task {
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            await self.coolIfIdle(after: idle)
        }
    }

    /// lets go of the weights and caches. it waits for nothing: running work keeps the engine warm.
    /// - returns: whether the engine is cold now
    @discardableResult
    func cool() async -> Bool {
        guard warm else { return true }
        guard run == nil, !cooling, state.jobs.isEmpty else { return false }
        cooling = true
        publish()
        // the host refuses if a piece is waiting, being read in or running
        let unloaded = await host.unload()
        cooling = false
        if unloaded {
            warm = false
            loadedKey = nil
            prepareFeed.cooled()
            idleCheck?.cancel()
        }
        let waiters = coolWaiters
        coolWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
        publish()
        return unloaded
    }

    // MARK: jobs

    func register(_ kind: JobState.Kind, phase: JobState.Phase) -> UUID {
        let job = JobState(id: UUID(), kind: kind, phase: phase, progress: nil, audioDuration: .zero, startedAt: Date())
        state.jobs.append(job)
        idleCheck?.cancel()
        publish()
        return job.id
    }

    func update(_ id: UUID, phase: JobState.Phase? = nil, progress: Double?? = nil, audio: Duration? = nil) {
        guard let index = state.jobs.firstIndex(where: { $0.id == id }) else { return }
        if let phase { state.jobs[index].phase = phase }
        if let progress {
            // pieces report from many tasks, so a late, smaller fraction is ignored
            if let new = progress, let old = state.jobs[index].progress, new < old {} else { state.jobs[index].progress = progress }
        }
        if let audio { state.jobs[index].audioDuration = audio }
        publish()
    }

    func end(_ id: UUID, error: FourierError? = nil) {
        state.jobs.removeAll { $0.id == id }
        if let error, error != .cancelled { state.error = error } else if error == nil { state.error = nil }
        publish()
        touch()
    }

    func setSpeech(_ activity: SpeechActivity) {
        guard state.speech != activity else { return }
        state.speech = activity
        publish()
    }

    // MARK: microphone

    static func readMicrophone(device preferred: String?, capture: MicrophoneState.Capture) -> MicrophoneState {
        var mic = MicrophoneState()
        mic.permission = switch MicrophoneAccess.permission {
        case .notDetermined: .notDetermined
        case .granted: .granted
        case .denied: .denied
        case .restricted: .restricted
        }
        mic.devices = InputDevices.list().map { InputDevice(id: $0.uid, name: $0.name, isDefault: $0.isDefault) }
        mic.device = mic.devices.first { $0.id == preferred } ?? mic.devices.first { $0.isDefault }
        mic.isSetupValid = MicrophoneAccess.setupProblems().isEmpty
        mic.capture = capture
        return mic
    }

    /// reads permission, devices and setup again. never prompts.
    func refreshMicrophone() -> MicrophoneState {
        state.microphone = Self.readMicrophone(device: activeInputUID, capture: state.microphone.capture)
        microphoneStates.send(state.microphone)
        publish()
        return state.microphone
    }

    func setCapture(_ capture: MicrophoneState.Capture, inputUID: String?? = nil) {
        if let inputUID { activeInputUID = inputUID }
        state.microphone.capture = capture
        _ = refreshMicrophone()
    }

    func watchDevices() {
        guard deviceWatch == nil else { return }
        deviceWatch = Task.detached(priority: .utility) {
            for await _ in InputDevices.changes() {
                _ = await Runtime.shared.refreshMicrophone()
            }
        }
    }

    /// asks only if the user was never asked, as the handling allows. throws for anything but granted.
    func ensureMicrophone(_ handling: PermissionHandling) async throws {
        if let problem = MicrophoneAccess.setupProblems().first {
            throw FourierError.wrap(capture: problem)
        }
        var permission = refreshMicrophone().permission
        if permission == .notDetermined {
            switch handling {
            case .automatic:
                _ = await MicrophoneAccess.request()
                permission = refreshMicrophone().permission
            case .manual:
                throw FourierError.microphoneNotRequested
            case .custom(let handler):
                _ = await handler()
                permission = refreshMicrophone().permission
            }
        }
        switch permission {
        case .granted: watchDevices()
        case .denied: throw FourierError.microphoneDenied
        case .restricted: throw FourierError.microphoneRestricted
        case .notDetermined: throw FourierError.microphoneNotRequested
        }
    }

    // MARK: pressure

    private func startMonitors() {
        guard !monitorsStarted else { return }
        monitorsStarted = true
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.normal, .warning, .critical], queue: .global(qos: .utility))
        source.setEventHandler { [weak source] in
            guard let raw = source?.data.rawValue else { return }
            Task { await Runtime.shared.memoryPressure(DispatchSource.MemoryPressureEvent(rawValue: raw)) }
        }
        source.resume()
        memorySource = source
        let center = NotificationCenter.default
        observers.append(center.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil) { _ in
            Task { await Runtime.shared.pressureChanged() }
        })
        observers.append(center.addObserver(forName: .NSProcessInfoPowerStateDidChange, object: nil, queue: nil) { _ in
            Task { await Runtime.shared.pressureChanged() }
        })
        // the app's did-become-active, by name, so this file needs no ui framework
        observers.append(center.addObserver(forName: Self.didBecomeActive, object: nil, queue: nil) { _ in
            Task { _ = await Runtime.shared.refreshMicrophone() }
        })
        #if os(iOS)
        // ios warns apps before it ends them for memory; let go of an idle engine at once
        observers.append(center.addObserver(forName: Notification.Name("UIApplicationDidReceiveMemoryWarningNotification"), object: nil, queue: nil) { _ in
            Task { await Runtime.shared.coolIfIdle() }
        })
        #endif
        applyPressure()
    }

    #if os(macOS)
    private static let didBecomeActive = Notification.Name("NSApplicationDidBecomeActiveNotification")
    #else
    private static let didBecomeActive = Notification.Name("UIApplicationDidBecomeActiveNotification")
    #endif

    private func memoryPressure(_ event: DispatchSource.MemoryPressureEvent) async {
        memoryWarning = event.contains(.warning) || event.contains(.critical)
        applyPressure()
        if event.contains(.critical) {
            await coolIfIdle()
        } else if event.contains(.warning), keepWarm == .automatic {
            await coolIfIdle()
        }
    }

    private func pressureChanged() async {
        applyPressure()
        let thermal = ProcessInfo.processInfo.thermalState
        if keepWarm == .automatic, thermal == .serious || thermal == .critical {
            await coolIfIdle()
        }
    }

    private func applyPressure() {
        let thermal = ProcessInfo.processInfo.thermalState
        let pressure: SystemPressure =
            if thermal == .serious || thermal == .critical { .hot }
            else if memoryWarning { .busy }
            else if ProcessInfo.processInfo.isLowPowerModeEnabled { .lowPower }
            else { .normal }
        Task { await host.setUnderPressure(pressure != .normal) }
        if state.pressure != pressure {
            state.pressure = pressure
            publish()
        }
    }

    // MARK: metrics

    func recordPiece(start: Double, end: Double, audioSeconds: Double, tokens: Int, firstText: Double?) {
        finished.append((start, end, audioSeconds, tokens))
        let cutoff = now() - 10
        finished.removeAll { $0.end < cutoff }
        if let firstText { lastTimeToFirstText = .seconds(firstText) }
    }

    func recordLive(timeToFirstText: Duration? = nil, draftDelay: Duration? = nil, settleDelay: Duration? = nil) {
        if let timeToFirstText { lastTimeToFirstText = timeToFirstText }
        if let draftDelay { lastDraftDelay = draftDelay }
        if let settleDelay { lastSettleDelay = settleDelay }
    }

    private func metricsReadersChanged() {
        let count = metricsFeed.readerCount
        if count > 0, metricsTask == nil {
            metricsTask = Task.detached(priority: .utility) {
                while !Task.isCancelled {
                    let metrics = await Runtime.shared.collectMetrics()
                    Runtime.shared.metricsFeed.send(metrics)
                    try? await Task.sleep(for: .milliseconds(250))
                }
            }
        } else if count == 0 {
            metricsTask?.cancel()
            metricsTask = nil
        }
    }

    func collectMetrics() async -> FourierMetrics {
        let snapshot = await host.snapshot()
        var metrics = FourierMetrics()
        let time = now()
        let recent = finished.filter { time - $0.end < 10 }
        if let first = recent.map(\.start).min(), let last = recent.map(\.end).max(), last > first {
            metrics.realtimeSpeed = recent.reduce(0) { $0 + $1.audio } / (last - first)
        }
        metrics.decoderStepsPerSecond = snapshot.stepsPerSecond
        // one token per running piece per step; beams are not extra tokens
        metrics.tokensPerSecond = snapshot.stepsPerSecond * Double(snapshot.activePieces)
        metrics.timeToFirstText = lastTimeToFirstText
        metrics.draftDelay = lastDraftDelay
        metrics.settleDelay = lastSettleDelay
        metrics.batchWidth = snapshot.batchRows
        metrics.maxBatchWidth = snapshot.maxRows
        metrics.queuedPieces = snapshot.queuedPieces
        metrics.activePieces = snapshot.activePieces
        // the neural engine has no busy measure, so it is left out rather than shown as 0
        metrics.chipBusy = [.gpu: snapshot.gpuBusy, .cpu: snapshot.cpuBusy]
        metrics.appMemory = appMemoryFootprint()
        metrics.weightMemory = snapshot.weightBytes
        metrics.cacheMemory = snapshot.cacheBytes
        metrics.workload = workload()
        metrics.device = snapshot.device
        return metrics
    }

    private func workload() -> Workload {
        let jobs = state.jobs
        if jobs.isEmpty { return .idle }
        if state.pressure != .normal { return .underLoad }
        let live = jobs.contains { $0.kind == .listen }
        let files = jobs.filter { $0.kind == .transcribe }
        if live { return files.isEmpty ? .realtime : .realtimePlusFiles }
        if files.count > 1 { return .manyFiles }
        guard let file = files.first else { return .realtime }
        let seconds = file.audioDuration.inSeconds
        return seconds < 5 ? .short : seconds <= 30 ? .medium : seconds <= 120 ? .long : .veryLong
    }
}

// MARK: prepare run

/// what a prepare makes ready: callers with the same key share a run and a loaded engine.
struct PrepareKey: Equatable, Sendable {
    var directory: URL
    var repository: String
    var revision: String
    var sha256: String?
    var encoder: Chip

    init(_ options: FourierOptions) {
        directory = options.model.resolvedDirectory.standardizedFileURL
        repository = options.model.repository
        revision = options.model.revision
        sha256 = options.model.sha256?.lowercased()
        encoder = Runtime.encoderChip(for: options.compute)
    }
}

/// one shared prepare. any number of callers join it; the work stops only when the last one cancels.
final class PrepareRun: @unchecked Sendable {
    /// thrown by `join()` when the run was already given up by its last waiter
    struct Abandoned: Error {}

    let tracker: PrepareTracker
    let key: PrepareKey
    private let lock = NSLock()
    private var waiters: [UUID: CheckedContinuation<Void, any Error>] = [:]
    /// callers waiting for the run to end, whatever its outcome, without keeping it alive
    private var watchers: [UUID: CheckedContinuation<Void, any Error>] = [:]
    private var outcome: Result<Void, FourierError>?
    /// the last waiter left, so the run is winding down and nobody may join it
    private var abandoned = false
    private var _task: Task<Void, Never>?

    init(tracker: PrepareTracker, key: PrepareKey) {
        self.tracker = tracker
        self.key = key
    }

    var task: Task<Void, Never>? {
        get { lock.withLock { _task } }
        set { lock.withLock { _task = newValue } }
    }

    func join() async throws {
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let (ready, stop): (Result<Void, any Error>?, Bool) = lock.withLock {
                    if let outcome { return (outcome.mapError { $0 as any Error }, false) }
                    if abandoned { return (.failure(Abandoned()), false) }
                    if Task.isCancelled {
                        if waiters.isEmpty { abandoned = true }
                        return (.failure(FourierError.cancelled), waiters.isEmpty)
                    }
                    waiters[id] = continuation
                    return (nil, false)
                }
                if stop { task?.cancel() }
                if let ready { continuation.resume(with: ready) }
            }
        } onCancel: {
            leave(id)
        }
    }

    /// waits for the run to end, however it ends. throws only if the caller is cancelled.
    func waitOut() async throws {
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                let (ended, cancelled): (Bool, Bool) = lock.withLock {
                    if outcome != nil { return (true, false) }
                    if Task.isCancelled { return (false, true) }
                    watchers[id] = continuation
                    return (false, false)
                }
                if ended { continuation.resume() } else if cancelled { continuation.resume(throwing: FourierError.cancelled) }
            }
        } onCancel: {
            let continuation = lock.withLock { watchers.removeValue(forKey: id) }
            continuation?.resume(throwing: FourierError.cancelled)
        }
    }

    /// a waiter gave up. the run stops when nobody is left.
    private func leave(_ id: UUID) {
        let (continuation, empty, task): (CheckedContinuation<Void, any Error>?, Bool, Task<Void, Never>?) = lock.withLock {
            guard let continuation = waiters.removeValue(forKey: id) else { return (nil, false, _task) }
            let empty = waiters.isEmpty && outcome == nil
            if empty { abandoned = true }
            return (continuation, empty, _task)
        }
        continuation?.resume(throwing: FourierError.cancelled)
        if empty { task?.cancel() }
    }

    func complete(_ error: FourierError?) {
        let (waiting, watching): ([CheckedContinuation<Void, any Error>], [CheckedContinuation<Void, any Error>]) = lock.withLock {
            outcome = error.map { .failure($0) } ?? .success(())
            defer { waiters.removeAll(); watchers.removeAll() }
            return (Array(waiters.values), Array(watchers.values))
        }
        for waiter in waiting {
            if let error { waiter.resume(throwing: error) } else { waiter.resume() }
        }
        for watcher in watching { watcher.resume() }
    }
}

/// hands `prepareProgress` readers the current run, or the next one, or 1.0 when the model is ready.
final class PrepareFeed: Sendable {
    private struct State {
        var run: PrepareRun?
        var ready = false
        var waiting: [UUID: AsyncStream<PrepareProgress>.Continuation] = [:]
    }

    private let state = Locked(State())

    func stream() -> AsyncStream<PrepareProgress> {
        let (stream, continuation) = AsyncStream.makeStream(of: PrepareProgress.self, bufferingPolicy: .bufferingNewest(1))
        let id = UUID()
        let (run, ready): (PrepareRun?, Bool) = state.update { state in
            if state.run == nil, !state.ready { state.waiting[id] = continuation }
            return (state.run, state.ready)
        }
        if let run {
            forward(run, to: continuation)
        } else if ready {
            continuation.yield(PrepareProgress(fraction: 1, stage: .ready))
            continuation.finish()
        } else {
            continuation.onTermination = { [weak self] _ in self?.state.update { $0.waiting[id] = nil } }
        }
        return stream
    }

    private func forward(_ run: PrepareRun, to continuation: AsyncStream<PrepareProgress>.Continuation) {
        let task = Task.detached(priority: .utility) {
            for await progress in run.tracker.feed.stream() { continuation.yield(progress) }
            continuation.finish()
        }
        continuation.onTermination = { _ in task.cancel() }
    }

    func begin(_ run: PrepareRun) {
        let waiting = state.update { state -> [AsyncStream<PrepareProgress>.Continuation] in
            state.run = run
            state.ready = false
            defer { state.waiting.removeAll() }
            return Array(state.waiting.values)
        }
        for continuation in waiting { forward(run, to: continuation) }
    }

    func end(_ run: PrepareRun, ready: Bool) {
        state.update { state in
            if state.run === run { state.run = nil }
            state.ready = ready
        }
    }

    func cooled() {
        state.update { $0.ready = false }
    }
}

/// measured compile and warm-up times from this device, so the estimated parts of the bar fit it.
struct PrepareTimings: Codable {
    var compile: Double = 2
    var warmUp: Double = 1

    static var url: URL { ModelLocation.supportDirectory.appendingPathComponent("prepare-timings.json") }

    static func load() -> PrepareTimings {
        guard let data = try? Data(contentsOf: url), let timings = try? JSONDecoder().decode(PrepareTimings.self, from: data) else {
            return PrepareTimings()
        }
        return timings
    }

    func save() {
        try? FileManager.default.createDirectory(at: Self.url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(self) { try? data.write(to: Self.url, options: .atomic) }
    }
}
