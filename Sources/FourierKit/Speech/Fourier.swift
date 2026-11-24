import Foundation

/// speech to text on this mac. one engine per process: every instance shares the same weights.
///
/// safe to call from the main actor. every heavy call is async and leaves the caller's actor at once;
/// streams are produced off the main actor and coalesced to at most one value per frame.
public final class Fourier: Sendable {
    /// the sdk's version. the package itself is versioned by its git tag, which matches this.
    public static let version = "0.0.1"

    public let options: FourierOptions
    private let runtime = Runtime.shared

    public init(_ options: FourierOptions = FourierOptions()) {
        self.options = options
    }

    // MARK: prepare

    /// downloads, verifies, loads and warms the model. optional: every mode prepares on first use.
    /// calling it again joins the same run; the work stops only when the last waiting caller cancels.
    @concurrent
    public func prepare() async throws {
        do {
            try await runtime.ensureReady(options)
        } catch {
            throw FourierError.wrap(error)
        }
    }

    /// the current or next prepare run, from 0 to 1. ends when the run ends; a ready model gives 1 at once.
    public var prepareProgress: AsyncStream<PrepareProgress> {
        runtime.prepareFeed.stream()
    }

    /// loads and warms the model now, the same as prepare().
    @concurrent
    public func warm() async throws {
        await runtime.setKeepWarm(options.keepWarm)
        try await prepare()
    }

    /// lets go of the model's memory if nothing is running. the next use warms it again.
    @concurrent
    public func cool() async {
        await runtime.cool()
    }

    // MARK: transcribe

    /// the model's exact text for a file. up to 120 s is one pass; longer audio is cut at quiet points.
    @concurrent
    public func transcribe(contentsOf url: URL) async throws -> Transcript {
        try await wait(for: transcriptionJob(contentsOf: url))
    }

    @concurrent
    public func transcribe(_ recording: Recording) async throws -> Transcript {
        try await wait(for: transcriptionJob(recording))
    }

    /// mono samples at any rate; they are resampled to 16 khz if needed.
    @concurrent
    public func transcribe(samples: [Float], sampleRate: Double = 16_000) async throws -> Transcript {
        try await wait(for: transcriptionJob(samples: samples, sampleRate: sampleRate))
    }

    /// starts a transcription with its own progress, result and cancel.
    public func transcriptionJob(contentsOf url: URL) -> TranscriptionJob {
        TranscriptionJob(input: .file(url), options: options)
    }

    public func transcriptionJob(_ recording: Recording) -> TranscriptionJob {
        TranscriptionJob(input: .samples(recording.samples, sampleRate: Recording.sampleRate), options: options)
    }

    public func transcriptionJob(samples: [Float], sampleRate: Double = 16_000) -> TranscriptionJob {
        TranscriptionJob(input: .samples(samples, sampleRate: sampleRate), options: options)
    }

    private func wait(for job: TranscriptionJob) async throws -> Transcript {
        try await withTaskCancellationHandler {
            try await job.result
        } onCancel: {
            job.cancel()
        }
    }

    // MARK: record

    /// raw capture with a live level. asks for the microphone the first time, as `options.permissions` allows.
    @concurrent
    public func record() async throws -> Recorder {
        let source = options.audioSource
        do {
            if source.usesMicrophone { try await runtime.ensureMicrophone(options.permissions) }
            let job = await runtime.register(.record, phase: .capturing)
            let input: AudioInput
            do {
                input = try await source.start(AudioSourceRequest(
                    purpose: .record, voiceProcessing: false, automaticGain: false,
                    inputDevice: options.inputDevice, preRoll: options.preRoll))
            } catch {
                await runtime.end(job, error: FourierError.wrap(error))
                throw error
            }
            if source.usesMicrophone { await runtime.setCapture(.on, inputUID: .some(options.inputDevice)) }
            // the engine warms while recording, so the transcript can start at once; never a download
            if runtime.modelLooksPresent(options) {
                let options = options
                Task.detached(priority: .utility) { try? await Runtime.shared.ensureReady(options) }
            }
            return Recorder(input: input, usesMicrophone: source.usesMicrophone, job: job)
        } catch {
            throw FourierError.wrap(error)
        }
    }

    // MARK: listen

    /// live text while you speak, with the default listen options.
    @concurrent
    public func listen() async throws -> LiveSession {
        try await listen(options.listen)
    }

    @concurrent
    public func listen(_ listenOptions: ListenOptions) async throws -> LiveSession {
        guard listenOptions.settleAfter >= 1 else { throw FourierError.invalidOptions("settleAfter must be at least 1") }
        let source = listenOptions.audioSource ?? options.audioSource
        do {
            if source.usesMicrophone { try await runtime.ensureMicrophone(options.permissions) }
            try await runtime.ensureReady(options)
            let draftPlan = try await DecodePlan.make(listenOptions.decoding.draft, priority: .live)
            let finalPlan = try await DecodePlan.make(listenOptions.decoding.final, priority: .file)
            guard let tokenizer = await EngineHost.shared.tokenizer else { throw FourierError.engineFailure("the model is not loaded") }
            let job = await runtime.register(.listen, phase: .capturing)
            let channel = EventChannel()
            let engine = LiveEngine(
                id: job, options: listenOptions, inputDevice: listenOptions.inputDevice ?? options.inputDevice,
                channel: channel, usesMicrophone: source.usesMicrophone, tokenizer: tokenizer,
                draftPlan: draftPlan, finalPlan: finalPlan)
            do {
                try await engine.start(source: source)
            } catch {
                await runtime.end(job, error: FourierError.wrap(error))
                throw error
            }
            return LiveSession(id: job, engine: engine, channel: channel)
        } catch {
            throw FourierError.wrap(error)
        }
    }

    // MARK: state

    /// the latest snapshot
    public var state: FourierState {
        get async { await runtime.snapshot }
    }

    /// every change, coalesced to one per frame, newest value only
    public var states: AsyncStream<FourierState> {
        runtime.states.stream()
    }

    /// live numbers about four times a second, collected only while something reads them
    public var metrics: AsyncStream<FourierMetrics> {
        runtime.metricsFeed.stream()
    }

    // MARK: microphone

    /// the latest microphone snapshot. never prompts.
    public var microphone: MicrophoneState {
        get async { await runtime.refreshMicrophone() }
    }

    /// permission, capture and device changes. never prompts.
    public var microphoneStates: AsyncStream<MicrophoneState> {
        let stream = runtime.microphoneStates.stream()
        Task { await Runtime.shared.watchDevices() }
        return stream
    }

    /// input level while record() or listen() runs, 0 (silence) to 1 (full scale), newest value only
    public var microphoneLevels: AsyncStream<Float> {
        runtime.levels.stream()
    }

    /// shows the system prompt only if the user was never asked, and returns the permission.
    @concurrent
    public func requestMicrophone() async -> MicrophoneState.Permission {
        if MicrophoneAccess.permission == .notDetermined, MicrophoneAccess.setupProblems().isEmpty {
            _ = await MicrophoneAccess.request()
        }
        return await runtime.refreshMicrophone().permission
    }

    /// opens privacy & security > microphone in system settings.
    @discardableResult
    public func openMicrophoneSettings() -> Bool {
        MicrophoneAccess.openSettings()
    }
}
