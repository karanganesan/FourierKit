import Foundation

// MARK: protocols

/// where record() and listen() get audio. the default is `MicrophoneSource`.
///
/// a source delivers 16 khz mono float chunks, in order, off the main thread.
public protocol AudioSource: Sendable {
    /// true if starting needs microphone permission; the sdk then asks first (per `PermissionHandling`)
    var usesMicrophone: Bool { get }
    /// starts delivering audio. called off the main thread.
    func start(_ request: AudioSourceRequest) async throws -> AudioInput
}

public struct AudioSourceRequest: Sendable, Hashable {
    public enum Purpose: String, Sendable, Hashable {
        case listen, record
    }

    public var purpose: Purpose
    /// apple's voice processing: noise suppression and echo cancellation
    public var voiceProcessing: Bool
    /// the system's automatic gain, only with voice processing
    public var automaticGain: Bool
    /// core audio device uid, nil follows the system default input
    public var inputDevice: String?
    public var preRoll: Duration

    public init(purpose: Purpose, voiceProcessing: Bool, automaticGain: Bool, inputDevice: String?, preRoll: Duration) {
        self.purpose = purpose
        self.voiceProcessing = voiceProcessing
        self.automaticGain = automaticGain
        self.inputDevice = inputDevice
        self.preRoll = preRoll
    }
}

/// a running source.
public struct AudioInput: Sendable {
    /// 16 khz mono chunks, in order. ends when the source stops or runs out.
    public var samples: AsyncStream<[Float]>
    public var events: AsyncStream<AudioInputEvent>
    /// stops the source. the samples stream ends after the audio already captured.
    public var stop: @Sendable () async -> Void

    public init(samples: AsyncStream<[Float]>, events: AsyncStream<AudioInputEvent>, stop: @escaping @Sendable () async -> Void) {
        self.samples = samples
        self.events = events
        self.stop = stop
    }
}

public enum AudioInputEvent: Sendable, Hashable {
    case capture(MicrophoneState.Capture)
    /// the route changed and capture was rebuilt
    case inputChanged(audioMayBeLost: Bool)
    case deviceUnavailable(String)
    case voiceProcessingUnavailable(String)
    /// samples lost because the reader fell behind
    case samplesDropped(Int)
}

/// cleans live audio. one copy per session; it runs off the main thread and never on the realtime audio thread.
public protocol NoiseProcessor: Sendable {
    /// 16 khz mono in, the same count out
    mutating func process(_ samples: [Float]) -> [Float]
}

/// finds speech in live audio. one copy per session, fed off the main thread.
public protocol SpeechDetecting: Sendable {
    /// called once before the first samples
    mutating func configure(pauseToEndPhrase: Duration)
    /// feeds 16 khz mono samples of any count, in order. positions count samples from the start of the session.
    mutating func feed(_ samples: [Float]) -> [SpeechBoundary]
    /// ends speech now if it is going, e.g. when the input stops
    mutating func flush() -> SpeechBoundary?
}

extension SpeechDetecting {
    public mutating func configure(pauseToEndPhrase: Duration) {}
}

public enum SpeechBoundary: Sendable, Hashable {
    /// first sample of speech
    case started(at: Int)
    /// one past the last voiced sample
    case ended(at: Int)
}

// MARK: default parts

/// the system microphone, rebuilt automatically when the input changes.
public struct MicrophoneSource: AudioSource {
    public init() {}

    public var usesMicrophone: Bool { true }

    public func start(_ request: AudioSourceRequest) async throws -> AudioInput {
        var options = CaptureOptions()
        options.voiceProcessing = request.voiceProcessing
        options.automaticGainControl = request.automaticGain
        options.preferredDeviceUID = request.inputDevice
        options.preRollSeconds = request.preRoll.inSeconds
        let capture = MicrophoneCapture(options: options)
        do {
            // start blocks briefly while the audio engine comes up
            try await offload { try capture.start() }
        } catch let error as CaptureError {
            capture.stop()
            throw FourierError.wrap(capture: error)
        }
        let (events, continuation) = AsyncStream.makeStream(of: AudioInputEvent.self, bufferingPolicy: .bufferingNewest(64))
        let forward = Task.detached(priority: .userInitiated) {
            for await event in capture.events {
                switch event {
                case .stateChanged(let state):
                    switch state {
                    case .off: continuation.yield(.capture(.off))
                    case .starting: continuation.yield(.capture(.starting))
                    case .on: continuation.yield(.capture(.on))
                    case .interrupted: continuation.yield(.capture(.interrupted))
                    case .failed(let error): continuation.yield(.capture(.failed(FourierError.wrap(capture: error))))
                    }
                case .configurationChanged(let lost): continuation.yield(.inputChanged(audioMayBeLost: lost))
                case .preferredDeviceUnavailable(let uid): continuation.yield(.deviceUnavailable(uid))
                case .voiceProcessingUnavailable(let reason): continuation.yield(.voiceProcessingUnavailable(reason))
                case .samplesDropped(let count): continuation.yield(.samplesDropped(count))
                }
            }
            continuation.finish()
        }
        _ = forward
        return AudioInput(samples: capture.chunks, events: events) {
            await offload { capture.stop() }
        }
    }
}

/// plays a file or samples in as if spoken, without a microphone or a permission prompt.
public struct FileAudioSource: AudioSource {
    private enum Content: Sendable {
        case url(URL)
        case samples([Float], Double)
    }

    private let content: Content
    /// true paces the audio at real speed, false delivers it as fast as it is read
    public var realtime: Bool
    /// chunk length delivered on each step
    public var chunk: Duration = .milliseconds(20)

    public init(url: URL, realtime: Bool = true) {
        content = .url(url)
        self.realtime = realtime
    }

    public init(samples: [Float], sampleRate: Double = 16_000, realtime: Bool = true) {
        content = .samples(samples, sampleRate)
        self.realtime = realtime
    }

    public var usesMicrophone: Bool { false }

    public func start(_ request: AudioSourceRequest) async throws -> AudioInput {
        let samples: [Float]
        switch content {
        case .url(let url):
            do {
                samples = try await offload { try AudioFileReader.read(url) }
            } catch {
                throw FourierError.wrap(error)
            }
        case .samples(let raw, let rate):
            samples = rate == 16_000 ? raw : await offload { Resampler.resample(raw, from: rate) }
        }
        let (stream, continuation) = AsyncStream.makeStream(of: [Float].self, bufferingPolicy: .unbounded)
        let (events, eventContinuation) = AsyncStream.makeStream(of: AudioInputEvent.self)
        let step = max(1, Int(chunk.inSeconds * 16_000))
        let realtime = realtime
        let player = Task.detached(priority: .userInitiated) {
            eventContinuation.yield(.capture(.on))
            let clock = ContinuousClock()
            let started = clock.now
            var position = 0
            while position < samples.count, !Task.isCancelled {
                let end = min(samples.count, position + step)
                if realtime {
                    // the chunk is delivered once it has been "heard"
                    let due = started + .seconds(Double(end) / 16_000)
                    try? await Task.sleep(until: due, clock: clock)
                    if Task.isCancelled { break }
                }
                continuation.yield(Array(samples[position..<end]))
                position = end
            }
            continuation.finish()
            eventContinuation.yield(.capture(.off))
            eventContinuation.finish()
        }
        return AudioInput(samples: stream, events: events) {
            player.cancel()
            _ = await player.value
        }
    }
}

/// the built-in speech detector: a fast energy gate with an adaptive noise floor.
///
/// apple's sound analysis classifier could veto false starts here; it is not wired in yet.
public struct EnergySpeechDetector: SpeechDetecting {
    private var detector: SpeechDetector
    /// nil follows the listen option
    public var pauseToEndPhrase: Duration?
    private var configured = false

    public init(pauseToEndPhrase: Duration? = nil) {
        self.pauseToEndPhrase = pauseToEndPhrase
        var options = SpeechDetector.Options()
        if let pauseToEndPhrase { options.silenceSeconds = pauseToEndPhrase.inSeconds }
        detector = SpeechDetector(options: options)
    }

    public mutating func configure(pauseToEndPhrase pause: Duration) {
        var options = SpeechDetector.Options()
        options.silenceSeconds = (pauseToEndPhrase ?? pause).inSeconds
        detector = SpeechDetector(options: options)
        configured = true
    }

    public mutating func feed(_ samples: [Float]) -> [SpeechBoundary] {
        detector.feed(samples).map { event in
            switch event {
            case .started(let at): .started(at: at)
            case .ended(let at): .ended(at: at)
            }
        }
    }

    public mutating func flush() -> SpeechBoundary? {
        detector.flush().map { event in
            switch event {
            case .started(let at): .started(at: at)
            case .ended(let at): .ended(at: at)
            }
        }
    }
}
