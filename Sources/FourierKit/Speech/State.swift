import Foundation

// MARK: compute vocabulary

/// a step of the pipeline that runs on one chip.
public enum Stage: String, Sendable, Hashable, CaseIterable, Codable {
    /// log-mel features from 16 khz audio
    case mel
    case encoder
    case decoder
    /// greedy or beam search, and text
    case search
}

public enum Chip: String, Sendable, Hashable, CaseIterable, Codable {
    case cpu
    case gpu
    /// the neural engine
    case npu
}

public enum Precision: String, Sendable, Hashable, CaseIterable, Codable {
    case automatic
    case float32
    case float16
    /// bf16 weights read as stored, with 32-bit accumulation
    case bfloat16
}

// MARK: state

/// one snapshot of everything the sdk is doing. an app can draw its whole ui from it.
public struct FourierState: Sendable, Equatable {
    public var engine: EngineState = .cold
    /// nil when not preparing
    public var prepare: PrepareProgress?
    /// each running listen, record and transcribe
    public var jobs: [JobState] = []
    public var microphone: MicrophoneState = MicrophoneState()
    public var speech: SpeechActivity = .silent
    public var pressure: SystemPressure = .normal
    /// planned chip for each stage
    public var placement: [Stage: Chip] = [:]
    /// precision for each stage
    public var precision: [Stage: Precision] = [:]
    /// the last error that stopped prepare or a job, cleared by the next success
    public var error: FourierError?

    public init() {}
}

public enum EngineState: String, Sendable, Hashable {
    /// not loaded
    case cold
    /// checking, downloading or verifying the model
    case preparing
    /// compiling, loading and warming up
    case loading
    /// loaded and ready to start at once
    case warm
    /// running work
    case busy
    /// letting go of memory
    case cooling
    case failed
}

public struct JobState: Sendable, Hashable, Identifiable {
    public enum Kind: String, Sendable, Hashable {
        case transcribe, record, listen
    }

    public enum Phase: String, Sendable, Hashable {
        /// waiting for the model to be ready
        case preparing
        /// reading or decoding the audio
        case reading
        /// capturing from the input
        case capturing
        /// running through the model
        case running
        /// waiting for the last exact passes
        case finishing
    }

    public var id: UUID
    public var kind: Kind
    public var phase: Phase
    /// 0...1 when it can be known, nil for open-ended work like listening
    public var progress: Double?
    /// audio heard or read so far
    public var audioDuration: Duration
    public var startedAt: Date
}

public enum SpeechActivity: String, Sendable, Hashable {
    case silent, speaking
}

public enum SystemPressure: String, Sendable, Hashable {
    case normal
    /// memory is getting tight
    case busy
    /// the mac is hot
    case hot
    /// low power mode is on
    case lowPower
}

// MARK: metrics

public enum Workload: String, Sendable, Hashable {
    case idle
    /// under 5 s
    case short
    /// 5 to 30 s
    case medium
    /// 30 to 120 s
    case long
    /// past 120 s
    case veryLong
    case manyFiles
    case realtime
    case realtimePlusFiles
    /// hot, on battery saver or short of memory
    case underLoad
}

/// live numbers, collected only while something reads them.
public struct FourierMetrics: Sendable, Equatable {
    /// seconds of audio finished per second of wall time, over the last few seconds
    public var realtimeSpeed: Double = 0
    public var tokensPerSecond: Double = 0
    public var decoderStepsPerSecond: Double = 0
    public var timeToFirstText: Duration?
    /// how far live draft text is behind the voice
    public var draftDelay: Duration?
    /// how long live text takes to settle
    public var settleDelay: Duration?
    public var workload: Workload = .idle
    /// decoder rows in the current step
    public var batchWidth: Int = 0
    /// the widest batch allowed right now
    public var maxBatchWidth: Int = 0
    public var queuedPieces: Int = 0
    public var activePieces: Int = 0
    /// share of recent time each chip spent on engine calls, 0...1
    public var chipBusy: [Chip: Double] = [:]
    /// the app's physical footprint
    public var appMemory: UInt64 = 0
    public var weightMemory: UInt64 = 0
    /// decoder key-value cache in use
    public var cacheMemory: UInt64 = 0
    /// what the engine measured on this device and the sizes it chose from that, nil until it loads
    public var device: DeviceReport?

    public init() {}
}

/// the device facts the engine measured when it loaded, and what it derived from them.
public struct DeviceReport: Sendable, Hashable {
    // measured
    public var gpuName: String
    public var physicalMemory: UInt64
    /// how much more the app could use when the engine loaded
    public var memoryHeadroom: UInt64
    public var processorCount: Int
    public var performanceCores: Int?
    public var maxBufferLength: UInt64
    public var gpuWorkingSet: UInt64
    public var hasUnifiedMemory: Bool
    public var maxThreadsPerThreadgroup: Int
    public var hasNeuralEngine: Bool

    // derived
    /// 1 when the checkpoint fits one gpu buffer, more when it is wrapped in windows
    public var weightBuffers: Int
    public var cacheSlots: Int
    public var cacheBytes: UInt64
    public var maxBatchWidth: Int
    public var pressuredBatchWidth: Int
    public var encoderChunkGroup: Int
    public var automaticPieceLength: Duration
    public var offersNeuralEngine: Bool

    init(_ profile: DeviceProfile, _ limits: EngineLimits, weightBuffers: Int, slotCapacity: Int) {
        gpuName = profile.gpuName
        physicalMemory = UInt64(profile.physicalMemory)
        memoryHeadroom = UInt64(profile.memoryHeadroom)
        processorCount = profile.activeProcessorCount
        performanceCores = profile.performanceCores
        maxBufferLength = UInt64(profile.maxBufferLength)
        gpuWorkingSet = UInt64(profile.recommendedMaxWorkingSetSize)
        hasUnifiedMemory = profile.hasUnifiedMemory
        maxThreadsPerThreadgroup = profile.maxThreadsPerThreadgroup
        hasNeuralEngine = profile.hasNeuralEngine
        self.weightBuffers = weightBuffers
        cacheSlots = slotCapacity
        cacheBytes = UInt64(slotCapacity * limits.bytesPerSlot)
        maxBatchWidth = limits.maxRows
        pressuredBatchWidth = limits.pressuredMaxRows
        encoderChunkGroup = limits.convGroup
        automaticPieceLength = .seconds(limits.automaticPieceSeconds)
        offersNeuralEngine = limits.offersNeuralEngine
    }
}

// MARK: microphone

public struct MicrophoneState: Sendable, Hashable {
    public enum Permission: String, Sendable, Hashable {
        case notDetermined, granted, denied, restricted
    }

    public enum Capture: Sendable, Hashable {
        case off, starting, on, interrupted
        case failed(FourierError)
    }

    public var permission: Permission = .notDetermined
    public var capture: Capture = .off
    /// the current input, nil if there is none
    public var device: InputDevice?
    /// available inputs
    public var devices: [InputDevice] = []
    /// the usage text and the audio entitlement are present
    public var isSetupValid: Bool = true

    public init() {}
}

public struct InputDevice: Sendable, Hashable, Identifiable {
    /// the core audio uid, stable across launches
    public var id: String
    public var name: String
    /// the system default input
    public var isDefault: Bool

    public init(id: String, name: String, isDefault: Bool) {
        self.id = id
        self.name = name
        self.isDefault = isDefault
    }
}

// MARK: prepare

/// one smooth value from 0 to 1 covering download to warm-up.
public struct PrepareProgress: Sendable, Hashable {
    public enum Stage: String, Sendable, Hashable, CaseIterable {
        case check, download, verify, convert, compile, load, warmUp, ready

        /// lowercase, for display
        public var name: String {
            switch self {
            case .check: "checking files"
            case .download: "downloading"
            case .verify: "verifying"
            case .convert: "converting"
            case .compile: "compiling"
            case .load: "loading"
            case .warmUp: "warming up"
            case .ready: "ready"
            }
        }

        /// starting shares of the bar, from the plan
        var share: Double {
            switch self {
            case .check: 0.01
            case .download: 0.60
            case .verify: 0.05
            case .convert: 0.20
            case .compile: 0.08
            case .load: 0.03
            case .warmUp: 0.03
            case .ready: 0
            }
        }

        var order: Int { Self.allCases.firstIndex(of: self)! }

        var start: Double {
            Self.allCases.prefix { $0 != self }.reduce(0) { $0 + $1.share }
        }
    }

    /// 0...1, never goes backward, 1 only once warm-up succeeded
    public var fraction: Double
    public var stage: Stage
    /// during download
    public var bytesDone: Int64
    public var bytesTotal: Int64
    /// nil until it can be estimated
    public var timeRemaining: Duration?

    public init(fraction: Double, stage: Stage, bytesDone: Int64 = 0, bytesTotal: Int64 = 0, timeRemaining: Duration? = nil) {
        self.fraction = fraction
        self.stage = stage
        self.bytesDone = bytesDone
        self.bytesTotal = bytesTotal
        self.timeRemaining = timeRemaining
    }
}
