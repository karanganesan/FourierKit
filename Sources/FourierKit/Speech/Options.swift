import Foundation

/// every knob, each with a default. `FourierOptions()` is the recommended setup.
public struct FourierOptions: Sendable {
    // compute
    public var compute = ComputeOptions()

    // decoding, for transcribe and record. the default is the checkpoint's own settings, which is the exact path.
    public var decoding = DecodingOptions()

    // audio
    /// where record() and listen() get audio. the default is the system microphone.
    public var audioSource: any AudioSource = MicrophoneSource()
    /// core audio device uid, nil follows the system default input
    public var inputDevice: String?
    /// how audio longer than 120 s is cut into pieces
    public var chunking: Chunking = .automatic

    /// the defaults for listen(). the audio cleanup knobs below forward to it.
    public var listen = ListenOptions()

    /// noise suppression for live audio. files and recordings are never cleaned.
    public var noise: NoiseProcessing {
        get { listen.noise }
        set { listen.noise = newValue }
    }
    public var echoCancellation: AudioSwitch {
        get { listen.echoCancellation }
        set { listen.echoCancellation = newValue }
    }
    public var gain: Gain {
        get { listen.gain }
        set { listen.gain = newValue }
    }
    /// nil uses the built-in energy gate
    public var speechDetector: (any SpeechDetecting)? {
        get { listen.speechDetector }
        set { listen.speechDetector = newValue }
    }
    public var preRoll: Duration {
        get { listen.preRoll }
        set { listen.preRoll = newValue }
    }

    // lifecycle
    public var keepWarm: KeepWarm = .automatic
    public var permissions: PermissionHandling = .automatic

    // model
    public var model = ModelOptions()
    /// where the model files come from. the default downloads the pinned checkpoint and checks its hash.
    public var modelStore: any ModelStore = HuggingFaceModelStore()

    public init() {}
}

// MARK: compute

public struct ComputeOptions: Sendable, Hashable {
    public enum PlacementPolicy: String, Sendable, Hashable {
        /// the fastest measured chip for each stage
        case automatic
        /// prefer the neural engine where a path exists
        case npu
        /// prefer the lowest power path
        case efficiency
    }

    public var policy: PlacementPolicy = .automatic
    /// hard pins. a pin no shipped path can satisfy fails prepare with `FourierError.unsupportedPin`.
    public var chips: [Stage: Chip] = [:]
    /// hard pins. a pin no shipped path can satisfy fails prepare with `FourierError.unsupportedPrecision`.
    public var precision: [Stage: Precision] = [:]

    public init(policy: PlacementPolicy = .automatic, chips: [Stage: Chip] = [:], precision: [Stage: Precision] = [:]) {
        self.policy = policy
        self.chips = chips
        self.precision = precision
    }
}

// MARK: decoding

public struct DecodingOptions: Sendable, Hashable {
    public enum Search: Sendable, Hashable {
        /// the checkpoint's own search (beam 3), as the reference runs it
        case automatic
        case greedy
        case beam(Int)
    }

    public var search: Search = .automatic
    /// nil uses the checkpoint's limit (512)
    public var maxTokens: Int?
    /// no-repeat n-gram size. nil uses the checkpoint's (8), 0 turns it off.
    public var repeatGuard: Int?
    /// a name or code, e.g. "English" or "en"
    public var language: String = "English"
    /// text the model reads before the audio, e.g. names and terms. nil is the exact path.
    public var context: String?

    public init(search: Search = .automatic, maxTokens: Int? = nil, repeatGuard: Int? = nil,
                language: String = "English", context: String? = nil) {
        self.search = search
        self.maxTokens = maxTokens
        self.repeatGuard = repeatGuard
        self.language = language
        self.context = context
    }

    /// the checkpoint's settings: the exact path
    public static let reference = DecodingOptions()
    /// greedy, for live drafts
    public static let fast = DecodingOptions(search: .greedy)
}

// MARK: audio

public enum Chunking: Sendable, Hashable {
    /// audio up to 120 s runs in one pass; longer audio is cut at quiet points into pieces of at most 30 s
    case automatic
    /// longer audio is cut into pieces of at most this length (1 to 120 s). audio up to 120 s is never cut.
    case maxPieceLength(Duration)
}

public enum AudioSwitch: String, Sendable, Hashable {
    case automatic, on, off
}

public enum Gain: Sendable, Hashable {
    /// the system's automatic gain while voice processing runs, unity otherwise
    case automatic
    /// a fixed gain in decibels, applied after capture
    case fixed(decibels: Float)
}

/// how live audio is cleaned before it reaches the model.
public enum NoiseProcessing: Sendable {
    /// apple's voice processing on the microphone: noise suppression, echo cancellation and gain
    case automatic
    case off
    /// your processor runs on each chunk, off the main thread and never on the realtime audio thread
    case custom(any NoiseProcessor)
}

// MARK: lifecycle

public enum KeepWarm: String, Sendable, Hashable {
    /// stays loaded after prepare(); cools after 5 idle minutes, under memory pressure or when the mac is hot
    case automatic
    /// stays loaded until cool() or critical memory pressure
    case always
    /// lets go of the model as soon as nothing is running
    case never
}

public enum PermissionHandling: Sendable {
    /// the first listen() or record() asks for the microphone
    case automatic
    /// the sdk never prompts. call requestMicrophone() yourself.
    case manual
    /// called when the permission was never asked; return the permission after your own flow
    case custom(@Sendable () async -> MicrophoneState.Permission)
}

// MARK: model

public struct ModelOptions: Sendable, Hashable {
    /// where the model is stored. nil is application support/FourierKit/models.
    public var directory: URL?
    public var repository: String = ModelLocation.repository
    public var revision: String = ModelLocation.revision
    /// the weights' sha256. nil uses the pinned one, which only fits the pinned revision.
    public var sha256: String?
    /// never touches the network. prepare fails if the model is not on this mac.
    public var offline = false

    /// `revision` nil is the pinned checkpoint
    public init(directory: URL? = nil, revision: String? = nil, sha256: String? = nil, offline: Bool = false) {
        self.directory = directory
        self.revision = revision ?? ModelLocation.revision
        self.sha256 = sha256
        self.offline = offline
    }

    /// the pinned checkpoint
    public static var pinnedRevision: String { ModelLocation.revision }
    public static var defaultDirectory: URL { ModelLocation.defaultDirectory }

    /// the folder that holds the files for this repository and revision
    var resolvedDirectory: URL {
        if let directory { return directory }
        let root = ModelLocation.supportDirectory
            .appendingPathComponent("models/\(repository.replacingOccurrences(of: "/", with: "--"))/\(revision)", isDirectory: true)
        return root
    }
}

// MARK: listen

public struct ListenOptions: Sendable {
    public enum UpdateInterval: Sendable, Hashable {
        /// starts near 300 ms and follows measured pass time, so delay never grows
        case automatic
        case fixed(Duration)
    }

    public enum FinalPass: String, Sendable, Hashable {
        /// each phrase gets an exact pass as soon as it closes, so finish() only waits for the last one
        case perPhrase
        /// one exact pass over everything heard, at finish()
        case fullAudio
    }

    public struct Decoding: Sendable, Hashable {
        /// live drafts: fast
        public var draft: DecodingOptions = .fast
        /// the final transcript: the checkpoint's settings
        public var final: DecodingOptions = .reference
        public init(draft: DecodingOptions = .fast, final: DecodingOptions = .reference) {
            self.draft = draft
            self.final = final
        }
    }

    /// show the changing guess as well as settled text
    public var drafts = true
    public var updateInterval: UpdateInterval = .automatic
    /// agreeing passes before text settles
    public var settleAfter = 2
    /// silence that closes a phrase
    public var pauseToEndPhrase: Duration = .milliseconds(600)
    /// longest phrase before a forced settle
    public var maxPhraseLength: Duration = .seconds(24)
    public var noise: NoiseProcessing = .automatic
    /// voice processing cancels echo and suppresses noise together; automatic follows `noise`
    public var echoCancellation: AudioSwitch = .automatic
    public var gain: Gain = .automatic
    /// nil uses the built-in energy gate with `pauseToEndPhrase`
    public var speechDetector: (any SpeechDetecting)?
    /// audio kept from before speech starts
    public var preRoll: Duration = .milliseconds(500)
    /// `.never` lets go of the model when the session ends
    public var keepWarm: KeepWarm = .automatic
    public var decoding = Decoding()
    public var finalPass: FinalPass = .perPhrase
    /// nil uses the Fourier's audio source
    public var audioSource: (any AudioSource)?
    /// core audio device uid, nil uses the Fourier's
    public var inputDevice: String?

    /// a name or code for drafts and the final pass
    public var language: String {
        get { decoding.final.language }
        set {
            decoding.final.language = newValue
            decoding.draft.language = newValue
        }
    }

    public init() {}
}
