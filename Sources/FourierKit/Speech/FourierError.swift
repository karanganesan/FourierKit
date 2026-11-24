import Foundation

/// everything the sdk can throw. each case names the real cause, and its description says what to do.
public enum FourierError: Error, Sendable, Hashable, LocalizedError, CustomStringConvertible {
    // microphone
    /// the user said no to the microphone
    case microphoneDenied
    /// a device policy blocks the microphone
    case microphoneRestricted
    /// permissions are `.manual` and the app has not asked yet
    case microphoneNotRequested
    /// there is no input device
    case noMicrophone
    /// the input device went away and capture could not recover
    case deviceRemoved
    /// the audio system refused to capture
    case captureFailed(String)
    /// the app is missing a usage description or an entitlement
    case setupMissing(SetupProblem)

    // model
    case downloadFailed(String)
    /// the downloaded weights do not match the pinned hash, even after a fresh download
    case hashMismatch
    case notEnoughDisk(needed: Int64, available: Int64)
    /// offline mode is on and the model is not on this mac
    case offlineModelMissing
    /// a revision other than the pinned one needs its own sha256 to be trusted
    case unverifiedRevision(String)
    case modelInvalid(String)

    // compute
    case unsupportedPin(stage: Stage, chip: Chip)
    case unsupportedPrecision(stage: Stage, precision: Precision)

    // input
    case unreadableAudio(String)
    case unsupportedLanguage(String)
    case invalidOptions(String)

    // running
    case cancelled
    /// the session already finished or was cancelled
    case sessionEnded
    case engineFailure(String)

    public enum SetupProblem: String, Sendable, Hashable {
        /// `NSMicrophoneUsageDescription` is missing from the app's info.plist
        case missingUsageDescription
        /// the app is sandboxed without `com.apple.security.device.audio-input`
        case missingAudioInputEntitlement
    }

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .microphoneDenied:
            "microphone access is off. turn it on in system settings > privacy & security > microphone."
        case .microphoneRestricted:
            "microphone access is blocked by a device policy. ask the person who manages this mac."
        case .microphoneNotRequested:
            "microphone access has not been asked for yet. call requestMicrophone() first."
        case .noMicrophone:
            "no microphone was found. connect one and try again."
        case .deviceRemoved:
            "the microphone was removed. connect it again or choose another input."
        case .captureFailed(let reason):
            "the microphone could not start (\(reason.lowercased())). try again or choose another input."
        case .setupMissing(.missingUsageDescription):
            "the app's info.plist has no NSMicrophoneUsageDescription. add one to use the microphone."
        case .setupMissing(.missingAudioInputEntitlement):
            "the app is sandboxed without com.apple.security.device.audio-input. add the entitlement to use the microphone."
        case .downloadFailed(let reason):
            "the model download failed (\(reason.lowercased())). check the connection and try again; it resumes where it stopped."
        case .hashMismatch:
            "the downloaded model did not match its checksum twice. check the connection and try again."
        case .notEnoughDisk(let needed, let available):
            "not enough disk space for the model: \(Self.gigabytes(needed)) needed, \(Self.gigabytes(available)) free. free some space and try again."
        case .offlineModelMissing:
            "offline mode is on and the model is not on this mac. turn offline mode off once to download it."
        case .unverifiedRevision(let revision):
            "revision \(revision) has no known checksum. pass its sha256 in model options to use it."
        case .modelInvalid(let reason):
            "the model files could not be read (\(reason.lowercased())). delete the model folder and prepare again."
        case .unsupportedPin(let stage, let chip):
            "no shipped path runs the \(stage.rawValue) on the \(chip.rawValue). remove the pin or use automatic placement."
        case .unsupportedPrecision(let stage, let precision):
            "no shipped path runs the \(stage.rawValue) in \(precision.rawValue). remove the pin or use automatic precision."
        case .unreadableAudio(let reason):
            "the audio could not be read (\(reason.lowercased())). use a file core audio can open."
        case .unsupportedLanguage(let language):
            "\(language.lowercased()) is not a language the model knows. use english or leave it unset."
        case .invalidOptions(let reason):
            "the options are not valid: \(reason.lowercased())."
        case .cancelled:
            "cancelled."
        case .sessionEnded:
            "this session has already ended. start a new one."
        case .engineFailure(let reason):
            "the engine failed (\(reason.lowercased())). try again; if it keeps failing, call cool() and retry."
        }
    }

    private static func gigabytes(_ bytes: Int64) -> String {
        String(format: "%.1f gb", Double(bytes) / 1e9)
    }

    /// maps anything thrown below the sdk to the case that names its cause.
    static func wrap(_ error: any Error) -> FourierError {
        switch error {
        case let error as FourierError: return error
        case is CancellationError: return .cancelled
        case let error as URLError where error.code == .cancelled: return .cancelled
        case HostFailure.cancelled: return .cancelled
        case HostFailure.notLoaded: return .engineFailure("the model is not loaded")
        case HostFailure.engine(let reason): return .engineFailure(reason)
        case let error as AudioFileError:
            switch error {
            case .unreadable(let reason), .conversionFailed(let reason): return .unreadableAudio(reason)
            }
        case let error as CaptureError: return wrap(capture: error)
        default: return .engineFailure(String(describing: error))
        }
    }

    static func wrap(capture error: CaptureError) -> FourierError {
        switch error {
        case .permissionNotRequested: .microphoneNotRequested
        case .permissionDenied: .microphoneDenied
        case .permissionRestricted: .microphoneRestricted
        case .missingUsageDescription: .setupMissing(.missingUsageDescription)
        case .missingAudioInputEntitlement: .setupMissing(.missingAudioInputEntitlement)
        case .noInputDevice: .noMicrophone
        case .engineFailed(let reason): .captureFailed(reason)
        }
    }
}
