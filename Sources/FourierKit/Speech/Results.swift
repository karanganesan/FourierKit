import Foundation

// MARK: transcript

/// the final text of a transcribe, record or listen.
public struct Transcript: Sendable, Hashable {
    /// the model's exact text, pieces joined with single spaces
    public var text: String
    /// the language the model was asked for
    public var language: String
    /// pieces or phrases, in order, each with its audio range
    public var pieces: [TranscriptPiece]
    /// length of the audio
    public var duration: Duration
    public var metrics: TranscriptMetrics

    public init(text: String, language: String, pieces: [TranscriptPiece], duration: Duration, metrics: TranscriptMetrics) {
        self.text = text
        self.language = language
        self.pieces = pieces
        self.duration = duration
        self.metrics = metrics
    }
}

public struct TranscriptPiece: Sendable, Hashable {
    public var text: String
    public var start: Duration
    public var end: Duration
    /// the same range as 16 khz sample indices into the audio
    public var samples: Range<Int>
    /// false if the piece ended without a stop token: it hit a `maxTokens` the caller set, or the
    /// checkpoint's limit and could not be split further, or it is live text standing in (see `isExact`)
    public var reachedStop: Bool
    /// false in a live session when this phrase's exact pass failed twice, so its settled live text
    /// stands in. always true for files and recordings, which throw instead.
    public var isExact: Bool

    public init(text: String, samples: Range<Int>, reachedStop: Bool = true, isExact: Bool = true) {
        self.text = text
        self.samples = samples
        self.start = .seconds(Double(samples.lowerBound) / 16_000)
        self.end = .seconds(Double(samples.upperBound) / 16_000)
        self.reachedStop = reachedStop
        self.isExact = isExact
    }
}

public struct TranscriptMetrics: Sendable, Hashable {
    /// seconds of audio per second of wall time, from the moment the engine was ready and the audio
    /// read to the result, so a cold first call's prepare (or download) is not counted. for a live
    /// session, where audio arrives in real time, it is the audio decoded over the time the engine
    /// worked on it.
    public var realtimeSpeed: Double = 0
    /// from the moment the engine was ready and the audio read to the first decoded token. for a
    /// live session, from the start of the session.
    public var timeToFirstText: Duration?
    /// the whole call, prepare included
    public var elapsed: Duration = .zero
    /// tokens generated, without the prompt
    public var tokens: Int = 0
    public var decoderSteps: Int = 0
    /// the chip each stage was set up to run on when the transcript was made: what was loaded, not
    /// a measurement of where the work ran
    public var placement: [Stage: Chip] = [:]
    /// the precision each stage was set up to run in, the same way
    public var precision: [Stage: Precision] = [:]
    public var widestBatch: Int = 0
    /// how many times a piece that hit the checkpoint's token limit was split in two and decoded
    /// again. a `maxTokens` the caller set is a cap and never splits.
    public var tokenLimitSplits: Int = 0
    /// time spent in each stage, summed over pieces
    public var melTime: Duration = .zero
    public var encodeTime: Duration = .zero
    public var prefillTime: Duration = .zero
    public var decodeTime: Duration = .zero

    public init() {}
}

/// the text of a file or a recording while it is still being transcribed.
public struct TranscriptionUpdate: Sendable, Hashable {
    /// text of the finished pieces from the start of the audio. never changes, only grows.
    public var settled: String
    /// what the engine currently has after that. may change, may be empty.
    public var draft: String
    /// settled + draft, the best full text so far
    public var text: String {
        settled.isEmpty || draft.isEmpty ? settled + draft : settled + " " + draft
    }

    public init(settled: String, draft: String) {
        self.settled = settled
        self.draft = draft
    }
}

/// raw 16 khz mono audio from record().
public struct Recording: Sendable, Hashable {
    public static let sampleRate = 16_000.0

    public var samples: [Float]
    public var startedAt: Date
    /// samples lost because the reader fell behind, normally 0
    public var droppedSamples: Int

    public var duration: Duration { .seconds(Double(samples.count) / Self.sampleRate) }

    public init(samples: [Float], startedAt: Date = Date(), droppedSamples: Int = 0) {
        self.samples = samples
        self.startedAt = startedAt
        self.droppedSamples = droppedSamples
    }
}

// MARK: live

public enum LiveEvent: Sendable, Equatable {
    /// the input is live
    case started
    /// input loudness for a meter, 0 (silence, -60 dbfs or quieter) to 1 (full scale). newest value only.
    case level(Float)
    /// voice detected
    case speechStarted
    /// new settled and draft text. a full snapshot: a slow reader gets the newest one.
    case update(LiveUpdate)
    /// a pause or the length limit closed a phrase
    case phraseEnded(LivePhrase)
    /// fell behind, input changed, and the like
    case warning(LiveWarning)
}

public struct LiveUpdate: Sendable, Hashable {
    /// never changes, only grows
    public var settled: String
    /// may change, may be empty
    public var draft: String
    /// settled + draft, the best full text so far
    public var text: String
    /// added to `settled` since the last update this reader got
    public var newlySettled: String
    public var phrases: [LivePhrase]
    /// how much audio has been heard
    public var audioTime: Duration
    /// how far the text is behind the voice
    public var delay: Duration

    public init(settled: String, draft: String, text: String, newlySettled: String, phrases: [LivePhrase],
                audioTime: Duration, delay: Duration) {
        self.settled = settled
        self.draft = draft
        self.text = text
        self.newlySettled = newlySettled
        self.phrases = phrases
        self.audioTime = audioTime
        self.delay = delay
    }
}

public struct LivePhrase: Sendable, Hashable, Identifiable {
    public var id: Int
    /// never changes, only grows
    public var settled: String
    /// empty once the phrase is closed
    public var draft: String
    public var start: Duration
    public var end: Duration
    /// a pause or the length limit closed it, and all of it is settled
    public var isClosed: Bool

    public var text: String {
        if settled.isEmpty { return draft }
        if draft.isEmpty { return settled }
        return settled + draft
    }

    public init(id: Int, settled: String, draft: String, start: Duration, end: Duration, isClosed: Bool) {
        self.id = id
        self.settled = settled
        self.draft = draft
        self.start = start
        self.end = end
        self.isClosed = isClosed
    }
}

public enum LiveWarning: Sendable, Hashable, CustomStringConvertible {
    /// passes took longer than the update interval, so updates slowed down. no audio was dropped.
    case fellBehind(updateInterval: Duration)
    /// audio was lost, e.g. while the input was rebuilt
    case audioDropped(Duration)
    /// the input route changed and capture was rebuilt
    case inputChanged(audioMayBeLost: Bool)
    /// the chosen input is gone, so capture follows the system default
    case inputDeviceUnavailable(String)
    /// voice processing could not start, so audio is not cleaned
    case voiceProcessingUnavailable(String)
    /// capture stopped for a moment
    case inputInterrupted
    /// the audio source has no more audio
    case inputEnded
    /// a phrase hit the length limit and was settled early
    case phraseTooLong

    public var description: String {
        switch self {
        case .fellBehind(let interval):
            "\(DeviceWords.thisDevice) fell behind, so text now updates every \(interval.inMilliseconds) ms. no audio was dropped."
        case .audioDropped(let duration):
            "\(duration.inMilliseconds) ms of audio was lost."
        case .inputChanged(let lost):
            lost ? "the input changed and some audio may be missing." : "the input changed."
        case .inputDeviceUnavailable:
            "the chosen microphone is not available, so the system default is used."
        case .voiceProcessingUnavailable:
            "voice processing could not start, so the audio is not cleaned."
        case .inputInterrupted:
            "the microphone was interrupted."
        case .inputEnded:
            "the audio source ended."
        case .phraseTooLong:
            "a long phrase was settled early."
        }
    }
}
