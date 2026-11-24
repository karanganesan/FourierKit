import Foundation

/// a fast energy gate for live 16 khz audio.
///
/// it works on 10 ms frames, keeps an adaptive noise floor, and uses two thresholds (hysteresis):
/// speech starts after `startFrames` loud frames and ends after `silenceSeconds` without a voiced frame.
/// deterministic: the same samples in any chunking give the same events.
struct SpeechDetector: Sendable {
    struct Options: Sendable, Equatable {
        /// silence that ends speech
        var silenceSeconds: Double = 0.6
        /// level above the noise floor that can start speech
        var startMarginDB: Float = 12
        /// level above the noise floor that keeps speech going
        var continueMarginDB: Float = 6
        /// consecutive loud frames needed to start speech
        var startFrames: Int = 3
        /// frames quieter than this never count as speech, whatever the floor
        var absoluteFloorDB: Float = -60
        /// how fast the floor rises toward louder background, in db per second
        var floorRiseDBPerSecond: Float = 3
        /// how fast the floor falls toward quieter background, as a fraction per frame
        var floorFallRate: Float = 0.2
        /// starting floor; nil learns it from the first frame, which suits a mic opened before anyone speaks
        var initialFloorDB: Float? = nil

        init() {}
    }

    enum Event: Sendable, Equatable {
        /// first sample of the run of frames that started speech
        case started(at: Int)
        /// one past the last voiced sample
        case ended(at: Int)
    }

    static let frameLength = 160
    static let sampleRate = 16_000.0

    let options: Options
    private(set) var isSpeaking = false
    /// current noise floor in dbfs, nil before the first frame
    private(set) var noiseFloorDB: Float?
    /// absolute position of the next sample fed
    private(set) var position: Int

    private var pending = [Float]()
    private var loudRun = 0
    private var loudRunStart = 0
    private var lastVoicedEnd = 0
    private let silenceFrames: Int

    init(options: Options = Options(), startPosition: Int = 0) {
        self.options = options
        noiseFloorDB = options.initialFloorDB
        position = startPosition
        silenceFrames = max(1, Int((options.silenceSeconds * Self.sampleRate / Double(Self.frameLength)).rounded()))
        pending.reserveCapacity(Self.frameLength)
    }

    /// feeds samples of any count and returns the events they caused, in order.
    mutating func feed(_ samples: [Float]) -> [Event] {
        samples.withUnsafeBufferPointer { feed($0) }
    }

    mutating func feed(_ samples: UnsafeBufferPointer<Float>) -> [Event] {
        var events = [Event]()
        var index = 0
        let n = Self.frameLength
        while index < samples.count {
            let take = min(n - pending.count, samples.count - index)
            pending.append(contentsOf: samples[index..<(index + take)])
            index += take
            if pending.count == n {
                let frameStart = position + index - n
                if let event = step(level: Self.levelDB(pending), frameStart: frameStart) {
                    events.append(event)
                }
                pending.removeAll(keepingCapacity: true)
            }
        }
        position += samples.count
        return events
    }

    /// ends speech now if it is going, e.g. when capture stops.
    mutating func flush() -> Event? {
        guard isSpeaking else { return nil }
        isSpeaking = false
        loudRun = 0
        return .ended(at: lastVoicedEnd)
    }

    private mutating func step(level: Float, frameStart: Int) -> Event? {
        let frameEnd = frameStart + Self.frameLength
        let floor = noiseFloorDB ?? level
        let loud = level > options.absoluteFloorDB && level > floor + options.startMarginDB
        let voiced = level > options.absoluteFloorDB && level > floor + options.continueMarginDB

        // a minimum follower: quick to fall into pauses, slow to rise, so speech barely lifts it
        // while a new steady background is learned within seconds
        let rise = options.floorRiseDBPerSecond * Float(Self.frameLength) / Float(Self.sampleRate)
        noiseFloorDB = level < floor ? floor + (level - floor) * options.floorFallRate : min(level, floor + rise)

        if isSpeaking {
            if voiced { lastVoicedEnd = frameEnd }
            if frameEnd - lastVoicedEnd >= silenceFrames * Self.frameLength {
                isSpeaking = false
                loudRun = 0
                return .ended(at: lastVoicedEnd)
            }
            return nil
        }

        guard loud else {
            loudRun = 0
            return nil
        }
        if loudRun == 0 { loudRunStart = frameStart }
        loudRun += 1
        guard loudRun >= options.startFrames else { return nil }
        isSpeaking = true
        lastVoicedEnd = frameEnd
        return .started(at: loudRunStart)
    }

    static func levelDB(_ frame: [Float]) -> Float {
        var sum: Float = 0
        for x in frame { sum += x * x }
        let rms = (sum / Float(frame.count)).squareRoot()
        return 20 * log10(max(rms, 1e-9))
    }
}
