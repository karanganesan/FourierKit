import FourierKit
import Foundation
import Observation

/// one listen session at a time, on the main actor: start, stop, live text, level and the final transcript.
///
/// `FourierLiveView` drives one of these. apps that put start and stop elsewhere (a toolbar, a menu, a key)
/// share the same model with the view.
@Observable
@MainActor
public final class FourierListenModel {
    public enum Phase: Sendable, Hashable {
        case idle
        /// preparing the model, asking for the microphone, starting capture
        case starting
        case listening
        /// waiting for the last phrase's exact pass
        case finishing
    }

    public var speech: Fourier
    /// options for the next session. nil uses `speech.options.listen`.
    public var options: ListenOptions?
    /// called with the exact transcript when a session finishes
    @ObservationIgnored public var onFinish: (@MainActor (Transcript) -> Void)?

    public private(set) var phase: Phase = .idle
    /// the newest live text, nil before the first update of a session
    public private(set) var update: LiveUpdate?
    /// input level for a meter, 0...1
    public private(set) var level: Float = 0
    /// voice heard in the current phrase
    public private(set) var isSpeaking = false
    /// the exact transcript of the last finished session
    public private(set) var transcript: Transcript?
    /// the newest warning of the current session
    public private(set) var warning: LiveWarning?
    /// why the last start or finish failed
    public private(set) var error: FourierError?
    /// when the current session started capturing
    public private(set) var startedAt: Date?

    public var isRunning: Bool { phase != .idle }

    /// the best text to show: the final transcript once finished, otherwise settled and draft text
    public var settledText: String { transcript?.text ?? update?.settled ?? "" }
    public var draftText: String { transcript == nil ? update?.draft ?? "" : "" }

    @ObservationIgnored private var session: LiveSession?
    @ObservationIgnored private var reader: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    public init(_ speech: Fourier = Fourier(), options: ListenOptions? = nil) {
        self.speech = speech
        self.options = options
    }

    /// starts listening. prepares the model and asks for the microphone if needed.
    public func start() {
        guard phase == .idle else { return }
        generation += 1
        let generation = generation
        phase = .starting
        update = nil
        transcript = nil
        warning = nil
        error = nil
        level = 0
        isSpeaking = false
        let speech = speech
        let options = options ?? speech.options.listen
        Task {
            do {
                let session = try await speech.listen(options)
                guard self.generation == generation, self.phase == .starting else {
                    session.cancel()
                    return
                }
                self.session = session
                self.phase = .listening
                self.startedAt = Date()
                self.read(session, generation: generation)
            } catch {
                guard self.generation == generation else { return }
                self.error = Self.wrap(error)
                self.phase = .idle
            }
        }
    }

    /// stops listening and waits for the exact transcript.
    public func stop() {
        switch phase {
        case .starting:
            generation += 1
            phase = .idle
        case .listening:
            finish()
        case .idle, .finishing:
            break
        }
    }

    /// starts when idle, stops when running
    public func toggle() {
        phase == .idle ? start() : stop()
    }

    /// stops and throws the session away
    public func cancel() {
        generation += 1
        session?.cancel()
        session = nil
        reader?.cancel()
        reader = nil
        phase = .idle
        level = 0
        isSpeaking = false
    }

    /// clears the text of the last session
    public func clear() {
        guard phase == .idle else { return }
        update = nil
        transcript = nil
        warning = nil
        error = nil
    }

    private func finish() {
        guard let session else { return }
        phase = .finishing
        let generation = generation
        Task {
            do {
                let transcript = try await session.finish()
                guard self.generation == generation else { return }
                self.transcript = transcript
                self.end()
                self.onFinish?(transcript)
            } catch {
                guard self.generation == generation else { return }
                self.error = Self.wrap(error)
                self.end()
            }
        }
    }

    private func end() {
        session = nil
        phase = .idle
        level = 0
        isSpeaking = false
    }

    private func read(_ session: LiveSession, generation: Int) {
        reader = Task {
            do {
                for try await event in session.events {
                    guard self.generation == generation else { return }
                    self.handle(event)
                }
            } catch {
                guard self.generation == generation, self.phase == .listening else { return }
                self.error = Self.wrap(error)
            }
        }
    }

    private func handle(_ event: LiveEvent) {
        switch event {
        case .started:
            break
        case .level(let value):
            if level != value { level = value }
        case .speechStarted:
            isSpeaking = true
        case .update(let new):
            update = new
        case .phraseEnded:
            isSpeaking = false
        case .warning(let new):
            warning = new
            // a file or custom source ran out: finish on our own, as if the user pressed stop
            if new == .inputEnded, phase == .listening { finish() }
        }
    }

    private static func wrap(_ error: any Error) -> FourierError {
        (error as? FourierError) ?? .engineFailure(String(describing: error))
    }
}
