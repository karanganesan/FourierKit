import FourierKit
import FourierKitUI
import Foundation
import Observation

/// record, then transcribe what was recorded.
@Observable
@MainActor
final class RecordModel {
    enum Phase: Equatable {
        case idle, starting, recording, transcribing
    }

    var speech: Fourier
    /// another audio source for the next recording, e.g. a file in automation. nil uses the speech's source.
    @ObservationIgnored var source: (any AudioSource)?

    private(set) var phase: Phase = .idle
    private(set) var level: Float = 0
    private(set) var startedAt: Date?
    private(set) var fraction: Double = 0
    /// the text so far while a job runs: finished pieces settled, the rest a draft
    private(set) var update: TranscriptionUpdate?
    private(set) var transcript: Transcript?
    private(set) var error: FourierError?

    @ObservationIgnored var onFinish: ((Transcript) -> Void)?
    @ObservationIgnored private var recorder: Recorder?
    @ObservationIgnored private var levels: Task<Void, Never>?
    @ObservationIgnored private var drafts: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    var isRunning: Bool { phase != .idle }

    init(_ speech: Fourier) {
        self.speech = speech
    }

    var micStatus: FourierMicStatus {
        switch phase {
        case .idle: .idle
        case .starting: .starting
        case .recording: .running
        case .transcribing: .finishing
        }
    }

    func toggle() {
        switch phase {
        case .idle: start()
        case .recording: stop()
        case .starting:
            generation += 1
            phase = .idle
        case .transcribing: break
        }
    }

    func start() {
        guard phase == .idle else { return }
        generation += 1
        let generation = generation
        phase = .starting
        transcript = nil
        error = nil
        level = 0
        fraction = 0
        update = nil
        var speech = speech
        if let source {
            var options = speech.options
            options.audioSource = source
            speech = Fourier(options)
        }
        Task {
            do {
                let recorder = try await speech.record()
                guard self.generation == generation, self.phase == .starting else {
                    await recorder.cancel()
                    return
                }
                self.recorder = recorder
                self.startedAt = Date()
                self.phase = .recording
                // draft text while recording. the exact transcript replaces it after stop.
                self.drafts = Task {
                    for await value in recorder.drafts {
                        guard self.generation == generation, self.phase == .recording else { return }
                        self.update = value
                    }
                }
                self.levels = Task {
                    for await level in recorder.levels {
                        guard self.generation == generation else { return }
                        if self.level != level { self.level = level }
                    }
                    // the source ran out on its own: stop as if the user pressed stop
                    if self.generation == generation, self.phase == .recording { self.stop() }
                }
            } catch {
                guard self.generation == generation else { return }
                self.error = (error as? FourierError) ?? .captureFailed(String(describing: error))
                self.phase = .idle
            }
        }
    }

    func stop() {
        guard phase == .recording, let recorder else { return }
        phase = .transcribing
        level = 0
        let generation = generation
        let speech = speech
        self.recorder = nil
        Task {
            let recording = await recorder.stop()
            guard self.generation == generation else { return }
            let job = speech.transcriptionJob(recording)
            let text = Task {
                for await value in job.updates {
                    guard self.generation == generation else { return }
                    self.update = value
                }
            }
            let progress = Task {
                for await value in job.progress {
                    guard self.generation == generation else { return }
                    if value > self.fraction { self.fraction = value }
                }
            }
            do {
                let transcript = try await job.result
                progress.cancel(); text.cancel()
                guard self.generation == generation else { return }
                self.transcript = transcript
                self.phase = .idle
                self.onFinish?(transcript)
            } catch {
                progress.cancel(); text.cancel()
                guard self.generation == generation else { return }
                self.error = (error as? FourierError) ?? .engineFailure(String(describing: error))
                self.phase = .idle
            }
        }
    }

    func clear() {
        guard phase == .idle else { return }
        transcript = nil
        error = nil
    }
}
