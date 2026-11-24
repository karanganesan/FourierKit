import FourierKit
import FourierKitUI
import Foundation
import Observation

/// one file at a time: progress, then the exact transcript.
@Observable
@MainActor
final class TranscribeModel {
    var speech: Fourier
    /// asks the screen to show the open panel
    var choosing = false

    private(set) var fileName: String?
    private(set) var fraction: Double = 0
    private(set) var isRunning = false
    private(set) var transcript: Transcript?
    private(set) var error: FourierError?

    @ObservationIgnored var onFinish: ((Transcript, String) -> Void)?
    @ObservationIgnored private var job: TranscriptionJob?
    @ObservationIgnored private var generation = 0

    init(_ speech: Fourier) {
        self.speech = speech
    }

    static let audioExtensions: Set<String> = ["wav", "wave", "aif", "aiff", "aifc", "caf", "m4a", "mp3", "mp4", "mov", "aac", "flac", "m4v"]

    func transcribe(_ url: URL) {
        cancel()
        generation += 1
        let generation = generation
        let name = url.lastPathComponent
        fileName = name
        fraction = 0
        transcript = nil
        error = nil
        isRunning = true
        let speech = speech
        Task {
            // files from the open panel or a drop stay readable while the job runs
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            let job = speech.transcriptionJob(contentsOf: url)
            self.job = job
            let progress = Task {
                for await value in job.progress {
                    guard self.generation == generation else { return }
                    if value > self.fraction { self.fraction = value }
                }
            }
            do {
                let transcript = try await job.result
                progress.cancel()
                guard self.generation == generation else { return }
                self.fraction = 1
                self.transcript = transcript
                self.isRunning = false
                self.job = nil
                self.onFinish?(transcript, name)
            } catch {
                progress.cancel()
                guard self.generation == generation else { return }
                let error = (error as? FourierError) ?? .engineFailure(String(describing: error))
                if error != .cancelled { self.error = error }
                self.isRunning = false
                self.job = nil
            }
        }
    }

    func cancel() {
        job?.cancel()
        job = nil
        if isRunning {
            generation += 1
            isRunning = false
            fileName = nil
        }
    }

    func clear() {
        cancel()
        fileName = nil
        transcript = nil
        error = nil
        fraction = 0
    }
}
