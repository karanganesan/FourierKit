import Foundation

/// the golden test's window into the engine: one piece in, tokens and every ranked candidate out.
/// it is spi, not api, so the test can run against a release build without the kit growing
/// public surface for it.
@_spi(Golden)
public enum GoldenProbe {
    public struct Run: Sendable {
        public var tokens: [Int32]
        public var text: String
        public var reachedStop: Bool
        public var steps: Int
        public var widestBatch: Int
        public var queued, mel, encode, prefill, decode, firstToken, total: Double
        /// per step, the kept candidates in rank order
        public var scores: [[Float]]
        /// per step, beam * vocabulary + token for each kept candidate
        public var indices: [[Int32]]
    }

    public static let vocabulary = 151_936
    public static var defaultModelDirectory: URL { ModelLocation.defaultDirectory }

    /// - returns: seconds taken, the beam count and the number of banned tokens in the loaded defaults
    public static func load(modelDirectory: URL) async throws -> (seconds: Double, beams: Int, bans: Int) {
        let started = EngineHost.now()
        try await EngineHost.shared.load(modelDirectory: modelDirectory)
        let defaults = await EngineHost.shared.defaults
        return (EngineHost.now() - started, defaults.numBeams, defaults.bannedTokens.count)
    }

    /// the files of the checkpoint that are missing from a directory, empty when it is complete
    public static func missingModelFiles(in directory: URL) -> [String] {
        ModelLocation.files.filter { !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    public static func unload() async {
        await EngineHost.shared.unload()
    }

    /// "gpu" or "npu" for the conv front, encoder and projector
    public static func setEncoderChip(_ chip: String) async throws {
        try await EngineHost.shared.setEncoderChip(EngineHost.EncoderChip(rawValue: chip) ?? .gpu)
    }

    /// share of the neural engine encoder's operations core ml planned on the neural engine
    public static func neuralShare() async -> Double? {
        await EngineHost.shared.neuralPlan.map { $0.operationShare("neural engine") }
    }

    public static func setMaxRows(_ rows: Int) async {
        await EngineHost.shared.setMaxRows(rows)
    }

    /// one piece with the checkpoint's own generation defaults, as the reference runs it.
    public static func transcribe(samples: [Float], candidates: Int) async throws -> Run {
        let settings = await EngineHost.shared.defaults
        return try await transcribe(samples: samples, settings: settings, candidates: candidates)
    }

    /// one piece with the checkpoint's generation defaults but another beam count (1 is greedy), so
    /// batches of any total width can be built. these runs are not comparable to the reference.
    public static func transcribe(samples: [Float], beams: Int, candidates: Int) async throws -> Run {
        var settings = await EngineHost.shared.defaults
        settings.numBeams = beams
        return try await transcribe(samples: samples, settings: settings, candidates: candidates)
    }

    private static func transcribe(samples: [Float], settings: GenerationSettings, candidates: Int) async throws -> Run {
        let host = EngineHost.shared
        let result = try await host.transcribe(PieceRequest(samples: samples, settings: settings, recordCandidates: candidates))
        let timings = result.timings
        return Run(
            tokens: result.tokens, text: result.text, reachedStop: result.reachedStop, steps: result.steps,
            widestBatch: result.widestBatch,
            queued: timings.queued, mel: timings.mel, encode: timings.encode, prefill: timings.prefill,
            decode: timings.decode, firstToken: timings.firstToken, total: timings.total,
            scores: result.candidates.map { $0.map(\.score) },
            indices: result.candidates.map { $0.map { Int32($0.beam * vocabulary) + $0.token } }
        )
    }

    /// "gpu" or "npu": where the host runs the conv front, encoder and projector right now
    public static func encoderChip() async -> String {
        await EngineHost.shared.encoderChip.rawValue
    }

    /// waits until `count` pieces are queued or running in the host, so a test can fix the order
    /// pieces enter a batch in. false if that did not happen within `timeout` seconds.
    public static func waitForPieces(_ count: Int, timeout: Double = 30) async -> Bool {
        let deadline = now() + timeout
        while now() < deadline {
            let snapshot = await EngineHost.shared.snapshot()
            if snapshot.activePieces + snapshot.queuedPieces >= count { return true }
            try? await Task.sleep(for: .milliseconds(1))
        }
        return false
    }

    /// what the engine makes of one clip at three points of the pipeline
    public struct Stages: Sendable {
        /// log-mel, row-major `[bin][paddedFrame]`
        public var mel: [Float]
        public var melBins: Int
        public var melFrames: Int
        public var melPaddedFrames: Int
        /// encoder output, row-major `[audioToken][width]`
        public var audio: [Float]
        public var audioTokens: Int
        /// raw logits of the first decoder step
        public var firstLogits: [Float]
    }

    /// mel, encoder output and first-step logits for each clip, from an engine of its own with a small
    /// decoder cache (same code and weights as the host's, which must be idle). for diagnostics.
    public static func stages(samples clips: [[Float]], modelDirectory: URL) async throws -> [Stages] {
        let settings = try GenerationSettings.load(generationConfigJSON: modelDirectory.appendingPathComponent("generation_config.json"))
        let tokenizer = try Tokenizer(contentsOf: modelDirectory.appendingPathComponent("tokenizer.json"))
        let engine = try MetalEngine(modelDirectory: modelDirectory, settings: settings, slotCapacity: 2048)
        var out: [Stages] = []
        for samples in clips {
            let mel = LogMel.features(samples: samples)
            let audio = try await engine.encode(mel: mel.data, frames: mel.frames, paddedFrames: mel.paddedFrames)
            let prompt = Prompt.transcription(audioTokens: audio.count, language: "English", context: nil, tokenizer: tokenizer)
            let (piece, first) = try await engine.prefill(prompt: prompt, audio: audio, beams: 1, maxNewTokens: 1)
            let logits = Array(first.values)
            engine.release(piece)
            out.append(Stages(
                mel: mel.data, melBins: mel.bins, melFrames: mel.frames, melPaddedFrames: mel.paddedFrames,
                audio: audio.values(), audioTokens: audio.count, firstLogits: logits))
        }
        return out
    }

    public static func readAudio(_ url: URL) throws -> [Float] {
        try AudioFileReader.read(url)
    }

    public static func floats(in url: URL, named name: String) throws -> [Float] {
        try SafeTensors(url: url).floats(name)
    }

    public static func ints(in url: URL, named name: String) throws -> [Int32] {
        try SafeTensors(url: url).ints(name)
    }

    public static func now() -> Double { EngineHost.now() }
}
