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
        let host = EngineHost.shared
        let settings = await host.defaults
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
