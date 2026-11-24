import Foundation

/// where the pinned checkpoint lives on this mac.
enum ModelLocation {
    static let repository = "oruk/oruk-fourier-2b"
    static let revision = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
    static let weightsSHA256 = "e64765ffdd90c83704c0951e0d88b7f2ed84499af5c208f7ddf8d922cf08441c"
    static let weightsBytes: Int64 = 4_076_193_080
    /// the files the engine reads. only the weights are large.
    static let files = ["config.json", "generation_config.json", "tokenizer.json", "model.safetensors"]

    static var supportDirectory: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("FourierKit", isDirectory: true)
    }

    static func directory(in root: URL, revision: String = revision) -> URL {
        root.appendingPathComponent("models/oruk--oruk-fourier-2b/\(revision)", isDirectory: true)
    }

    static var defaultDirectory: URL { directory(in: supportDirectory) }
}
