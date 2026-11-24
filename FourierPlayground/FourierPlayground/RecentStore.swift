import FourierKit
import Foundation
import Observation

/// one saved transcript. `text` is the model's exact text; screens lowercase it for display only.
struct RecentTranscript: Codable, Hashable, Identifiable, Sendable {
    var id = UUID()
    var date = Date()
    var mode: Mode
    var text: String
    /// the file's name for transcribe, nil otherwise
    var source: String?
    /// audio length in seconds
    var duration: Double
    var speed: Double

    var title: String {
        let words = text.split(whereSeparator: \.isWhitespace).prefix(6).joined(separator: " ")
        return words.isEmpty ? "no speech" : words.lowercased()
    }
}

/// recent transcripts, newest first, kept as json in application support/FourierPlayground/recents.json.
/// reads and writes happen off the main actor.
@Observable
@MainActor
final class RecentStore {
    private(set) var items: [RecentTranscript] = []
    private(set) var isLoaded = false
    @ObservationIgnored private let writer = RecentFile()

    func load() async {
        let loaded = await writer.read()
        // anything added before the file was read stays on top
        let ids = Set(items.map(\.id))
        items = items + loaded.filter { !ids.contains($0.id) }
        isLoaded = true
    }

    func add(_ item: RecentTranscript) {
        items.insert(item, at: 0)
        save()
    }

    func item(_ id: RecentTranscript.ID) -> RecentTranscript? {
        items.first { $0.id == id }
    }

    /// removes the item and returns it with its old position, for undo
    @discardableResult
    func delete(_ id: RecentTranscript.ID) -> (RecentTranscript, Int)? {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return nil }
        let item = items.remove(at: index)
        save()
        return (item, index)
    }

    func restore(_ item: RecentTranscript, at index: Int) {
        guard !items.contains(where: { $0.id == item.id }) else { return }
        items.insert(item, at: min(index, items.count))
        save()
    }

    private func save() {
        let items = items
        let writer = writer
        let previous = lastWrite
        // chained, so writes land in order
        lastWrite = Task.detached(priority: .utility) {
            await previous?.value
            await writer.write(items)
        }
    }

    @ObservationIgnored private var lastWrite: Task<Void, Never>?
}

/// the file itself, on its own actor so writes stay in order and off the main thread
private actor RecentFile {
    private var url: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("FourierPlayground/recents.json")
    }

    func read() -> [RecentTranscript] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return (try? decoder.decode([RecentTranscript].self, from: data)) ?? []
    }

    func write(_ items: [RecentTranscript]) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(items) else { return }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }
}
