import Foundation
import Observation
import os

private let log = Logger(subsystem: "com.karanganesan.FourierPlayground", category: "recents")

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

extension RecentTranscript {
    enum CodingKeys: String, CodingKey { case id, date, mode, text, source, duration, speed }

    /// only the text is essential. every other field has a default, so a file written by an older or
    /// a newer build still loads, and a mode this build does not know becomes transcribe.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(String.self, forKey: .text)
        id = (try? c.decodeIfPresent(UUID.self, forKey: .id)) ?? UUID()
        date = (try? c.decodeIfPresent(Date.self, forKey: .date)) ?? Date()
        mode = (try? c.decodeIfPresent(String.self, forKey: .mode)).flatMap(Mode.init(rawValue:)) ?? .transcribe
        source = (try? c.decodeIfPresent(String.self, forKey: .source)) ?? nil
        duration = (try? c.decodeIfPresent(Double.self, forKey: .duration)) ?? 0
        speed = (try? c.decodeIfPresent(Double.self, forKey: .speed)) ?? 0
    }
}

/// a transcript that was deleted, kept for 30 days so it can come back, also after a quit
struct DeletedTranscript: Codable, Hashable, Identifiable, Sendable {
    var item: RecentTranscript
    var deletedAt = Date()
    var id: UUID { item.id }

    init(_ item: RecentTranscript, deletedAt: Date = Date()) {
        self.item = item
        self.deletedAt = deletedAt
    }

    enum CodingKeys: String, CodingKey { case item, deletedAt }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        item = try c.decode(RecentTranscript.self, forKey: .item)
        deletedAt = (try? c.decodeIfPresent(Date.self, forKey: .deletedAt)) ?? Date()
    }
}

/// what a stored list needs from its elements
protocol StoredEntry: Codable, Sendable, Identifiable where ID == UUID {
    /// newest first
    var sortDate: Date { get }
    /// dropped when a list is read or written
    var isExpired: Bool { get }
}

extension StoredEntry {
    var isExpired: Bool { false }
}

extension RecentTranscript: StoredEntry {
    var sortDate: Date { date }
}

extension DeletedTranscript: StoredEntry {
    static let retention: TimeInterval = 30 * 24 * 3600
    var sortDate: Date { deletedAt }
    var isExpired: Bool { deletedAt.timeIntervalSinceNow < -Self.retention }
}

/// recent transcripts, newest first, kept as json in application support/FourierPlayground/recents.json,
/// and the recently deleted ones next to it. reads and writes happen off the main actor.
///
/// nothing here may lose a saved transcript:
/// - a file that cannot be read is moved aside, never written over, and the rolling backup takes its place
/// - one bad element does not cost the others
/// - nothing is written before the first read has finished
/// - a write merges with what is on disk first, so a second running copy of the app keeps its additions
@Observable
@MainActor
final class RecentStore {
    private(set) var items: [RecentTranscript] = []
    /// deleted in the last 30 days, newest first
    private(set) var deleted: [DeletedTranscript] = []
    private(set) var isLoaded = false
    /// the last write failed. the data stays in memory and the next change tries again.
    private(set) var saveFailed = false

    @ObservationIgnored private let recentsFile: StoredList<RecentTranscript>
    @ObservationIgnored private let deletedFile: StoredList<DeletedTranscript>
    /// removed from the recent list in this run, so a merge with the file never brings them back
    @ObservationIgnored private var removedFromRecents: Set<UUID> = []
    /// restored from the deleted list in this run
    @ObservationIgnored private var removedFromDeleted: Set<UUID> = []
    @ObservationIgnored private var changedBeforeLoad = false
    @ObservationIgnored private var lastWrite: Task<Void, Never>?
    @ObservationIgnored private var reportedFailure = false

    /// automated launches (any `-fourier…` argument) keep their own list, so test runs never add to,
    /// or remove from, the transcripts a person made
    init(directory: URL? = nil, name: String? = nil) {
        let base = directory ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FourierPlayground", isDirectory: true)
        let name = name ?? (LaunchOptions.current.isAutomated ? "recents.automation" : "recents")
        recentsFile = StoredList(directory: base, name: name)
        deletedFile = StoredList(directory: base, name: "\(name).deleted")
    }

    func load() async {
        guard !isLoaded else { return }
        let loaded = await recentsFile.load()
        let loadedDeleted = await deletedFile.load()
        // anything added before the file was read stays, and is written once the read is done
        let have = Set(items.map(\.id)).union(removedFromRecents)
        items = Self.newestFirst(items + loaded.filter { !have.contains($0.id) })
        let haveDeleted = Set(deleted.map(\.id)).union(removedFromDeleted)
        deleted = Self.newestFirst(deleted + loadedDeleted.filter { !haveDeleted.contains($0.id) })
        isLoaded = true
        if changedBeforeLoad { save() }
    }

    func add(_ item: RecentTranscript) {
        items = Self.newestFirst(items + [item])
        removedFromRecents.remove(item.id)
        save()
    }

    func item(_ id: RecentTranscript.ID) -> RecentTranscript? {
        items.first { $0.id == id }
    }

    /// moves the item to recently deleted and returns it with its old position, for undo
    @discardableResult
    func delete(_ id: RecentTranscript.ID) -> (RecentTranscript, Int)? {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return nil }
        let item = items.remove(at: index)
        removedFromRecents.insert(id)
        removedFromDeleted.remove(id)
        deleted = Self.newestFirst(deleted.filter { $0.id != id } + [DeletedTranscript(item)])
        save()
        return (item, index)
    }

    /// puts an item back, at its old position when it is known (undo)
    func restore(_ item: RecentTranscript, at index: Int? = nil) {
        guard !items.contains(where: { $0.id == item.id }) else { return }
        if let index { items.insert(item, at: min(index, items.count)) } else { items = Self.newestFirst(items + [item]) }
        removedFromRecents.remove(item.id)
        removedFromDeleted.insert(item.id)
        deleted.removeAll { $0.id == item.id }
        save()
    }

    /// brings a transcript back from recently deleted
    func restoreDeleted(_ id: UUID) {
        guard let entry = deleted.first(where: { $0.id == id }) else { return }
        restore(entry.item)
    }

    private static func newestFirst<Entry: StoredEntry>(_ entries: [Entry]) -> [Entry] {
        var seen = Set<UUID>()
        return entries
            .filter { seen.insert($0.id).inserted }
            .sorted { ($0.sortDate, $0.id.uuidString) > ($1.sortDate, $1.id.uuidString) }
    }

    private func save() {
        guard isLoaded else {
            changedBeforeLoad = true
            return
        }
        changedBeforeLoad = false
        let items = items, deleted = deleted
        let removedFromRecents = removedFromRecents, removedFromDeleted = removedFromDeleted
        let recentsFile = recentsFile, deletedFile = deletedFile
        let previous = lastWrite
        // chained, so writes land in order
        lastWrite = Task { [weak self] in
            await previous?.value
            var failure: (any Error)?
            do {
                let merged = try await recentsFile.save(items, dropping: removedFromRecents)
                self?.adopt(merged)
            } catch {
                failure = error
            }
            do {
                let merged = try await deletedFile.save(deleted, dropping: removedFromDeleted)
                self?.adoptDeleted(merged)
            } catch {
                failure = error
            }
            self?.finishWrite(failure)
        }
    }

    /// what another running copy added shows up here too
    private func adopt(_ merged: [RecentTranscript]) {
        let known = Set(items.map(\.id))
        let extra = merged.filter { !known.contains($0.id) && !removedFromRecents.contains($0.id) }
        if !extra.isEmpty { items = Self.newestFirst(items + extra) }
    }

    private func adoptDeleted(_ merged: [DeletedTranscript]) {
        let known = Set(deleted.map(\.id))
        let extra = merged.filter { !known.contains($0.id) && !removedFromDeleted.contains($0.id) }
        if !extra.isEmpty { deleted = Self.newestFirst(deleted + extra) }
    }

    private func finishWrite(_ failure: (any Error)?) {
        saveFailed = failure != nil
        if let failure {
            // once per streak of failures, so a full disk does not flood the log
            if !reportedFailure { log.error("could not save transcripts: \(String(describing: failure), privacy: .public)") }
            reportedFailure = true
        } else {
            reportedFailure = false
        }
    }

    /// waits for the queued writes, for tests
    func flush() async {
        await lastWrite?.value
    }
}

private struct Lossy<Element: Decodable>: Decodable {
    let value: Element?
    init(from decoder: Decoder) {
        value = try? Element(from: decoder)
    }
}

enum StoredListError: Error {
    /// a write came before the first read finished
    case notLoaded
    /// the unreadable file could not be moved aside, so nothing is written over it
    case blocked
}

/// one json list on disk, on its own actor so writes stay in order and off the main thread.
///
/// files, next to each other: `<name>.json`, `<name>.backup.json` (the previous good copy, written
/// before each overwrite) and `<name>.unreadable-<time>.json` (files that could not be read, the last
/// five kept).
actor StoredList<Entry: StoredEntry> {
    private enum Read {
        case missing
        case unreadable
        case entries([Entry], dropped: Int)
    }

    private let directory: URL
    private let name: String
    private var hasLoaded = false
    /// the main file is unreadable and could not be moved: it is never written
    private var blocked = false

    init(directory: URL, name: String) {
        self.directory = directory
        self.name = name
    }

    private var main: URL { directory.appendingPathComponent("\(name).json") }
    private var backup: URL { directory.appendingPathComponent("\(name).backup.json") }

    func load() -> [Entry] {
        defer { hasLoaded = true }
        switch read(main) {
        case .entries(let entries, let dropped):
            // keep a copy of the file with the rows that did not decode, then carry on with the rest
            if dropped > 0 { _ = setAside(main, copy: true) }
            return entries.filter { !$0.isExpired }
        case .missing:
            return fromBackup()
        case .unreadable:
            if !setAside(main, copy: false) { blocked = true }
            return fromBackup()
        }
    }

    /// merges with the file as it is now, so a second running copy of the app keeps what it added,
    /// then writes. returns the merged list. `dropping` are ids this run deleted on purpose.
    func save(_ entries: [Entry], dropping: Set<UUID>) throws -> [Entry] {
        guard hasLoaded else { throw StoredListError.notLoaded }
        guard !blocked else { throw StoredListError.blocked }
        var onDisk: [Entry] = []
        var good: Data?
        switch read(main) {
        case .entries(let found, let dropped):
            onDisk = found
            good = try? Data(contentsOf: main)
            if dropped > 0 { _ = setAside(main, copy: true) }
        case .missing:
            break
        case .unreadable:
            guard setAside(main, copy: false) else {
                blocked = true
                throw StoredListError.blocked
            }
        }
        var byID: [UUID: Entry] = [:]
        for entry in onDisk { byID[entry.id] = entry }
        for entry in entries { byID[entry.id] = entry }
        for id in dropping { byID[id] = nil }
        let merged = byID.values.filter { !$0.isExpired }
            .sorted { ($0.sortDate, $0.id.uuidString) > ($1.sortDate, $1.id.uuidString) }

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(merged)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        // the backup is the last good file, taken before it is replaced. an empty list is not worth a backup.
        if let good, !onDisk.isEmpty {
            do { try good.write(to: backup, options: .atomic) } catch { log.error("could not write the backup: \(String(describing: error), privacy: .public)") }
        }
        try data.write(to: main, options: .atomic)
        return merged
    }

    private func read(_ url: URL) -> Read {
        guard FileManager.default.fileExists(atPath: url.path) else { return .missing }
        guard let data = try? Data(contentsOf: url) else { return .unreadable }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let rows = try? decoder.decode([Lossy<Entry>].self, from: data) else { return .unreadable }
        let entries = rows.compactMap(\.value)
        return .entries(entries, dropped: rows.count - entries.count)
    }

    private func fromBackup() -> [Entry] {
        switch read(backup) {
        case .entries(let entries, _):
            log.notice("restored \(entries.count) from the backup")
            return entries.filter { !$0.isExpired }
        case .missing:
            return []
        case .unreadable:
            _ = setAside(backup, copy: true)
            return []
        }
    }

    /// moves (or copies) a file out of the way under a name that says when, and keeps the last five
    private func setAside(_ url: URL, copy: Bool) -> Bool {
        let stamp = ISO8601DateFormatter().string(from: Date()).filter { $0.isNumber || $0 == "T" }
        let target = directory.appendingPathComponent("\(name).unreadable-\(stamp)-\(UUID().uuidString.prefix(4)).json")
        do {
            if copy { try FileManager.default.copyItem(at: url, to: target) } else { try FileManager.default.moveItem(at: url, to: target) }
        } catch {
            log.error("could not set \(url.lastPathComponent, privacy: .public) aside: \(String(describing: error), privacy: .public)")
            return false
        }
        log.notice("kept \(url.lastPathComponent, privacy: .public) as \(target.lastPathComponent, privacy: .public)")
        let prefix = "\(name).unreadable-"
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []).filter { $0.hasPrefix(prefix) }.sorted()
        for old in names.dropLast(5) { try? FileManager.default.removeItem(at: directory.appendingPathComponent(old)) }
        return true
    }
}
