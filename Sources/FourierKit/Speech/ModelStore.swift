import CryptoKit
import Foundation

/// where the model files come from. the default downloads the pinned checkpoint and checks its hash.
public protocol ModelStore: Sendable {
    /// makes the files in `request` ready on disk and returns their folder. reports progress for the
    /// check, download, verify and convert stages. called off the main thread; cancellation keeps partial files.
    func prepare(_ request: ModelRequest, progress: @escaping @Sendable (ModelStoreProgress) -> Void) async throws -> URL
}

public struct ModelRequest: Sendable, Hashable {
    public var repository: String
    public var revision: String
    public var directory: URL
    /// every file the engine reads, the weights included
    public var files: [String]
    public var weightsFile: String
    public var weightsSHA256: String
    /// nil when unknown
    public var weightsBytes: Int64?
    public var offline: Bool

    public init(repository: String, revision: String, directory: URL, files: [String], weightsFile: String,
                weightsSHA256: String, weightsBytes: Int64?, offline: Bool) {
        self.repository = repository
        self.revision = revision
        self.directory = directory
        self.files = files
        self.weightsFile = weightsFile
        self.weightsSHA256 = weightsSHA256
        self.weightsBytes = weightsBytes
        self.offline = offline
    }
}

public struct ModelStoreProgress: Sendable, Hashable {
    /// check, download, verify or convert
    public var stage: PrepareProgress.Stage
    /// 0...1 within the stage
    public var fraction: Double
    public var bytesDone: Int64
    public var bytesTotal: Int64

    public init(stage: PrepareProgress.Stage, fraction: Double, bytesDone: Int64 = 0, bytesTotal: Int64 = 0) {
        self.stage = stage
        self.fraction = fraction
        self.bytesDone = bytesDone
        self.bytesTotal = bytesTotal
    }
}

/// downloads from hugging face with resume, checks the weights' sha256, and remembers a verified file.
public struct HuggingFaceModelStore: ModelStore {
    public var endpoint: URL
    /// download attempts after a failed hash
    public var hashAttempts = 2
    /// network retries per attempt, each resuming from the last written byte
    public var networkRetries = 5

    public init(endpoint: URL = URL(string: "https://huggingface.co")!) {
        self.endpoint = endpoint
    }

    static let markerSuffix = ".verified"
    /// room left over after the download, so the disk is not filled to the brim
    static let diskMargin: Int64 = 512 * 1024 * 1024
    /// a guess for the small files, which are not counted in the bar
    static let smallFilesEstimate: Int64 = 16 * 1024 * 1024

    public func prepare(_ request: ModelRequest, progress: @escaping @Sendable (ModelStoreProgress) -> Void) async throws -> URL {
        let fm = FileManager.default
        let directory = request.directory
        let weights = directory.appendingPathComponent(request.weightsFile)
        let part = weights.appendingPathExtension("part")
        let total = request.weightsBytes ?? 0

        // check
        progress(ModelStoreProgress(stage: .check, fraction: 0, bytesTotal: total))
        var missingSmall: [String] = []
        let small = request.files.filter { $0 != request.weightsFile }
        for (index, name) in small.enumerated() {
            let url = directory.appendingPathComponent(name)
            if Self.size(of: url) <= 0 { missingSmall.append(name) }
            progress(ModelStoreProgress(stage: .check, fraction: Double(index + 1) / Double(small.count + 1), bytesTotal: total))
        }
        var verified = Self.isVerified(weights, sha256: request.weightsSHA256)
        progress(ModelStoreProgress(stage: .check, fraction: 1, bytesTotal: total))

        // a weights file that was never verified is hashed before anything is downloaded. it is only
        // removed once its hash is known not to match, or its known size rules a match out, and
        // never offline, where it may be all there is.
        if !verified, fm.fileExists(atPath: weights.path) {
            let size = Self.size(of: weights)
            if total == 0 || size == total {
                progress(ModelStoreProgress(stage: .download, fraction: 1, bytesDone: size, bytesTotal: max(total, size)))
                if try await Self.verify(weights, request: request, progress: progress) {
                    Self.writeMarker(for: weights, sha256: request.weightsSHA256)
                    verified = true
                } else if !request.offline {
                    try? fm.removeItem(at: weights)
                }
            } else if request.offline {
                // the wrong size for the pinned file, and nothing may be changed offline
            } else if size < total, !fm.fileExists(atPath: part.path) {
                // a partial file under the final name continues as the part file
                try? fm.moveItem(at: weights, to: part)
            } else {
                try? fm.removeItem(at: weights)
            }
        }

        if verified && missingSmall.isEmpty {
            let size = Self.size(of: weights)
            progress(ModelStoreProgress(stage: .download, fraction: 1, bytesDone: size, bytesTotal: size))
            progress(ModelStoreProgress(stage: .verify, fraction: 1, bytesDone: size, bytesTotal: size))
            // the engine reads the raw bf16 file as it is, so the default path converts nothing
            progress(ModelStoreProgress(stage: .convert, fraction: 1, bytesDone: size, bytesTotal: size))
            return directory
        }

        if request.offline { throw FourierError.offlineModelMissing }

        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        ModelLocation.excludeFromBackup(directory)
        try Self.checkDisk(directory: directory, needed: (verified ? 0 : Self.remainingBytes(weights: weights, part: part, total: total))
            + (missingSmall.isEmpty ? 0 : Self.smallFilesEstimate))

        let session = Self.makeSession()
        defer { session.finishTasksAndInvalidate() }

        for name in missingSmall {
            try Task.checkCancellation()
            try await downloadSmall(name, request: request, session: session)
        }

        if verified {
            progress(ModelStoreProgress(stage: .convert, fraction: 1, bytesDone: total, bytesTotal: total))
            return directory
        }

        var attempt = 0
        while true {
            attempt += 1
            let size = try await downloadWeights(request: request, part: part, session: session, progress: progress)
            progress(ModelStoreProgress(stage: .download, fraction: 1, bytesDone: size, bytesTotal: max(total, size)))
            if try await Self.verify(part, request: request, progress: progress) {
                try? fm.removeItem(at: weights)
                try fm.moveItem(at: part, to: weights)
                Self.writeMarker(for: weights, sha256: request.weightsSHA256)
                break
            }
            // the bar holds while the file downloads again
            try? fm.removeItem(at: part)
            if attempt >= hashAttempts { throw FourierError.hashMismatch }
        }
        progress(ModelStoreProgress(stage: .convert, fraction: 1, bytesDone: total, bytesTotal: total))
        return directory
    }

    // MARK: network

    /// a foreground session: on ios the download runs while the app is open. a background
    /// `URLSessionConfiguration.background` session with range resumes would go here.
    static func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        // nothing that identifies the user or the app
        configuration.httpAdditionalHeaders = ["User-Agent": "FourierKit"]
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        configuration.timeoutIntervalForRequest = 60
        configuration.waitsForConnectivity = false
        return URLSession(configuration: configuration)
    }

    func fileURL(_ name: String, request: ModelRequest) -> URL {
        endpoint.appendingPathComponent(request.repository)
            .appendingPathComponent("resolve")
            .appendingPathComponent(request.revision)
            .appendingPathComponent(name)
    }

    private func downloadSmall(_ name: String, request: ModelRequest, session: URLSession) async throws {
        let target = request.directory.appendingPathComponent(name)
        var lastError: (any Error)?
        for _ in 0...networkRetries {
            try Task.checkCancellation()
            do {
                let (data, response) = try await session.data(from: fileURL(name, request: request))
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    throw FourierError.downloadFailed("\(name): http \((response as? HTTPURLResponse)?.statusCode ?? 0)")
                }
                let part = target.appendingPathExtension("part")
                try data.write(to: part, options: .atomic)
                try? FileManager.default.removeItem(at: target)
                try FileManager.default.moveItem(at: part, to: target)
                return
            } catch let error as URLError where error.code == .cancelled {
                throw CancellationError()
            } catch let error as FourierError {
                throw error
            } catch {
                if Task.isCancelled { throw CancellationError() }
                lastError = error
                try await Task.sleep(for: .seconds(1))
            }
        }
        throw FourierError.downloadFailed(lastError.map { "\(name): \($0.localizedDescription)" } ?? name)
    }

    /// downloads the weights into `part`, resuming from its last byte. returns the final size.
    private func downloadWeights(request: ModelRequest, part: URL, session: URLSession,
                                 progress: @escaping @Sendable (ModelStoreProgress) -> Void) async throws -> Int64 {
        let total = request.weightsBytes ?? 0
        var failures = 0
        while true {
            try Task.checkCancellation()
            let have = Self.size(of: part)
            if total > 0, have == total { return have }
            if total > 0, have > total { try? FileManager.default.removeItem(at: part) }
            do {
                let size = try await RangeDownload.run(
                    url: fileURL(request.weightsFile, request: request), to: part, session: session
                ) { written, expected in
                    let all = total > 0 ? total : expected
                    progress(ModelStoreProgress(stage: .download, fraction: all > 0 ? Double(written) / Double(all) : 0,
                                                bytesDone: written, bytesTotal: all))
                }
                if total == 0 || size == total { return size }
                failures += 1
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as FourierError {
                throw error
            } catch {
                if Task.isCancelled { throw CancellationError() }
                failures += 1
                if failures > networkRetries { throw FourierError.downloadFailed(error.localizedDescription) }
                try await Task.sleep(for: .seconds(min(30, 1 << failures)))
            }
            if failures > networkRetries { throw FourierError.downloadFailed("the download kept stopping early") }
        }
    }

    // MARK: verification

    /// hashes the whole file in large reads and compares with the pinned sha256.
    static func verify(_ url: URL, request: ModelRequest, progress: @escaping @Sendable (ModelStoreProgress) -> Void) async throws -> Bool {
        let total = size(of: url)
        let digest = try await sha256(of: url) { done in
            progress(ModelStoreProgress(stage: .verify, fraction: total > 0 ? Double(done) / Double(total) : 1,
                                        bytesDone: done, bytesTotal: total))
        }
        return digest == request.weightsSHA256.lowercased()
    }

    static func sha256(of url: URL, progress: @escaping @Sendable (Int64) -> Void) async throws -> String {
        let task = Task.detached(priority: .utility) { () throws -> String in
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hasher = SHA256()
            var done: Int64 = 0
            while true {
                try Task.checkCancellation()
                let chunk = try autoreleasepool { try handle.read(upToCount: 16 << 20) } ?? Data()
                if chunk.isEmpty { break }
                hasher.update(data: chunk)
                done += Int64(chunk.count)
                progress(done)
            }
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }
        return try await withTaskCancellationHandler { try await task.value } onCancel: { task.cancel() }
    }

    private struct Marker: Codable {
        var size: Int64
        var modified: Double
        var sha256: String
    }

    static func markerURL(for weights: URL) -> URL {
        weights.deletingLastPathComponent().appendingPathComponent(weights.lastPathComponent + markerSuffix)
    }

    /// true when a marker records this exact file (size and modification time) as hashed and matching.
    static func isVerified(_ weights: URL, sha256: String) -> Bool {
        guard let data = try? Data(contentsOf: markerURL(for: weights)),
              let marker = try? JSONDecoder().decode(Marker.self, from: data),
              let file = attributes(of: weights)
        else { return false }
        return marker.sha256 == sha256.lowercased() && marker.size == file.size && abs(marker.modified - file.modified) < 0.001
    }

    static func writeMarker(for weights: URL, sha256: String) {
        guard let file = attributes(of: weights),
              let data = try? JSONEncoder().encode(Marker(size: file.size, modified: file.modified, sha256: sha256.lowercased()))
        else { return }
        try? data.write(to: markerURL(for: weights), options: .atomic)
    }

    // MARK: disk

    /// read fresh from the file system each time; url resource values are cached and go stale while a file grows.
    static func size(of url: URL) -> Int64 {
        attributes(of: url)?.size ?? 0
    }

    static func attributes(of url: URL) -> (size: Int64, modified: Double)? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = (attributes[.size] as? NSNumber)?.int64Value,
              let modified = attributes[.modificationDate] as? Date
        else { return nil }
        return (size, modified.timeIntervalSince1970)
    }

    static func remainingBytes(weights: URL, part: URL, total: Int64) -> Int64 {
        guard total > 0 else { return 0 }
        return max(0, total - size(of: part))
    }

    static func checkDisk(directory: URL, needed: Int64) throws {
        guard needed > 0 else { return }
        let values = try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
        let available = values?.volumeAvailableCapacityForImportantUsage ?? Int64(values?.volumeAvailableCapacity ?? Int.max)
        if available < needed + diskMargin {
            throw FourierError.notEnoughDisk(needed: needed + diskMargin, available: available)
        }
    }
}

/// one ranged get that streams straight into a file, resuming from the file's current size.
final class RangeDownload: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    // everything below is touched only on the session's serial delegate queue, or before the task starts
    private let handle: FileHandle
    private var written: Int64
    private let requestedOffset: Int64
    private var expected: Int64 = 0
    private var failure: (any Error)?
    private let onBytes: @Sendable (Int64, Int64) -> Void
    private var continuation: CheckedContinuation<Int64, any Error>?
    private var lastReport: Double = 0

    private init(handle: FileHandle, offset: Int64, onBytes: @escaping @Sendable (Int64, Int64) -> Void) {
        self.handle = handle
        written = offset
        requestedOffset = offset
        self.onBytes = onBytes
    }

    /// returns the file's size when the server says the body is complete.
    static func run(url: URL, to file: URL, session: URLSession,
                    onBytes: @escaping @Sendable (Int64, Int64) -> Void) async throws -> Int64 {
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        let offset = try handle.seekToEnd()
        var request = URLRequest(url: url)
        if offset > 0 { request.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        let download = RangeDownload(handle: handle, offset: Int64(offset), onBytes: onBytes)
        let task = session.dataTask(with: request)
        task.delegate = download
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                download.continuation = continuation
                task.resume()
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            failure = FourierError.downloadFailed("not an http response")
            return completionHandler(.cancel)
        }
        switch http.statusCode {
        case 206:
            expected = written + max(0, http.expectedContentLength)
        case 200:
            // the server ignored the range, so the file starts over; the bar holds meanwhile
            if requestedOffset > 0 {
                try? handle.truncate(atOffset: 0)
                try? handle.seek(toOffset: 0)
                written = 0
            }
            expected = max(0, http.expectedContentLength)
        case 416:
            // nothing past the end: the file is already whole
            expected = written
            return completionHandler(.cancel)
        default:
            failure = FourierError.downloadFailed("http \(http.statusCode)")
            return completionHandler(.cancel)
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do {
            try handle.write(contentsOf: data)
            written += Int64(data.count)
        } catch {
            failure = FourierError.downloadFailed("could not write the model file: \(error.localizedDescription)")
            dataTask.cancel()
            return
        }
        let time = now()
        if time - lastReport > 0.05 {
            lastReport = time
            onBytes(written, expected)
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        try? handle.synchronize()
        onBytes(written, expected)
        guard let continuation else { return }
        self.continuation = nil
        if let failure {
            continuation.resume(throwing: failure)
        } else if let error = error as? URLError, error.code == .cancelled {
            if expected == written, expected > 0 {
                continuation.resume(returning: written)
            } else {
                continuation.resume(throwing: CancellationError())
            }
        } else if let error {
            continuation.resume(throwing: error)
        } else {
            continuation.resume(returning: written)
        }
    }
}
