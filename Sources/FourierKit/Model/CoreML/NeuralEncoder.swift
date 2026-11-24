import CoreML
import Foundation

/// what core ml planned for each operation at load. a plan is a preference: only a trace proves where work ran.
struct PlacementReport: Codable, Sendable, CustomStringConvertible {
    struct Share: Codable, Sendable {
        var operations = 0
        /// sum of core ml's estimated relative cost
        var cost = 0.0
    }

    var devices: [String: Share] = [:]
    /// op types planned off the neural engine, with counts
    var offNeuralEngine: [String: Int] = [:]
    var operations = 0
    var totalCost = 0.0

    func operationShare(_ device: String) -> Double { operations == 0 ? 0 : Double(devices[device]?.operations ?? 0) / Double(operations) }
    func costShare(_ device: String) -> Double { totalCost == 0 ? 0 : (devices[device]?.cost ?? 0) / totalCost }

    var description: String {
        var lines = ["\(operations) ops planned"]
        for device in devices.keys.sorted() {
            let share = devices[device]!
            lines.append(String(format: "  %@: %d ops (%.1f%%), %.1f%% of estimated cost", device, share.operations,
                                operationShare(device) * 100, costShare(device) * 100))
        }
        if !offNeuralEngine.isEmpty {
            lines.append("  off the neural engine: " + offNeuralEngine.sorted { $0.key < $1.key }.map { "\($0.key) x\($0.value)" }.joined(separator: ", "))
        }
        return lines.joined(separator: "\n")
    }
}

/// the audio encoder on core ml, written from the checkpoint at runtime and aimed at the neural engine.
final class NeuralEncoder: @unchecked Sendable {
    enum Precision: String, Sendable {
        /// float16 weights and math: what the neural engine computes in
        case float16
        /// float32 weights and math, a diagnostic for the cpu
        case float32
    }

    enum Failure: Error, CustomStringConvertible {
        case input(String)
        case output(String)
        var description: String {
            switch self {
            case .input(let why): "the encoder input is not valid: \(why)"
            case .output(let why): "core ml returned an unexpected encoder output: \(why)"
            }
        }
    }

    /// what the cache records next to the compiled model
    private struct Record: Codable {
        var format: Int
        var fingerprint: String
        var precision: String
        var loss: Float16Loss
        /// plans by compute units raw value, since loading a plan costs as much as loading the model
        var plans: [String: PlacementReport] = [:]
    }

    /// seconds spent on each startup step. write and compile are zero when the cache was used,
    /// and plan is zero when the cache already held a plan for these compute units.
    struct Startup: Sendable {
        var write = 0.0
        var compile = 0.0
        var load = 0.0
        var plan = 0.0
        var cached = false
    }

    let precision: Precision
    let layout: EncoderGraph.Layout
    let computeUnits: MLComputeUnits
    let compiledURL: URL
    private(set) var plan = PlacementReport()
    private(set) var loss = Float16Loss()
    private(set) var startup = Startup()
    var inexactWeightCount: Int { loss.total }
    private let model: MLModel

    init(weights: SafeTensors, cacheDirectory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
         precision: Precision = .float16, layout: EncoderGraph.Layout = .channels) async throws {
        self.precision = precision
        self.layout = layout
        self.computeUnits = computeUnits
        let files = FileManager.default
        try files.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
        let stem = "encoder-\(precision.rawValue)-\(layout.rawValue)"
        compiledURL = cacheDirectory.appendingPathComponent(stem + ".mlmodelc", isDirectory: true)
        let recordURL = cacheDirectory.appendingPathComponent(stem + ".json")
        let fingerprint = Self.fingerprint(weights)

        var startup = Startup()
        var record: Record
        if let data = try? Data(contentsOf: recordURL),
           let saved = try? JSONDecoder().decode(Record.self, from: data),
           saved.format == EncoderGraph.formatVersion, saved.fingerprint == fingerprint,
           saved.precision == precision.rawValue, files.fileExists(atPath: compiledURL.path) {
            record = saved
            startup.cached = true
        } else {
            try? files.removeItem(at: recordURL)
            try? files.removeItem(at: compiledURL)
            let package = MLPackageWriter(url: cacheDirectory.appendingPathComponent(stem + ".mlpackage", isDirectory: true))
            var clock = ContinuousClock.now
            let graph = EncoderGraph(weights: weights, compute: precision == .float16 ? .float16 : .float32, layout: layout)
            let loss = try graph.write(to: package)
            startup.write = clock.duration(to: .now).seconds
            clock = .now
            let compiled = try await MLModel.compileModel(at: package.url)
            try files.moveItem(at: compiled, to: compiledURL)
            try? files.removeItem(at: compiled.deletingLastPathComponent())
            try? files.removeItem(at: package.url)
            startup.compile = clock.duration(to: .now).seconds
            record = Record(format: EncoderGraph.formatVersion, fingerprint: fingerprint, precision: precision.rawValue, loss: loss)
        }
        loss = record.loss

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        var clock = ContinuousClock.now
        model = try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
        startup.load = clock.duration(to: .now).seconds
        let unitsKey = String(computeUnits.rawValue)
        if let saved = record.plans[unitsKey] {
            plan = saved
        } else {
            clock = .now
            plan = try await Self.placement(compiledURL, configuration)
            startup.plan = clock.duration(to: .now).seconds
            record.plans[unitsKey] = plan
            try JSONEncoder().encode(record).write(to: recordURL, options: .atomic)
        }
        self.startup = startup
    }

    /// same contract as MetalEngine.encode: mel is [bin][paddedFrame], zero padded to whole 1-second chunks.
    /// returns [tokens][2048] row-major.
    func encode(mel: [Float], frames: Int, paddedFrames: Int) async throws -> [Float] {
        let bins = EncoderGraph.bins
        let chunkFrames = EncoderGraph.chunkFrames
        guard paddedFrames % chunkFrames == 0, mel.count == bins * paddedFrames, frames <= paddedFrames else {
            throw Failure.input("mel must be [\(bins)][a multiple of \(chunkFrames)] and frames must fit")
        }
        let tokens = Self.tokenCount(frames: frames)
        guard tokens > 0 else { throw Failure.input("the audio is empty") }
        let window = EncoderGraph.windowTokens
        let windows = (tokens + window - 1) / window
        let chunks = paddedFrames / chunkFrames
        let audioWidth = 2048
        var output = [Float](repeating: 0, count: tokens * audioWidth)

        try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            for index in 0..<windows {
                group.addTask {
                    let melArray = try MLMultiArray(shape: [8, 1, 128, 100], dataType: .float32)
                    let maskArray = try MLMultiArray(shape: [1, 1, NSNumber(value: window)], dataType: .float32)
                    melArray.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, strides in
                        pointer.update(repeating: 0)
                        mel.withUnsafeBufferPointer { source in
                            for chunk in 0..<EncoderGraph.chunks {
                                let global = index * EncoderGraph.chunks + chunk
                                guard global < chunks else { break }
                                for bin in 0..<bins {
                                    let from = source.baseAddress! + bin * paddedFrames + global * chunkFrames
                                    (pointer.baseAddress! + chunk * strides[0] + bin * strides[2]).update(from: from, count: chunkFrames)
                                }
                            }
                        }
                    }
                    let valid = min(window, tokens - index * window)
                    maskArray.withUnsafeMutableBufferPointer(ofType: Float.self) { pointer, strides in
                        for token in 0..<window { pointer[token * strides[2]] = token < valid ? 0 : EncoderGraph.maskValue }
                    }
                    let input = try MLDictionaryFeatureProvider(dictionary: [
                        "mel": MLFeatureValue(multiArray: melArray), "mask": MLFeatureValue(multiArray: maskArray),
                    ])
                    let result = try await self.model.prediction(from: input)
                    guard let audio = result.featureValue(for: "audio")?.multiArrayValue, audio.dataType == .float32,
                          audio.shape.count == 2, audio.shape[1].intValue == audioWidth
                    else { throw Failure.output("no float32 [\(window)][\(audioWidth)] audio") }
                    var rows = [Float](repeating: 0, count: valid * audioWidth)
                    audio.withUnsafeBufferPointer(ofType: Float.self) { pointer in
                        let stride = audio.strides[0].intValue
                        for row in 0..<valid {
                            for column in 0..<audioWidth { rows[row * audioWidth + column] = pointer[row * stride + column] }
                        }
                    }
                    return (index, rows)
                }
            }
            for try await (index, rows) in group {
                let start = index * window * audioWidth
                output.replaceSubrange(start..<start + rows.count, with: rows)
            }
        }
        return output
    }

    /// how many audio tokens a clip with this many valid mel frames becomes
    static func tokenCount(frames: Int) -> Int {
        let rest = frames % EncoderGraph.chunkFrames
        var tail = rest
        if tail > 0 { for _ in 0..<3 { tail = (tail - 1) / 2 + 1 } }
        return frames / EncoderGraph.chunkFrames * EncoderGraph.tokensPerChunk + tail
    }

    /// identifies the checkpoint without reading 4 gb: file length plus a hash of the tensor header,
    /// which holds every name, shape and offset. the download itself is sha256-checked by prepare().
    private static func fingerprint(_ weights: SafeTensors) -> String {
        let headerLength = Int(weights.base.loadUnaligned(as: UInt64.self).littleEndian)
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        let bytes = UnsafeRawBufferPointer(start: weights.base, count: 8 + headerLength)
        for byte in bytes {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return "\(weights.fileLength)-\(String(hash, radix: 16))"
    }

    private static func placement(_ url: URL, _ configuration: MLModelConfiguration) async throws -> PlacementReport {
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
        var report = PlacementReport()
        guard case let .program(program) = plan.modelStructure, let main = program.functions["main"] else { return report }
        for operation in main.block.operations where operation.operatorName != "const" {
            let device: String
            switch plan.deviceUsage(for: operation)?.preferred {
            case .neuralEngine?: device = "neural engine"
            case .gpu?: device = "gpu"
            case .cpu?: device = "cpu"
            default: device = "unplanned"
            }
            let cost = plan.estimatedCost(of: operation)?.weight ?? 0
            report.operations += 1
            report.totalCost += cost
            report.devices[device, default: .init()].operations += 1
            report.devices[device, default: .init()].cost += cost
            if device != "neural engine" { report.offNeuralEngine[operation.operatorName, default: 0] += 1 }
        }
        return report
    }
}

private extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) * 1e-18 }
}
