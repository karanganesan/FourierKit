import CoreML
import Darwin
import Foundation
import Metal
#if os(iOS)
import os
#endif

/// what the device running the engine has, measured with real queries when the engine loads.
/// every size and default the engine picks comes from this through `EngineLimits`.
struct DeviceProfile: Sendable, Hashable {
    var physicalMemory: Int
    var activeProcessorCount: Int
    /// `hw.perflevel0.physicalcpu`, nil where the system does not report it
    var performanceCores: Int?
    /// how much more this process may use before the system stops it
    var memoryHeadroom: Int
    var gpuName: String
    var maxBufferLength: Int
    var recommendedMaxWorkingSetSize: Int
    var hasUnifiedMemory: Bool
    var maxThreadsPerThreadgroup: Int
    var hasNeuralEngine: Bool

    static func measure(device: MTLDevice) -> DeviceProfile {
        DeviceProfile(
            physicalMemory: Int(clamping: ProcessInfo.processInfo.physicalMemory),
            activeProcessorCount: cpu.activeProcessorCount,
            performanceCores: cpu.performanceCores,
            memoryHeadroom: availableMemory(),
            gpuName: device.name,
            maxBufferLength: device.maxBufferLength,
            recommendedMaxWorkingSetSize: Int(clamping: device.recommendedMaxWorkingSetSize),
            hasUnifiedMemory: device.hasUnifiedMemory,
            maxThreadsPerThreadgroup: device.maxThreadsPerThreadgroup.width,
            hasNeuralEngine: hasNeuralEngine
        )
    }

    /// the cpu facts, read once. the mel fan-out needs them before any engine exists.
    static let cpu: (activeProcessorCount: Int, performanceCores: Int?) = {
        var cores: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let found = sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, nil, 0) == 0 && cores > 0
        return (ProcessInfo.processInfo.activeProcessorCount, found ? Int(cores) : nil)
    }()

    /// whether core ml lists a neural engine on this device; the simulator has none.
    static let hasNeuralEngine: Bool = MLComputeDevice.allComputeDevices.contains {
        if case .neuralEngine = $0 { true } else { false }
    }

    /// the process's headroom. ios reports its real limit. a mac has no per-process limit, so it is
    /// physical memory less an eighth for the system and other apps (32 gb → 28 gb).
    private static func availableMemory() -> Int {
        let physical = Int(clamping: ProcessInfo.processInfo.physicalMemory)
        let shared = physical - physical / 8
        #if os(iOS)
        // zero in the simulator, which runs as a mac process without a limit
        let reported = os_proc_available_memory()
        return reported > 0 ? reported : shared
        #else
        return shared
        #endif
    }
}

/// the engine's sizes and defaults, derived from the profile, the checkpoint and the generation
/// settings. the caps are the values tuned and exactness-tested on an m4 with 32 gb, which
/// derives exactly them; smaller devices derive less.
struct EngineLimits: Sendable, Hashable {
    static let slotCap = 12_288
    static let rowCap = 12
    static let convGroupCap = 8
    /// the longest piece read in one pass, and the shortest piece the cache must always hold
    static let onePassSeconds = 120
    /// prompt tokens around the audio: chat turns, language and some context, rounded up
    static let promptAllowance = 64

    /// the checkpoint is wrapped in several page-aligned buffers instead of one
    var weightsWindowed: Bool
    /// decoder cache slots in the arena, and the bytes of one slot
    var slotCapacity: Int
    var bytesPerSlot: Int
    /// the widest decoder batch in rows, and the width under thermal, power or memory pressure
    var maxRows: Int
    var pressuredMaxRows: Int
    /// 1-second chunks per conv command buffer in the gpu encoder
    var convGroup: Int
    /// piece length for long audio with automatic chunking
    var automaticPieceSeconds: Double
    /// whether the neural engine encoder can be pinned at all
    var offersNeuralEngine: Bool
    /// the largest work buffers one two-minute piece needs, held back from the cache arena
    var scratchReserve: Int

    /// before any engine has loaded, long audio is cut like on the tested mac.
    static let current = Locked<EngineLimits?>(nil)
    static var automaticPieceSeconds: Double { current.value?.automaticPieceSeconds ?? 30 }

    init(profile: DeviceProfile, shape: ModelShape, mappedLength: Int, weightBytes: Int, settings: GenerationSettings) throws {
        let beams = max(1, settings.numBeams)
        offersNeuralEngine = profile.hasNeuralEngine
        // one metal buffer cannot be longer than maxBufferLength (m4 32 gb: about 22 gb > 4.08 gb file)
        weightsWindowed = mappedLength > profile.maxBufferLength

        // a slot holds one token's key and value in every layer: 28 layers × 1,024 × 4 bytes × 2 = 229,376 bytes
        bytesPerSlot = shape.decoderLayers * shape.keyWidth * 4 * 2
        // a piece takes its prompt plus a row per beam per new token: 120 s → 1,560 audio tokens + 64 + 3 × 512 = 3,160
        func slots(seconds: Int) -> Int {
            Self.promptAllowance + LogMel.audioTokenCount(frames: seconds * 100) + beams * settings.maxNewTokens
        }
        let minimumSlots = (slots(seconds: Self.onePassSeconds) + 255) / 256 * 256

        let convChunk = Self.convBytesPerChunk(shape)
        scratchReserve = Self.scratchBytes(shape: shape, seconds: Self.onePassSeconds, convGroup: Self.convGroupCap,
                                           convBytesPerChunk: convChunk, stepRows: Self.rowCap)

        // the mapped weights count against the headroom, then the work buffers; the cache arena may use
        // three quarters of the rest, keeping a margin for the app. m4 32 gb: (28 gib − 4.08 gb − reserve) × 3/4
        // is about 82,000 slots, so the cap applies.
        let spare = max(0, profile.memoryHeadroom - weightBytes - scratchReserve)
        let fitting = spare / 4 * 3 / bytesPerSlot / 256 * 256
        slotCapacity = min(Self.slotCap, fitting)
        guard slotCapacity >= minimumSlots else {
            throw EngineFailure.deviceLimit(
                "this device leaves \(Self.gigabytes(spare)) for the decoder cache after the weights, and one two-minute piece needs \(Self.gigabytes(minimumSlots * bytesPerSlot))")
        }

        // rows: whole 30-second pieces (64 + 390 + 3 × 512 = 1,990 slots) the arena holds, times their beams;
        // m4: 12,288 / 1,990 = 6 pieces = 18 rows. a row also needs about one megabyte of step buffers
        // inside the gpu working set beside the weights and arena. at least one piece always runs.
        let pieces = slotCapacity / slots(seconds: 30)
        let perRow = shape.vocabulary * 4 + shape.heads * 4 * 4096 + (8 * shape.width + 2 * shape.inner) * 4
        let workingSpare = max(0, profile.recommendedMaxWorkingSetSize - weightBytes - slotCapacity * bytesPerSlot - scratchReserve)
        maxRows = max(beams, min(Self.rowCap, pieces * beams, workingSpare / perRow))
        // under pressure, half the width but never less than one piece: m4 12 → 6
        pressuredMaxRows = max(beams, maxRows / 2)

        // eight 1-second chunks of conv buffers are about 208 mb; fewer only if that passes a quarter of the spare
        convGroup = max(1, min(Self.convGroupCap, spare / 4 / convChunk))

        // with room for two pieces at once, long audio is cut into 30 s pieces that overlap on the gpu;
        // with room for one, cutting buys nothing, so pieces are as long as one pass allows
        automaticPieceSeconds = pieces >= 2 ? 30 : Double(Self.onePassSeconds)
    }

    func maxRows(underPressure: Bool) -> Int { underPressure ? pressuredMaxRows : maxRows }

    /// conv work buffers for one 1-second chunk: input, three patch and three activation buffers,
    /// and the flattened rows (about 26 mb).
    static func convBytesPerChunk(_ shape: ModelShape) -> Int {
        var heights = [shape.melBins], widths = [shape.chunkFrames]
        for _ in 0..<3 {
            heights.append((heights.last! - 1) / 2 + 1)
            widths.append((widths.last! - 1) / 2 + 1)
        }
        let cells = (0...3).map { heights[$0] * widths[$0] }
        let c = shape.convChannels
        let floats = cells[0] + cells[1] * 9 + cells[1] * c + cells[2] * c * 9 + cells[2] * c
            + cells[3] * c * 9 + cells[3] * c + shape.tokensPerChunk * c * heights[3]
        return floats * 4
    }

    /// the work buffers kept for one piece of `seconds`: its prefill, its encoder, the conv groups,
    /// a full decoder step and the 64 mb widened-weight buffer. they are all held at once.
    static func scratchBytes(shape: ModelShape, seconds: Int, convGroup: Int, convBytesPerChunk: Int, stepRows: Int) -> Int {
        let audio = LogMel.audioTokenCount(frames: seconds * 100)
        let rows = audio + promptAllowance
        let context = (rows + 63) / 64 * 64
        let prefill = rows * 4 * (6 * shape.width + 2 * shape.keyWidth + 2 * shape.inner + shape.heads * context)
        let window = (shape.windowTokens + 63) / 64 * 64
        let encoder = audio * 4 * (7 * shape.encoderWidth + shape.encoderInner + shape.encoderHeads * window + shape.audioWidth)
        let step = stepRows * 4 * (shape.vocabulary + 8 * shape.width + 2 * shape.inner + shape.heads * 4096)
        return prefill + encoder + convGroup * convBytesPerChunk + step + (64 << 20)
    }

    static func gigabytes(_ bytes: Int) -> String {
        String(format: "%.2f gb", Double(bytes) / 1e9)
    }
}
