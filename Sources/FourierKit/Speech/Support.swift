import Darwin
import Foundation
import Synchronization

extension Duration {
    var inSeconds: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }

    var inMilliseconds: Int { Int((inSeconds * 1000).rounded()) }
}

private let workQueue = DispatchQueue(label: "fourierkit.speech.work", qos: .userInitiated, attributes: .concurrent)

/// runs blocking work on a background queue, so neither the caller's actor nor the cooperative pool waits on it.
func offload<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { continuation in
        workQueue.async {
            continuation.resume(with: Result { try work() })
        }
    }
}

func offload<T: Sendable>(_ work: @escaping @Sendable () -> T) async -> T {
    await withCheckedContinuation { continuation in
        workQueue.async { continuation.resume(returning: work()) }
    }
}

/// the words user-facing text uses for this kind of device and its settings app.
enum DeviceWords {
    #if os(macOS)
    static let thisDevice = "this mac"
    static let settings = "system settings"
    #else
    static let thisDevice = "this device"
    static let settings = "settings"
    #endif
}

func now() -> Double { EngineHost.now() }

/// a value behind a lock, readable from any thread.
final class Locked<Value: Sendable>: Sendable {
    private let mutex: Mutex<Value>
    init(_ value: Value) { mutex = Mutex(value) }
    var value: Value { mutex.withLock { $0 } }
    @discardableResult
    func update<R: Sendable>(_ body: (inout Value) -> R) -> R { mutex.withLock { body(&$0) } }
}

/// fans one value out to any number of readers, newest value only, at most one update per `interval`.
///
/// a new reader gets the latest value at once. sending never blocks and never runs reader code.
final class Broadcaster<Value: Sendable>: Sendable {
    private struct State {
        var latest: Value?
        var readers: [UUID: AsyncStream<Value>.Continuation] = [:]
        var flushScheduled = false
        var lastFlush: Double = 0
        var finished = false
    }

    private let state: Mutex<State>
    private let interval: Double
    private let queue: DispatchQueue
    private let onReadersChange: (@Sendable (Int) -> Void)?

    init(initial: Value? = nil, interval: Duration = .milliseconds(16), onReadersChange: (@Sendable (Int) -> Void)? = nil) {
        state = Mutex(State(latest: initial))
        self.interval = interval.inSeconds
        queue = DispatchQueue(label: "fourierkit.speech.broadcast", qos: .userInitiated)
        self.onReadersChange = onReadersChange
    }

    var latest: Value? { state.withLock { $0.latest } }
    var readerCount: Int { state.withLock { $0.readers.count } }

    func stream() -> AsyncStream<Value> {
        let (stream, continuation) = AsyncStream.makeStream(of: Value.self, bufferingPolicy: .bufferingNewest(1))
        let id = UUID()
        let (count, finished): (Int, Bool) = state.withLock { state in
            if state.finished { return (state.readers.count, true) }
            if let latest = state.latest { continuation.yield(latest) }
            state.readers[id] = continuation
            return (state.readers.count, false)
        }
        if finished {
            continuation.finish()
            return stream
        }
        continuation.onTermination = { [weak self] _ in
            guard let self else { return }
            let count = self.state.withLock { state -> Int in
                state.readers[id] = nil
                return state.readers.count
            }
            self.onReadersChange?(count)
        }
        onReadersChange?(count)
        return stream
    }

    /// stores the value and delivers it now or at the next frame, whichever keeps the cap.
    func send(_ value: Value) {
        let delay: Double? = state.withLock { state in
            state.latest = value
            guard !state.flushScheduled, !state.finished else { return nil }
            state.flushScheduled = true
            return max(0, state.lastFlush + interval - now())
        }
        guard let delay else { return }
        if delay == 0 {
            queue.async { self.flush() }
        } else {
            queue.asyncAfter(deadline: .now() + delay) { self.flush() }
        }
    }

    /// delivers the value at once, bypassing the frame cap.
    func sendNow(_ value: Value) {
        let readers = state.withLock { state -> [AsyncStream<Value>.Continuation] in
            state.latest = value
            state.lastFlush = now()
            return Array(state.readers.values)
        }
        for reader in readers { reader.yield(value) }
    }

    private func flush() {
        let (value, readers) = state.withLock { state -> (Value?, [AsyncStream<Value>.Continuation]) in
            state.flushScheduled = false
            state.lastFlush = now()
            return (state.latest, Array(state.readers.values))
        }
        guard let value else { return }
        for reader in readers { reader.yield(value) }
    }

    /// ends every reader's stream. later readers get the latest value and an ended stream.
    func finish(with value: Value? = nil) {
        let readers = state.withLock { state -> [AsyncStream<Value>.Continuation] in
            if let value { state.latest = value }
            state.finished = true
            let readers = Array(state.readers.values)
            state.readers.removeAll()
            return readers
        }
        for reader in readers {
            if let value { reader.yield(value) }
            reader.finish()
        }
    }
}

/// the app's physical footprint, the number activity monitor shows.
func appMemoryFootprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : 0
}

/// rms of a chunk mapped to a meter value: 0 at -60 dbfs or quieter, 1 at full scale.
func meterLevel(_ samples: [Float]) -> Float {
    guard !samples.isEmpty else { return 0 }
    var sum: Float = 0
    for x in samples { sum += x * x }
    let rms = (sum / Float(samples.count)).squareRoot()
    let db = 20 * log10(max(rms, 1e-9))
    return min(1, max(0, (db + 60) / 60))
}

/// runs `operation` and returns as soon as the caller is cancelled, even if the operation has not seen it yet.
///
/// the host only notices a cancel between decoder steps and misses one that lands while a piece is
/// being read in, so the caller stops waiting at once and the piece is left to finish on its own.
func returningOnCancel<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) async throws -> T {
    let slot = Locked<CheckedContinuation<T, any Error>?>(nil)
    let inner = Locked<Task<Void, Never>?>(nil)
    @Sendable func take() -> CheckedContinuation<T, any Error>? {
        slot.update { slot in defer { slot = nil }; return slot }
    }
    return try await withTaskCancellationHandler {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T, any Error>) in
            slot.update { $0 = continuation }
            if Task.isCancelled {
                take()?.resume(throwing: CancellationError())
                return
            }
            inner.update {
                $0 = Task {
                    do {
                        let value = try await operation()
                        take()?.resume(returning: value)
                    } catch {
                        take()?.resume(throwing: error)
                    }
                }
            }
        }
    } onCancel: {
        inner.value?.cancel()
        take()?.resume(throwing: CancellationError())
    }
}
