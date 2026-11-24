import Foundation
import Synchronization

/// single-producer single-consumer ring of samples.
///
/// the writer never locks, allocates or waits, so it is safe on the realtime audio thread.
/// positions are absolute: `totalWritten` only grows, so a reader can ask for audio from before
/// it noticed speech (pre-roll) as long as that audio has not been overwritten yet.
final class RingBuffer: @unchecked Sendable {
    enum ReadResult: Equatable, Sendable {
        /// `count` samples were copied, starting at the requested position
        case read(count: Int)
        /// the requested audio was overwritten; the oldest position still held is `oldest`
        case overrun(oldest: Int)
    }

    let capacity: Int
    private let mask: Int
    private let storage: UnsafeMutablePointer<Float>
    /// end of the last published write
    private let written = Atomic<Int>(0)
    /// end of the write in progress, stored before any sample is touched (seqlock style)
    private let reserved = Atomic<Int>(0)

    /// `minimumCapacity` is rounded up to a power of two.
    init(minimumCapacity: Int) {
        var size = 1
        while size < max(minimumCapacity, 2) { size <<= 1 }
        capacity = size
        mask = size - 1
        storage = .allocate(capacity: size)
        storage.initialize(repeating: 0, count: size)
    }

    deinit {
        storage.deallocate()
    }

    /// total samples ever written, the absolute position one past the newest sample.
    var totalWritten: Int { written.load(ordering: .acquiring) }

    /// oldest absolute position still held.
    var oldestAvailable: Int { max(0, totalWritten - capacity) }

    // MARK: writer (realtime safe)

    /// copies samples in and publishes them. if `count` exceeds the capacity only the newest samples are kept.
    func write(_ samples: UnsafePointer<Float>, count: Int) {
        guard count > 0 else { return }
        // only the writer stores this counter, so a relaxed load of our own value is enough
        let start = written.load(ordering: .relaxed)
        reserved.store(start + count, ordering: .relaxed)
        // keeps the reservation ahead of the sample stores below
        atomicMemoryFence(ordering: .releasing)
        let skip = max(0, count - capacity)
        var position = start + skip
        var source = samples + skip
        var remaining = count - skip
        while remaining > 0 {
            let index = position & mask
            let n = min(remaining, capacity - index)
            (storage + index).update(from: source, count: n)
            position += n
            source += n
            remaining -= n
        }
        written.store(start + count, ordering: .releasing)
    }

    // MARK: reader

    /// copies up to `count` samples starting at absolute `position` into `destination`.
    /// fewer are copied when less has been written. the copy is validated after the fact, so a
    /// writer lapping the reader mid-copy is reported as an overrun rather than torn audio.
    func read(from position: Int, count: Int, into destination: UnsafeMutablePointer<Float>) -> ReadResult {
        let end = written.load(ordering: .acquiring)
        let begin = max(0, position)
        if end - begin > capacity {
            return .overrun(oldest: end - capacity)
        }
        let n = max(0, min(count, end - begin))
        var copied = 0
        while copied < n {
            let index = (begin + copied) & mask
            let chunk = min(n - copied, capacity - index)
            (destination + copied).update(from: storage + index, count: chunk)
            copied += chunk
        }
        // a write in flight past `begin + capacity` may have touched what we copied
        atomicMemoryFence(ordering: .acquiring)
        let after = reserved.load(ordering: .relaxed)
        if after - begin > capacity {
            return .overrun(oldest: max(0, after - capacity))
        }
        return .read(count: n)
    }

    /// array form of `read(from:count:into:)`; returns nil on overrun.
    func read(from position: Int, count: Int) -> [Float]? {
        let available = max(0, min(count, totalWritten - max(0, position)))
        var out = [Float](repeating: 0, count: available)
        let result = out.withUnsafeMutableBufferPointer {
            read(from: position, count: available, into: $0.baseAddress!)
        }
        switch result {
        case .read(let n):
            if n < out.count { out.removeLast(out.count - n) }
            return out
        case .overrun:
            return nil
        }
    }

    /// the newest `count` samples (or fewer if not that many were written), with their start position.
    func readNewest(_ count: Int) -> (position: Int, samples: [Float])? {
        let end = totalWritten
        let n = min(count, end, capacity)
        let start = end - n
        guard let samples = read(from: start, count: n) else { return nil }
        return (start, samples)
    }
}
