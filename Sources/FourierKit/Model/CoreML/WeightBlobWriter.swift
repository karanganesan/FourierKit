import Foundation

/// writes core ml's weight.bin: a 64-byte file header, then per tensor a 64-byte metadata record
/// followed by its data. records and data both start on 64-byte boundaries.
/// a const op points at the metadata record's offset, not at the data.
final class WeightBlobWriter {
    enum ElementType: UInt32 {
        case float16 = 1
        case float32 = 2
    }

    private let handle: FileHandle
    private(set) var position: UInt64 = 64
    private(set) var count: UInt32 = 0
    private static let alignment: UInt64 = 64
    private static let sentinel: UInt32 = 0xDEAD_BEEF

    init(url: URL) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        try handle.write(contentsOf: Data(count: Int(Self.alignment)))
    }

    /// appends one tensor's raw little-endian bytes and returns the offset a blob reference uses
    func append(_ type: ElementType, bytes: UnsafeRawBufferPointer) throws -> UInt64 {
        let record = position
        let dataStart = record + Self.alignment
        var metadata = Data(count: Int(Self.alignment))
        metadata.withUnsafeMutableBytes { raw in
            raw.storeBytes(of: Self.sentinel.littleEndian, toByteOffset: 0, as: UInt32.self)
            raw.storeBytes(of: type.rawValue.littleEndian, toByteOffset: 4, as: UInt32.self)
            raw.storeBytes(of: UInt64(bytes.count).littleEndian, toByteOffset: 8, as: UInt64.self)
            raw.storeBytes(of: dataStart.littleEndian, toByteOffset: 16, as: UInt64.self)
        }
        try handle.write(contentsOf: metadata)
        try handle.write(contentsOf: bytes)
        let end = dataStart + UInt64(bytes.count)
        let padded = (end + Self.alignment - 1) / Self.alignment * Self.alignment
        if padded > end { try handle.write(contentsOf: Data(count: Int(padded - end))) }
        position = padded
        count += 1
        return record
    }

    /// fills in the header: blob count, then storage format version 2
    func finish() throws {
        try handle.seek(toOffset: 0)
        var header = Data(count: 8)
        header.withUnsafeMutableBytes { raw in
            raw.storeBytes(of: count.littleEndian, toByteOffset: 0, as: UInt32.self)
            raw.storeBytes(of: UInt32(2).littleEndian, toByteOffset: 4, as: UInt32.self)
        }
        try handle.write(contentsOf: header)
        try handle.close()
    }
}
