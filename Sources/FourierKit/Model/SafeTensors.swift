import Foundation

/// a memory-mapped safetensors file. nothing is copied: tensors are byte ranges of the mapping.
final class SafeTensors: @unchecked Sendable {
    struct Tensor: Sendable {
        var name: String
        var dtype: String
        var shape: [Int]
        /// byte offset from the start of the file
        var offset: Int
        var byteCount: Int
        var count: Int { shape.reduce(1, *) }
    }

    enum Failure: Error, CustomStringConvertible {
        case unreadable(String)
        case missing(String)
        case unexpected(String)

        var description: String {
            switch self {
            case .unreadable(let why): "the weights file could not be read: \(why)"
            case .missing(let name): "the weights file has no tensor named \(name)"
            case .unexpected(let why): "the weights file is not what the engine expects: \(why)"
            }
        }
    }

    let url: URL
    /// start of the mapping, page aligned
    let base: UnsafeMutableRawPointer
    /// length of the mapping, rounded up to a whole page so metal can wrap it without a copy
    let mappedLength: Int
    let fileLength: Int
    private(set) var tensors: [String: Tensor] = [:]

    init(url: URL) throws {
        self.url = url
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else { throw Failure.unreadable(String(cString: strerror(errno))) }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { throw Failure.unreadable(String(cString: strerror(errno))) }
        fileLength = Int(info.st_size)
        guard fileLength > 8 else { throw Failure.unreadable("the file is empty") }
        let page = Int(getpagesize())
        mappedLength = (fileLength + page - 1) / page * page
        guard let mapping = mmap(nil, mappedLength, PROT_READ, MAP_PRIVATE, descriptor, 0), mapping != MAP_FAILED else {
            throw Failure.unreadable(String(cString: strerror(errno)))
        }
        base = mapping

        let headerLength = Int(base.loadUnaligned(as: UInt64.self).littleEndian)
        guard headerLength > 0, 8 + headerLength <= fileLength else { throw Failure.unreadable("bad header length") }
        let header = Data(bytesNoCopy: base + 8, count: headerLength, deallocator: .none)
        guard let entries = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw Failure.unreadable("bad header")
        }
        let dataStart = 8 + headerLength
        for (name, value) in entries where name != "__metadata__" {
            guard let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
            else { throw Failure.unreadable("bad entry for \(name)") }
            guard dataStart + offsets[1] <= fileLength else { throw Failure.unreadable("\(name) runs past the end of the file") }
            tensors[name] = Tensor(
                name: name, dtype: dtype, shape: shape,
                offset: dataStart + offsets[0], byteCount: offsets[1] - offsets[0]
            )
        }
    }

    deinit {
        munmap(base, mappedLength)
    }

    func tensor(_ name: String) throws -> Tensor {
        guard let tensor = tensors[name] else { throw Failure.missing(name) }
        return tensor
    }

    func has(_ name: String) -> Bool { tensors[name] != nil }

    func pointer(_ tensor: Tensor) -> UnsafeRawPointer { UnsafeRawPointer(base + tensor.offset) }

    /// reads a float32, float64 or int32 tensor as an array. used by the golden test, not by the engine.
    func floats(_ name: String) throws -> [Float] {
        let tensor = try tensor(name)
        let source = pointer(tensor)
        switch tensor.dtype {
        case "F32":
            return [Float](unsafeUninitializedCapacity: tensor.count) { buffer, count in
                memcpy(buffer.baseAddress!, source, tensor.count * 4)
                count = tensor.count
            }
        case "F64":
            return (0..<tensor.count).map { Float(source.loadUnaligned(fromByteOffset: $0 * 8, as: Double.self)) }
        case "BF16":
            return (0..<tensor.count).map {
                Float(bitPattern: UInt32(source.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self)) << 16)
            }
        default:
            throw Failure.unexpected("\(name) is \(tensor.dtype), not a float tensor")
        }
    }

    func doubles(_ name: String) throws -> [Double] {
        let tensor = try tensor(name)
        let source = pointer(tensor)
        switch tensor.dtype {
        case "F64": return (0..<tensor.count).map { source.loadUnaligned(fromByteOffset: $0 * 8, as: Double.self) }
        case "F32": return (0..<tensor.count).map { Double(source.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self)) }
        default: throw Failure.unexpected("\(name) is \(tensor.dtype), not a float tensor")
        }
    }

    func ints(_ name: String) throws -> [Int32] {
        let tensor = try tensor(name)
        let source = pointer(tensor)
        switch tensor.dtype {
        case "I32": return (0..<tensor.count).map { source.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self) }
        case "I64": return (0..<tensor.count).map { Int32(source.loadUnaligned(fromByteOffset: $0 * 8, as: Int64.self)) }
        default: throw Failure.unexpected("\(name) is \(tensor.dtype), not an integer tensor")
        }
    }
}
