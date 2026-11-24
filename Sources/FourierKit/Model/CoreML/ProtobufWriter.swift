import Foundation

/// the few protobuf wire rules core ml model files need: varints and length-delimited fields.
/// fields are appended in call order, which protobuf readers accept for any order.
struct ProtobufWriter {
    private(set) var data = Data()

    private mutating func varint(_ value: UInt64) {
        var rest = value
        while rest >= 0x80 {
            data.append(UInt8(truncatingIfNeeded: rest) | 0x80)
            rest >>= 7
        }
        data.append(UInt8(rest))
    }

    private mutating func key(_ field: Int, _ wire: UInt64) {
        varint(UInt64(field) << 3 | wire)
    }

    /// int32, int64, uint64, bool and enum fields all share wire type 0. negative ints take ten bytes.
    mutating func int(_ field: Int, _ value: Int) {
        key(field, 0)
        varint(UInt64(bitPattern: Int64(value)))
    }

    mutating func bool(_ field: Int, _ value: Bool) {
        int(field, value ? 1 : 0)
    }

    mutating func bytes(_ field: Int, _ value: Data) {
        key(field, 2)
        varint(UInt64(value.count))
        data.append(value)
    }

    mutating func string(_ field: Int, _ value: String) {
        bytes(field, Data(value.utf8))
    }

    mutating func message(_ field: Int, _ value: ProtobufWriter) {
        bytes(field, value.data)
    }

    mutating func message(_ field: Int, _ build: (inout ProtobufWriter) -> Void) {
        var nested = ProtobufWriter()
        build(&nested)
        bytes(field, nested.data)
    }

    mutating func packedInts(_ field: Int, _ values: [Int]) {
        var packed = ProtobufWriter()
        for value in values { packed.varint(UInt64(bitPattern: Int64(value))) }
        bytes(field, packed.data)
    }

    mutating func packedFloats(_ field: Int, _ values: [Float]) {
        var packed = Data(capacity: values.count * 4)
        for value in values {
            withUnsafeBytes(of: value.bitPattern.littleEndian) { packed.append(contentsOf: $0) }
        }
        bytes(field, packed)
    }

    /// a map entry: key in field 1, message value in field 2
    mutating func mapEntry(_ field: Int, key: String, value: ProtobufWriter) {
        message(field) { entry in
            entry.string(1, key)
            entry.message(2, value)
        }
    }
}
