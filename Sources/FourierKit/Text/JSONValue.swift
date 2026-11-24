import Foundation

/// a small json reader for the model's config files.
/// foundation's JSONSerialization bridges every key through NSString, which is slow for the
/// 150k-entry vocabulary, so this parses straight from the utf-8 bytes.
enum JSONValue: Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([(key: String, value: JSONValue)])

    struct ParseError: Error, CustomStringConvertible {
        var offset: Int
        var reason: String
        var description: String { "json parse error at byte \(offset): \(reason)" }
    }

    static func parse(contentsOf url: URL) throws -> JSONValue {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        return try parse(data)
    }

    static func parse(_ data: Data) throws -> JSONValue {
        try data.withUnsafeBytes { raw in
            var reader = Reader(bytes: raw.bindMemory(to: UInt8.self))
            reader.skipSpace()
            let value = try reader.value()
            reader.skipSpace()
            guard reader.position == reader.bytes.count else { throw reader.error("trailing data") }
            return value
        }
    }

    subscript(key: String) -> JSONValue? {
        guard case .object(let members) = self else { return nil }
        return members.first(where: { $0.key == key })?.value
    }

    var string: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    var number: Double? {
        if case .number(let n) = self { return n }
        return nil
    }

    var bool: Bool? {
        if case .bool(let b) = self { return b }
        return nil
    }

    var array: [JSONValue]? {
        if case .array(let a) = self { return a }
        return nil
    }

    var members: [(key: String, value: JSONValue)]? {
        if case .object(let m) = self { return m }
        return nil
    }

    var isNull: Bool {
        if case .null = self { return true }
        return false
    }

    /// an integer, or each integer of an array of integers.
    var integers: [Int]? {
        switch self {
        case .number(let n): return Int(exactly: n).map { [$0] }
        case .array(let items):
            var out: [Int] = []
            out.reserveCapacity(items.count)
            for item in items {
                guard let n = item.number, let i = Int(exactly: n) else { return nil }
                out.append(i)
            }
            return out
        default: return nil
        }
    }
}

private struct Reader {
    let bytes: UnsafeBufferPointer<UInt8>
    var position = 0
    var scratch: [UInt8] = []

    init(bytes: UnsafeBufferPointer<UInt8>) {
        self.bytes = bytes
        scratch.reserveCapacity(256)
    }

    func error(_ reason: String) -> JSONValue.ParseError {
        JSONValue.ParseError(offset: position, reason: reason)
    }

    mutating func skipSpace() {
        while position < bytes.count {
            switch bytes[position] {
            case 0x20, 0x0A, 0x0D, 0x09: position += 1
            default: return
            }
        }
    }

    mutating func expect(_ byte: UInt8) throws {
        guard position < bytes.count, bytes[position] == byte else {
            throw error("expected '\(Character(Unicode.Scalar(byte)))'")
        }
        position += 1
    }

    mutating func literal(_ text: StaticString) throws {
        let count = text.utf8CodeUnitCount
        guard position + count <= bytes.count else { throw error("unexpected end") }
        let start = text.utf8Start
        for i in 0..<count where bytes[position + i] != start[i] {
            throw error("bad literal")
        }
        position += count
    }

    mutating func value() throws -> JSONValue {
        guard position < bytes.count else { throw error("unexpected end") }
        switch bytes[position] {
        case UInt8(ascii: "{"):
            position += 1
            var members: [(key: String, value: JSONValue)] = []
            skipSpace()
            if position < bytes.count, bytes[position] == UInt8(ascii: "}") {
                position += 1
                return .object(members)
            }
            while true {
                skipSpace()
                let key = try string()
                skipSpace()
                try expect(UInt8(ascii: ":"))
                skipSpace()
                let v = try value()
                members.append((key, v))
                skipSpace()
                guard position < bytes.count else { throw error("unexpected end") }
                if bytes[position] == UInt8(ascii: ",") { position += 1; continue }
                try expect(UInt8(ascii: "}"))
                return .object(members)
            }
        case UInt8(ascii: "["):
            position += 1
            var items: [JSONValue] = []
            skipSpace()
            if position < bytes.count, bytes[position] == UInt8(ascii: "]") {
                position += 1
                return .array(items)
            }
            while true {
                skipSpace()
                items.append(try value())
                skipSpace()
                guard position < bytes.count else { throw error("unexpected end") }
                if bytes[position] == UInt8(ascii: ",") { position += 1; continue }
                try expect(UInt8(ascii: "]"))
                return .array(items)
            }
        case UInt8(ascii: "\""):
            return .string(try string())
        case UInt8(ascii: "t"):
            try literal("true")
            return .bool(true)
        case UInt8(ascii: "f"):
            try literal("false")
            return .bool(false)
        case UInt8(ascii: "n"):
            try literal("null")
            return .null
        // python's json writes these for non-finite floats
        case UInt8(ascii: "N"):
            try literal("NaN")
            return .number(.nan)
        case UInt8(ascii: "I"):
            try literal("Infinity")
            return .number(.infinity)
        case UInt8(ascii: "-") where position + 1 < bytes.count && bytes[position + 1] == UInt8(ascii: "I"):
            position += 1
            try literal("Infinity")
            return .number(-.infinity)
        default:
            return .number(try number())
        }
    }

    mutating func number() throws -> Double {
        let start = position
        var isInteger = true
        var integer: Int64 = 0
        var negative = false
        if position < bytes.count, bytes[position] == UInt8(ascii: "-") {
            negative = true
            position += 1
        }
        while position < bytes.count {
            let b = bytes[position]
            if b >= 0x30 && b <= 0x39 {
                if isInteger {
                    let (m, o1) = integer.multipliedReportingOverflow(by: 10)
                    let (a, o2) = m.addingReportingOverflow(Int64(b - 0x30))
                    if o1 || o2 { isInteger = false } else { integer = a }
                }
            } else if b == UInt8(ascii: ".") || b == UInt8(ascii: "e") || b == UInt8(ascii: "E")
                        || b == UInt8(ascii: "+") || b == UInt8(ascii: "-") {
                isInteger = false
            } else {
                break
            }
            position += 1
        }
        guard position > start else { throw error("bad value") }
        if isInteger, integer < (1 << 53) { return Double(negative ? -integer : integer) }
        let text = String(decoding: UnsafeBufferPointer(rebasing: bytes[start..<position]), as: UTF8.self)
        guard let d = Double(text) else { throw error("bad number") }
        return d
    }

    mutating func hex4() throws -> UInt32 {
        guard position + 4 <= bytes.count else { throw error("bad escape") }
        var v: UInt32 = 0
        for _ in 0..<4 {
            let b = bytes[position]
            let d: UInt32
            switch b {
            case 0x30...0x39: d = UInt32(b - 0x30)
            case 0x41...0x46: d = UInt32(b - 0x41 + 10)
            case 0x61...0x66: d = UInt32(b - 0x61 + 10)
            default: throw error("bad hex digit")
            }
            v = v << 4 | d
            position += 1
        }
        return v
    }

    mutating func string() throws -> String {
        try expect(UInt8(ascii: "\""))
        let start = position
        // fast path: no escapes
        while position < bytes.count {
            let b = bytes[position]
            if b == UInt8(ascii: "\"") {
                let s = String(decoding: UnsafeBufferPointer(rebasing: bytes[start..<position]), as: UTF8.self)
                position += 1
                return s
            }
            if b == UInt8(ascii: "\\") { break }
            position += 1
        }
        scratch.removeAll(keepingCapacity: true)
        scratch.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[start..<position]))
        while position < bytes.count {
            let b = bytes[position]
            position += 1
            if b == UInt8(ascii: "\"") {
                return String(decoding: scratch, as: UTF8.self)
            }
            guard b == UInt8(ascii: "\\") else {
                scratch.append(b)
                continue
            }
            guard position < bytes.count else { break }
            let e = bytes[position]
            position += 1
            switch e {
            case UInt8(ascii: "\""): scratch.append(0x22)
            case UInt8(ascii: "\\"): scratch.append(0x5C)
            case UInt8(ascii: "/"): scratch.append(0x2F)
            case UInt8(ascii: "b"): scratch.append(0x08)
            case UInt8(ascii: "f"): scratch.append(0x0C)
            case UInt8(ascii: "n"): scratch.append(0x0A)
            case UInt8(ascii: "r"): scratch.append(0x0D)
            case UInt8(ascii: "t"): scratch.append(0x09)
            case UInt8(ascii: "u"):
                var code = try hex4()
                if code >= 0xD800 && code < 0xDC00,
                   position + 6 <= bytes.count,
                   bytes[position] == UInt8(ascii: "\\"), bytes[position + 1] == UInt8(ascii: "u") {
                    let save = position
                    position += 2
                    let low = try hex4()
                    if low >= 0xDC00 && low < 0xE000 {
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    } else {
                        position = save
                    }
                }
                let scalar = Unicode.Scalar(code) ?? "\u{FFFD}"
                scratch.append(contentsOf: Array(String(scalar).utf8))
            default:
                throw error("bad escape")
            }
        }
        throw error("unterminated string")
    }
}
