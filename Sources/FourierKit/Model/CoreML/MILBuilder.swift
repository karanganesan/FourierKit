import Foundation

/// mil data type codes from MIL.proto
enum MILType: Int {
    case bool = 1
    case string = 2
    case float16 = 10
    case float32 = 11
    case int32 = 23

    var castName: String {
        switch self {
        case .float16: "fp16"
        case .float32: "fp32"
        case .int32: "int32"
        case .bool: "bool"
        case .string: "string"
        }
    }

    /// ArrayFeatureType.ArrayDataType for model inputs and outputs
    var arrayCode: Int {
        switch self {
        case .float16: 65552
        case .float32: 65568
        case .int32: 131_104
        default: 0
        }
    }
}

/// a named value in the program with its static type and shape. an empty shape is a scalar.
struct MILValue {
    let name: String
    let type: MILType
    let shape: [Int]
}

/// records the operations of one ml program function in order, writing large constants to the weight blob.
final class MILBuilder {
    /// type of floats inside the graph
    let compute: MILType
    private(set) var operations: [ProtobufWriter] = []
    private let blobs: WeightBlobWriter
    private var used: [String: Int] = [:]
    /// checkpoint values that did not survive narrowing to float16
    private(set) var loss = Float16Loss()
    static let blobPath = "@model_path/weights/weight.bin"

    init(compute: MILType, blobs: WeightBlobWriter) {
        self.compute = compute
        self.blobs = blobs
    }

    /// names are made unique, since mil requires every value to be defined once
    func unique(_ base: String) -> String {
        let clean = String(base.map { $0.isLetter || $0.isNumber || $0 == "_" ? $0 : "_" })
        let n = used[clean, default: 0]
        used[clean] = n + 1
        return n == 0 ? clean : "\(clean)_\(n)"
    }

    // MARK: encoding

    static func valueType(_ type: MILType, _ shape: [Int]) -> ProtobufWriter {
        var tensor = ProtobufWriter()
        tensor.int(1, type.rawValue)
        if !shape.isEmpty {
            tensor.int(2, shape.count)
            for size in shape {
                tensor.message(3) { dimension in
                    dimension.message(1) { $0.int(1, size) }
                }
            }
        }
        var value = ProtobufWriter()
        value.message(1, tensor)
        return value
    }

    static func namedValueType(_ name: String, _ type: MILType, _ shape: [Int]) -> ProtobufWriter {
        var named = ProtobufWriter()
        named.string(1, name)
        named.message(2, valueType(type, shape))
        return named
    }

    private static func immediate(_ type: MILType, _ shape: [Int], tensor: ProtobufWriter) -> ProtobufWriter {
        var value = ProtobufWriter()
        value.message(2, valueType(type, shape))
        value.message(3) { $0.message(1, tensor) }
        return value
    }

    private static func stringValue(_ text: String) -> ProtobufWriter {
        var tensor = ProtobufWriter()
        tensor.message(4) { $0.string(1, text) }
        return immediate(.string, [], tensor: tensor)
    }

    private func appendConst(_ base: String, _ type: MILType, _ shape: [Int], value: ProtobufWriter) -> MILValue {
        let name = unique(base)
        var operation = ProtobufWriter()
        operation.string(1, "const")
        operation.message(3, Self.namedValueType(name, type, shape))
        operation.mapEntry(5, key: "val", value: value)
        operation.mapEntry(5, key: "name", value: Self.stringValue(name))
        operations.append(operation)
        return MILValue(name: name, type: type, shape: shape)
    }

    // MARK: constants

    func ints(_ values: [Int], scalar: Bool = false, name: String = "ints") -> MILValue {
        var tensor = ProtobufWriter()
        tensor.message(2) { $0.packedInts(1, values) }
        return appendConst(name, .int32, scalar ? [] : [values.count], value: Self.immediate(.int32, scalar ? [] : [values.count], tensor: tensor))
    }

    func bool(_ flag: Bool, name: String = "flag") -> MILValue {
        var tensor = ProtobufWriter()
        tensor.message(3) { $0.packedInts(1, [flag ? 1 : 0]) }
        return appendConst(name, .bool, [], value: Self.immediate(.bool, [], tensor: tensor))
    }

    func string(_ text: String, name: String = "text") -> MILValue {
        appendConst(name, .string, [], value: Self.stringValue(text))
    }

    /// a small float constant stored inline at the graph's float type
    func scalar(_ x: Float, name: String = "scalar") -> MILValue {
        var tensor = ProtobufWriter()
        if compute == .float16 {
            var raw = Data()
            withUnsafeBytes(of: Float16(x).bitPattern.littleEndian) { raw.append(contentsOf: $0) }
            tensor.message(7) { $0.bytes(1, raw) }
        } else {
            tensor.message(1) { $0.packedFloats(1, [x]) }
        }
        return appendConst(name, compute, [], value: Self.immediate(compute, [], tensor: tensor))
    }

    /// a float tensor written to the weight blob at the graph's float type.
    /// `fill` writes the values as float32; they are narrowed here when the graph is float16.
    func tensor(_ base: String, shape: [Int], checkpoint: Bool = true, fill: (UnsafeMutableBufferPointer<Float>) throws -> Void) throws -> MILValue {
        let count = shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        try values.withUnsafeMutableBufferPointer { try fill($0) }
        let offset: UInt64
        if compute == .float16 {
            let half = values.map { Float16($0) }
            if checkpoint { loss.add(values, half) }
            offset = try half.withUnsafeBytes { try blobs.append(.float16, bytes: $0) }
        } else {
            offset = try values.withUnsafeBytes { try blobs.append(.float32, bytes: $0) }
        }
        var value = ProtobufWriter()
        value.message(2, Self.valueType(compute, shape))
        value.message(5) { blob in
            blob.string(1, Self.blobPath)
            blob.int(2, Int(offset))
        }
        return appendConst(base, compute, shape, value: value)
    }

    // MARK: operations

    /// appends one operation with named inputs and a single output
    @discardableResult
    func op(_ type: String, _ inputs: KeyValuePairs<String, MILValue>, shape: [Int], output: MILType? = nil, name: String? = nil) -> MILValue {
        let result = unique(name ?? type)
        let resultType = output ?? compute
        var operation = ProtobufWriter()
        operation.string(1, type)
        for (key, value) in inputs {
            var argument = ProtobufWriter()
            argument.message(1) { $0.string(1, value.name) }
            operation.mapEntry(2, key: key, value: argument)
        }
        operation.message(3, Self.namedValueType(result, resultType, shape))
        operation.mapEntry(5, key: "name", value: Self.stringValue(result))
        operations.append(operation)
        return MILValue(name: result, type: resultType, shape: shape)
    }
}

/// how many values changed when stored as float16, by kind
struct Float16Loss: Codable, Sendable {
    /// finite values beyond float16's range, which became infinite
    var overflow = 0
    /// nonzero values that became zero
    var flushed = 0
    /// values that kept a nonzero finite value but lost bits, mostly by landing in the subnormal range
    var rounded = 0
    var total: Int { overflow + flushed + rounded }

    mutating func add(_ exact: [Float], _ half: [Float16]) {
        for i in exact.indices where Float(half[i]) != exact[i] {
            if half[i].isInfinite { overflow += 1 } else if half[i] == 0 { flushed += 1 } else { rounded += 1 }
        }
    }
}

// shape-checked helpers for the ops the encoder uses
extension MILBuilder {
    func cast(_ x: MILValue, to type: MILType, name: String? = nil) -> MILValue {
        op("cast", ["x": x, "dtype": string(type.castName, name: "dtype")], shape: x.shape, output: type, name: name)
    }

    /// 2d convolution, nchw, square kernel, symmetric padding
    func conv(_ x: MILValue, weight: MILValue, bias: MILValue?, stride: Int, pad: Int, name: String) -> MILValue {
        precondition(x.shape.count == 4 && weight.shape.count == 4 && weight.shape[1] == x.shape[1])
        let k = weight.shape[2]
        let h = (x.shape[2] + 2 * pad - k) / stride + 1
        let w = (x.shape[3] + 2 * pad - k) / stride + 1
        let strides = ints([stride, stride], name: "strides")
        let padType = string("custom", name: "pad_type")
        let pads = ints([pad, pad, pad, pad], name: "pad")
        let dilations = ints([1, 1], name: "dilations")
        let groups = ints([1], scalar: true, name: "groups")
        let shape = [x.shape[0], weight.shape[0], h, w]
        if let bias {
            return op("conv", ["x": x, "weight": weight, "bias": bias, "strides": strides, "pad_type": padType,
                               "pad": pads, "dilations": dilations, "groups": groups], shape: shape, name: name)
        }
        return op("conv", ["x": x, "weight": weight, "strides": strides, "pad_type": padType,
                           "pad": pads, "dilations": dilations, "groups": groups], shape: shape, name: name)
    }

    /// y = x w^T + b over the last axis; weight is [out][in]
    func linear(_ x: MILValue, weight: MILValue, bias: MILValue?, name: String) -> MILValue {
        precondition(x.shape.last == weight.shape[1])
        let shape = x.shape.dropLast() + [weight.shape[0]]
        if let bias { return op("linear", ["x": x, "weight": weight, "bias": bias], shape: Array(shape), name: name) }
        return op("linear", ["x": x, "weight": weight], shape: Array(shape), name: name)
    }

    func matmul(_ x: MILValue, _ y: MILValue, transposeX: Bool = false, transposeY: Bool = false, name: String) -> MILValue {
        let r = x.shape.count
        let m = transposeX ? x.shape[r - 1] : x.shape[r - 2]
        let kx = transposeX ? x.shape[r - 2] : x.shape[r - 1]
        let n = transposeY ? y.shape[y.shape.count - 2] : y.shape.last!
        let ky = transposeY ? y.shape.last! : y.shape[y.shape.count - 2]
        precondition(kx == ky)
        let shape = Array(x.shape.dropLast(2)) + [m, n]
        return op("matmul", ["x": x, "y": y, "transpose_x": bool(transposeX, name: "transpose_x"),
                             "transpose_y": bool(transposeY, name: "transpose_y")], shape: shape, name: name)
    }

    func reshape(_ x: MILValue, _ shape: [Int], name: String = "reshape") -> MILValue {
        precondition(x.shape.reduce(1, *) == shape.reduce(1, *))
        return op("reshape", ["x": x, "shape": ints(shape, name: "shape")], shape: shape, name: name)
    }

    func transpose(_ x: MILValue, _ perm: [Int], name: String = "transpose") -> MILValue {
        op("transpose", ["x": x, "perm": ints(perm, name: "perm")], shape: perm.map { x.shape[$0] }, name: name)
    }

    func add(_ x: MILValue, _ y: MILValue, name: String = "add") -> MILValue {
        op("add", ["x": x, "y": y], shape: broadcast(x.shape, y.shape), name: name)
    }

    func mul(_ x: MILValue, _ y: MILValue, name: String = "mul") -> MILValue {
        op("mul", ["x": x, "y": y], shape: broadcast(x.shape, y.shape), name: name)
    }

    func layerNorm(_ x: MILValue, axis: Int, gamma: MILValue, beta: MILValue, epsilon: Float, name: String) -> MILValue {
        op("layer_norm", ["x": x, "axes": ints([axis], name: "axes"), "gamma": gamma, "beta": beta,
                          "epsilon": scalar(epsilon, name: "epsilon")], shape: x.shape, name: name)
    }

    func gelu(_ x: MILValue, name: String) -> MILValue {
        op("gelu", ["x": x, "mode": string("EXACT", name: "mode")], shape: x.shape, name: name)
    }

    func softmax(_ x: MILValue, axis: Int, name: String) -> MILValue {
        op("softmax", ["x": x, "axis": ints([axis], scalar: true, name: "axis")], shape: x.shape, name: name)
    }

    private func broadcast(_ a: [Int], _ b: [Int]) -> [Int] {
        let rank = max(a.count, b.count)
        let a = Array(repeating: 1, count: rank - a.count) + a
        let b = Array(repeating: 1, count: rank - b.count) + b
        return zip(a, b).map { x, y in
            precondition(x == y || x == 1 || y == 1, "shapes do not broadcast")
            return max(x, y)
        }
    }
}
