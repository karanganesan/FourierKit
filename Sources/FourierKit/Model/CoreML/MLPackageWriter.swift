import Foundation

/// lays out an .mlpackage by hand: Manifest.json, the model protobuf and the weight blob.
struct MLPackageWriter {
    struct Feature {
        var name: String
        var type: MILType
        var shape: [Int]
    }

    let url: URL

    var specificationDirectory: URL { url.appendingPathComponent("Data/com.apple.CoreML", isDirectory: true) }
    var weightsURL: URL { specificationDirectory.appendingPathComponent("weights/weight.bin") }

    /// removes any old package and makes the folders, so the weight blob can be streamed in first
    func prepare() throws {
        let files = FileManager.default
        try? files.removeItem(at: url)
        try files.createDirectory(at: weightsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    }

    /// writes the model protobuf and the manifest once the graph and blob are done.
    /// opset CoreML8 with specification version 9 is the macos 15 level, enough for every op used here.
    func finish(inputs: [Feature], outputs: [Feature], operations: [ProtobufWriter], metadata: [String: String]) throws {
        let opset = "CoreML8"
        var block = ProtobufWriter()
        for output in outputs { block.string(2, output.name) }
        for operation in operations { block.message(3, operation) }

        var function = ProtobufWriter()
        for input in inputs { function.message(1, MILBuilder.namedValueType(input.name, input.type, input.shape)) }
        function.string(2, opset)
        function.mapEntry(3, key: opset, value: block)

        var program = ProtobufWriter()
        program.int(1, 1)
        program.mapEntry(2, key: "main", value: function)

        var description = ProtobufWriter()
        for input in inputs { description.message(1, Self.feature(input)) }
        for output in outputs { description.message(10, Self.feature(output)) }
        description.message(100) { meta in
            for key in metadata.keys.sorted() {
                meta.message(100) { entry in
                    entry.string(1, key)
                    entry.string(2, metadata[key]!)
                }
            }
        }

        var model = ProtobufWriter()
        model.int(1, 9)
        model.message(2, description)
        model.message(502, program)
        try model.data.write(to: specificationDirectory.appendingPathComponent("model.mlmodel"))

        let modelID = UUID().uuidString
        let weightsID = UUID().uuidString
        let manifest: [String: Any] = [
            "fileFormatVersion": "1.0.0",
            "itemInfoEntries": [
                modelID: ["author": "com.apple.CoreML", "description": "CoreML Model Specification",
                          "name": "model.mlmodel", "path": "com.apple.CoreML/model.mlmodel"],
                weightsID: ["author": "com.apple.CoreML", "description": "CoreML Model Weights",
                            "name": "weights", "path": "com.apple.CoreML/weights"],
            ],
            "rootModelIdentifier": modelID,
        ]
        let json = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: url.appendingPathComponent("Manifest.json"))
    }

    private static func feature(_ feature: Feature) -> ProtobufWriter {
        var description = ProtobufWriter()
        description.string(1, feature.name)
        description.message(3) { type in
            type.message(5) { array in
                array.packedInts(1, feature.shape)
                array.int(2, feature.type.arrayCode)
            }
        }
        return description
    }
}
