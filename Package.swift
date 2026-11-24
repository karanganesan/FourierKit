// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "FourierKit",
    platforms: [.macOS(.v26), .iOS(.v26)],
    products: [
        .library(name: "FourierKit", targets: ["FourierKit"]),
        .library(name: "FourierKitUI", targets: ["FourierKitUI"]),
    ],
    targets: [
        .target(
            name: "FourierKit",
            path: "Sources/FourierKit"
        ),
        .target(
            name: "FourierKitUI",
            dependencies: ["FourierKit"],
            path: "Sources/FourierKitUI",
            resources: [.copy("Resources/Fonts")]
        ),
        .testTarget(
            name: "FourierKitTests",
            dependencies: ["FourierKit"],
            path: "Tests/FourierKitTests"
        ),
    ],
    swiftLanguageModes: [.v6]
)
