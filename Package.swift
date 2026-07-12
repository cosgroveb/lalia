// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Lalia",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "Lalia", targets: ["Lalia"]),
        .executable(name: "LaliaSpeechSmoke", targets: ["LaliaSpeechSmoke"]),
    ],
    targets: [
        .executableTarget(name: "Lalia"),
        .executableTarget(name: "LaliaSpeechSmoke"),
        .testTarget(name: "LaliaTests", dependencies: ["Lalia"]),
    ]
)
