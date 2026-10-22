// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Lalia",
    platforms: [.macOS(.v26)],
    products: [
        .executable(name: "Lalia", targets: ["Lalia"]),
    ],
    targets: [
        .executableTarget(name: "Lalia"),
        .testTarget(name: "LaliaTests", dependencies: ["Lalia"]),
    ]
)
