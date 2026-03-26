// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PrivateAgentBenchmarks",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "EndToEndBench",
            dependencies: [
                .product(name: "FlashMoEBridge", package: "PrivateAgent"),
                .product(name: "ModelPack", package: "PrivateAgent"),
            ],
            path: "."
        ),
    ]
)
