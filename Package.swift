// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PrivateAgent",
    platforms: [
        .iOS(.v18),
        .macOS(.v15)
    ],
    products: [
        .library(name: "FlashMoECore", targets: ["FlashMoECore"]),
        .library(name: "FlashMoEMetal", targets: ["FlashMoEMetal"]),
        .library(name: "TurboQuantCore", targets: ["TurboQuantCore"]),
        .library(name: "TurboQuantMetal", targets: ["TurboQuantMetal"]),
        .library(name: "FlashMoERuntime", targets: ["FlashMoERuntime"]),
        .library(name: "ModelPack", targets: ["ModelPack"]),
        .library(name: "ModelHub", targets: ["ModelHub"]),
        .library(name: "FlashMoEBridge", targets: ["FlashMoEBridge"]),
        .library(name: "PrivateAgentUI", targets: ["PrivateAgentUI"]),
    ],
    targets: [
        // ── C ABI Foundation ──
        .target(
            name: "FlashMoECore",
            path: "Sources/FlashMoECore",
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include"),
                .define("PA_VERSION_MAJOR", to: "0"),
                .define("PA_VERSION_MINOR", to: "1"),
            ]
        ),

        // ── Metal Compute ──
        .target(
            name: "FlashMoEMetal",
            dependencies: ["FlashMoECore"],
            path: "Sources/FlashMoEMetal",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
            ]
        ),
        .target(
            name: "TurboQuantCore",
            dependencies: ["FlashMoECore"],
            path: "Sources/TurboQuantCore",
            publicHeadersPath: "include"
        ),
        .target(
            name: "TurboQuantMetal",
            dependencies: ["TurboQuantCore", "FlashMoECore"],
            path: "Sources/TurboQuantMetal",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Metal"),
            ]
        ),

        // ── Runtime ──
        .target(
            name: "FlashMoERuntime",
            dependencies: ["FlashMoECore", "FlashMoEMetal", "TurboQuantCore", "TurboQuantMetal"],
            path: "Sources/FlashMoERuntime",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
            ]
        ),

        // ── Swift Layers ──
        .target(
            name: "ModelPack",
            dependencies: ["FlashMoECore"],
            path: "Sources/ModelPack"
        ),
        .target(
            name: "ModelHub",
            dependencies: ["ModelPack"],
            path: "Sources/ModelHub"
        ),
        .target(
            name: "FlashMoEBridge",
            dependencies: ["FlashMoERuntime", "ModelPack"],
            path: "Sources/FlashMoEBridge"
        ),
        .target(
            name: "PrivateAgentUI",
            dependencies: ["FlashMoEBridge", "ModelHub"],
            path: "Sources/PrivateAgentUI"
        ),

        // ── Tests ──
        .testTarget(
            name: "FlashMoECoreTests",
            dependencies: ["FlashMoECore"]
        ),
        .testTarget(
            name: "TurboQuantCoreTests",
            dependencies: ["TurboQuantCore"]
        ),
        .testTarget(
            name: "ModelPackTests",
            dependencies: ["ModelPack"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
