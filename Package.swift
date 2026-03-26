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

        // ── Vendored flash-moe engine (unity build) ──
        .target(
            name: "FlashMoEVendor",
            dependencies: ["FlashMoECore"],
            path: "Vendor/flash-moe",
            sources: ["FlashMoEEngine.m"],
            publicHeadersPath: "public",
            cSettings: [
                .headerSearchPath("."),
                .define("CHAT_MODE", to: "1"),
                .define("ACCELERATE_NEW_LAPACK"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedLibrary("compression"),
            ]
        ),

        // ── Runtime ──
        .target(
            name: "FlashMoERuntime",
            dependencies: ["FlashMoECore", "FlashMoEMetal", "TurboQuantCore", "TurboQuantMetal", "FlashMoEVendor"],
            path: "Sources/FlashMoERuntime",
            publicHeadersPath: "include",
            cSettings: [
                .define("PA_USE_REAL_ENGINE"),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedLibrary("compression"),
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
        .testTarget(
            name: "FlashMoEBridgeTests",
            dependencies: ["FlashMoEBridge", "ModelPack"],
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "ModelHubTests",
            dependencies: ["ModelHub"]
        ),
    ]
)
