# PrivateAgent Plan 1: Package Scaffold + C ABI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the SPM package structure with all targets, define the stable C ABI structs, and verify the build compiles on iOS.

**Architecture:** Single `Package.swift` with 9 library targets + 2 test targets. The iOS app target lives in a lightweight `PrivateAgent.xcodeproj` that imports the SPM package (SPM cannot produce `.app` bundles with `Info.plist` for iOS directly). C ABI defined in `FlashMoECore/include/` as the stable interface between all modules. No inference logic yet — just the skeleton, headers, and build verification.

**Tech Stack:** Swift 6.0, C17, Metal 3, SPM, iOS 18.0+

**Spec:** `docs/superpowers/specs/2026-03-26-privateagent-design.md`

**Depends on:** Nothing (this is the foundation)

**Produces:** Compilable SPM package with all target boundaries defined, C ABI headers, and a passing `swift build` on macOS. iOS simulator build via `xcodebuild` is a stretch goal for this plan (requires Xcode project setup which may be deferred to Plan 2).

**Note on iOS app target:** SPM cannot produce `.app` bundles for iOS. The iOS app will use a minimal Xcode project that depends on the SPM package. This plan focuses on the SPM package; the Xcode project is added in Task 6.

---

## File Structure

```
PrivateAgent/
├── Package.swift
├── Apps/
│   └── PrivateAgentiOS/
│       ├── PrivateAgentApp.swift
│       └── Info.plist
├── Sources/
│   ├── FlashMoECore/
│   │   ├── include/
│   │   │   ├── pa_types.h              # PA_TensorRef, PA_QuantizedKVDesc, PA_ModelDesc
│   │   │   ├── pa_status.h             # PA_Status enum
│   │   │   └── FlashMoECore.h          # umbrella header
│   │   ├── pa_status.c                  # status string implementation
│   │   └── pa_types.c                  # validation helpers
│   ├── FlashMoEMetal/
│   │   ├── include/
│   │   │   └── FlashMoEMetal.h
│   │   └── FlashMoEMetal_stub.m        # placeholder ObjC file for Metal target
│   ├── TurboQuantCore/
│   │   ├── include/
│   │   │   └── TurboQuantCore.h
│   │   └── tq_types.c                  # placeholder
│   ├── TurboQuantMetal/
│   │   ├── include/
│   │   │   └── TurboQuantMetal.h
│   │   └── TurboQuantMetal_stub.m      # placeholder
│   ├── FlashMoERuntime/
│   │   ├── include/
│   │   │   └── FlashMoERuntime.h       # session, loader, pager API
│   │   └── FlashMoERuntime_stub.c      # placeholder
│   ├── ModelPack/
│   │   ├── ModelManifest.swift
│   │   └── PAModelDescBridge.swift     # Swift → PA_ModelDesc conversion
│   ├── ModelHub/
│   │   └── ModelHub_stub.swift         # placeholder
│   ├── FlashMoEBridge/
│   │   └── PrivateAgentEngine.swift    # @MainActor @Observable skeleton
│   └── PrivateAgentUI/
│       └── ContentView.swift           # minimal "Hello PrivateAgent" view
├── Tests/
│   ├── FlashMoECoreTests/
│   │   └── PATypesTests.swift          # C struct layout tests
│   └── TurboQuantCoreTests/
│       └── TQTypesTests.swift          # TQ descriptor tests
└── Benchmarks/
    └── .gitkeep
```

---

### Task 1: Initialize git repo + Package.swift

**Files:**
- Create: `PrivateAgent/Package.swift`
- Create: `PrivateAgent/.gitignore`
- Create: `PrivateAgent/README.md`

- [ ] **Step 1: Create project directory and git repo**

```bash
mkdir -p ~/Developer/PrivateAgent
cd ~/Developer/PrivateAgent
git init
```

- [ ] **Step 2: Create .gitignore**

```gitignore
# Xcode
*.xcodeproj/
xcuserdata/
DerivedData/
build/
*.pbxuser
*.mode1v3
*.mode2v3
*.perspectivev3
*.xcuserstate

# SPM
.build/
.swiftpm/
Package.resolved

# OS
.DS_Store
Thumbs.db

# Models (downloaded, not committed)
Models/
*.bin
*.safetensors
*.gguf
```

- [ ] **Step 3: Create Package.swift with all targets**

```swift
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
    ]
)
```

- [ ] **Step 4: Create minimal README.md**

```markdown
# PrivateAgent

Offline LLM chat for iPhone. Runs Qwen3.5-35B-A3B on-device via Flash-MoE (SSD expert streaming) + TurboQuant (KV cache compression).

## Build

```bash
swift build
```

## Status

Early development. See `docs/superpowers/specs/` for the design spec.
```

- [ ] **Step 5: Commit**

```bash
git add Package.swift .gitignore README.md
git commit -m "feat: initialize SPM package with target graph"
```

---

### Task 2: Define PA_TensorRef + PA_Status + PA_QuantizedKVDesc

**Files:**
- Create: `Sources/FlashMoECore/include/pa_status.h`
- Create: `Sources/FlashMoECore/include/pa_types.h`
- Create: `Sources/FlashMoECore/include/FlashMoECore.h`
- Create: `Sources/FlashMoECore/pa_types.c`

- [ ] **Step 1: Write pa_status.h**

```c
#ifndef PA_STATUS_H
#define PA_STATUS_H

typedef enum {
    PA_STATUS_OK = 0,
    PA_STATUS_ERROR_GENERIC = -1,
    PA_STATUS_ERROR_OOM = -2,
    PA_STATUS_ERROR_IO = -3,
    PA_STATUS_ERROR_INVALID_MODEL = -4,
    PA_STATUS_ERROR_METAL_INIT = -5,
    PA_STATUS_ERROR_LOAD_FAILED = -6,
    PA_STATUS_CONTEXT_EXHAUSTED = -10,
    PA_STATUS_CANCELLED = -11,
    PA_STATUS_THROTTLED = -12,
} PA_Status;

/// Human-readable string for a status code. Never returns NULL.
const char *pa_status_string(PA_Status status);

#endif // PA_STATUS_H
```

- [ ] **Step 2: Write pa_types.h**

```c
#ifndef PA_TYPES_H
#define PA_TYPES_H

#include <stdint.h>
#include <stddef.h>

// ── Storage kinds ──
typedef enum {
    PA_STORAGE_CPU = 0,
    PA_STORAGE_METAL_BUFFER = 1,
    PA_STORAGE_QUANTIZED_KV = 2,
} PA_StorageKind;

// ── Quantization schemes ──
typedef enum {
    PA_QUANT_NONE = 0,
    PA_QUANT_Q2 = 2,
    PA_QUANT_Q3 = 3,
    PA_QUANT_Q4 = 4,
    PA_QUANT_TQ3 = 103,    // TurboQuant 3-bit
    PA_QUANT_TQ3_5 = 107,  // TurboQuant 3.5-bit (custom)
    PA_QUANT_TQ4 = 104,    // TurboQuant 4-bit
} PA_QuantScheme;

// ── Tensor reference ──
typedef struct {
    void *data;
    uint64_t byte_offset;
    uint32_t dtype;          // 0=f32, 1=f16, 2=bf16, 3=i32, 4=u8
    uint32_t rank;
    uint32_t shape[4];
    uint32_t stride[4];
    uint32_t storage_kind;   // PA_StorageKind
    uint32_t quant_scheme;   // PA_QuantScheme
} PA_TensorRef;

// ── TurboQuant KV descriptor ──
typedef enum {
    PA_AUX_NONE = 0,
    PA_AUX_GLOBAL = 1,
    PA_AUX_PER_BLOCK = 2,
} PA_AuxParamsKind;

typedef enum {
    PA_TRANSFORM_NONE = 0,
    PA_TRANSFORM_STRUCTURED_ROTATION = 1,
    PA_TRANSFORM_HADAMARD = 2,
} PA_TransformKind;

typedef struct {
    uint16_t key_bits_x2;        // bits * 2: 7 = 3.5 bits, 6 = 3 bits, 8 = 4 bits
    uint16_t value_bits_x2;      // separate from key_bits; same encoding
    uint32_t block_size;
    uint32_t transform_kind;     // PA_TransformKind
    uint64_t transform_seed;     // deterministic, reproducible
    uint32_t residual_bits;      // QJL bits (typically 1)
    uint64_t main_codes_offset;  // byte offset into KV buffer
    uint64_t aux_params_offset;  // 0 if aux_params_kind == PA_AUX_NONE
    uint64_t qjl_bits_offset;
    uint32_t aux_params_kind;    // PA_AuxParamsKind
} PA_QuantizedKVDesc;

// ── Layer type ──
typedef enum {
    PA_LAYER_GDN = 0,           // GatedDeltaNet (linear attention, O(1) state)
    PA_LAYER_FULL_ATTN = 1,     // Full attention (needs KV cache)
} PA_LayerType;

// ── Model descriptor (C ABI, populated by Swift ModelPack) ──
#define PA_MAX_LAYERS 128
#define PA_MAX_PATH 1024

typedef struct {
    // Paths
    char model_dir[PA_MAX_PATH];         // root model directory
    char weights_path[PA_MAX_PATH];      // model_weights.bin
    char tokenizer_path[PA_MAX_PATH];    // tokenizer.bin

    // Architecture
    uint32_t num_layers;
    uint32_t num_experts;
    uint32_t active_experts_k;           // top-K active per token
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t num_attn_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t moe_intermediate;
    uint32_t max_position_embeddings;
    float rms_norm_eps;
    PA_LayerType layer_types[PA_MAX_LAYERS];

    // Expert layout
    uint32_t expert_quant_bits;          // 2, 3, or 4
    uint32_t dense_quant_bits;
    uint64_t expert_size_each;           // bytes per expert shard

    // TurboQuant defaults
    uint16_t default_key_bits_x2;        // 0 = TQ disabled
    uint16_t default_value_bits_x2;
    uint32_t default_tq_block_size;
    uint32_t default_transform_kind;
    uint64_t default_transform_seed;

    // Manifest version
    uint32_t manifest_version;           // 1 for v1
} PA_ModelDesc;

/// Validate a PA_ModelDesc. Returns PA_STATUS_OK if valid.
int pa_model_desc_validate(const PA_ModelDesc *desc);

/// Validate a PA_TensorRef. Returns PA_STATUS_OK if valid.
int pa_tensor_ref_validate(const PA_TensorRef *ref);

/// Count full-attention layers in a PA_ModelDesc.
uint32_t pa_model_desc_full_attn_count(const PA_ModelDesc *desc);

/// Count GDN layers in a PA_ModelDesc.
uint32_t pa_model_desc_gdn_count(const PA_ModelDesc *desc);

/// Get effective bits as float from x2 encoding. E.g. 7 → 3.5f
static inline float pa_bits_from_x2(uint16_t bits_x2) {
    return (float)bits_x2 / 2.0f;
}

#endif // PA_TYPES_H
```

- [ ] **Step 3: Write FlashMoECore.h umbrella**

```c
#ifndef FLASHMOE_CORE_H
#define FLASHMOE_CORE_H

#include "pa_status.h"
#include "pa_types.h"

#endif // FLASHMOE_CORE_H
```

- [ ] **Step 4: Write pa_status.c**

```c
#include "pa_status.h"

const char *pa_status_string(PA_Status status) {
    switch (status) {
        case PA_STATUS_OK:                  return "OK";
        case PA_STATUS_ERROR_GENERIC:       return "Generic error";
        case PA_STATUS_ERROR_OOM:           return "Out of memory";
        case PA_STATUS_ERROR_IO:            return "I/O error";
        case PA_STATUS_ERROR_INVALID_MODEL: return "Invalid model";
        case PA_STATUS_ERROR_METAL_INIT:    return "Metal initialization failed";
        case PA_STATUS_ERROR_LOAD_FAILED:   return "Model load failed";
        case PA_STATUS_CONTEXT_EXHAUSTED:   return "Context exhausted";
        case PA_STATUS_CANCELLED:           return "Cancelled";
        case PA_STATUS_THROTTLED:           return "Throttled";
        default:                            return "Unknown status";
    }
}
```

- [ ] **Step 5: Write pa_types.c implementation**

```c
#include "pa_types.h"
#include "pa_status.h"
#include <string.h>

int pa_model_desc_validate(const PA_ModelDesc *desc) {
    if (!desc) return PA_STATUS_ERROR_GENERIC;
    if (desc->num_layers == 0 || desc->num_layers > PA_MAX_LAYERS)
        return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->hidden_dim == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->vocab_size == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->num_kv_heads == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->head_dim == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->model_dir[0] == '\0') return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->manifest_version == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    return PA_STATUS_OK;
}

int pa_tensor_ref_validate(const PA_TensorRef *ref) {
    if (!ref) return PA_STATUS_ERROR_GENERIC;
    if (ref->rank == 0 || ref->rank > 4) return PA_STATUS_ERROR_GENERIC;
    for (uint32_t i = 0; i < ref->rank; i++) {
        if (ref->shape[i] == 0) return PA_STATUS_ERROR_GENERIC;
    }
    return PA_STATUS_OK;
}

uint32_t pa_model_desc_full_attn_count(const PA_ModelDesc *desc) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < desc->num_layers; i++) {
        if (desc->layer_types[i] == PA_LAYER_FULL_ATTN) count++;
    }
    return count;
}

uint32_t pa_model_desc_gdn_count(const PA_ModelDesc *desc) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < desc->num_layers; i++) {
        if (desc->layer_types[i] == PA_LAYER_GDN) count++;
    }
    return count;
}
```

- [ ] **Step 6: Commit**

```bash
git add Sources/FlashMoECore/
git commit -m "feat: define C ABI — PA_TensorRef, PA_QuantizedKVDesc, PA_ModelDesc, PA_Status"
```

---

### Task 3: Create all stub source files for remaining targets

All targets need source files before tests can compile. Create stubs first, tests next.

**Files:**
- Create: `Sources/FlashMoEMetal/include/FlashMoEMetal.h`
- Create: `Sources/FlashMoEMetal/FlashMoEMetal_stub.m`
- Create: `Sources/TurboQuantCore/include/TurboQuantCore.h`
- Create: `Sources/TurboQuantCore/tq_types.c`
- Create: `Sources/TurboQuantMetal/include/TurboQuantMetal.h`
- Create: `Sources/TurboQuantMetal/TurboQuantMetal_stub.m`
- Create: `Sources/FlashMoERuntime/include/FlashMoERuntime.h`
- Create: `Sources/FlashMoERuntime/FlashMoERuntime_stub.c`
- Create: `Sources/ModelPack/ModelManifest.swift`
- Create: `Sources/ModelPack/PAModelDescBridge.swift`
- Create: `Sources/ModelHub/ModelHub_stub.swift`
- Create: `Sources/FlashMoEBridge/PrivateAgentEngine.swift`
- Create: `Sources/PrivateAgentUI/ContentView.swift`
- Create: `Benchmarks/.gitkeep`

- [ ] **Step 1: FlashMoEMetal stubs**

`Sources/FlashMoEMetal/include/FlashMoEMetal.h`:
```c
#ifndef FLASHMOE_METAL_H
#define FLASHMOE_METAL_H

#include "FlashMoECore.h"

// Dense-path GPU kernel encoder builders.
// Implementations will be added in Plan 3.

#endif // FLASHMOE_METAL_H
```

`Sources/FlashMoEMetal/FlashMoEMetal_stub.m`:
```objc
#import <Foundation/Foundation.h>
// FlashMoEMetal placeholder. Metal kernels added in Plan 3.
```

- [ ] **Step 2: TurboQuantCore stubs**

`Sources/TurboQuantCore/include/TurboQuantCore.h`:
```c
#ifndef TURBOQUANT_CORE_H
#define TURBOQUANT_CORE_H

#include "FlashMoECore.h"

// TurboQuant KV cache compression — CPU reference implementations.
// Added in Plan 5.

#endif // TURBOQUANT_CORE_H
```

`Sources/TurboQuantCore/tq_types.c`:
```c
#include "TurboQuantCore.h"
// TurboQuantCore placeholder. Reference kernels added in Plan 5.
```

- [ ] **Step 3: TurboQuantMetal stubs**

`Sources/TurboQuantMetal/include/TurboQuantMetal.h`:
```c
#ifndef TURBOQUANT_METAL_H
#define TURBOQUANT_METAL_H

#include "TurboQuantCore.h"

// TurboQuant Metal compute kernels for compressed-domain attention.
// Added in Plan 5.

#endif // TURBOQUANT_METAL_H
```

`Sources/TurboQuantMetal/TurboQuantMetal_stub.m`:
```objc
#import <Foundation/Foundation.h>
// TurboQuantMetal placeholder. Metal kernels added in Plan 5.
```

- [ ] **Step 4: FlashMoERuntime stubs**

`Sources/FlashMoERuntime/include/FlashMoERuntime.h`:
```c
#ifndef FLASHMOE_RUNTIME_H
#define FLASHMOE_RUNTIME_H

#include "FlashMoECore.h"

// Session state machine, memory planner, expert pager, layer scheduler.
// Implementations added in Plan 3.

/// Opaque runtime session handle.
typedef struct PA_Session PA_Session;

/// Session states.
typedef enum {
    PA_SESSION_IDLE = 0,
    PA_SESSION_LOADING = 1,
    PA_SESSION_PREFILL = 2,
    PA_SESSION_DECODE = 3,
    PA_SESSION_DONE = 4,
    PA_SESSION_CANCELLED = 5,
    PA_SESSION_THROTTLED = 6,
    PA_SESSION_RECOVERING_MEMORY = 7,
} PA_SessionState;

#endif // FLASHMOE_RUNTIME_H
```

`Sources/FlashMoERuntime/FlashMoERuntime_stub.c`:
```c
#include "FlashMoERuntime.h"
// FlashMoERuntime placeholder. Session logic added in Plan 3.
```

- [ ] **Step 5: ModelPack Swift stubs**

`Sources/ModelPack/ModelManifest.swift`:
```swift
import Foundation

/// Parsed model manifest from config.json + privateagent-manifest.json.
/// Responsible for reading model metadata and producing a PA_ModelDesc
/// for the C runtime.
public struct ModelManifest: Sendable {
    public let modelDir: URL
    public let manifestVersion: UInt32

    // Placeholder — full parsing added in Plan 2.
    public init(modelDir: URL) {
        self.modelDir = modelDir
        self.manifestVersion = 1
    }
}
```

`Sources/ModelPack/PAModelDescBridge.swift`:
```swift
import Foundation
import FlashMoECore

/// Converts a Swift ModelManifest into a C PA_ModelDesc struct
/// for passing into the FlashMoERuntime.
public enum PAModelDescBridge {

    /// Build a PA_ModelDesc from a ModelManifest.
    /// Full implementation in Plan 2.
    public static func makeModelDesc(from manifest: ModelManifest) -> PA_ModelDesc {
        var desc = PA_ModelDesc()
        desc.manifest_version = manifest.manifestVersion
        return desc
    }
}
```

- [ ] **Step 6: ModelHub stub**

`Sources/ModelHub/ModelHub_stub.swift`:
```swift
import Foundation

/// Model download + local storage management.
/// Full implementation in Plan 4.
public enum ModelHub {
    public static let version = "0.1.0"
}
```

- [ ] **Step 7: FlashMoEBridge skeleton**

Note: `@Observable` + `@MainActor` is intentional per spec. All callers must be `@MainActor`,
which is the intended design since only SwiftUI views consume this type.

`Sources/FlashMoEBridge/PrivateAgentEngine.swift`:
```swift
import Foundation
import Observation
import FlashMoECore
import FlashMoERuntime
import ModelPack

/// Engine states exposed to UI.
public enum EngineState: Sendable, Equatable {
    case idle
    case loading
    case ready
    case generating
    case cancelled
    case throttled(String)
    case recoveringMemory
    case error(String)
}

/// Placeholder for generation statistics snapshot.
public struct GenerationStats: Sendable {
    public let tokensPerSecond: Double
    public let tokensGenerated: Int
    public let ttftMs: Double

    public init(tokensPerSecond: Double = 0, tokensGenerated: Int = 0, ttftMs: Double = 0) {
        self.tokensPerSecond = tokensPerSecond
        self.tokensGenerated = tokensGenerated
        self.ttftMs = ttftMs
    }
}

/// Events streamed during generation.
/// Terminal errors go through AsyncThrowingStream throw, NOT via an .error case.
public enum GenerationEvent: Sendable {
    case prefillProgress(tokens: Int, total: Int)
    case token(text: String, id: Int)
    case thinkingStart
    case thinkingEnd
    case contextExhausted(policy: String)   // policy decided by Runtime
    case throttled(reason: String)
    case finished(stats: GenerationStats)
}

/// Prompt input variants.
public enum PromptInput: Sendable {
    case formattedPrompt(String)
    case tokenIDs([Int32])
}

/// Main engine facade — @MainActor for SwiftUI binding.
/// Full implementation in Plan 2.
@MainActor
@Observable
public final class PrivateAgentEngine {
    public private(set) var state: EngineState = .idle
    public private(set) var modelInfo: String?

    public init() {}
}
```

- [ ] **Step 8: PrivateAgentUI minimal view**

`Sources/PrivateAgentUI/ContentView.swift`:
```swift
import SwiftUI
import FlashMoEBridge

public struct ContentView: View {
    @State private var engine = PrivateAgentEngine()

    public init() {}

    public var body: some View {
        NavigationStack {
            VStack {
                Text("PrivateAgent")
                    .font(.largeTitle)
                Text("Engine: \(engine.state == .idle ? "Ready" : "...")")
                    .foregroundStyle(.secondary)
            }
            .navigationTitle("PrivateAgent")
        }
    }
}
```

- [ ] **Step 9: Create Benchmarks/.gitkeep**

```bash
mkdir -p Benchmarks
touch Benchmarks/.gitkeep
```

- [ ] **Step 10: Commit all stubs**

```bash
git add Sources/ Benchmarks/
git commit -m "feat: add stub source files for all 9 targets"
```

---

### Task 4: Write C ABI tests

All stub targets now exist, so tests can compile and run.

**Files:**
- Create: `Tests/FlashMoECoreTests/PATypesTests.swift`
- Create: `Tests/TurboQuantCoreTests/TQTypesTests.swift`

- [ ] **Step 1: Write PATypesTests.swift**

```swift
import Testing
@testable import FlashMoECore

@Suite("PA_Types C ABI Tests")
struct PATypesTests {

    @Test("PA_TensorRef has expected size and layout")
    func tensorRefLayout() {
        #expect(MemoryLayout<PA_TensorRef>.size > 0)
        // Verify we can create and validate
        var ref = PA_TensorRef()
        ref.rank = 2
        ref.shape.0 = 4
        ref.shape.1 = 8
        ref.dtype = 0  // f32
        ref.storage_kind = UInt32(PA_STORAGE_CPU.rawValue)
        ref.quant_scheme = UInt32(PA_QUANT_NONE.rawValue)
        #expect(pa_tensor_ref_validate(&ref) == PA_STATUS_OK.rawValue)
    }

    @Test("PA_TensorRef validation rejects rank 0")
    func tensorRefRejectsRank0() {
        var ref = PA_TensorRef()
        ref.rank = 0
        #expect(pa_tensor_ref_validate(&ref) != PA_STATUS_OK.rawValue)
    }

    @Test("PA_TensorRef validation rejects rank > 4")
    func tensorRefRejectsRank5() {
        var ref = PA_TensorRef()
        ref.rank = 5
        #expect(pa_tensor_ref_validate(&ref) != PA_STATUS_OK.rawValue)
    }

    @Test("PA_ModelDesc validation passes for valid desc")
    func modelDescValid() {
        var desc = PA_ModelDesc()
        withUnsafeMutablePointer(to: &desc.model_dir) { ptr in
            let raw = UnsafeMutableRawPointer(ptr)
            let bound = raw.bindMemory(to: CChar.self, capacity: Int(PA_MAX_PATH))
            "/tmp/model".withCString { src in
                strcpy(bound, src)
            }
        }
        desc.num_layers = 60
        desc.hidden_dim = 2048
        desc.vocab_size = 151936
        desc.num_kv_heads = 2
        desc.head_dim = 256
        desc.manifest_version = 1
        #expect(pa_model_desc_validate(&desc) == PA_STATUS_OK.rawValue)
    }

    @Test("PA_ModelDesc validation rejects 0 layers")
    func modelDescRejectsZeroLayers() {
        var desc = PA_ModelDesc()
        desc.num_layers = 0
        #expect(pa_model_desc_validate(&desc) != PA_STATUS_OK.rawValue)
    }

    @Test("PA_ModelDesc validation rejects > PA_MAX_LAYERS")
    func modelDescRejectsTooManyLayers() {
        var desc = PA_ModelDesc()
        desc.num_layers = UInt32(PA_MAX_LAYERS) + 1
        #expect(pa_model_desc_validate(&desc) != PA_STATUS_OK.rawValue)
    }

    @Test("pa_model_desc_full_attn_count counts correctly")
    func fullAttnCount() {
        var desc = PA_ModelDesc()
        desc.num_layers = 5
        // layers 0,1,2 = GDN, 3,4 = full_attn
        desc.layer_types.0 = PA_LAYER_GDN
        desc.layer_types.1 = PA_LAYER_GDN
        desc.layer_types.2 = PA_LAYER_GDN
        desc.layer_types.3 = PA_LAYER_FULL_ATTN
        desc.layer_types.4 = PA_LAYER_FULL_ATTN
        #expect(pa_model_desc_full_attn_count(&desc) == 2)
        #expect(pa_model_desc_gdn_count(&desc) == 3)
    }

    @Test("pa_bits_from_x2 converts correctly")
    func bitsFromX2() {
        #expect(pa_bits_from_x2(7) == 3.5)
        #expect(pa_bits_from_x2(6) == 3.0)
        #expect(pa_bits_from_x2(8) == 4.0)
    }

    @Test("PA_QuantizedKVDesc key/value bits encoding")
    func kvDescBitsEncoding() {
        var kv = PA_QuantizedKVDesc()
        kv.key_bits_x2 = 7    // 3.5 bits
        kv.value_bits_x2 = 8  // 4.0 bits
        #expect(pa_bits_from_x2(kv.key_bits_x2) == 3.5)
        #expect(pa_bits_from_x2(kv.value_bits_x2) == 4.0)
    }

    @Test("pa_status_string returns non-null for all codes")
    func statusStrings() {
        let codes: [PA_Status] = [
            PA_STATUS_OK, PA_STATUS_ERROR_GENERIC, PA_STATUS_ERROR_OOM,
            PA_STATUS_ERROR_IO, PA_STATUS_ERROR_INVALID_MODEL,
            PA_STATUS_ERROR_METAL_INIT, PA_STATUS_ERROR_LOAD_FAILED,
            PA_STATUS_CONTEXT_EXHAUSTED, PA_STATUS_CANCELLED, PA_STATUS_THROTTLED,
        ]
        for code in codes {
            let str = pa_status_string(code)
            #expect(str != nil)
            #expect(String(cString: str!) != "Unknown status")
        }
    }
}
```

- [ ] **Step 2: Write TQTypesTests.swift**

```swift
import Testing
@testable import TurboQuantCore

@Suite("TurboQuant Types Tests")
struct TQTypesTests {

    @Test("TurboQuantCore target compiles and links FlashMoECore")
    func targetCompiles() {
        // If this test runs, the target graph is correct
        #expect(true)
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cd ~/Developer/PrivateAgent
swift test 2>&1 | tail -10
```

Expected: All tests PASS (FlashMoECore from Task 2, all other stubs from Task 3).

- [ ] **Step 4: Commit test files**

```bash
git add Tests/
git commit -m "test: add C ABI struct layout and validation tests"
```

---


### Task 5: Verify full build + run tests (all stubs created in Task 3)

**Files:** None (verification only)

- [ ] **Step 1: Build all targets for macOS**

```bash
cd ~/Developer/PrivateAgent
swift build 2>&1 | tail -5
```

Expected: `Build complete!`

- [ ] **Step 2: Run all tests**

```bash
swift test 2>&1 | tail -20
```

Expected: All tests pass (PATypesTests + TQTypesTests).

- [ ] **Step 3: Build for iOS simulator**

```bash
# Discover available schemes (SPM generates per-target, not per-package):
xcodebuild -list 2>&1 | head -20
# Build a library target for iOS simulator:
xcodebuild build \
    -scheme FlashMoECore \
    -destination 'platform=iOS Simulator,name=iPhone 16 Pro' \
    -quiet 2>&1 | tail -5
```

Note: If direct build fails, open `Package.swift` in Xcode first to let it generate schemes, then retry.

- [ ] **Step 4: Verify target dependency graph**

```bash
swift package show-dependencies --format json 2>&1 | head -30
```

Expected: All 9 targets visible with correct dependency edges.

- [ ] **Step 5: Commit any build fixes**

```bash
git add -A
git status
# Only commit if there are fixes needed
git commit -m "fix: resolve build issues from initial scaffold" --allow-empty
```

---

### Task 6: Copy spec + create plan index

**Files:**
- Copy: `docs/superpowers/specs/2026-03-26-privateagent-design.md` into repo
- Create: `docs/superpowers/plans/README.md`

- [ ] **Step 1: Copy spec into repo**

```bash
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp ~/docs/superpowers/specs/2026-03-26-privateagent-design.md docs/superpowers/specs/
```

- [ ] **Step 2: Create plan index**

`docs/superpowers/plans/README.md`:
```markdown
# Implementation Plans

| # | Plan | Status | Depends On |
|---|------|--------|------------|
| 1 | Package scaffold + C ABI | In progress | — |
| 2 | ModelPack → Bridge → Runtime load chain | Pending | Plan 1 |
| 3 | FlashMoERuntime skeleton + benchmarks | Pending | Plan 2 |
| 4 | ModelHub background download | Pending | Plan 2 |
| 5 | TurboQuant CPU reference → Metal | Pending | Plan 3 |
```

- [ ] **Step 3: Commit docs**

```bash
git add docs/
git commit -m "docs: add design spec and implementation plan index"
```

---

## Summary

After completing Plan 1, the repo will have:

- Compilable SPM package with 9 targets and correct dependency graph
- Stable C ABI headers: `PA_TensorRef`, `PA_QuantizedKVDesc`, `PA_ModelDesc`, `PA_Status`
- Passing unit tests for all C struct validation
- Stub files for every target so downstream plans can start immediately
- Design spec and plan index committed

**Next:** Plan 2 (ModelPack → Bridge → Runtime load chain) builds on this scaffold to implement model loading.
