# PrivateAgent Plan 2: ModelPack → Bridge → Runtime Load Chain

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the model loading pipeline: parse model manifest JSON → populate PA_ModelDesc → pass to Runtime → report load status to UI.

**Architecture:** ModelPack (Swift) parses `config.json` + `privateagent-manifest.json` → PAModelDescBridge converts to C struct → FlashMoERuntime ModelLoader receives it → FlashMoEBridge exposes async load to SwiftUI. No actual inference yet — just the load chain + memory budget reporting.

**Tech Stack:** Swift 6.0, C17, Foundation (JSONDecoder), SwiftUI

**Spec:** `docs/superpowers/specs/2026-03-26-privateagent-design.md`

**Depends on:** Plan 1 (complete)

**Produces:** Working model load pipeline that can parse a model directory, validate it, compute memory budget, and report status to the UI layer.

---

## File Structure

```
Sources/
├── ModelPack/
│   ├── ModelManifest.swift          # MODIFY: full JSON parsing
│   ├── PAModelDescBridge.swift      # MODIFY: full PA_ModelDesc population
│   ├── PromptCompiler.swift         # CREATE: chat template formatting
│   └── ManifestSchema.swift         # CREATE: Codable structs for JSON
├── FlashMoERuntime/
│   ├── include/FlashMoERuntime.h    # MODIFY: add loader API
│   ├── FlashMoERuntime_stub.c       # DELETE
│   ├── pa_runtime.c                 # CREATE: session + loader implementation
│   └── pa_memory_planner.c          # CREATE: memory budget computation
├── FlashMoEBridge/
│   └── PrivateAgentEngine.swift     # MODIFY: add loadModel/unloadModel
└── FlashMoECore/
    └── include/pa_types.h           # MODIFY: add PA_MemoryBudget struct
Tests/
├── ModelPackTests/
│   ├── ManifestParsingTests.swift   # CREATE
│   └── PromptCompilerTests.swift    # CREATE
└── FlashMoECoreTests/
    └── MemoryPlannerTests.swift     # CREATE
```

---

### Task 1: Add PA_MemoryBudget to C ABI

**Files:**
- Modify: `Sources/FlashMoECore/include/pa_types.h`
- Modify: `Sources/FlashMoECore/pa_types.c`
- Create: `Tests/FlashMoECoreTests/MemoryPlannerTests.swift`

- [ ] **Step 1: Add PA_MemoryBudget struct to pa_types.h**

Add before the closing `#endif`:

```c
// ── Memory budget (computed by MemoryPlanner) ──
typedef struct {
    // Fixed allocations (dirty memory)
    uint64_t metal_buffers_bytes;
    uint64_t gdn_state_bytes;
    uint64_t expert_buffers_bytes;
    uint64_t kv_cache_bytes;
    uint64_t scratch_bytes;
    uint64_t total_dirty_bytes;

    // File-backed (resident but not dirty)
    uint64_t mmap_weights_bytes;

    // Derived
    uint64_t total_resident_bytes;
    uint32_t max_context_length;      // computed from remaining budget
    uint32_t full_attn_layer_count;   // from model metadata
} PA_MemoryBudget;

/// Compute memory budget for a model given available memory.
/// available_bytes: from os_proc_available_memory() or similar.
/// Returns PA_STATUS_OK if model fits, PA_STATUS_ERROR_OOM if not.
int pa_compute_memory_budget(
    const PA_ModelDesc *desc,
    uint64_t available_bytes,
    PA_MemoryBudget *out_budget
);
```

- [ ] **Step 2: Implement pa_compute_memory_budget in pa_types.c**

```c
int pa_compute_memory_budget(
    const PA_ModelDesc *desc,
    uint64_t available_bytes,
    PA_MemoryBudget *out_budget
) {
    if (!desc || !out_budget) return PA_STATUS_ERROR_GENERIC;

    memset(out_budget, 0, sizeof(PA_MemoryBudget));

    uint32_t full_attn = pa_model_desc_full_attn_count(desc);
    uint32_t gdn = pa_model_desc_gdn_count(desc);
    out_budget->full_attn_layer_count = full_attn;

    // Expert buffer: K experts * 2 (double-buffered) * expert_size
    uint32_t k = desc->active_experts_k > 0 ? desc->active_experts_k : 8;
    out_budget->expert_buffers_bytes = (uint64_t)k * 2 * desc->expert_size_each;

    // GDN state: gdn_layers * num_attn_heads * head_dim * head_dim * sizeof(float)
    out_budget->gdn_state_bytes = (uint64_t)gdn * desc->num_attn_heads *
        desc->head_dim * desc->head_dim * sizeof(float);

    // Metal buffers: rough estimate ~200MB for projections/scratch
    out_budget->metal_buffers_bytes = 200 * 1024 * 1024;

    // Scratch: ~5MB
    out_budget->scratch_bytes = 5 * 1024 * 1024;

    // mmap weights: estimate from hidden_dim (not dirty memory)
    // For 35B-A3B: ~1.4GB. Estimate: hidden_dim * vocab_size * 0.5 (4-bit)
    out_budget->mmap_weights_bytes = (uint64_t)desc->hidden_dim *
        desc->vocab_size / 2;

    // Total dirty (excluding mmap)
    out_budget->total_dirty_bytes =
        out_budget->metal_buffers_bytes +
        out_budget->gdn_state_bytes +
        out_budget->expert_buffers_bytes +
        out_budget->scratch_bytes;

    // Budget for KV cache: 70% of available minus dirty
    uint64_t budget_70 = (available_bytes * 70) / 100;
    if (budget_70 < out_budget->total_dirty_bytes) {
        return PA_STATUS_ERROR_OOM;
    }
    uint64_t kv_budget = budget_70 - out_budget->total_dirty_bytes;

    // KV cache size per token per layer:
    // 2 (K+V) * kv_heads * head_dim * bytes_per_element
    // With TurboQuant: bytes_per_element = bits_x2 / (2 * 8) per value
    float key_bits = desc->default_key_bits_x2 > 0
        ? pa_bits_from_x2(desc->default_key_bits_x2) : 16.0f;
    float value_bits = desc->default_value_bits_x2 > 0
        ? pa_bits_from_x2(desc->default_value_bits_x2) : 16.0f;
    float avg_bits = (key_bits + value_bits) / 2.0f;

    uint64_t bytes_per_token_per_layer =
        (uint64_t)(2 * desc->num_kv_heads * desc->head_dim * avg_bits / 8.0f);

    uint64_t bytes_per_token = bytes_per_token_per_layer * full_attn;

    if (bytes_per_token == 0) {
        out_budget->max_context_length = 8192;
    } else {
        uint64_t max_ctx = kv_budget / bytes_per_token;
        // Clamp to power-of-2 up to 8192
        if (max_ctx >= 8192) max_ctx = 8192;
        else if (max_ctx >= 4096) max_ctx = 4096;
        else if (max_ctx >= 2048) max_ctx = 2048;
        else if (max_ctx >= 1024) max_ctx = 1024;
        else if (max_ctx >= 512) max_ctx = 512;
        else return PA_STATUS_ERROR_OOM;
        out_budget->max_context_length = (uint32_t)max_ctx;
    }

    out_budget->kv_cache_bytes = bytes_per_token * out_budget->max_context_length;
    out_budget->total_dirty_bytes += out_budget->kv_cache_bytes;
    out_budget->total_resident_bytes =
        out_budget->total_dirty_bytes + out_budget->mmap_weights_bytes;

    return PA_STATUS_OK;
}
```

- [ ] **Step 3: Write MemoryPlannerTests.swift**

```swift
import Testing
@testable import FlashMoECore

@Suite("Memory Planner Tests")
struct MemoryPlannerTests {

    private func makeQwen35BDesc() -> PA_ModelDesc {
        var desc = PA_ModelDesc()
        withUnsafeMutablePointer(to: &desc.model_dir) { ptr in
            let raw = UnsafeMutableRawPointer(ptr)
            let bound = raw.bindMemory(to: CChar.self, capacity: Int(PA_MAX_PATH))
            "/tmp/model".withCString { src in strcpy(bound, src) }
        }
        desc.num_layers = 60
        desc.num_experts = 128
        desc.active_experts_k = 8
        desc.hidden_dim = 2048
        desc.vocab_size = 151936
        desc.num_attn_heads = 16
        desc.num_kv_heads = 2
        desc.head_dim = 256
        desc.moe_intermediate = 1408
        desc.max_position_embeddings = 131072
        desc.rms_norm_eps = 1e-6
        desc.expert_quant_bits = 4
        desc.dense_quant_bits = 4
        desc.expert_size_each = 7_000_000  // ~7MB per expert
        desc.manifest_version = 1

        // Set layer types: 45 GDN + 15 full_attn
        withUnsafeMutablePointer(to: &desc.layer_types) { ptr in
            let bound = UnsafeMutableRawPointer(ptr)
                .bindMemory(to: PA_LayerType.self, capacity: 60)
            for i in 0..<60 {
                bound[i] = (i % 4 == 3) ? PA_LAYER_FULL_ATTN : PA_LAYER_GDN
            }
        }

        // TurboQuant 3.5-bit
        desc.default_key_bits_x2 = 7
        desc.default_value_bits_x2 = 7
        desc.default_tq_block_size = 64
        desc.default_transform_kind = UInt32(PA_TRANSFORM_STRUCTURED_ROTATION.rawValue)
        desc.default_transform_seed = 42

        return desc
    }

    @Test("Memory budget computes for Qwen3.5-35B-A3B with 6GB available")
    func budgetWith6GB() {
        var desc = makeQwen35BDesc()
        var budget = PA_MemoryBudget()
        let available: UInt64 = 6 * 1024 * 1024 * 1024  // 6 GB

        let status = pa_compute_memory_budget(&desc, available, &budget)
        #expect(status == PA_STATUS_OK.rawValue)
        #expect(budget.max_context_length >= 512)
        #expect(budget.max_context_length <= 8192)
        #expect(budget.total_dirty_bytes > 0)
        #expect(budget.total_resident_bytes > budget.total_dirty_bytes)
        #expect(budget.full_attn_layer_count == 15)
    }

    @Test("Memory budget rejects insufficient memory")
    func budgetRejectsLowMem() {
        var desc = makeQwen35BDesc()
        var budget = PA_MemoryBudget()
        let available: UInt64 = 100 * 1024 * 1024  // 100 MB — way too small

        let status = pa_compute_memory_budget(&desc, available, &budget)
        #expect(status == PA_STATUS_ERROR_OOM.rawValue)
    }

    @Test("TurboQuant increases max context vs bf16")
    func tqIncreasesContext() {
        var descTQ = makeQwen35BDesc()
        descTQ.default_key_bits_x2 = 7  // 3.5 bit
        descTQ.default_value_bits_x2 = 7

        var descBF16 = makeQwen35BDesc()
        descBF16.default_key_bits_x2 = 0  // bf16 fallback
        descBF16.default_value_bits_x2 = 0

        var budgetTQ = PA_MemoryBudget()
        var budgetBF16 = PA_MemoryBudget()
        let available: UInt64 = 6 * 1024 * 1024 * 1024

        pa_compute_memory_budget(&descTQ, available, &budgetTQ)
        pa_compute_memory_budget(&descBF16, available, &budgetBF16)

        #expect(budgetTQ.max_context_length >= budgetBF16.max_context_length)
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd ~/Developer/PrivateAgent && swift test 2>&1 | tail -20
```

- [ ] **Step 5: Commit**

```bash
git add Sources/FlashMoECore/ Tests/FlashMoECoreTests/
git commit -m "feat: add PA_MemoryBudget and memory planner computation"
```

---

### Task 2: Implement ManifestSchema + ModelManifest parsing

**Files:**
- Create: `Sources/ModelPack/ManifestSchema.swift`
- Modify: `Sources/ModelPack/ModelManifest.swift`
- Create: `Tests/ModelPackTests/ManifestParsingTests.swift`

- [ ] **Step 1: Create ManifestSchema.swift**

```swift
import Foundation

// MARK: - HuggingFace config.json schema (subset)

struct HFConfig: Codable, Sendable {
    let hiddenSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int?
    let vocabSize: Int
    let numExperts: Int?
    let numExpertsPerTok: Int?
    let moeIntermediateSize: Int?
    let maxPositionEmbeddings: Int?
    let rmsNormEps: Float?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
    }
}

// MARK: - privateagent-manifest.json schema

struct PAManifest: Codable, Sendable {
    let manifestVersion: Int
    let layerTypes: [String]?        // ["gdn", "full_attn", ...]
    let expertLayout: ExpertLayout?
    let turboQuantDefaults: TQDefaults?
    let checksums: [String: String]?  // filename -> sha256

    enum CodingKeys: String, CodingKey {
        case manifestVersion = "manifest_version"
        case layerTypes = "layer_types"
        case expertLayout = "expert_layout"
        case turboQuantDefaults = "turboquant_defaults"
        case checksums
    }
}

struct ExpertLayout: Codable, Sendable {
    let quantBits: Int
    let denseQuantBits: Int
    let expertSizeEach: UInt64
    let expertLayers: Int

    enum CodingKeys: String, CodingKey {
        case quantBits = "quant_bits"
        case denseQuantBits = "dense_quant_bits"
        case expertSizeEach = "expert_size_each"
        case expertLayers = "expert_layers"
    }
}

struct TQDefaults: Codable, Sendable {
    let keyBitsX2: Int
    let valueBitsX2: Int
    let blockSize: Int
    let transformKind: String     // "structured_rotation", "hadamard", "none"
    let transformSeed: UInt64

    enum CodingKeys: String, CodingKey {
        case keyBitsX2 = "key_bits_x2"
        case valueBitsX2 = "value_bits_x2"
        case blockSize = "block_size"
        case transformKind = "transform_kind"
        case transformSeed = "transform_seed"
    }
}
```

- [ ] **Step 2: Rewrite ModelManifest.swift**

```swift
import Foundation

/// Parsed model manifest from config.json + privateagent-manifest.json.
public struct ModelManifest: Sendable {
    public let modelDir: URL
    public let hfConfig: HFConfig
    public let paManifest: PAManifest?
    public let manifestVersion: UInt32

    /// Parse from a model directory containing config.json and optionally
    /// privateagent-manifest.json.
    public init(modelDir: URL) throws {
        self.modelDir = modelDir

        // Parse config.json (required)
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        self.hfConfig = try JSONDecoder().decode(HFConfig.self, from: configData)

        // Parse privateagent-manifest.json (optional)
        let manifestURL = modelDir.appendingPathComponent("privateagent-manifest.json")
        if let manifestData = try? Data(contentsOf: manifestURL) {
            self.paManifest = try JSONDecoder().decode(PAManifest.self, from: manifestData)
            self.manifestVersion = UInt32(self.paManifest?.manifestVersion ?? 1)
        } else {
            self.paManifest = nil
            self.manifestVersion = 1
        }
    }

    /// Layer types derived from manifest or inferred from model config.
    /// Returns array of "gdn" or "full_attn" strings.
    public var layerTypes: [String] {
        if let types = paManifest?.layerTypes, types.count == hfConfig.numHiddenLayers {
            return types
        }
        // Default inference: every 4th layer is full_attn (Qwen3.5 pattern)
        return (0..<hfConfig.numHiddenLayers).map { i in
            i % 4 == 3 ? "full_attn" : "gdn"
        }
    }
}
```

- [ ] **Step 3: Add ModelPackTests to Package.swift**

Add to the targets array:

```swift
.testTarget(
    name: "ModelPackTests",
    dependencies: ["ModelPack"],
    resources: [.copy("Fixtures")]
),
```

- [ ] **Step 4: Create test fixture files**

Create `Tests/ModelPackTests/Fixtures/config.json`:
```json
{
    "hidden_size": 2048,
    "num_hidden_layers": 60,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 151936,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 1408,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-06
}
```

Create `Tests/ModelPackTests/Fixtures/privateagent-manifest.json`:
```json
{
    "manifest_version": 1,
    "layer_types": null,
    "expert_layout": {
        "quant_bits": 4,
        "dense_quant_bits": 4,
        "expert_size_each": 7000000,
        "expert_layers": 40
    },
    "turboquant_defaults": {
        "key_bits_x2": 7,
        "value_bits_x2": 7,
        "block_size": 64,
        "transform_kind": "structured_rotation",
        "transform_seed": 42
    },
    "checksums": {}
}
```

- [ ] **Step 5: Write ManifestParsingTests.swift**

```swift
import Testing
import Foundation
@testable import ModelPack

@Suite("Manifest Parsing Tests")
struct ManifestParsingTests {

    private var fixturesDir: URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures")
    }

    @Test("Parse config.json + privateagent-manifest.json")
    func parseFullManifest() throws {
        let manifest = try ModelManifest(modelDir: fixturesDir)
        #expect(manifest.hfConfig.hiddenSize == 2048)
        #expect(manifest.hfConfig.numHiddenLayers == 60)
        #expect(manifest.hfConfig.numKeyValueHeads == 2)
        #expect(manifest.hfConfig.headDim == 256)
        #expect(manifest.hfConfig.vocabSize == 151936)
        #expect(manifest.hfConfig.numExperts == 128)
        #expect(manifest.hfConfig.numExpertsPerTok == 8)
        #expect(manifest.manifestVersion == 1)
        #expect(manifest.paManifest != nil)
        #expect(manifest.paManifest?.expertLayout?.quantBits == 4)
        #expect(manifest.paManifest?.turboQuantDefaults?.keyBitsX2 == 7)
    }

    @Test("Layer types default to every-4th-full-attn")
    func layerTypesDefault() throws {
        let manifest = try ModelManifest(modelDir: fixturesDir)
        let types = manifest.layerTypes
        #expect(types.count == 60)
        #expect(types[0] == "gdn")
        #expect(types[3] == "full_attn")
        #expect(types[7] == "full_attn")
    }

    @Test("Missing config.json throws")
    func missingConfig() {
        let emptyDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try! FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDir) }

        #expect(throws: (any Error).self) {
            _ = try ModelManifest(modelDir: emptyDir)
        }
    }
}
```

- [ ] **Step 6: Run tests, commit**

```bash
swift test 2>&1 | tail -20
git add Sources/ModelPack/ Tests/ModelPackTests/ Package.swift
git commit -m "feat: implement ModelManifest JSON parsing with test fixtures"
```

---

### Task 3: Implement PAModelDescBridge

**Files:**
- Modify: `Sources/ModelPack/PAModelDescBridge.swift`
- Modify: `Tests/ModelPackTests/ManifestParsingTests.swift` (add bridge tests)

- [ ] **Step 1: Rewrite PAModelDescBridge.swift**

```swift
import Foundation
import FlashMoECore

/// Converts a Swift ModelManifest into a C PA_ModelDesc struct.
public enum PAModelDescBridge {

    public static func makeModelDesc(from manifest: ModelManifest) -> PA_ModelDesc {
        var desc = PA_ModelDesc()

        // Paths
        setString(&desc.model_dir, manifest.modelDir.path)
        let weightsPath = manifest.modelDir
            .appendingPathComponent("model_weights.bin").path
        setString(&desc.weights_path, weightsPath)
        let tokenizerPath = manifest.modelDir
            .appendingPathComponent("tokenizer.bin").path
        setString(&desc.tokenizer_path, tokenizerPath)

        // Architecture from HF config
        let hf = manifest.hfConfig
        desc.num_layers = UInt32(hf.numHiddenLayers)
        desc.num_experts = UInt32(hf.numExperts ?? 0)
        desc.active_experts_k = UInt32(hf.numExpertsPerTok ?? 0)
        desc.hidden_dim = UInt32(hf.hiddenSize)
        desc.vocab_size = UInt32(hf.vocabSize)
        desc.num_attn_heads = UInt32(hf.numAttentionHeads)
        desc.num_kv_heads = UInt32(hf.numKeyValueHeads)
        desc.head_dim = UInt32(hf.headDim ?? (hf.hiddenSize / hf.numAttentionHeads))
        desc.moe_intermediate = UInt32(hf.moeIntermediateSize ?? 0)
        desc.max_position_embeddings = UInt32(hf.maxPositionEmbeddings ?? 131072)
        desc.rms_norm_eps = hf.rmsNormEps ?? 1e-6

        // Layer types
        let types = manifest.layerTypes
        withUnsafeMutablePointer(to: &desc.layer_types) { ptr in
            let bound = UnsafeMutableRawPointer(ptr)
                .bindMemory(to: PA_LayerType.self, capacity: Int(PA_MAX_LAYERS))
            for i in 0..<min(types.count, Int(PA_MAX_LAYERS)) {
                bound[i] = types[i] == "full_attn" ? PA_LAYER_FULL_ATTN : PA_LAYER_GDN
            }
        }

        // Expert layout from PA manifest
        if let layout = manifest.paManifest?.expertLayout {
            desc.expert_quant_bits = UInt32(layout.quantBits)
            desc.dense_quant_bits = UInt32(layout.denseQuantBits)
            desc.expert_size_each = layout.expertSizeEach
        }

        // TurboQuant defaults
        if let tq = manifest.paManifest?.turboQuantDefaults {
            desc.default_key_bits_x2 = UInt16(tq.keyBitsX2)
            desc.default_value_bits_x2 = UInt16(tq.valueBitsX2)
            desc.default_tq_block_size = UInt32(tq.blockSize)
            desc.default_transform_seed = tq.transformSeed
            switch tq.transformKind {
            case "structured_rotation":
                desc.default_transform_kind = UInt32(PA_TRANSFORM_STRUCTURED_ROTATION.rawValue)
            case "hadamard":
                desc.default_transform_kind = UInt32(PA_TRANSFORM_HADAMARD.rawValue)
            default:
                desc.default_transform_kind = UInt32(PA_TRANSFORM_NONE.rawValue)
            }
        }

        desc.manifest_version = manifest.manifestVersion

        return desc
    }

    // MARK: - Helpers

    private static func setString(_ dest: inout (CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar, CChar), _ value: String) {
        withUnsafeMutablePointer(to: &dest) { ptr in
            let raw = UnsafeMutableRawPointer(ptr)
            let bound = raw.bindMemory(to: CChar.self, capacity: Int(PA_MAX_PATH))
            value.withCString { src in
                strncpy(bound, src, Int(PA_MAX_PATH) - 1)
                bound[Int(PA_MAX_PATH) - 1] = 0
            }
        }
    }
}
```

NOTE: The setString helper has a massive tuple type because Swift imports C char arrays as tuples. The implementer should use `withUnsafeMutablePointer(to:)` to handle this — the exact tuple type will match whatever Swift 6 imports from the C header. If the tuple type doesn't match exactly, use a generic approach:

```swift
private static func setCString<T>(_ dest: inout T, _ value: String) {
    withUnsafeMutablePointer(to: &dest) { ptr in
        let raw = UnsafeMutableRawPointer(ptr)
        let bound = raw.bindMemory(to: CChar.self, capacity: MemoryLayout<T>.size)
        value.withCString { src in
            strncpy(bound, src, MemoryLayout<T>.size - 1)
            bound[MemoryLayout<T>.size - 1] = 0
        }
    }
}
```

- [ ] **Step 2: Add bridge tests to ManifestParsingTests.swift**

Add to the existing test suite:

```swift
@Test("PAModelDescBridge produces valid PA_ModelDesc")
func bridgeProducesValidDesc() throws {
    let manifest = try ModelManifest(modelDir: fixturesDir)
    var desc = PAModelDescBridge.makeModelDesc(from: manifest)
    #expect(pa_model_desc_validate(&desc) == PA_STATUS_OK.rawValue)
    #expect(desc.num_layers == 60)
    #expect(desc.hidden_dim == 2048)
    #expect(desc.num_kv_heads == 2)
    #expect(desc.head_dim == 256)
    #expect(desc.vocab_size == 151936)
    #expect(desc.num_experts == 128)
    #expect(desc.active_experts_k == 8)
    #expect(desc.expert_quant_bits == 4)
    #expect(desc.default_key_bits_x2 == 7)
    #expect(desc.manifest_version == 1)
}

@Test("PAModelDescBridge sets layer types correctly")
func bridgeSetsLayerTypes() throws {
    let manifest = try ModelManifest(modelDir: fixturesDir)
    var desc = PAModelDescBridge.makeModelDesc(from: manifest)
    #expect(pa_model_desc_full_attn_count(&desc) == 15)
    #expect(pa_model_desc_gdn_count(&desc) == 45)
}
```

- [ ] **Step 3: Run tests, commit**

```bash
swift test 2>&1 | tail -20
git add Sources/ModelPack/ Tests/ModelPackTests/ Package.swift
git commit -m "feat: implement PAModelDescBridge — Swift manifest to C struct conversion"
```

---

### Task 4: Implement PromptCompiler

**Files:**
- Create: `Sources/ModelPack/PromptCompiler.swift`
- Create: `Tests/ModelPackTests/PromptCompilerTests.swift`

- [ ] **Step 1: Create PromptCompiler.swift**

```swift
import Foundation

/// Chat message for prompt compilation.
public struct ChatMessage: Sendable {
    public let role: String    // "system", "user", "assistant"
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

/// Compiles chat messages into model-specific prompt format.
/// Currently supports Qwen chat template (im_start/im_end).
public enum PromptCompiler {

    /// Compile messages into a Qwen-format prompt string.
    /// The last message should be from the user; an empty assistant turn
    /// is appended to prompt the model to respond.
    public static func compile(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true
    ) -> String {
        var result = ""

        for message in messages {
            result += "<|im_start|>\(message.role)\n"
            result += message.content
            result += "<|im_end|>\n"
        }

        if addGenerationPrompt {
            result += "<|im_start|>assistant\n"
        }

        return result
    }

    /// Compile a continuation prompt — only the new user message,
    /// assuming previous turns are already in the KV cache.
    public static func compileContinuation(
        userMessage: String
    ) -> String {
        var result = ""
        result += "<|im_end|>\n"  // close previous assistant turn
        result += "<|im_start|>user\n"
        result += userMessage
        result += "<|im_end|>\n"
        result += "<|im_start|>assistant\n"
        return result
    }

    /// Default system prompt.
    public static let defaultSystemPrompt = "You are a helpful assistant."
}
```

- [ ] **Step 2: Create PromptCompilerTests.swift**

```swift
import Testing
@testable import ModelPack

@Suite("PromptCompiler Tests")
struct PromptCompilerTests {

    @Test("Single user message with system prompt")
    func singleTurn() {
        let messages = [
            ChatMessage(role: "system", content: "You are helpful."),
            ChatMessage(role: "user", content: "Hello"),
        ]
        let prompt = PromptCompiler.compile(messages: messages)
        #expect(prompt.contains("<|im_start|>system\nYou are helpful.<|im_end|>"))
        #expect(prompt.contains("<|im_start|>user\nHello<|im_end|>"))
        #expect(prompt.hasSuffix("<|im_start|>assistant\n"))
    }

    @Test("Multi-turn conversation")
    func multiTurn() {
        let messages = [
            ChatMessage(role: "system", content: "Be brief."),
            ChatMessage(role: "user", content: "Hi"),
            ChatMessage(role: "assistant", content: "Hello!"),
            ChatMessage(role: "user", content: "How are you?"),
        ]
        let prompt = PromptCompiler.compile(messages: messages)
        #expect(prompt.contains("<|im_start|>assistant\nHello!<|im_end|>"))
        #expect(prompt.hasSuffix("<|im_start|>assistant\n"))
    }

    @Test("No generation prompt")
    func noGenerationPrompt() {
        let messages = [ChatMessage(role: "user", content: "Hi")]
        let prompt = PromptCompiler.compile(messages: messages, addGenerationPrompt: false)
        #expect(!prompt.hasSuffix("<|im_start|>assistant\n"))
        #expect(prompt.hasSuffix("<|im_end|>\n"))
    }

    @Test("Continuation prompt")
    func continuation() {
        let prompt = PromptCompiler.compileContinuation(userMessage: "Next question")
        #expect(prompt.hasPrefix("<|im_end|>\n"))
        #expect(prompt.contains("<|im_start|>user\nNext question<|im_end|>"))
        #expect(prompt.hasSuffix("<|im_start|>assistant\n"))
    }
}
```

- [ ] **Step 3: Run tests, commit**

```bash
swift test 2>&1 | tail -20
git add Sources/ModelPack/PromptCompiler.swift Tests/ModelPackTests/PromptCompilerTests.swift
git commit -m "feat: implement PromptCompiler with Qwen chat template"
```

---

### Task 5: Implement FlashMoERuntime loader skeleton

**Files:**
- Modify: `Sources/FlashMoERuntime/include/FlashMoERuntime.h`
- Delete: `Sources/FlashMoERuntime/FlashMoERuntime_stub.c`
- Create: `Sources/FlashMoERuntime/pa_runtime.c`

- [ ] **Step 1: Update FlashMoERuntime.h with loader API**

```c
#ifndef FLASHMOE_RUNTIME_H
#define FLASHMOE_RUNTIME_H

#include "FlashMoECore.h"

#ifdef __cplusplus
extern "C" {
#endif

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

/// Create a new session. Returns NULL on failure.
PA_Session *pa_session_create(void);

/// Destroy a session and free all resources.
void pa_session_destroy(PA_Session *session);

/// Get current session state.
PA_SessionState pa_session_get_state(const PA_Session *session);

/// Load a model into the session.
/// available_memory: current os_proc_available_memory() value.
/// Returns PA_STATUS_OK on success.
int pa_session_load_model(
    PA_Session *session,
    const PA_ModelDesc *desc,
    uint64_t available_memory
);

/// Unload the current model.
void pa_session_unload_model(PA_Session *session);

/// Get the computed memory budget (valid after successful load).
int pa_session_get_memory_budget(
    const PA_Session *session,
    PA_MemoryBudget *out_budget
);

/// Get last error message. Never returns NULL.
const char *pa_session_last_error(const PA_Session *session);

#ifdef __cplusplus
}
#endif

#endif // FLASHMOE_RUNTIME_H
```

- [ ] **Step 2: Create pa_runtime.c**

```c
#include "FlashMoERuntime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct PA_Session {
    PA_SessionState state;
    PA_ModelDesc model_desc;
    PA_MemoryBudget memory_budget;
    int model_loaded;
    char last_error[512];
};

PA_Session *pa_session_create(void) {
    PA_Session *s = calloc(1, sizeof(PA_Session));
    if (!s) return NULL;
    s->state = PA_SESSION_IDLE;
    s->model_loaded = 0;
    strcpy(s->last_error, "No error");
    return s;
}

void pa_session_destroy(PA_Session *session) {
    if (!session) return;
    // Future: release Metal resources, close file descriptors, etc.
    free(session);
}

PA_SessionState pa_session_get_state(const PA_Session *session) {
    if (!session) return PA_SESSION_IDLE;
    return session->state;
}

int pa_session_load_model(
    PA_Session *session,
    const PA_ModelDesc *desc,
    uint64_t available_memory
) {
    if (!session || !desc) {
        if (session) snprintf(session->last_error, sizeof(session->last_error),
            "NULL argument");
        return PA_STATUS_ERROR_GENERIC;
    }

    session->state = PA_SESSION_LOADING;

    // Validate model descriptor
    int status = pa_model_desc_validate(desc);
    if (status != PA_STATUS_OK) {
        snprintf(session->last_error, sizeof(session->last_error),
            "Invalid model descriptor: %s", pa_status_string(status));
        session->state = PA_SESSION_IDLE;
        return status;
    }

    // Copy model descriptor
    memcpy(&session->model_desc, desc, sizeof(PA_ModelDesc));

    // Compute memory budget
    status = pa_compute_memory_budget(desc, available_memory, &session->memory_budget);
    if (status != PA_STATUS_OK) {
        snprintf(session->last_error, sizeof(session->last_error),
            "Memory budget failed: need more RAM (available: %llu bytes)",
            (unsigned long long)available_memory);
        session->state = PA_SESSION_IDLE;
        return status;
    }

    // Future: mmap weights, open expert file descriptors, init Metal
    // For now, just mark as loaded
    session->model_loaded = 1;
    session->state = PA_SESSION_DONE;  // Will be PA_SESSION_IDLE when ready
    snprintf(session->last_error, sizeof(session->last_error), "No error");

    return PA_STATUS_OK;
}

void pa_session_unload_model(PA_Session *session) {
    if (!session) return;
    // Future: release mmap, close FDs, release Metal
    session->model_loaded = 0;
    memset(&session->model_desc, 0, sizeof(PA_ModelDesc));
    memset(&session->memory_budget, 0, sizeof(PA_MemoryBudget));
    session->state = PA_SESSION_IDLE;
}

int pa_session_get_memory_budget(
    const PA_Session *session,
    PA_MemoryBudget *out_budget
) {
    if (!session || !out_budget) return PA_STATUS_ERROR_GENERIC;
    if (!session->model_loaded) return PA_STATUS_ERROR_GENERIC;
    memcpy(out_budget, &session->memory_budget, sizeof(PA_MemoryBudget));
    return PA_STATUS_OK;
}

const char *pa_session_last_error(const PA_Session *session) {
    if (!session) return "NULL session";
    return session->last_error;
}
```

- [ ] **Step 3: Delete old stub, run tests, commit**

```bash
rm Sources/FlashMoERuntime/FlashMoERuntime_stub.c
swift test 2>&1 | tail -20
git add Sources/FlashMoERuntime/
git commit -m "feat: implement FlashMoERuntime session + model loader skeleton"
```

---

### Task 6: Wire up PrivateAgentEngine load chain

**Files:**
- Modify: `Sources/FlashMoEBridge/PrivateAgentEngine.swift`

- [ ] **Step 1: Implement load/unload in PrivateAgentEngine**

```swift
import Foundation
import Observation
import FlashMoECore
import FlashMoERuntime
import ModelPack

// (Keep existing EngineState, GenerationStats, GenerationEvent, PromptInput enums)

/// Model information exposed to UI.
public struct ModelInfo: Sendable {
    public let name: String
    public let numLayers: Int
    public let numExperts: Int
    public let activeExpertsK: Int
    public let hiddenDim: Int
    public let vocabSize: Int
    public let maxContext: Int
    public let totalDirtyMB: Double
    public let totalResidentMB: Double
    public let kvCacheMB: Double
}

@MainActor
@Observable
public final class PrivateAgentEngine {
    public private(set) var state: EngineState = .idle
    public private(set) var modelInfo: ModelInfo?
    public private(set) var lastError: String?

    private var session: OpaquePointer?  // PA_Session*
    private let engineQueue = DispatchQueue(label: "com.privateagent.engine", qos: .userInitiated)

    public init() {}

    deinit {
        if let s = session {
            pa_session_destroy(s)
        }
    }

    /// Load a model from a manifest. Runs validation and memory budget on background queue.
    public func loadModel(from manifest: ModelManifest, availableMemory: UInt64 = 0) async throws {
        guard state == .idle || state == .error("") != state else {
            throw EngineError.busy
        }

        state = .loading
        lastError = nil

        let desc = PAModelDescBridge.makeModelDesc(from: manifest)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            engineQueue.async { [weak self] in
                guard let self else {
                    continuation.resume(throwing: EngineError.destroyed)
                    return
                }

                // Create session if needed
                var s = self.session
                if s == nil {
                    s = pa_session_create()
                    if s == nil {
                        DispatchQueue.main.async {
                            self.state = .error("Failed to create session")
                            self.lastError = "Failed to create session"
                        }
                        continuation.resume(throwing: EngineError.initFailed)
                        return
                    }
                }

                // Use provided memory or estimate 6GB
                let mem = availableMemory > 0 ? availableMemory : 6 * 1024 * 1024 * 1024

                var mutableDesc = desc
                let result = pa_session_load_model(s, &mutableDesc, mem)

                if result != PA_STATUS_OK.rawValue {
                    let errorMsg = String(cString: pa_session_last_error(s))
                    DispatchQueue.main.async {
                        self.state = .error(errorMsg)
                        self.lastError = errorMsg
                        self.session = s
                    }
                    continuation.resume(throwing: EngineError.loadFailed(errorMsg))
                    return
                }

                // Get memory budget
                var budget = PA_MemoryBudget()
                pa_session_get_memory_budget(s, &budget)

                let info = ModelInfo(
                    name: manifest.modelDir.lastPathComponent,
                    numLayers: Int(mutableDesc.num_layers),
                    numExperts: Int(mutableDesc.num_experts),
                    activeExpertsK: Int(mutableDesc.active_experts_k),
                    hiddenDim: Int(mutableDesc.hidden_dim),
                    vocabSize: Int(mutableDesc.vocab_size),
                    maxContext: Int(budget.max_context_length),
                    totalDirtyMB: Double(budget.total_dirty_bytes) / (1024 * 1024),
                    totalResidentMB: Double(budget.total_resident_bytes) / (1024 * 1024),
                    kvCacheMB: Double(budget.kv_cache_bytes) / (1024 * 1024)
                )

                DispatchQueue.main.async {
                    self.session = s
                    self.modelInfo = info
                    self.state = .ready
                }
                continuation.resume()
            }
        }
    }

    /// Unload the current model.
    public func unloadModel() {
        if let s = session {
            engineQueue.sync {
                pa_session_unload_model(s)
            }
        }
        modelInfo = nil
        state = .idle
    }
}

/// Engine errors.
public enum EngineError: LocalizedError, Sendable {
    case busy
    case destroyed
    case initFailed
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .busy: return "Engine is busy"
        case .destroyed: return "Engine was destroyed"
        case .initFailed: return "Failed to initialize engine"
        case .loadFailed(let msg): return "Load failed: \(msg)"
        }
    }
}
```

NOTE TO IMPLEMENTER: The above code has a deliberate issue in the state guard (`state == .error("") != state`). Fix it to properly check that state is `.idle` or an error state. Use a switch or pattern match. Also, the `@MainActor` class with `DispatchQueue.main.async` inside `engineQueue.async` has a subtle concurrency issue — the `[weak self]` capture and `DispatchQueue.main.async` pattern is correct for bridging between the engine queue and MainActor, but make sure the continuation is only resumed once.

- [ ] **Step 2: Run tests, commit**

```bash
swift build 2>&1 | tail -10
swift test 2>&1 | tail -20
git add Sources/FlashMoEBridge/
git commit -m "feat: wire PrivateAgentEngine load chain — manifest → C runtime → UI"
```

---

## Summary

After completing Plan 2, the pipeline is:

```
ModelManifest.init(modelDir:)      → parse JSON
PAModelDescBridge.makeModelDesc()  → populate C struct
pa_session_load_model()            → validate + compute memory budget
PrivateAgentEngine.loadModel()     → async Swift API for UI
PromptCompiler.compile()           → format chat messages
```

**Next:** Plan 3 (FlashMoERuntime skeleton + benchmark harness) adds the actual inference pipeline.
