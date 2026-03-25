# PrivateAgent: Offline LLM Chat for iPhone

**Date**: 2026-03-26
**Status**: Approved for implementation planning
**Target**: iPhone 16 Pro (A18 Pro, ~8GB RAM per teardown reports — not publicly disclosed by Apple; memory budget validated at runtime via `os_proc_available_memory()`)
**Goal**: Open-source offline chat app running Qwen3.5-35B-A3B via Flash-MoE + TurboQuant
**Repository**: github.com (public, open source)

---

## 1. Project Summary

PrivateAgent is a native iOS chat application that runs Qwen3.5-35B-A3B entirely on-device. It combines two complementary technologies:

- **Flash-MoE**: SSD expert streaming — only active experts (3B of 35B params) are loaded into RAM per token, the rest stream from NVMe storage on demand.
- **TurboQuant**: KV cache compression — attention keys/values are stored in compressed format, enabling longer context windows within iPhone's memory constraints. Note: Google's published TurboQuant describes 3-bit and 4-bit configurations. This project targets a custom 3.5-bit configuration (averaging 3-bit main codes + auxiliary overhead) that will be implemented and validated in-house; it is not a pre-existing TurboQuant variant. If 3.5-bit proves infeasible, fallback to published 3-bit or 4-bit.

The architecture is **FlashMoE-first, TurboQuant-native** — Flash-MoE owns the inference pipeline, TurboQuant participates in the attention fast path as a first-class citizen, not an afterthought.

### Core Principles

1. **Large-total / sparse-active, not dense fully-resident.** The model is 35B params but only 3B are active per token.
2. **TurboQuant is not a compress/decompress utility layer.** It participates in the attention fast path — decode scores Q against compressed K directly.
3. **SwiftData, UI, downloads never enter the hot path.** FlashMoERuntime is the core.
4. **Benchmarked from day one.** Structured benchmark suite with reproducible conditions.

---

## 2. Architecture

### Package Structure

Single `Package.swift`, multiple targets/library products. Not separate packages — cross-package headers, Metal bundling, and internal symbols would slow development for an experimental repo.

```
PrivateAgent/
├── Package.swift
├── Apps/
│   └── PrivateAgentiOS/              # iPhone 16 Pro app target
├── Sources/
│   ├── FlashMoECore/                 # C / Accelerate
│   │   ├── routing math (softmax, topK) + layer policy
│   │   ├── dense/MoE graph definitions and tensor/layout descriptors
│   │   ├── CPU reference kernels (correctness tests)
│   │   ├── GatedDeltaNet state machine (Accelerate BLAS)
│   │   ├── full-attention reference path (validation)
│   │   └── tokenizer (C BPE, single-header)
│   │
│   ├── FlashMoEMetal/                # Metal / ObjC
│   │   ├── dense-path GPU kernels: dequant, matvec, SwiGLU, RMSNorm, RoPE
│   │   ├── dense full-attention kernels
│   │   ├── MoE combine / residual fused kernels
│   │   ├── pipeline state cache
│   │   └── encoder builders for command buffers
│   │
│   ├── TurboQuantCore/               # C
│   │   ├── KV format definitions
│   │   ├── transform descriptors (structured transform, seed-driven, no dense R matrix)
│   │   ├── pack/unpack policy
│   │   ├── QJL residual correction logic
│   │   └── CPU reference implementations (tolerance-based validation)
│   │
│   ├── TurboQuantMetal/              # Metal / ObjC
│   │   ├── compressed KV write path (rotate + quantize + QJL pack)
│   │   ├── compressed-domain Q·K scoring (no full K decompression)
│   │   ├── V tile-wise dequant-and-accumulate (no full V materialization)
│   │   └── selective unpack / correction kernels
│   │
│   ├── FlashMoERuntime/              # C / ObjC
│   │   ├── Session state machine
│   │   │   states: idle → loading → prefill → decode → done
│   │   │            ↘ cancelled / throttled / recoveringMemory
│   │   │   cancelled during prefill → KV cache discarded, next generate starts fresh
│   │   │   cancelled during decode → KV cache retained up to last completed token
│   │   ├── MemoryPlanner
│   │   │   ├── multi-snapshot (load-time, post-load, pre-expand, on-pressure)
│   │   │   ├── dirty budget: Metal buffers + GDN state + expert buffers + KV + scratch
│   │   │   └── resident estimate: hot weight window + page cache window
│   │   ├── ExpertPager
│   │   │   ├── SSD pread fanout (GCD dispatch group, page-aligned, cache-io-split)
│   │   │   ├── page cache trust (no custom LRU, Trust the OS)
│   │   │   ├── next-layer prefetch overlap with current-layer GPU compute
│   │   │   └── I/O error policy: pread fail → retry once → PA_STATUS_IO_ERROR
│   │   │       → generation terminates with user-visible error
│   │   ├── LayerScheduler
│   │   │   ├── reads model metadata → dispatches per-layer backend
│   │   │   ├── assembles command buffers (CMD1/CMD2/CMD3 + TQ encoders)
│   │   │   └── GDN layers → Accelerate BLAS, full-attn layers → TurboQuant path
│   │   ├── KVCacheManager
│   │   │   ├── reads full-attn layer count from model metadata
│   │   │   ├── allocates compressed KV buffers (PA_QuantizedKVDesc driven)
│   │   │   ├── context length determined by MemoryPlanner budget
│   │   │   └── context exhausted → PA_STATUS_CONTEXT_EXHAUSTED
│   │   │       (reset vs sliding window is Runtime policy, not Bridge decision)
│   │   ├── ModelLoader
│   │   │   ├── receives PA_ModelDesc (C struct) populated by Bridge/ModelPack
│   │   │   │   PA_ModelDesc contains: paths, layer_types[], expert_layout,
│   │   │   │   kv_config, memory_profile — all C-native, no Swift dependency
│   │   │   ├── ModelPack (Swift) parses JSON → populates PA_ModelDesc → passes to Runtime
│   │   │   └── mmap weights → init Metal buffers based on PA_ModelDesc
│   │   ├── ThermalPolicy
│   │   │   ├── .nominal/.fair → normal
│   │   │   ├── .serious → reduce prefetch depth, fanout, GPU queue depth, UI update freq
│   │   │   ├── .critical → pause generation + UI overlay
│   │   │   └── also checks ProcessInfo.isLowPowerModeEnabled
│   │   └── MemoryPressurePolicy
│   │       ├── foreground warning: cancel gen → release expert buffers → keep KV
│   │       ├── second warning: release KV → UI notify "conversation reset"
│   │       └── scene background/disconnect: release expert buffers, KV per policy
│   │
│   ├── ModelPack/                    # Swift — model metadata only
│   │   ├── ModelManifest (config.json + privateagent-manifest.json v1)
│   │   │   manifest includes: manifest_version field for forward compat
│   │   │   ├── layer_types, expert_layout, kv_config
│   │   │   ├── memory_profile (pre-estimated)
│   │   │   └── recommended key_bits / value_bits (separate, per-layer configurable)
│   │   ├── TokenizerPack (tokenizer.bin loading)
│   │   ├── PromptCompiler (chat template, model semantics, not ViewModel responsibility)
│   │   └── Pack validator (integrity check)
│   │
│   ├── ModelHub/                     # Swift — download + storage lifecycle
│   │   ├── ModelCatalog (static + remote JSON; offline fallback: last-cached or bundled)
│   │   ├── DownloadManager
│   │   │   ├── Background URLSession (survives app suspend, NOT force-quit)
│   │   │   ├── per-file SHA-256 from privateagent-manifest.json (not HF API dependent)
│   │   │   ├── resume from interruption
│   │   │   ├── pre-download free space check
│   │   │   ├── download state persisted in Application Support (not UserDefaults)
│   │   │   └── relaunch handshake: AppDelegate.application(_:handleEventsForBackgroundURLSession:)
│   │   │       → rebind to in-flight tasks → reconcile ModelHub state
│   │   │       → acceptance test: suspend or system-terminate app mid-download, relaunch, verify resume + completion
│   │   └── ModelStorage (scan, validate, delete, space stats)
│   │
│   ├── FlashMoEBridge/               # Swift — thin facade
│   │   ├── PrivateAgentEngine (@MainActor @Observable, NOT @unchecked Sendable)
│   │   │   ├── state: EngineState (maps to Runtime Session states)
│   │   │   ├── modelInfo, generationStats, systemStatus
│   │   │   ├── loadModel(_ manifest:) async throws
│   │   │   ├── generate(input: PromptInput, config:) -> AsyncThrowingStream<GenerationEvent, Error>
│   │   │   ├── cancel(), resetConversation(), unloadModel()
│   │   │   └── PromptInput = .formattedPrompt(String) | .tokenIDs([Int32])
│   │   └── GenerationEvent
│   │       ├── .prefillProgress(tokens:total:)
│   │       ├── .token(text:id:)
│   │       ├── .thinkingStart / .thinkingEnd
│   │       ├── .contextExhausted(policy:)
│   │       ├── .throttled(reason:)
│   │       └── .finished(stats:)
│   │       (NO .error — terminal errors go through AsyncThrowingStream throw)
│   │
│   └── PrivateAgentUI/               # SwiftUI
│       ├── App/, Views/, DataModels/, Services/, Resources/
│       └── (see Section 6 below)
│
├── Benchmarks/
│   ├── expert-io/                    # pread throughput, page cache hit rate
│   ├── kv-attention/                 # TurboQuant encode/decode latency + accuracy
│   └── end-to-end/                   # tok/s, TTFT, memory peak, thermal trajectory
└── Tests/
    ├── FlashMoECoreTests/
    ├── TurboQuantCoreTests/
    └── PrivateAgentTests/
```

### Data Flow

```
User input
→ ChatSessionViewModel (@MainActor)
→ PromptCompiler (ModelPack — chat template formatting)
→ PrivateAgentEngine.generate(input: .formattedPrompt(...))
→ FlashMoERuntime Session
→ Prefill: FlashMoERuntime.prefill()
  → LayerScheduler dispatches per-layer:
    → GDN layers: FlashMoEMetal encoders + Accelerate BLAS
    → Full-attn layers: FlashMoEMetal encoders + TurboQuantMetal KV write
  → ExpertPager: pread active experts from SSD
→ Decode loop:
  → FlashMoE router selects top-k experts
  → ExpertPager prefetch/pread next-layer expert shards
  → Full-attn layers: TurboQuantMetal compressed-domain Q·K scoring
  → Full-attn layers: TurboQuantMetal V tile-wise dequant-accumulate
  → GDN layers: O(1) fixed state, no KV cache
→ Token callback
→ AsyncThrowingStream<GenerationEvent>
→ ChatSessionViewModel updates UI (100ms throttle)
→ SwiftData persistence (on sentence boundary / completion / checkpoint, NOT per-token)
```

### Model Weight Format (Qwen3.5-35B-A3B)

Follows flash-moe's pre-packed weight binary format. Compatible with existing HuggingFace weight files, provided a PrivateAgent sidecar manifest (`privateagent-manifest.json`) is published alongside them. Existing repos (e.g., `alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE`) contain the weight binaries but not the sidecar; a manifest-synthesis step or republished repos are needed for F1.

| Component | Format | On-disk size | Runtime |
|-----------|--------|-------------|---------|
| Dense weights (non-expert) | `model_weights.bin` (MLX 4-bit, mmap'd) | ~1.4 GB | file-backed, not dirty |
| Expert shards | `packed_experts/layer_XX.bin` (one per layer, 4-bit) | ~18 GB total (~450 MB each × 40 layers) | SSD-streamed on demand |
| Tokenizer | `tokenizer.bin` (C BPE pre-exported) | ~8 MB | loaded at init |
| Config | `config.json` + `privateagent-manifest.json` | ~300 KB | parsed at init |

Total on-disk: **~19.5 GB** (4-bit full) or **~13.4 GB** (tiered 4-bit/2-bit).

Future: Q3 GGUF experts (from Unsloth) would reduce expert storage to ~12 GB with near-4-bit quality.

### Memory Budget Breakdown (iPhone 16 Pro, operational target)

| Component | Dirty? | Estimated |
|-----------|--------|-----------|
| Dense weights (mmap'd) | No (file-backed) | ~1.4 GB resident working set |
| Metal buffers (projections, scratch) | Yes | ~200 MB |
| GDN state (40 layers × O(1)) | Yes | ~150 MB |
| Expert data double-buffers (K=8 × 2) | Yes | ~112 MB |
| KV cache (TQ3.5, 4096 context) | Yes | ~26 MB |
| Rotation descriptors + scratch | Yes | ~5 MB |
| **Total dirty** | | **~493 MB** |
| **Total resident (dirty + file-backed)** | | **~1.9 GB** |

Remaining for OS + page cache: ~6 GB (of ~8 GB). Page cache manages expert SSD streaming (~75% hit rate observed on flash-moe iOS port).

Target ceiling: **< 5 GB total app memory** (dirty + clean resident). This is an operational target based on observed iOS memory limits, not an Apple-published spec.

### Stable Interface: PA_TensorRef

```c
typedef struct {
    void *data;
    uint64_t byte_offset;
    uint32_t dtype;
    uint32_t rank;
    uint32_t shape[4];
    uint32_t stride[4];
    uint32_t storage_kind;   // cpu / metal_buffer / quantized_kv
    uint32_t quant_scheme;   // none / q4 / q2 / tq3_5 / tq4
} PA_TensorRef;
```

### Stable Interface: PA_QuantizedKVDesc

```c
typedef struct {
    uint16_t key_bits_x2;        // bits × 2: e.g. 7 = 3.5 bits, 6 = 3 bits, 8 = 4 bits
    uint16_t value_bits_x2;      // separate from key_bits; same encoding
    uint32_t block_size;
    uint32_t transform_kind;     // structured rotation type
    uint64_t transform_seed;     // deterministic, reproducible
    uint32_t residual_bits;      // QJL bits (typically 1)
    uint64_t main_codes_offset;  // into KV buffer
    uint64_t aux_params_offset;  // can be 0 (empty), global, or per-block
    uint64_t qjl_bits_offset;
    uint32_t aux_params_kind;    // none / global / per_block
} PA_QuantizedKVDesc;
```

KV layout: `main_codes | aux_params | qjl_bits` — aux_params can be empty, global, or per-block depending on TurboQuant variant. No hardcoded assumption of affine scale/zero_point.

---

## 3. TurboQuant Design

Based on [Google Research TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) and [mlx-vlm PR #858](https://github.com/Blaizzy/mlx-vlm/pull/858) (Draft as of 2026-03-25).

### Transform Strategy

Do NOT materialize dense random rotation matrices. Use `transform_kind + seed + head_dim` descriptor. Implementation uses structured transform or seed-driven generation to avoid initialization cost and memory overhead on iPhone.

### Write Path (per new token, full-attn layers only)

1. `tq_rotate_and_quantize` kernel: structured rotation → scalar quantize → pack main_codes
2. `tq_qjl_residual` kernel: compute quantization error → random projection → 1-bit pack

Output: compressed KV entry written directly to KV cache buffer.

### Read Path (per decode step, full-attn layers only)

1. Query prepare: `Q_transformed = Q @ R^T_structured` — cached in scratch buffer, computed once per decode step, reused across all KV tiles.
2. Score blocks: `tq_compressed_qk_score` — Q_transformed dot compressed K + QJL bias correction → attention scores (float). **No full K decompression.**
3. V gather: `tq_compressed_v_gather` — softmax(scores) @ compressed V via tile-wise dequant-and-accumulate. **No full V materialization.**

### Key Invariants

- **Decode MUST NOT fully decompress K.** Scoring happens in compressed domain.
- **V MUST use block/tile-level on-the-fly unpack only.** No expansion to full-precision cache.
- `key_bits` and `value_bits` are separately configurable (even if both default to 3.5 initially).
- Full-attn layer count comes from model metadata, not hardcoded constants.

### Memory Impact (iPhone 16 Pro, Qwen3.5-35B-A3B)

**Architecture assumptions (MUST be confirmed from model's `config.json` before implementation):**
- N full-attention layers, remaining layers are GDN (ratio from model metadata)
- 2 KV heads, 256 head dim (sourced from flash-moe iOS port; may differ — memory estimates will be revised if actual values differ)

Risk: GDN-to-full-attn layer ratio directly determines KV memory requirements. If the model has more full-attn layers than expected, context length targets shrink proportionally.

Assuming these values hold:

| Context | bf16 KV | TQ3.5 KV (~4.57x) | TQ3 (~5.33x) | TQ4 (~4x) |
|---------|---------|-------------------|-------------|-----------|
| 2048 | ~60 MB | ~13 MB | ~11 MB | ~15 MB |
| 4096 | ~120 MB | ~26 MB | ~23 MB | ~30 MB |
| 8192 | ~240 MB | ~53 MB | ~45 MB | ~60 MB |

TurboQuant enables ~4-5x context extension within the same memory budget. Exact ratios depend on final bit configuration (3, 3.5, or 4 bit) determined during implementation validation.

---

## 4. UI Design (PrivateAgentUI)

Native iOS style. NavigationStack based. No custom design system.

### Screen Structure

```
App Launch
├── ConversationListView          # Main screen, conversation list
│   ├── NavigationStack + List + .searchable
│   ├── Toolbar: + New Chat, Settings gear
│   └── → push ChatView
├── ChatView                      # Conversation screen
│   ├── ScrollView + LazyVStack (messages)
│   │   ├── UserBubble (right-aligned)
│   │   └── AssistantMessage (left-aligned)
│   │       ├── AttributedString Markdown (incremental, current message only)
│   │       ├── CodeBlockView (syntax highlight + copy button)
│   │       └── ThinkingDisclosure (<think> collapsible, fold state in view not DB)
│   ├── StatsBar (tok/s, prefill, context usage)
│   ├── InputBar (TextField axis: .vertical, 1-5 lines)
│   └── Toolbar: system prompt edit, share, export
├── ModelManagerView (push)       # Model management
│   ├── Downloaded models (size, quant, delete)
│   ├── Catalog (downloadable models)
│   ├── Download progress (per-file, pause/resume/cancel)
│   └── Free space display
└── SettingsView (push)           # Settings
    ├── Generation: max tokens, temperature, top-p
    ├── TurboQuant: key_bits, value_bits, context budget
    ├── Performance: expert K count, prefill batch size
    ├── Default system prompt
    ├── Export/Import conversations
    └── About / GitHub link
```

### Data Model (SwiftData)

```swift
@Model
class Conversation {
    @Attribute(.unique) var id: UUID
    var title: String              // auto: first 50 chars of user's first message; editable
    var systemPrompt: String
    var createdAt: Date
    var updatedAt: Date
    @Relationship(deleteRule: .cascade, inverse: \Message.conversation)
    var messages: [Message]
    var modelId: String
}

@Model
class Message {
    @Attribute(.unique) var id: UUID
    var conversation: Conversation?
    var role: MessageRole              // Codable enum: .user / .assistant
    var content: String
    var thinkingContent: String?       // <think> block, stored separately
    var ordinal: Int                   // explicit sort order, not store-dependent
    var createdAt: Date
    var tokenCount: Int?
    var stats: GenerationStatsSnapshot? // small Codable struct, nullable
}
```

### Persistence Rules

- UI redraws throttled at 100ms during streaming (only current assistant message re-parsed)
- SwiftData writes happen at: sentence boundary, generation pause/stop/complete, or periodic checkpoint (every ~5s) — **never per-token**
- Full conversation transcript is parsed once on load, not re-parsed during streaming

### Export / Import

- **Export**: JSON (full, reimportable) + Markdown (human-readable)
- **Import**: Files app picker → JSON parse → SwiftData write
- Custom UTType: `.privateagent-chat`
- Share single message via `ShareLink` with "Generated locally on iPhone with PrivateAgent" footer

---

## 5. Acceptance Criteria

### MVP Gate

| # | Criteria | Verification |
|---|----------|-------------|
| F1 | In-app download Qwen3.5-35B-A3B, SHA-256 pass | Checksum match post-download |
| F2 | Model load success, MemoryPlanner reports budget | UI shows memory profile |
| F3 | Single-turn: input → streaming reply → complete display | Manual test |
| F4 | Multi-turn: KV cache reuse, no history re-prefill | Turn 2+ TTFT << Turn 1 |
| F5 | Context exhausted triggers Runtime policy | Sustained conversation until trigger |
| F6 | Markdown rendering: headers, bold, italic, code fence | Send markdown-heavy prompt |
| F7 | Code block syntax highlight + copy button | Ask model to write code |
| F8 | `<think>` block collapsible | Enable thinking, test disclosure |
| F9 | Conversation list CRUD + search | Create/rename/delete/search |
| F10 | Per-conversation system prompt editing | Modify, confirm behavior change |
| F11 | Export JSON + Markdown, import JSON roundtrip | Roundtrip test |
| F12 | Share single message via ShareLink | Tap share, confirm output |
| F13 | Model management: progress, pause/resume, delete, space | Operational test |
| F14 | Settings: generation params + TurboQuant config | Change settings, confirm inference change |
| P1 | Decode speed >= 3 tok/s | Benchmark suite |
| P2 | TTFT < 15s (512-token prompt) | Benchmark suite |
| P3 | Context length >= 4096 with TurboQuant | Benchmark suite |
| P4 | App memory peak < 5GB (operational target — see memory budget breakdown) | Instruments Memory Graph + `task_vm_info.phys_footprint` logging (NOT `os_proc_available_memory`, which only reports remaining dirty headroom) |
| P5 | Model load time < 10s | Benchmark suite |
| T1 | Compressed KV pack/code path: bit-exact vs CPU reference | Unit test |
| T2 | Compressed-domain Q·K scoring: cosine similarity > 0.99 vs bf16 | Tolerance-based test (CPU/Metal rounding) |
| T3 | V tile-wise dequant-accumulate: max abs error < 0.01 vs bf16 | Tolerance-based test |
| T4 | KV memory compression ratio >= 4x vs bf16 | Memory measurement |

### Stretch Goals

| # | Criteria | Verification |
|---|----------|-------------|
| S1 | TTFT < 5s (2048 token prompt) | Benchmark suite |
| S2 | Context length >= 8192 | Benchmark suite |
| S3 | 5 min continuous generation without .critical thermal | Thermal trajectory log |
| S4 | Memory warning → graceful degrade, no crash | Simulated memory pressure |
| S5 | Needle-in-haystack at 4096 context: pass | Targeted prompt test |

### Benchmark Contract

All performance numbers measured under these conditions:

- **Device**: iPhone 16 Pro
- **iOS version**: documented per run
- **Battery**: > 50%, Low Power Mode OFF
- **Power**: NOT plugged in (to reflect real thermal behavior)
- **Thermal state**: cold start (device idle > 5 min before run)
- **Xcode**: NOT attached (no debug overhead, no Metal validation)
- **Build**: Release configuration
- **Prompt length**: documented per benchmark (default: 128 tokens input)
- **Runs**: 5 runs, report p50 and p95
- **Output**: JSON, stored in Benchmarks/ for historical comparison

---

## 6. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Pure C/Metal engine, not MLX | MLX cannot do SSD expert streaming; on 8GB device, framework overhead is fatal |
| Single Package.swift, multi-target | Avoid cross-package header/Metal bundling complexity for experimental repo |
| TurboQuant as first-class targets | Not a sub-module of FlashMoE — owns compressed attention path entirely |
| Structured transform descriptors | No dense rotation matrices in memory; seed-driven for reproducibility |
| K/V bits separately configurable | K and V have different distortion tolerance; enables per-layer tuning |
| KV layout: main_codes \| aux_params \| qjl_bits | No hardcoded affine quantization assumption; aux_params can be empty/global/per-block |
| Trust the OS page cache | No custom expert LRU — validated by flash-moe on both Mac and iPhone |
| MemoryPlanner multi-snapshot | os_proc_available_memory() is instantaneous advice, not a hard cap; re-check on every pressure signal |
| Dirty vs resident budget separation | mmap'd weights are file-backed, not dirty; mixing them inflates dirty estimate |
| SwiftData writes at checkpoints | Per-token persistence would throttle generation; write on sentence/stop/5s intervals |
| @MainActor @Observable for Bridge | UI-bound facade; Sendable belongs on runtime handle/actor underneath |
| PromptCompiler in ModelPack | Chat template is model semantics, not ViewModel responsibility |
| Background URLSession for downloads | Survives app suspend; documented limitation: force-quit cancels transfer |

---

## 7. Distribution

MVP targets **sideload / TestFlight** only. App Store submission is a future consideration — the post-install multi-GB model download may require App Store review guidance. App binary itself (C engine + Metal shaders + SwiftUI) is expected to be < 20 MB before model download.

---

## 8. External References

- [Flash-MoE iOS-App branch](https://github.com/Anemll/flash-moe/tree/iOS-App) — existing iOS port, unity build C/Metal engine
- [Flash-MoE iOS Port doc](https://github.com/Anemll/flash-moe/blob/iOS-App/FlashMoE-iOS/IOS_PORT.md) — problems solved, architecture decisions
- [Flash-MoE Review](https://github.com/Anemll/flash-moe/blob/iOS-App/FlashMoE-iOS/REVIEW.md) — engineering review, known issues, roadmap
- [Google TurboQuant blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — 2026-03-24, random rotation + 1-bit QJL residual
- [mlx-vlm TurboQuant PR #858](https://github.com/Blaizzy/mlx-vlm/pull/858) — Draft as of 2026-03-25, Metal kernels + correctness tests exist
- [Qwen3.5 Small Models](https://github.com/QwenLM/Qwen3.5) — 0.8B/2B/4B/9B on-device models
- [os_proc_available_memory](https://developer.apple.com/documentation/os/3191911-os_proc_available_memory) — instantaneous dirty limit advice
- [URLSession background](https://developer.apple.com/documentation/foundation/urlsessionconfiguration/background(withidentifier:)) — background download lifecycle
- [iPhone 16 Pro tech specs](https://support.apple.com/en-sg/121031) — A18 Pro; RAM (~8GB) from teardown reports, not officially published by Apple
