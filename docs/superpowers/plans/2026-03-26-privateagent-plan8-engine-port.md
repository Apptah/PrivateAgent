# PrivateAgent Plan 8: Flash-MoE Engine Port

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the flash-moe inference engine (infer.m + shaders.metal) into PrivateAgent, replacing mock token generation with real MoE inference.

**Architecture:** Unity build approach — include flash-moe's infer.m directly into FlashMoERuntime, adapt the C API to wrap PA_Session around FlashMoEContext. Metal shaders compile as part of FlashMoEMetal target. This avoids rewriting 7000 lines and maintains compatibility with upstream flash-moe improvements.

**Tech Stack:** C, Objective-C, Metal Shading Language, Accelerate framework

**Spec:** `docs/superpowers/specs/2026-03-26-privateagent-design.md`

**Depends on:** Plans 1-3 (C ABI + Runtime skeleton), Plan 7 (Xcode project)

**Produces:** Working inference engine that can load a Qwen3.5-35B-A3B model and generate real tokens on Apple Silicon.

**Key decision:** We clone the flash-moe iOS-App branch source files into a `Vendor/flash-moe/` directory within our repo. The unity build includes these files. This lets us track upstream changes while adapting the API.

---

### Task 1: Vendor flash-moe source files

**Files:**
- Create: `Vendor/flash-moe/` directory with extracted source files

- [ ] **Step 1: Clone flash-moe iOS-App branch**

```bash
cd ~/Developer/PrivateAgent
git clone --branch iOS-App --depth 1 https://github.com/Anemll/flash-moe.git /tmp/flash-moe-ios
```

- [ ] **Step 2: Copy required source files**

```bash
mkdir -p Vendor/flash-moe
cp /tmp/flash-moe-ios/metal_infer/infer.m Vendor/flash-moe/
cp /tmp/flash-moe-ios/metal_infer/shaders.metal Vendor/flash-moe/
cp /tmp/flash-moe-ios/metal_infer/tokenizer.h Vendor/flash-moe/
cp /tmp/flash-moe-ios/metal_infer/batched_prefill.h Vendor/flash-moe/
cp /tmp/flash-moe-ios/metal_infer/gguf_iq_shared.h Vendor/flash-moe/
cp /tmp/flash-moe-ios/FlashMoE-iOS/FlashMoEEngine/FlashMoEEngine.h Vendor/flash-moe/
cp /tmp/flash-moe-ios/FlashMoE-iOS/FlashMoEEngine/FlashMoEEngine.m Vendor/flash-moe/
```

- [ ] **Step 3: Add vendoring note**

Create `Vendor/flash-moe/VENDORED.md`:
```markdown
# Vendored Flash-MoE Source

Source: https://github.com/Anemll/flash-moe (iOS-App branch)
Date vendored: 2026-03-26
Commit: (record the actual commit hash)

These files are included via unity build in FlashMoERuntime.
Modifications are tracked in this repo's git history.
```

- [ ] **Step 4: Commit vendored files**

```bash
git add Vendor/
git commit -m "vendor: add flash-moe inference engine source files (iOS-App branch)"
```

---

### Task 2: Create FlashMoERuntime unity build wrapper

**Files:**
- Create: `Sources/FlashMoERuntime/pa_engine_wrapper.m` — unity build that includes infer.m
- Modify: `Sources/FlashMoERuntime/pa_runtime.c` — replace mock generation with real engine calls
- Modify: `Sources/FlashMoERuntime/include/FlashMoERuntime.h` — add engine-specific config

- [ ] **Step 1: Create pa_engine_wrapper.m**

This file uses the unity build pattern to include infer.m:

```objc
// Unity build — includes the entire flash-moe inference engine
// This approach avoids rewriting 7000 lines of C/Metal code.

#define CHAT_MODE 1
#define ACCELERATE_NEW_LAPACK

// Include the vendored engine
#include "../../Vendor/flash-moe/infer.m"
```

NOTE: This will require adjusting header search paths in Package.swift so the engine can find its headers (tokenizer.h, batched_prefill.h, etc.).

- [ ] **Step 2: Update Package.swift for FlashMoERuntime**

Add header search paths and source exclusions:

```swift
.target(
    name: "FlashMoERuntime",
    dependencies: ["FlashMoECore", "FlashMoEMetal", "TurboQuantCore", "TurboQuantMetal"],
    path: "Sources/FlashMoERuntime",
    exclude: [],
    publicHeadersPath: "include",
    cSettings: [
        .headerSearchPath("../../Vendor/flash-moe"),
        .define("CHAT_MODE", to: "1"),
        .define("ACCELERATE_NEW_LAPACK"),
    ],
    linkerSettings: [
        .linkedFramework("Metal"),
        .linkedFramework("Accelerate"),
        .linkedLibrary("compression"),
    ]
),
```

- [ ] **Step 3: Adapt pa_runtime.c to use real engine**

Replace mock generation with calls to FlashMoEEngine C API:
- `pa_session_load_model` → calls `flashmoe_create()` + `flashmoe_load()`
- `pa_session_generate` → calls `flashmoe_generate()` with token callback bridge
- `pa_session_cancel` → calls `flashmoe_cancel()`
- `pa_session_reset` → calls `flashmoe_reset()`
- Keep the PA_Session wrapper struct, store FlashMoEContext* inside it

- [ ] **Step 4: Handle Metal shader loading**

The engine needs shaders.metal compiled. Two approaches:
1. iOS: compile into default.metallib at build time (add to Sources build phase)
2. macOS: runtime compilation from source file

For SPM, Metal shaders need to be in a .metal file within the target's source directory. Copy or symlink shaders.metal.

- [ ] **Step 5: Build (expect many issues — this is iterative)**

```bash
swift build 2>&1 | head -50
```

Fix issues one by one:
- Missing headers → add header search paths
- Duplicate symbols → exclude conflicting files
- Metal compilation → configure shader handling
- Platform-specific code → add #if guards

- [ ] **Step 6: Commit working build**

```bash
git add Sources/ Package.swift
git commit -m "feat: integrate flash-moe inference engine via unity build"
```

---

### Task 3: Wire real engine into PA_Session

**Files:**
- Modify: `Sources/FlashMoERuntime/pa_runtime.c`

- [ ] **Step 1: Replace mock implementation**

The PA_Session struct should now hold a FlashMoEContext* and delegate all operations to it:

```c
struct PA_Session {
    PA_SessionState state;
    PA_ModelDesc model_desc;
    PA_MemoryBudget memory_budget;
    int model_loaded;
    char last_error[512];
    int cancelled;
    int32_t turn_count;
    PA_GenerationStats last_gen_stats;

    // Real engine context
    void *engine_ctx;  // FlashMoEContext*
};
```

- `pa_session_load_model`: create FlashMoEContext, configure FlashMoEConfig from PA_ModelDesc, call flashmoe_load
- `pa_session_generate`: bridge PA_TokenCallback → FlashMoETokenCallback, call flashmoe_generate
- `pa_session_unload_model`: call flashmoe_unload + flashmoe_destroy
- Keep the memory planner computation as-is (it runs before engine load)

- [ ] **Step 2: Test with mock model directory**

The engine will fail to load without real model files, but it should compile and the error handling path should work.

- [ ] **Step 3: Commit**

```bash
git add Sources/FlashMoERuntime/
git commit -m "feat: wire PA_Session to real FlashMoEContext engine"
```

---

### Task 4: Build verification + error handling

- [ ] **Step 1: Verify swift build passes**
- [ ] **Step 2: Verify swift test passes (integration tests now use real engine, will fail on model load — expected)**
- [ ] **Step 3: Update integration tests to handle engine errors gracefully**
- [ ] **Step 4: Commit**

```bash
git add Sources/ Tests/
git commit -m "fix: update tests for real engine integration"
```

---

## Notes

**This plan is inherently iterative.** The unity build of 7000+ lines of C/Metal will produce many compilation issues that need to be resolved one by one. The implementer should:

1. Start with the simplest possible inclusion
2. Fix errors incrementally
3. Use `#ifdef` guards liberally for platform differences
4. Keep the mock generation as a fallback (compile-time flag)

**Expected challenges:**
- infer.m uses static globals — need to ensure they don't conflict with PA_Session
- Metal shader compilation in SPM is not straightforward
- Some flash-moe code assumes macOS (file paths, Metal API differences)
- The 7000-line file may exceed SPM's compilation limits for a single TU

**Fallback:** If unity build proves too complex, switch to extracting only the core functions needed and creating a clean interface. This is more work but more maintainable.

## Summary

After Plan 8: PrivateAgent can load real Qwen3.5-35B-A3B models and generate actual tokens using the Flash-MoE SSD streaming engine. The mock generation is replaced with real inference.
