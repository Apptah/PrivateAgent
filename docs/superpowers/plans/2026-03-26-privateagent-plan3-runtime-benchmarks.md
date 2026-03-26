# PrivateAgent Plan 3: FlashMoERuntime Skeleton + Benchmark Harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend FlashMoERuntime with generation API (token callback, prefill/decode lifecycle), add benchmark harness for measuring performance, and integrate with PrivateAgentEngine's async generation stream.

**Architecture:** Runtime exposes C generation API (pa_session_generate) with token callbacks. Bridge wraps this in AsyncThrowingStream. Benchmark harness measures tok/s, TTFT, memory peak. No actual Metal inference yet — generation returns mock tokens for pipeline validation.

**Tech Stack:** Swift 6.0, C17, Foundation, Swift Testing

**Spec:** `docs/superpowers/specs/2026-03-26-privateagent-design.md`

**Depends on:** Plan 2 (complete)

**Produces:** Full generation pipeline skeleton (prompt → prefill → decode → token stream), benchmark CLI, and integration tests proving the end-to-end load → generate → stream flow works.

---

### Task 1: Add generation API to FlashMoERuntime

**Files:**
- Modify: `Sources/FlashMoERuntime/include/FlashMoERuntime.h`
- Modify: `Sources/FlashMoERuntime/pa_runtime.c`

- [ ] **Step 1: Add generation types and functions to FlashMoERuntime.h**

Add after existing declarations, before closing `}` and `#endif`:

```c
/// Token callback. Return 0 to continue, non-zero to cancel.
typedef int (*PA_TokenCallback)(
    const char *token_text,
    int32_t token_id,
    int32_t tokens_generated,
    double tokens_per_second,
    void *user_data
);

/// Generation configuration.
typedef struct {
    int32_t max_tokens;
    float temperature;
    float top_p;
    int32_t think_budget;       // max thinking tokens (0 = unlimited)
} PA_GenerationConfig;

/// Generation statistics (filled after generation completes).
typedef struct {
    double tokens_per_second;
    int32_t tokens_generated;
    double total_time_ms;
    double ttft_ms;             // time to first token
    double prefill_ms;
    int32_t prefill_tokens;
    double prefill_tps;
    uint64_t peak_memory_bytes; // from task_vm_info.phys_footprint
} PA_GenerationStats;

/// Generate tokens from a prompt. Blocks until complete or cancelled.
/// Returns number of tokens generated, or negative PA_Status on error.
int pa_session_generate(
    PA_Session *session,
    const char *prompt,
    const PA_GenerationConfig *config,
    PA_TokenCallback callback,
    void *user_data
);

/// Generate continuation — reuses KV cache from previous turns.
/// Returns tokens generated, -2 (PA_STATUS_CONTEXT_EXHAUSTED) if context full.
int pa_session_generate_continuation(
    PA_Session *session,
    const char *user_message,
    const PA_GenerationConfig *config,
    PA_TokenCallback callback,
    void *user_data
);

/// Cancel in-progress generation. Thread-safe.
void pa_session_cancel(PA_Session *session);

/// Reset conversation state (KV cache, position).
void pa_session_reset(PA_Session *session);

/// Get generation stats from last completed generation.
int pa_session_get_gen_stats(const PA_Session *session, PA_GenerationStats *out_stats);

/// Get current turn count (0 = no history).
int32_t pa_session_turn_count(const PA_Session *session);
```

- [ ] **Step 2: Implement mock generation in pa_runtime.c**

Add fields to PA_Session struct:
```c
struct PA_Session {
    // ... existing fields ...
    int cancelled;
    int32_t turn_count;
    PA_GenerationStats last_gen_stats;
};
```

Implement pa_session_generate with mock token generation (returns "Hello world" tokens):
```c
#include <time.h>
#include <unistd.h>

int pa_session_generate(
    PA_Session *session,
    const char *prompt,
    const PA_GenerationConfig *config,
    PA_TokenCallback callback,
    void *user_data
) {
    if (!session || !prompt || !config) return PA_STATUS_ERROR_GENERIC;
    if (!session->model_loaded) {
        snprintf(session->last_error, sizeof(session->last_error), "No model loaded");
        return PA_STATUS_ERROR_GENERIC;
    }

    session->state = PA_SESSION_PREFILL;
    session->cancelled = 0;

    // Mock prefill delay
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    usleep(50000); // 50ms simulated prefill

    session->state = PA_SESSION_DECODE;

    // Mock tokens
    const char *mock_tokens[] = {
        "Hello", " ", "!", " ", "I", "'m", " Private", "Agent", ",",
        " running", " locally", " on", " your", " device", ".", NULL
    };

    int32_t max = config->max_tokens > 0 ? config->max_tokens : 100;
    int32_t generated = 0;
    struct timespec decode_start;
    clock_gettime(CLOCK_MONOTONIC, &decode_start);

    for (int i = 0; mock_tokens[i] && generated < max; i++) {
        if (session->cancelled) {
            session->state = PA_SESSION_CANCELLED;
            break;
        }

        generated++;
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - decode_start.tv_sec) +
            (now.tv_nsec - decode_start.tv_nsec) / 1e9;
        double tps = elapsed > 0 ? generated / elapsed : 0;

        if (callback) {
            int ret = callback(mock_tokens[i], generated, generated, tps, user_data);
            if (ret != 0) {
                session->cancelled = 1;
                session->state = PA_SESSION_CANCELLED;
                break;
            }
        }
        usleep(10000); // 10ms per token (~100 tok/s mock)
    }

    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double decode_time = (end.tv_sec - decode_start.tv_sec) +
        (end.tv_nsec - decode_start.tv_nsec) / 1e9;

    session->last_gen_stats.tokens_generated = generated;
    session->last_gen_stats.total_time_ms = total * 1000;
    session->last_gen_stats.ttft_ms = 50.0; // mock prefill
    session->last_gen_stats.tokens_per_second = decode_time > 0 ? generated / decode_time : 0;
    session->last_gen_stats.prefill_ms = 50.0;
    session->last_gen_stats.prefill_tokens = 0;
    session->last_gen_stats.prefill_tps = 0;

    session->turn_count++;
    if (session->state != PA_SESSION_CANCELLED) {
        session->state = PA_SESSION_DONE;
    }
    return generated;
}

int pa_session_generate_continuation(
    PA_Session *session,
    const char *user_message,
    const PA_GenerationConfig *config,
    PA_TokenCallback callback,
    void *user_data
) {
    // For now, just delegate to full generate
    return pa_session_generate(session, user_message, config, callback, user_data);
}

void pa_session_cancel(PA_Session *session) {
    if (session) session->cancelled = 1;
}

void pa_session_reset(PA_Session *session) {
    if (!session) return;
    session->turn_count = 0;
    session->state = PA_SESSION_IDLE;
    memset(&session->last_gen_stats, 0, sizeof(PA_GenerationStats));
}

int pa_session_get_gen_stats(const PA_Session *session, PA_GenerationStats *out_stats) {
    if (!session || !out_stats) return PA_STATUS_ERROR_GENERIC;
    memcpy(out_stats, &session->last_gen_stats, sizeof(PA_GenerationStats));
    return PA_STATUS_OK;
}

int32_t pa_session_turn_count(const PA_Session *session) {
    return session ? session->turn_count : 0;
}
```

- [ ] **Step 3: Build and verify**

```bash
swift build 2>&1 | tail -5
swift test 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add Sources/FlashMoERuntime/
git commit -m "feat: add generation API to FlashMoERuntime with mock token output"
```

---

### Task 2: Wire generation into PrivateAgentEngine

**Files:**
- Modify: `Sources/FlashMoEBridge/PrivateAgentEngine.swift`

- [ ] **Step 1: Add generate method to PrivateAgentEngine**

Add to the class:

```swift
/// Generate tokens from a prompt, returning an async stream of events.
public func generate(input: PromptInput, config: GenerationConfig = .default) -> AsyncThrowingStream<GenerationEvent, Error> {
    AsyncThrowingStream { continuation in
        guard state == .ready else {
            continuation.finish(throwing: EngineError.busy)
            return
        }

        state = .generating

        let prompt: String
        switch input {
        case .formattedPrompt(let p): prompt = p
        case .tokenIDs: prompt = "" // TODO: token ID input
        }

        let cConfig = PA_GenerationConfig(
            max_tokens: Int32(config.maxTokens),
            temperature: config.temperature,
            top_p: config.topP,
            think_budget: Int32(config.thinkBudget)
        )

        // Bridge context for C callback
        final class CallbackContext: @unchecked Sendable {
            let continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
            init(_ c: AsyncThrowingStream<GenerationEvent, Error>.Continuation) {
                self.continuation = c
            }
        }
        let ctx = CallbackContext(continuation)
        let ctxPtr = Unmanaged.passRetained(ctx).toOpaque()

        continuation.onTermination = { [weak self] _ in
            if let s = self?.session {
                pa_session_cancel(s)
            }
        }

        engineQueue.async { [weak self] in
            guard let self, let s = self.session else {
                Unmanaged<CallbackContext>.fromOpaque(ctxPtr).release()
                continuation.finish(throwing: EngineError.destroyed)
                return
            }

            var mutableConfig = cConfig
            let result = pa_session_generate(
                s, prompt, &mutableConfig,
                { text, tokenId, tokensGenerated, tps, userData -> Int32 in
                    guard let userData, let text else { return 1 }
                    let ctx = Unmanaged<CallbackContext>.fromOpaque(userData)
                        .takeUnretainedValue()
                    let token = String(cString: text)
                    ctx.continuation.yield(.token(text: token, id: Int(tokenId)))
                    return 0
                },
                ctxPtr
            )

            Unmanaged<CallbackContext>.fromOpaque(ctxPtr).release()

            var stats = PA_GenerationStats()
            pa_session_get_gen_stats(s, &stats)

            let genStats = GenerationStats(
                tokensPerSecond: stats.tokens_per_second,
                tokensGenerated: Int(stats.tokens_generated),
                ttftMs: stats.ttft_ms
            )

            continuation.yield(.finished(stats: genStats))
            continuation.finish()

            DispatchQueue.main.async {
                self.state = .ready
            }
        }
    }
}

/// Cancel in-progress generation.
public func cancel() {
    if let s = session {
        pa_session_cancel(s)
    }
}

/// Reset conversation state.
public func resetConversation() {
    if let s = session {
        engineQueue.sync {
            pa_session_reset(s)
        }
    }
}
```

Also add GenerationConfig:

```swift
public struct GenerationConfig: Sendable {
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var thinkBudget: Int

    public static let `default` = GenerationConfig(
        maxTokens: 2048, temperature: 0.7, topP: 0.9, thinkBudget: 0
    )

    public init(maxTokens: Int = 2048, temperature: Float = 0.7, topP: Float = 0.9, thinkBudget: Int = 0) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.thinkBudget = thinkBudget
    }
}
```

- [ ] **Step 2: Build and verify**

```bash
swift build 2>&1 | tail -5
swift test 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add Sources/FlashMoEBridge/
git commit -m "feat: wire async generation stream into PrivateAgentEngine"
```

---

### Task 3: Create benchmark harness

**Files:**
- Create: `Benchmarks/end-to-end/BenchmarkRunner.swift`
- Create: `Benchmarks/end-to-end/Package.swift` (standalone executable)

- [ ] **Step 1: Create benchmark Package.swift**

```swift
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
```

- [ ] **Step 2: Create BenchmarkRunner.swift**

```swift
import Foundation
import FlashMoEBridge
import ModelPack

@main
struct BenchmarkRunner {
    static func main() async throws {
        print("=== PrivateAgent End-to-End Benchmark ===")
        print("Date: \(ISO8601DateFormatter().string(from: Date()))")
        print()

        // Check for model directory argument
        let modelDir: URL
        if CommandLine.arguments.count > 1 {
            modelDir = URL(fileURLWithPath: CommandLine.arguments[1])
        } else {
            // Use test fixtures for mock benchmark
            let fixturesPath = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Tests/ModelPackTests/Fixtures")
            modelDir = fixturesPath
        }

        print("Model dir: \(modelDir.path)")

        // Parse manifest
        let manifest: ModelManifest
        do {
            manifest = try ModelManifest(modelDir: modelDir)
            print("Model: \(manifest.hfConfig.numHiddenLayers) layers, \(manifest.hfConfig.vocabSize) vocab")
        } catch {
            print("ERROR: Failed to parse manifest: \(error)")
            return
        }

        // Load model
        let engine = await PrivateAgentEngine()
        do {
            try await engine.loadModel(from: manifest)
            if let info = await engine.modelInfo {
                print("Loaded: \(info.name)")
                print("  Max context: \(info.maxContext)")
                print("  Dirty memory: \(String(format: "%.1f", info.totalDirtyMB)) MB")
                print("  Resident memory: \(String(format: "%.1f", info.totalResidentMB)) MB")
                print("  KV cache: \(String(format: "%.1f", info.kvCacheMB)) MB")
            }
        } catch {
            print("ERROR: Failed to load: \(error)")
            return
        }

        // Generate
        print()
        print("--- Generation (mock) ---")
        let input = PromptInput.formattedPrompt("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n")

        var tokenCount = 0
        var allTokens = ""
        let start = ContinuousClock.now

        let stream = await engine.generate(input: input)
        do {
            for try await event in stream {
                switch event {
                case .token(let text, _):
                    tokenCount += 1
                    allTokens += text
                    print(text, terminator: "")
                case .finished(let stats):
                    print()
                    print()
                    print("--- Results ---")
                    print("Tokens: \(stats.tokensGenerated)")
                    print("Speed: \(String(format: "%.1f", stats.tokensPerSecond)) tok/s")
                    print("TTFT: \(String(format: "%.1f", stats.ttftMs)) ms")
                default:
                    break
                }
            }
        } catch {
            print("\nERROR during generation: \(error)")
        }

        let elapsed = ContinuousClock.now - start
        print("Wall time: \(elapsed)")
        print()

        // Output JSON for historical comparison
        let json: [String: Any] = [
            "date": ISO8601DateFormatter().string(from: Date()),
            "tokens": tokenCount,
            "wall_time_ms": elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000,
            "model_dir": modelDir.path,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: json, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            print("JSON output:")
            print(str)
        }

        await engine.unloadModel()
        print("Done.")
    }
}
```

- [ ] **Step 3: Build and run benchmark**

```bash
cd ~/Developer/PrivateAgent/Benchmarks/end-to-end
swift run EndToEndBench 2>&1
```

- [ ] **Step 4: Commit**

```bash
cd ~/Developer/PrivateAgent
rm Benchmarks/.gitkeep
git add Benchmarks/
git commit -m "feat: add end-to-end benchmark harness with mock generation"
```

---

### Task 4: Add integration test for full pipeline

**Files:**
- Create: `Tests/FlashMoEBridgeTests/IntegrationTests.swift`
- Modify: `Package.swift` (add FlashMoEBridgeTests target)

- [ ] **Step 1: Add test target to Package.swift**

```swift
.testTarget(
    name: "FlashMoEBridgeTests",
    dependencies: ["FlashMoEBridge", "ModelPack"],
    resources: [.copy("Fixtures")]
),
```

- [ ] **Step 2: Create test fixtures (symlink or copy)**

```bash
mkdir -p Tests/FlashMoEBridgeTests/Fixtures
cp Tests/ModelPackTests/Fixtures/config.json Tests/FlashMoEBridgeTests/Fixtures/
cp Tests/ModelPackTests/Fixtures/privateagent-manifest.json Tests/FlashMoEBridgeTests/Fixtures/
```

- [ ] **Step 3: Create IntegrationTests.swift**

```swift
import Testing
import Foundation
@testable import FlashMoEBridge
@testable import ModelPack

@Suite("Integration Tests")
struct IntegrationTests {

    private var fixturesDir: URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures")
    }

    @Test("Full load → generate → stream pipeline")
    @MainActor
    func fullPipeline() async throws {
        let manifest = try ModelManifest(modelDir: fixturesDir)
        let engine = PrivateAgentEngine()

        // Load
        try await engine.loadModel(from: manifest)
        #expect(engine.state == .ready)
        #expect(engine.modelInfo != nil)
        #expect(engine.modelInfo?.maxContext ?? 0 > 0)

        // Generate
        let input = PromptInput.formattedPrompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        var tokens: [String] = []
        var finished = false

        let stream = engine.generate(input: input)
        for try await event in stream {
            switch event {
            case .token(let text, _):
                tokens.append(text)
            case .finished(let stats):
                finished = true
                #expect(stats.tokensGenerated > 0)
                #expect(stats.tokensPerSecond > 0)
            default:
                break
            }
        }

        #expect(tokens.count > 0)
        #expect(finished)

        // Unload
        engine.unloadModel()
        #expect(engine.state == .idle)
        #expect(engine.modelInfo == nil)
    }

    @Test("Cancel during generation")
    @MainActor
    func cancelGeneration() async throws {
        let manifest = try ModelManifest(modelDir: fixturesDir)
        let engine = PrivateAgentEngine()
        try await engine.loadModel(from: manifest)

        let input = PromptInput.formattedPrompt("test")
        let stream = engine.generate(input: input, config: GenerationConfig(maxTokens: 1000))

        var tokenCount = 0
        for try await event in stream {
            if case .token = event {
                tokenCount += 1
                if tokenCount >= 3 {
                    engine.cancel()
                    break
                }
            }
        }

        #expect(tokenCount >= 3)
        engine.unloadModel()
    }
}
```

- [ ] **Step 4: Run all tests**

```bash
cd ~/Developer/PrivateAgent
swift test 2>&1 | tail -15
```

- [ ] **Step 5: Commit**

```bash
git add Tests/FlashMoEBridgeTests/ Package.swift
git commit -m "test: add integration tests for full load → generate → stream pipeline"
```

---

## Summary

After Plan 3:

- FlashMoERuntime has generation API (pa_session_generate, cancel, reset, stats)
- PrivateAgentEngine exposes AsyncThrowingStream<GenerationEvent> for SwiftUI
- Benchmark harness in Benchmarks/end-to-end/
- Integration tests proving load → generate → stream → unload works
- Mock token generation (real inference engine port is a separate effort)

**Next:** Plan 4 (ModelHub background download) and Plan 5 (TurboQuant) can proceed in parallel.
