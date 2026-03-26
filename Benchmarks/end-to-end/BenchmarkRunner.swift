import Foundation
import FlashMoEBridge
import ModelPack

// MARK: - JSON Summary

struct BenchmarkSummary: Codable {
    let modelName: String
    let numLayers: Int
    let vocabSize: Int
    let maxContext: Int
    let totalResidentMB: Double
    let kvCacheMB: Double
    let prompt: String
    let tokensGenerated: Int
    let tokensPerSecond: Double
    let ttftMs: Double
    let wallTimeMs: Double
    let output: String
    let timestamp: String
}

// MARK: - Main

@main
struct BenchmarkRunner {
    @MainActor
    static func main() async {
        let args = CommandLine.arguments
        // args[0] = executable path; args[1] (optional) = model directory
        let fixturesRelative = "Tests/ModelPackTests/Fixtures"

        let modelDir: URL
        if args.count > 1 {
            modelDir = URL(fileURLWithPath: args[1])
        } else {
            // Resolve relative to the repo root (two levels up from Benchmarks/end-to-end)
            let repoRoot = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()   // end-to-end/
                .deletingLastPathComponent()   // Benchmarks/
                .deletingLastPathComponent()   // repo root
            modelDir = repoRoot.appendingPathComponent(fixturesRelative)
        }

        print("=== PrivateAgent End-to-End Benchmark ===")
        print("Model directory: \(modelDir.path)")

        // ── 1. Parse manifest ──────────────────────────────────────────────
        let manifest: ModelManifest
        do {
            manifest = try ModelManifest(modelDir: modelDir)
        } catch {
            fputs("ERROR: Failed to parse ModelManifest: \(error)\n", stderr)
            exit(1)
        }

        let cfg = manifest.hfConfig
        print("\nManifest parsed:")
        print("  hidden_size       : \(cfg.hiddenSize)")
        print("  num_hidden_layers : \(cfg.numHiddenLayers)")
        print("  vocab_size        : \(cfg.vocabSize)")
        print("  max_position_emb  : \(cfg.maxPositionEmbeddings ?? 0)")
        if let experts = cfg.numExperts {
            print("  num_experts       : \(experts)")
            print("  experts_per_tok   : \(cfg.numExpertsPerTok ?? 0)")
        }

        // ── 2. Create engine & load model ──────────────────────────────────
        let engine = PrivateAgentEngine()
        print("\nLoading model…")
        let loadStart = ContinuousClock.now

        do {
            try await engine.loadModel(from: manifest)
        } catch {
            fputs("ERROR: loadModel failed: \(error)\n", stderr)
            exit(1)
        }

        // loadModel resumes the continuation before the DispatchQueue.main.async
        // block that sets modelInfo fires. One yield drains that pending work.
        await Task.yield()

        let loadElapsed = ContinuousClock.now - loadStart
        print("Model loaded in \(String(format: "%.1f", Double(loadElapsed.components.seconds) + Double(loadElapsed.components.attoseconds) * 1e-18))s")

        // ── 3. Print model info ────────────────────────────────────────────
        guard let info = engine.modelInfo else {
            fputs("ERROR: modelInfo unavailable after load\n", stderr)
            exit(1)
        }

        print("\nModel info:")
        print("  name              : \(info.name)")
        print("  layers            : \(info.numLayers)")
        print("  experts           : \(info.numExperts)  (active k=\(info.activeExpertsK))")
        print("  hidden dim        : \(info.hiddenDim)")
        print("  vocab size        : \(info.vocabSize)")
        print("  max context       : \(info.maxContext) tokens")
        print(String(format: "  resident memory   : %.1f MB", info.totalResidentMB))
        print(String(format: "  dirty memory      : %.1f MB", info.totalDirtyMB))
        print(String(format: "  KV-cache budget   : %.1f MB", info.kvCacheMB))

        // ── 4. Generate ────────────────────────────────────────────────────
        let prompt = "Hello, I am a private on-device language model. My purpose is"
        let genConfig = GenerationConfig(maxTokens: 64, temperature: 0.7, topP: 0.9, thinkBudget: 0)

        print("\nPrompt: \"\(prompt)\"")
        print("Generating (max \(genConfig.maxTokens) tokens)…\n")

        var outputTokens: [String] = []
        var finalStats: GenerationStats?
        let genStart = ContinuousClock.now

        do {
            let stream = engine.generate(.formattedPrompt(prompt), config: genConfig)
            for try await event in stream {
                switch event {
                case .token(let text, _):
                    print(text, terminator: "")
                    outputTokens.append(text)
                case .prefillProgress(let done, let total):
                    if total > 0 {
                        let pct = Int(Double(done) / Double(total) * 100)
                        fputs("\r[prefill \(pct)%] ", stderr)
                    }
                case .thinkingStart:
                    fputs("\n<thinking>\n", stderr)
                case .thinkingEnd:
                    fputs("\n</thinking>\n", stderr)
                case .contextExhausted(let policy):
                    fputs("\n[context exhausted, policy=\(policy)]\n", stderr)
                case .throttled(let reason):
                    fputs("\n[throttled: \(reason)]\n", stderr)
                case .finished(let stats):
                    finalStats = stats
                }
            }
        } catch EngineError.generationFailed(let msg) {
            fputs("\nERROR: generation failed: \(msg)\n", stderr)
            exit(1)
        } catch {
            fputs("\nERROR: \(error)\n", stderr)
            exit(1)
        }

        let genElapsed = ContinuousClock.now - genStart
        let wallMs = (Double(genElapsed.components.seconds) + Double(genElapsed.components.attoseconds) * 1e-18) * 1000.0

        print("\n")

        // ── 5. Print stats ─────────────────────────────────────────────────
        let stats = finalStats ?? GenerationStats()
        print("=== Stats ===")
        print(String(format: "  tokens generated  : %d", stats.tokensGenerated))
        print(String(format: "  tokens/sec        : %.2f", stats.tokensPerSecond))
        print(String(format: "  TTFT              : %.1f ms", stats.ttftMs))
        print(String(format: "  wall time         : %.1f ms", wallMs))

        // ── 6. JSON summary ────────────────────────────────────────────────
        let outputText = outputTokens.joined()
        let iso8601: String = {
            let fmt = ISO8601DateFormatter()
            fmt.formatOptions = [.withInternetDateTime]
            return fmt.string(from: Date())
        }()

        let summary = BenchmarkSummary(
            modelName: info.name,
            numLayers: info.numLayers,
            vocabSize: info.vocabSize,
            maxContext: info.maxContext,
            totalResidentMB: info.totalResidentMB,
            kvCacheMB: info.kvCacheMB,
            prompt: prompt,
            tokensGenerated: stats.tokensGenerated,
            tokensPerSecond: stats.tokensPerSecond,
            ttftMs: stats.ttftMs,
            wallTimeMs: wallMs,
            output: outputText,
            timestamp: iso8601
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let jsonData = try? encoder.encode(summary),
           let jsonStr = String(data: jsonData, encoding: .utf8) {
            print("\n=== JSON Summary ===")
            print(jsonStr)
        }

        print("\nBenchmark complete.")
    }
}
