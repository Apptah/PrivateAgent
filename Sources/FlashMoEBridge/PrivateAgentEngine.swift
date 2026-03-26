import Foundation
import Observation
import FlashMoECore
import FlashMoERuntime
import ModelPack

// MARK: - EngineState

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

// MARK: - GenerationStats

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

// MARK: - GenerationEvent

/// Events streamed during generation.
/// Terminal errors go through AsyncThrowingStream throw, NOT via an .error case.
public enum GenerationEvent: Sendable {
    case prefillProgress(tokens: Int, total: Int)
    case token(text: String, id: Int)
    case thinkingStart
    case thinkingEnd
    case contextExhausted(policy: String)
    case throttled(reason: String)
    case finished(stats: GenerationStats)
}

// MARK: - PromptInput

/// Prompt input variants.
public enum PromptInput: Sendable {
    case formattedPrompt(String)
    case tokenIDs([Int32])
}

// MARK: - ModelInfo

/// Model metadata exposed to UI after a successful load.
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

// MARK: - GenerationConfig

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

// MARK: - EngineError

public enum EngineError: LocalizedError, Sendable {
    case busy
    case destroyed
    case initFailed
    case loadFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .busy:                       return "Engine is busy"
        case .destroyed:                  return "Engine was destroyed"
        case .initFailed:                 return "Failed to initialize engine"
        case .loadFailed(let msg):        return "Load failed: \(msg)"
        case .generationFailed(let msg):  return "Generation failed: \(msg)"
        }
    }
}

// MARK: - PrivateAgentEngine

/// Main engine facade — @MainActor for SwiftUI binding.
@MainActor
@Observable
public final class PrivateAgentEngine {
    public private(set) var state: EngineState = .idle
    public private(set) var modelInfo: ModelInfo?
    public private(set) var lastError: String?

    // @ObservationIgnored + nonisolated(unsafe): session is an opaque C pointer,
    // not tracked by @Observable. All off-actor access is serialized via engineQueue.
    @ObservationIgnored
    nonisolated(unsafe) private var session: OpaquePointer?  // PA_Session*
    private let engineQueue = DispatchQueue(label: "com.privateagent.engine", qos: .userInitiated)

    public init() {}

    deinit {
        if let s = session {
            pa_session_destroy(s)
        }
    }

    public func loadModel(from manifest: ModelManifest, availableMemory: UInt64 = 0) async throws {
        guard state == .idle || isErrorState else {
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

                var s = self.session
                if s == nil {
                    s = pa_session_create()
                    guard s != nil else {
                        DispatchQueue.main.async {
                            self.state = .error("Failed to create session")
                            self.lastError = "Failed to create session"
                        }
                        continuation.resume(throwing: EngineError.initFailed)
                        return
                    }
                }

                let mem = availableMemory > 0 ? availableMemory : 6 * 1024 * 1024 * 1024

                var mutableDesc = desc
                let result = pa_session_load_model(s, &mutableDesc, mem)

                if result != Int32(PA_STATUS_OK.rawValue) {
                    let errorMsg = String(cString: pa_session_last_error(s))
                    DispatchQueue.main.async {
                        self.state = .error(errorMsg)
                        self.lastError = errorMsg
                        self.session = s
                    }
                    continuation.resume(throwing: EngineError.loadFailed(errorMsg))
                    return
                }

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
                    totalDirtyMB: Double(budget.total_dirty_bytes) / (1024.0 * 1024.0),
                    totalResidentMB: Double(budget.total_resident_bytes) / (1024.0 * 1024.0),
                    kvCacheMB: Double(budget.kv_cache_bytes) / (1024.0 * 1024.0)
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

    public func unloadModel() {
        if let s = session {
            engineQueue.sync {
                pa_session_unload_model(s)
            }
        }
        modelInfo = nil
        state = .idle
    }

    /// Stream tokens from the model for the given prompt.
    /// The returned `AsyncThrowingStream` emits `.token`, `.prefillProgress`, etc.
    /// and terminates with `.finished` on success or a thrown `EngineError` on failure.
    public func generate(
        _ input: PromptInput,
        config: GenerationConfig = .default
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            // Must be called on @MainActor, so state access is safe here.
            guard self.state == .ready else {
                continuation.finish(throwing: EngineError.busy)
                return
            }
            guard let s = self.session else {
                continuation.finish(throwing: EngineError.initFailed)
                return
            }
            self.state = .generating

            let prompt: String
            switch input {
            case .formattedPrompt(let text):
                prompt = text
            case .tokenIDs(let ids):
                // Encode as space-separated decimal IDs; runtime handles token-id prompts.
                prompt = ids.map { String($0) }.joined(separator: " ")
            }

            let cConfig = PA_GenerationConfig(
                max_tokens:   Int32(config.maxTokens),
                temperature:  config.temperature,
                top_p:        config.topP,
                think_budget: Int32(config.thinkBudget)
            )

            // Bridge: retain the continuation context so the C callback can reach it.
            final class CallbackContext: @unchecked Sendable {
                let continuation: AsyncThrowingStream<GenerationEvent, Error>.Continuation
                init(_ c: AsyncThrowingStream<GenerationEvent, Error>.Continuation) {
                    self.continuation = c
                }
            }
            let ctx = CallbackContext(continuation)
            let rawCtx = Unmanaged.passRetained(ctx).toOpaque()

            // Register cancellation *before* dispatching work.
            continuation.onTermination = { [weak self] _ in
                guard let self else { return }
                if let s = self.session {
                    pa_session_cancel(s)
                }
            }

            self.engineQueue.async { [weak self] in
                let tokenCallback: PA_TokenCallback = { tokenText, tokenID, _, tps, userData in
                    let ctx = Unmanaged<CallbackContext>.fromOpaque(userData!).takeUnretainedValue()
                    let text = tokenText.map { String(cString: $0) } ?? ""
                    ctx.continuation.yield(.token(text: text, id: Int(tokenID)))
                    return 0  // 0 = continue
                }

                var mutableConfig = cConfig
                let result = pa_session_generate(s, prompt, &mutableConfig, tokenCallback, rawCtx)

                // Collect stats regardless of result.
                var rawStats = PA_GenerationStats()
                pa_session_get_gen_stats(s, &rawStats)
                let stats = GenerationStats(
                    tokensPerSecond: rawStats.tokens_per_second,
                    tokensGenerated: Int(rawStats.tokens_generated),
                    ttftMs: rawStats.ttft_ms
                )

                // Release the retained context now that the C side is done.
                Unmanaged<CallbackContext>.fromOpaque(rawCtx).release()

                // Capture error string on engine queue before switching to main
                let errorMsg: String? = (result != Int32(PA_STATUS_OK.rawValue))
                    ? String(cString: pa_session_last_error(s))
                    : nil

                DispatchQueue.main.async {
                    guard let self else { return }
                    if let errorMsg {
                        self.state = .error(errorMsg)
                        self.lastError = errorMsg
                        continuation.finish(throwing: EngineError.generationFailed(errorMsg))
                    } else {
                        self.state = .ready
                        continuation.yield(.finished(stats: stats))
                        continuation.finish()
                    }
                }
            }
        }
    }

    /// Cancel any in-progress generation.
    public func cancel() {
        guard let s = session else { return }
        engineQueue.async {
            pa_session_cancel(s)
        }
    }

    /// Reset multi-turn conversation state (clears KV-cache history).
    public func resetConversation() {
        guard let s = session else { return }
        engineQueue.sync {
            pa_session_reset(s)
        }
    }

    private var isErrorState: Bool {
        if case .error = state { return true }
        return false
    }
}
