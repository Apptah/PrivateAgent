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

// MARK: - EngineError

public enum EngineError: LocalizedError, Sendable {
    case busy
    case destroyed
    case initFailed
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .busy:                return "Engine is busy"
        case .destroyed:           return "Engine was destroyed"
        case .initFailed:          return "Failed to initialize engine"
        case .loadFailed(let msg): return "Load failed: \(msg)"
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

    private var isErrorState: Bool {
        if case .error = state { return true }
        return false
    }
}
