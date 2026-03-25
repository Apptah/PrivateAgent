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
    case contextExhausted(policy: String)
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
