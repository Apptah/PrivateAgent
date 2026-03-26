import Foundation
import SwiftData
import Observation
import FlashMoEBridge
import ModelPack
import ModelHub

@MainActor
@Observable
final class ChatViewModel {
    let conversationId: UUID
    private let modelContext: ModelContext
    private let engine: PrivateAgentEngine

    var inputText: String = ""
    var isGenerating: Bool = false
    var streamingText: String = ""
    var currentStats: String = ""

    private(set) var conversation: Conversation?
    private var generationTask: Task<Void, Never>?
    private var lastBatchTime: Date = .distantPast

    init(conversationId: UUID, modelContext: ModelContext, engine: PrivateAgentEngine) {
        self.conversationId = conversationId
        self.modelContext = modelContext
        self.engine = engine
        loadConversation()
    }

    private func loadConversation() {
        let id = conversationId
        let descriptor = FetchDescriptor<Conversation>(
            predicate: #Predicate { $0.id == id }
        )
        conversation = try? modelContext.fetch(descriptor).first
    }

    var sortedMessages: [Message] {
        (conversation?.messages ?? []).sorted { $0.ordinal < $1.ordinal }
    }

    func sendMessage() {
        guard !inputText.isEmpty, !isGenerating else { return }
        guard let conversation else { return }

        // Auto-load model if engine isn't ready
        if engine.state == .idle {
            Task {
                await autoLoadModel()
                // Retry send after load
                if engine.state == .ready {
                    sendMessageInternal(conversation: conversation)
                }
            }
            return
        }
        if engine.state != .ready {
            currentStats = "Engine state: \(engine.state)"
            return
        }

        sendMessageInternal(conversation: conversation)
    }

    private func autoLoadModel() async {
        let storage = ModelStorage()
        let models = (try? await storage.listModels()) ?? []
        guard let modelDir = models.first else {
            currentStats = "No model downloaded. Go to Models to download one."
            return
        }
        currentStats = "Loading model..."
        do {
            let manifest = try ModelManifest(modelDir: modelDir)
            try await engine.loadModel(from: manifest)
            currentStats = "Model loaded!"
        } catch {
            currentStats = "Load failed: \(error.localizedDescription)"
        }
    }

    private func sendMessageInternal(conversation: Conversation) {

        let text = inputText
        inputText = ""

        // 1. Create user message
        let userOrdinal = conversation.messages.count
        let userMessage = Message(role: .user, content: text, ordinal: userOrdinal)
        userMessage.conversation = conversation
        modelContext.insert(userMessage)

        // 2. Auto-generate title from first user message
        if conversation.title == "New Chat" {
            conversation.title = String(text.prefix(50))
        }

        // 3. Create assistant message placeholder
        let assistantOrdinal = conversation.messages.count
        let assistantMessage = Message(role: .assistant, content: "", ordinal: assistantOrdinal)
        assistantMessage.conversation = conversation
        modelContext.insert(assistantMessage)

        try? modelContext.save()

        // 4. Compile prompt from conversation history
        let systemPrompt = conversation.systemPrompt
        let history = sortedMessages
            .filter { $0.id != assistantMessage.id }
            .map { ChatMessage(role: $0.role.rawValue, content: $0.content) }

        var chatMessages: [ChatMessage] = [
            ChatMessage(role: "system", content: systemPrompt)
        ]
        chatMessages.append(contentsOf: history)
        let prompt = PromptCompiler.compile(messages: chatMessages, addGenerationPrompt: true)

        // 5. Stream generation
        isGenerating = true
        streamingText = ""
        currentStats = ""
        lastBatchTime = Date()

        let stream = engine.generate(.formattedPrompt(prompt))

        generationTask = Task { [weak self] in
            guard let self else { return }
            var accumulated = ""
            var inThinking = false
            var tokenCount = 0
            let genStartTime = Date()

            do {
                for try await event in stream {
                    guard !Task.isCancelled else { break }

                    switch event {
                    case .token(let text, _):
                        accumulated += text
                        self.streamingText = accumulated
                        tokenCount += 1

                        // Update live tok/s every token
                        let elapsed = Date().timeIntervalSince(genStartTime)
                        if elapsed > 0.1 {
                            let tps = Double(tokenCount) / elapsed
                            self.currentStats = String(format: "%d tokens • %.1f tok/s", tokenCount, tps)
                        }

                        // Batch-write to SwiftData every 500ms
                        let now = Date()
                        if now.timeIntervalSince(self.lastBatchTime) >= 0.5 {
                            assistantMessage.content = accumulated
                            self.lastBatchTime = now
                        }

                    case .thinkingStart:
                        inThinking = true

                    case .thinkingEnd:
                        inThinking = false

                    case .prefillProgress(let done, let total):
                        if total > 0 {
                            self.currentStats = "Prefill \(done)/\(total)"
                        }

                    case .throttled(let reason):
                        self.currentStats = "Throttled: \(reason)"

                    case .contextExhausted:
                        self.currentStats = "Context full"

                    case .finished(let stats):
                        assistantMessage.content = accumulated
                        let finalTps = stats.tokensPerSecond > 0
                            ? stats.tokensPerSecond
                            : (Date().timeIntervalSince(genStartTime) > 0
                               ? Double(tokenCount) / Date().timeIntervalSince(genStartTime)
                               : 0)
                        if finalTps > 0 {
                            self.currentStats = String(format: "%d tokens • %.1f tok/s • TTFT %.0fms",
                                                       stats.tokensGenerated > 0 ? stats.tokensGenerated : tokenCount,
                                                       finalTps,
                                                       stats.ttftMs)
                        }
                        conversation.updatedAt = Date()
                        try? self.modelContext.save()
                    }
                }
            } catch {
                assistantMessage.content = accumulated.isEmpty
                    ? "Error: \(error.localizedDescription)"
                    : accumulated + "\n\n[Error: \(error.localizedDescription)]"
                try? self.modelContext.save()
            }

            // Final flush
            if assistantMessage.content != accumulated && !accumulated.isEmpty {
                assistantMessage.content = accumulated
                try? self.modelContext.save()
            }

            self.isGenerating = false
            self.streamingText = ""
            _ = inThinking  // suppress unused warning
        }
    }

    func cancel() {
        engine.cancel()
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        streamingText = ""
    }
}
