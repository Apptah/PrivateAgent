import Foundation

public struct ChatMessage: Sendable {
    public let role: String    // "system", "user", "assistant"
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public enum PromptCompiler {
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

    public static func compileContinuation(userMessage: String) -> String {
        var result = ""
        result += "<|im_end|>\n"
        result += "<|im_start|>user\n"
        result += userMessage
        result += "<|im_end|>\n"
        result += "<|im_start|>assistant\n"
        return result
    }

    public static let defaultSystemPrompt = "You are a helpful assistant."
}
