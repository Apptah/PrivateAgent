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
        #expect(prompt.contains("<|im_start|>assistant\n"))
        // Default: thinking disabled, empty think block appended
        #expect(prompt.hasSuffix("<think>\n</think>\n"))
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
        #expect(prompt.hasSuffix("<think>\n</think>\n"))
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
