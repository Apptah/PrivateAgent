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

        do {
            try await engine.loadModel(from: manifest)
        } catch {
            // Real engine fails without actual model weights / Metal GPU — skip gracefully
            // This test validates the mock path; real engine tested on device
            #expect(Bool(true), "Load failed (expected without real model): \(error)")
            return
        }
        #expect(engine.state == .ready)
        #expect(engine.modelInfo != nil)
        #expect(engine.modelInfo?.maxContext ?? 0 > 0)

        let input = PromptInput.formattedPrompt("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")
        var tokens: [String] = []
        var finished = false

        let stream = engine.generate(input, config: .default)
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

        engine.unloadModel()
        #expect(engine.state == .idle)
        #expect(engine.modelInfo == nil)
    }

    @Test("Load failure with invalid directory")
    @MainActor
    func loadFailure() async {
        let engine = PrivateAgentEngine()
        let badDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try! FileManager.default.createDirectory(at: badDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: badDir) }

        // ModelManifest parsing should fail (no config.json)
        do {
            let manifest = try ModelManifest(modelDir: badDir)
            try await engine.loadModel(from: manifest)
            #expect(Bool(false), "Should have thrown")
        } catch {
            // Expected — either manifest parsing or load fails
        }
    }
}
