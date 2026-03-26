import Testing
import Foundation
@testable import ModelHub

@Suite("ModelStorage Tests")
struct ModelStorageTests {

    @Test("List models in empty storage")
    func emptyStorage() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("PATest-\(UUID().uuidString)")
        let storage = ModelStorage(storageDir: dir)
        let models = try await storage.listModels()
        #expect(models.isEmpty)
        try? FileManager.default.removeItem(at: dir)
    }

    @Test("Model directory path")
    func modelDir() async {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("PATest-\(UUID().uuidString)")
        let storage = ModelStorage(storageDir: dir)
        let modelDir = await storage.modelDir(for: "test-model")
        #expect(modelDir.lastPathComponent == "test-model")
        try? FileManager.default.removeItem(at: dir)
    }

    @Test("Available space is positive")
    func availableSpace() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("PATest-\(UUID().uuidString)")
        let storage = ModelStorage(storageDir: dir)
        let space = try await storage.availableSpaceBytes()
        #expect(space > 0)
        try? FileManager.default.removeItem(at: dir)
    }

    @Test("SHA-256 verification")
    func sha256Verify() throws {
        let tmpFile = FileManager.default.temporaryDirectory.appendingPathComponent("sha256test-\(UUID().uuidString).txt")
        let testData = "Hello, World!\n".data(using: .utf8)!
        try testData.write(to: tmpFile)
        defer { try? FileManager.default.removeItem(at: tmpFile) }

        // SHA-256 of "Hello, World!\n"
        let expected = "c98c24b677eff44860afea6f493bbaec5bb1c4cbb209c6fc2bbb47f66ff2ad31"
        #expect(try DownloadManager.verifySHA256(fileURL: tmpFile, expected: expected))
        #expect(try !DownloadManager.verifySHA256(fileURL: tmpFile, expected: "0000"))
    }

    @Test("Empty expected hash skips verification")
    func sha256EmptySkips() throws {
        let tmpFile = FileManager.default.temporaryDirectory.appendingPathComponent("sha256empty-\(UUID().uuidString).txt")
        try "any content".data(using: .utf8)!.write(to: tmpFile)
        defer { try? FileManager.default.removeItem(at: tmpFile) }

        // Empty expected string should return true (no checksum to verify)
        #expect(try DownloadManager.verifySHA256(fileURL: tmpFile, expected: ""))
    }
}
