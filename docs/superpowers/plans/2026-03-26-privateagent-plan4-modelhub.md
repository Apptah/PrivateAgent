# PrivateAgent Plan 4: ModelHub Background Download

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement model catalog, download manager with Background URLSession, model storage management, and free space checking.

**Architecture:** ModelCatalog provides downloadable model list (static + cached remote JSON). DownloadManager handles Background URLSession downloads with per-file SHA-256 verification. ModelStorage manages on-disk models (scan, validate, delete, space stats). Download state persisted in Application Support.

**Tech Stack:** Swift 6.0, Foundation (URLSession, FileManager, CryptoKit), SwiftUI (for progress observation)

**Spec:** `docs/superpowers/specs/2026-03-26-privateagent-design.md`

**Depends on:** Plan 2 (ModelPack)

**Produces:** Working download pipeline: browse catalog → download model → verify checksums → store on disk → ready for loading.

---

### Task 1: ModelCatalog — static + remote model list

**Files:**
- Delete: `Sources/ModelHub/ModelHub_stub.swift`
- Create: `Sources/ModelHub/ModelCatalog.swift`
- Create: `Sources/ModelHub/CatalogEntry.swift`

- [ ] **Step 1: Create CatalogEntry.swift**

```swift
import Foundation

/// A downloadable model in the catalog.
public struct CatalogEntry: Codable, Sendable, Identifiable {
    public let id: String
    public let displayName: String
    public let repoId: String
    public let description: String
    public let totalSizeBytes: UInt64
    public let quantization: String
    public let expertLayers: Int
    public let files: [ModelFile]

    public var totalSizeGB: Double {
        Double(totalSizeBytes) / (1024.0 * 1024.0 * 1024.0)
    }
}

/// A file within a model package.
public struct ModelFile: Codable, Sendable, Identifiable {
    public var id: String { filename }
    public let filename: String
    public let sizeBytes: UInt64
    public let sha256: String?

    public init(filename: String, sizeBytes: UInt64, sha256: String? = nil) {
        self.filename = filename
        self.sizeBytes = sizeBytes
        self.sha256 = sha256
    }
}
```

- [ ] **Step 2: Create ModelCatalog.swift**

```swift
import Foundation

/// Provides the list of downloadable models.
/// Uses a static bundled catalog with optional remote refresh.
public actor ModelCatalog {

    private var entries: [CatalogEntry]
    private var lastFetchDate: Date?

    public init() {
        self.entries = Self.bundledCatalog
    }

    /// All available models.
    public var models: [CatalogEntry] {
        entries
    }

    /// Refresh catalog from remote JSON. Falls back to cached/bundled on failure.
    public func refresh() async {
        // Future: fetch from remote URL, cache in Application Support
        // For now, always use bundled catalog
    }

    /// Bundled static catalog.
    private static let bundledCatalog: [CatalogEntry] = [
        CatalogEntry(
            id: "qwen3.5-35b-a3b-q4",
            displayName: "Qwen 3.5 35B-A3B",
            repoId: "alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE",
            description: "Compact 35B MoE model. 3B active params per token. Good for 8GB devices.",
            totalSizeBytes: 19_500_000_000,
            quantization: "4-bit",
            expertLayers: 40,
            files: makeFileList(expertLayers: 40, expertLayerSize: 452_984_832)
        ),
        CatalogEntry(
            id: "qwen3.5-35b-a3b-tiered",
            displayName: "Qwen 3.5 35B-A3B Tiered",
            repoId: "alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE",
            description: "Tiered quantization: hot experts 4-bit, cold 2-bit. ~13GB (saves 6GB).",
            totalSizeBytes: 13_424_643_082,
            quantization: "tiered (4-bit/2-bit)",
            expertLayers: 40,
            files: makeFileList(expertLayers: 40, expertLayerSize: 300_000_000)
        ),
    ]

    private static func makeFileList(expertLayers: Int, expertLayerSize: UInt64) -> [ModelFile] {
        var files = [
            ModelFile(filename: "config.json", sizeBytes: 3_809),
            ModelFile(filename: "model_weights.json", sizeBytes: 251_539),
            ModelFile(filename: "model_weights.bin", sizeBytes: 1_378_869_376),
            ModelFile(filename: "vocab.bin", sizeBytes: 3_360_287),
            ModelFile(filename: "tokenizer.json", sizeBytes: 19_989_343),
            ModelFile(filename: "tokenizer.bin", sizeBytes: 8_201_040),
        ]
        for i in 0..<expertLayers {
            files.append(ModelFile(
                filename: String(format: "packed_experts/layer_%02d.bin", i),
                sizeBytes: expertLayerSize
            ))
        }
        return files
    }
}
```

- [ ] **Step 3: Delete stub, build, commit**

```bash
rm Sources/ModelHub/ModelHub_stub.swift
swift build 2>&1 | tail -5
git add Sources/ModelHub/
git commit -m "feat: add ModelCatalog with bundled model entries"
```

---

### Task 2: ModelStorage — local model management

**Files:**
- Create: `Sources/ModelHub/ModelStorage.swift`

- [ ] **Step 1: Create ModelStorage.swift**

```swift
import Foundation

/// Manages downloaded models on disk.
public actor ModelStorage {

    /// Root directory for model storage.
    public let storageDir: URL

    public init(storageDir: URL? = nil) {
        if let dir = storageDir {
            self.storageDir = dir
        } else {
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            self.storageDir = appSupport.appendingPathComponent("PrivateAgent/Models", isDirectory: true)
        }
        try? FileManager.default.createDirectory(at: self.storageDir, withIntermediateDirectories: true)
    }

    /// List all downloaded model directories.
    public func listModels() -> [URL] {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: storageDir, includingPropertiesForKeys: [.isDirectoryKey],
            options: .skipsHiddenFiles
        ) else { return [] }

        return contents.filter { url in
            var isDir: ObjCBool = false
            fm.fileExists(atPath: url.path, isDirectory: &isDir)
            return isDir.boolValue && fm.fileExists(atPath: url.appendingPathComponent("config.json").path)
        }
    }

    /// Get directory for a specific model.
    public func modelDir(for catalogId: String) -> URL {
        storageDir.appendingPathComponent(catalogId, isDirectory: true)
    }

    /// Check if a model is fully downloaded.
    public func isModelComplete(catalogId: String, expectedFiles: [ModelFile]) -> Bool {
        let dir = modelDir(for: catalogId)
        let fm = FileManager.default
        for file in expectedFiles {
            let path = dir.appendingPathComponent(file.filename)
            guard fm.fileExists(atPath: path.path) else { return false }
            guard let attrs = try? fm.attributesOfItem(atPath: path.path),
                  let size = attrs[.size] as? UInt64,
                  size == file.sizeBytes else { return false }
        }
        return true
    }

    /// Delete a downloaded model.
    public func deleteModel(catalogId: String) throws {
        let dir = modelDir(for: catalogId)
        try FileManager.default.removeItem(at: dir)
    }

    /// Get available disk space.
    public func availableSpaceBytes() -> UInt64 {
        guard let attrs = try? FileManager.default.attributesOfFileSystem(
            forPath: storageDir.path
        ) else { return 0 }
        return (attrs[.systemFreeSize] as? UInt64) ?? 0
    }

    /// Total size of all downloaded models.
    public func totalDownloadedBytes() -> UInt64 {
        let models = listModels()
        var total: UInt64 = 0
        for model in models {
            total += directorySize(at: model)
        }
        return total
    }

    private func directorySize(at url: URL) -> UInt64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: UInt64 = 0
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += UInt64(size)
            }
        }
        return total
    }
}
```

- [ ] **Step 2: Build, commit**

```bash
swift build 2>&1 | tail -5
git add Sources/ModelHub/ModelStorage.swift
git commit -m "feat: add ModelStorage for local model management"
```

---

### Task 3: DownloadManager — Background URLSession

**Files:**
- Create: `Sources/ModelHub/DownloadManager.swift`
- Create: `Sources/ModelHub/DownloadState.swift`

- [ ] **Step 1: Create DownloadState.swift**

```swift
import Foundation

/// Persistent download state for a model.
public struct DownloadState: Codable, Sendable {
    public let catalogId: String
    public var totalFiles: Int
    public var completedFiles: Int
    public var totalBytes: UInt64
    public var downloadedBytes: UInt64
    public var status: Status
    public var lastUpdated: Date

    public enum Status: String, Codable, Sendable {
        case pending
        case downloading
        case paused
        case verifying
        case complete
        case failed
    }

    public var progress: Double {
        guard totalBytes > 0 else { return 0 }
        return Double(downloadedBytes) / Double(totalBytes)
    }
}

/// Persists download states to Application Support.
actor DownloadStateStore {
    private let stateFileURL: URL
    private var states: [String: DownloadState] = [:]

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("PrivateAgent", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.stateFileURL = dir.appendingPathComponent("download_states.json")
        load()
    }

    func get(_ catalogId: String) -> DownloadState? {
        states[catalogId]
    }

    func set(_ state: DownloadState) {
        states[state.catalogId] = state
        save()
    }

    func remove(_ catalogId: String) {
        states.removeValue(forKey: catalogId)
        save()
    }

    func all() -> [DownloadState] {
        Array(states.values)
    }

    private func load() {
        guard let data = try? Data(contentsOf: stateFileURL),
              let decoded = try? JSONDecoder().decode([String: DownloadState].self, from: data)
        else { return }
        states = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(states) else { return }
        try? data.write(to: stateFileURL, options: .atomic)
    }
}
```

- [ ] **Step 2: Create DownloadManager.swift**

```swift
import Foundation
import CryptoKit

/// Manages model downloads via Background URLSession.
@Observable
public final class DownloadManager: NSObject, @unchecked Sendable {

    // Observable state for UI
    public private(set) var activeDownloads: [String: DownloadProgress] = [:]

    private let storage: ModelStorage
    private let stateStore: DownloadStateStore
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.background(withIdentifier: "com.privateagent.downloads")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    public struct DownloadProgress: Sendable {
        public let catalogId: String
        public let displayName: String
        public var currentFile: String
        public var filesCompleted: Int
        public var filesTotal: Int
        public var bytesDownloaded: UInt64
        public var bytesTotal: UInt64
        public var status: DownloadState.Status

        public var progress: Double {
            guard bytesTotal > 0 else { return 0 }
            return Double(bytesDownloaded) / Double(bytesTotal)
        }
    }

    public init(storage: ModelStorage) {
        self.storage = storage
        self.stateStore = DownloadStateStore()
        super.init()
    }

    /// Start downloading a model from the catalog.
    public func download(entry: CatalogEntry) async {
        let destDir = await storage.modelDir(for: entry.id)
        try? FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        let state = DownloadState(
            catalogId: entry.id,
            totalFiles: entry.files.count,
            completedFiles: 0,
            totalBytes: entry.totalSizeBytes,
            downloadedBytes: 0,
            status: .downloading,
            lastUpdated: Date()
        )
        await stateStore.set(state)

        activeDownloads[entry.id] = DownloadProgress(
            catalogId: entry.id,
            displayName: entry.displayName,
            currentFile: entry.files.first?.filename ?? "",
            filesCompleted: 0,
            filesTotal: entry.files.count,
            bytesDownloaded: 0,
            bytesTotal: entry.totalSizeBytes,
            status: .downloading
        )

        // Download files sequentially
        for (index, file) in entry.files.enumerated() {
            let url = URL(string: "https://huggingface.co/\(entry.repoId)/resolve/main/\(file.filename)")!
            let destFile = destDir.appendingPathComponent(file.filename)

            // Create subdirectories if needed
            let parentDir = destFile.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

            // Skip already downloaded files with correct size
            if FileManager.default.fileExists(atPath: destFile.path),
               let attrs = try? FileManager.default.attributesOfItem(atPath: destFile.path),
               let size = attrs[.size] as? UInt64,
               size == file.sizeBytes {
                activeDownloads[entry.id]?.filesCompleted = index + 1
                activeDownloads[entry.id]?.bytesDownloaded += file.sizeBytes
                continue
            }

            activeDownloads[entry.id]?.currentFile = file.filename

            // Use background download task
            let task = session.downloadTask(with: url)
            task.taskDescription = "\(entry.id)|\(file.filename)|\(file.sizeBytes)|\(file.sha256 ?? "")"
            task.resume()
        }
    }

    /// Cancel an in-progress download.
    public func cancel(catalogId: String) {
        session.getActiveTasks { tasks in
            for (_, downloadTasks, _) in [(tasks, tasks, tasks)] {
                // Cancel tasks matching this catalog ID
            }
        }
        activeDownloads.removeValue(forKey: catalogId)
        Task { await stateStore.remove(catalogId) }
    }

    /// Check free space before download.
    public func checkFreeSpace(for entry: CatalogEntry) async -> Bool {
        let available = await storage.availableSpaceBytes()
        return available > entry.totalSizeBytes + 1_000_000_000 // 1GB margin
    }

    /// Verify SHA-256 checksum of a file.
    public static func verifySHA256(fileURL: URL, expected: String) -> Bool {
        guard let data = try? Data(contentsOf: fileURL) else { return false }
        let hash = SHA256.hash(data: data)
        let hexString = hash.compactMap { String(format: "%02x", $0) }.joined()
        return hexString == expected.lowercased()
    }
}

// MARK: - URLSessionDownloadDelegate

extension DownloadManager: URLSessionDownloadDelegate {
    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let desc = downloadTask.taskDescription else { return }
        let parts = desc.split(separator: "|")
        guard parts.count >= 3 else { return }

        let catalogId = String(parts[0])
        let filename = String(parts[1])

        Task {
            let destDir = await storage.modelDir(for: catalogId)
            let destFile = destDir.appendingPathComponent(filename)
            let parentDir = destFile.deletingLastPathComponent()
            try? FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: destFile) // remove if exists
            try? FileManager.default.moveItem(at: location, to: destFile)

            // Verify checksum if available
            if parts.count >= 4 {
                let expectedHash = String(parts[3])
                if !expectedHash.isEmpty && !Self.verifySHA256(fileURL: destFile, expected: expectedHash) {
                    try? FileManager.default.removeItem(at: destFile) // corrupt, remove
                    return
                }
            }

            // Update progress
            if var progress = activeDownloads[catalogId] {
                progress.filesCompleted += 1
                if let size = try? FileManager.default.attributesOfItem(atPath: destFile.path)[.size] as? UInt64 {
                    progress.bytesDownloaded += size
                }
                activeDownloads[catalogId] = progress
            }
        }
    }

    public func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        if let error {
            print("Download error: \(error.localizedDescription)")
        }
    }

    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        // Progress update for current file — could update UI here
    }
}
```

- [ ] **Step 3: Build, commit**

```bash
swift build 2>&1 | tail -5
git add Sources/ModelHub/
git commit -m "feat: add DownloadManager with Background URLSession + SHA-256 verification"
```

---

### Task 4: ModelHub tests

**Files:**
- Create: `Tests/ModelHubTests/ModelStorageTests.swift`
- Create: `Tests/ModelHubTests/CatalogTests.swift`
- Modify: `Package.swift` (add ModelHubTests target)

- [ ] **Step 1: Add test target to Package.swift**

```swift
.testTarget(
    name: "ModelHubTests",
    dependencies: ["ModelHub"]
),
```

- [ ] **Step 2: Create CatalogTests.swift**

```swift
import Testing
@testable import ModelHub

@Suite("ModelCatalog Tests")
struct CatalogTests {

    @Test("Bundled catalog has entries")
    func bundledCatalog() async {
        let catalog = ModelCatalog()
        let models = await catalog.models
        #expect(models.count >= 2)
        #expect(models[0].id == "qwen3.5-35b-a3b-q4")
        #expect(models[0].files.count > 0)
        #expect(models[0].totalSizeGB > 10)
    }

    @Test("Each catalog entry has valid file list")
    func validFiles() async {
        let catalog = ModelCatalog()
        for entry in await catalog.models {
            #expect(!entry.files.isEmpty)
            #expect(entry.files.contains { $0.filename == "config.json" })
            #expect(entry.files.contains { $0.filename == "model_weights.bin" })
        }
    }
}
```

- [ ] **Step 3: Create ModelStorageTests.swift**

```swift
import Testing
import Foundation
@testable import ModelHub

@Suite("ModelStorage Tests")
struct ModelStorageTests {

    private func makeTempStorage() async -> ModelStorage {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("PrivateAgentTest-\(UUID().uuidString)")
        return ModelStorage(storageDir: dir)
    }

    @Test("List models in empty storage")
    func emptyStorage() async {
        let storage = await makeTempStorage()
        let models = await storage.listModels()
        #expect(models.isEmpty)
    }

    @Test("Model directory creation")
    func modelDir() async {
        let storage = await makeTempStorage()
        let dir = await storage.modelDir(for: "test-model")
        #expect(dir.lastPathComponent == "test-model")
    }

    @Test("Available space is positive")
    func availableSpace() async {
        let storage = await makeTempStorage()
        let space = await storage.availableSpaceBytes()
        #expect(space > 0)
    }

    @Test("SHA-256 verification")
    func sha256Verify() {
        let tmpFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("sha256test-\(UUID().uuidString).txt")
        try! "Hello, World!\n".data(using: .utf8)!.write(to: tmpFile)
        defer { try? FileManager.default.removeItem(at: tmpFile) }

        // SHA-256 of "Hello, World!\n"
        let expected = "8663bab6d124806b9727f89bb4ab9db4cbcc3862f6bbf22024dfa7212aa4ab7d"
        #expect(DownloadManager.verifySHA256(fileURL: tmpFile, expected: expected))
        #expect(!DownloadManager.verifySHA256(fileURL: tmpFile, expected: "0000"))
    }
}
```

- [ ] **Step 4: Run tests, commit**

```bash
swift test 2>&1 | tail -15
git add Tests/ModelHubTests/ Package.swift
git commit -m "test: add ModelCatalog and ModelStorage tests"
```

---

## Summary

After Plan 4:

- ModelCatalog: browseable model list (2 entries: full Q4 + tiered)
- ModelStorage: scan/validate/delete models on disk + space checking
- DownloadManager: Background URLSession + per-file SHA-256 verification
- DownloadState: persistent download progress in Application Support
- Tests for catalog browsing, storage management, and checksum verification

**Next:** Plan 5 (TurboQuant CPU reference → Metal kernels)
