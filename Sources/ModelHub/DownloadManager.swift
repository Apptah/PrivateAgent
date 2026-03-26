// DownloadManager.swift
// ModelHub

import CryptoKit
import Foundation
import Observation

// MARK: - DownloadManager

@Observable
public final class DownloadManager: NSObject {

    // MARK: - DownloadProgress

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

    // MARK: Lifecycle

    public override init() {
        let config = URLSessionConfiguration.background(
            withIdentifier: "com.privateagent.downloads"
        )
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true

        // Placeholder; real session assigned after super.init()
        _session = nil
        super.init()
        _session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    // MARK: Public

    /// Active in-progress downloads keyed by catalogId.
    public private(set) var activeDownloads: [String: DownloadProgress] = [:]

    // MARK: - Download lifecycle

    /// Begin downloading all files for the given catalog entry.
    public func download(entry: CatalogEntry) async throws {
        guard try await checkFreeSpace(for: entry) else {
            throw DownloadError.insufficientDiskSpace
        }

        let storage = ModelStorage()
        let destDir = await storage.modelDir(for: entry.id)
        let fm = FileManager.default
        try fm.createDirectory(at: destDir, withIntermediateDirectories: true)

        // Pre-create subdirectories for expert files (e.g. packed_experts/, packed_experts_tiered/)
        for file in entry.files {
            let fileURL = destDir.appendingPathComponent(file.filename)
            let parentDir = fileURL.deletingLastPathComponent()
            if parentDir != destDir && !fm.fileExists(atPath: parentDir.path) {
                try fm.createDirectory(at: parentDir, withIntermediateDirectories: true)
            }
        }

        var progress = DownloadProgress(
            catalogId: entry.id,
            displayName: entry.displayName,
            currentFile: entry.files.first?.filename ?? "",
            filesCompleted: 0,
            filesTotal: entry.files.count,
            bytesDownloaded: 0,
            bytesTotal: entry.totalSizeBytes,
            status: .downloading
        )
        activeDownloads[entry.id] = progress
        completedBytes[entry.id] = 0

        await stateStore.set(DownloadState(
            catalogId: entry.id,
            totalFiles: entry.files.count,
            completedFiles: 0,
            totalBytes: entry.totalSizeBytes,
            downloadedBytes: 0,
            status: .downloading,
            lastUpdated: Date()
        ))

        for file in entry.files {
            // Skip files already fully downloaded
            let destURL = destDir.appendingPathComponent(file.filename)
            if FileManager.default.fileExists(atPath: destURL.path) {
                progress.filesCompleted += 1
                progress.bytesDownloaded += file.sizeBytes
                progress.currentFile = file.filename
                activeDownloads[entry.id] = progress
                continue
            }

            let downloadURL = huggingFaceURL(repoId: entry.repoId, filename: file.filename)
            var request = URLRequest(url: downloadURL)
            request.setValue("PrivateAgent/1.0", forHTTPHeaderField: "User-Agent")

            let task = session.downloadTask(with: request)
            // Encode metadata into task description for delegate recovery
            task.taskDescription = [
                entry.id,
                file.filename,
                "\(file.sizeBytes)",
                file.sha256 ?? "",
            ].joined(separator: "|")

            taskMap[task.taskIdentifier] = (catalogId: entry.id, file: file, destDir: destDir)
            task.resume()
        }
    }

    /// Cancel all download tasks for the given catalogId.
    public func cancel(catalogId: String) {
        session.getAllTasks { tasks in
            for task in tasks {
                guard let desc = task.taskDescription,
                      desc.hasPrefix(catalogId + "|")
                else { continue }
                task.cancel()
            }
        }
        activeDownloads.removeValue(forKey: catalogId)
        Task {
            await self.stateStore.remove(catalogId)
        }
    }

    // MARK: - Disk space

    /// Returns true when at least `entry.totalSizeBytes` + 1 GB is available.
    public func checkFreeSpace(for entry: CatalogEntry) async throws -> Bool {
        let storage = ModelStorage()
        let available = try await storage.availableSpaceBytes()
        let required = entry.totalSizeBytes + 1_073_741_824 // +1 GB margin
        return available >= required
    }

    // MARK: - SHA-256 verification

    /// Returns true when the file's SHA-256 digest matches `expected` (hex string).
    public static func verifySHA256(fileURL: URL, expected: String) throws -> Bool {
        guard !expected.isEmpty else { return true } // no checksum to verify
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        let digest = SHA256.hash(data: data)
        let hex = digest.compactMap { String(format: "%02x", $0) }.joined()
        return hex.lowercased() == expected.lowercased()
    }

    // MARK: Private

    private var _session: URLSession?
    private var session: URLSession { _session! }

    private let stateStore = DownloadStateStore()

    /// Maps URLSession task identifier → (catalogId, file metadata, destination directory).
    private var taskMap: [Int: (catalogId: String, file: ModelFile, destDir: URL)] = [:]

    /// Tracks cumulative bytes written per active task (by task identifier).
    private var taskBytesWritten: [Int: Int64] = [:]

    /// Tracks total bytes from completed files per catalog entry.
    private var completedBytes: [String: UInt64] = [:]

    private func huggingFaceURL(repoId: String, filename: String) -> URL {
        // https://huggingface.co/<owner>/<repo>/resolve/main/<filename>
        URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filename)")!
    }
}

// MARK: - URLSessionDownloadDelegate

extension DownloadManager: URLSessionDownloadDelegate {

    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let info = taskMap[downloadTask.taskIdentifier] else { return }
        let destURL = info.destDir.appendingPathComponent(info.file.filename)

        do {
            // Move temp file to final destination
            if FileManager.default.fileExists(atPath: destURL.path) {
                try FileManager.default.removeItem(at: destURL)
            }
            try FileManager.default.moveItem(at: location, to: destURL)

            // Verify SHA-256 if available
            if let expectedHash = info.file.sha256, !expectedHash.isEmpty {
                let valid = try DownloadManager.verifySHA256(fileURL: destURL, expected: expectedHash)
                if !valid {
                    try? FileManager.default.removeItem(at: destURL)
                    markFailed(catalogId: info.catalogId, reason: "SHA-256 mismatch for \(info.file.filename)")
                    return
                }
            }

            // Update progress
            DispatchQueue.main.async {
                self.recordFileCompleted(
                    catalogId: info.catalogId,
                    file: info.file
                )
            }
        } catch {
            markFailed(catalogId: info.catalogId, reason: error.localizedDescription)
        }

        taskBytesWritten.removeValue(forKey: downloadTask.taskIdentifier)
        taskMap.removeValue(forKey: downloadTask.taskIdentifier)
    }

    public func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let error else { return }
        guard let info = taskMap[task.taskIdentifier] else { return }
        taskBytesWritten.removeValue(forKey: task.taskIdentifier)
        taskMap.removeValue(forKey: task.taskIdentifier)
        DispatchQueue.main.async {
            self.markFailed(catalogId: info.catalogId, reason: error.localizedDescription)
        }
    }

    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let info = taskMap[downloadTask.taskIdentifier] else { return }

        // Store cumulative bytes for this task
        taskBytesWritten[downloadTask.taskIdentifier] = totalBytesWritten

        // Compute total: completed files + all active tasks' cumulative bytes
        let completed = completedBytes[info.catalogId] ?? 0
        let activeTotal = taskMap
            .filter { $0.value.catalogId == info.catalogId }
            .reduce(UInt64(0)) { sum, entry in
                sum + UInt64(max(0, taskBytesWritten[entry.key] ?? 0))
            }
        let totalDownloaded = completed + activeTotal
        let currentFile = info.file.filename

        DispatchQueue.main.async {
            if var progress = self.activeDownloads[info.catalogId] {
                progress.bytesDownloaded = totalDownloaded
                progress.currentFile = currentFile
                self.activeDownloads[info.catalogId] = progress
            }
        }
    }

    // MARK: Private helpers

    private func recordFileCompleted(catalogId: String, file: ModelFile) {
        // Add this file's size to completed bytes tracker
        completedBytes[catalogId, default: 0] += file.sizeBytes

        if var progress = activeDownloads[catalogId] {
            progress.filesCompleted += 1
            progress.bytesDownloaded = completedBytes[catalogId] ?? 0

            if progress.filesCompleted >= progress.filesTotal {
                progress.status = .complete
                activeDownloads.removeValue(forKey: catalogId)
                completedBytes.removeValue(forKey: catalogId)
            } else {
                activeDownloads[catalogId] = progress
            }
        }

        Task {
            if var state = await stateStore.get(catalogId) {
                state.completedFiles += 1
                state.status = state.completedFiles >= state.totalFiles ? .complete : .downloading
                state.lastUpdated = Date()
                await stateStore.set(state)
            }
        }
    }

    private func markFailed(catalogId: String, reason: String) {
        if var progress = activeDownloads[catalogId] {
            progress.status = .failed
            activeDownloads[catalogId] = progress
        }
        Task {
            if var state = await stateStore.get(catalogId) {
                state.status = .failed
                state.lastUpdated = Date()
                await stateStore.set(state)
            }
        }
    }
}

// MARK: - DownloadError

public enum DownloadError: Error, LocalizedError {
    case insufficientDiskSpace
    case sha256Mismatch(filename: String)

    public var errorDescription: String? {
        switch self {
        case .insufficientDiskSpace:
            return "Not enough free disk space to download this model (requires 1 GB headroom)."
        case .sha256Mismatch(let filename):
            return "SHA-256 verification failed for \(filename)."
        }
    }
}
