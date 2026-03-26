// DownloadManager.swift
// ModelHub

import CryptoKit
import Foundation
import Observation

@Observable
public final class DownloadManager: @unchecked Sendable {

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

    public init() {}

    // MARK: Public

    public private(set) var activeDownloads: [String: DownloadProgress] = [:]

    /// Download all files for a catalog entry sequentially with progress.
    public func download(entry: CatalogEntry) async throws {
        guard try await checkFreeSpace(for: entry) else {
            throw DownloadError.insufficientDiskSpace
        }

        let storage = ModelStorage()
        let destDir = await storage.modelDir(for: entry.id)
        let fm = FileManager.default
        try fm.createDirectory(at: destDir, withIntermediateDirectories: true)

        // Pre-create subdirectories
        for file in entry.files {
            let fileURL = destDir.appendingPathComponent(file.filename)
            let parentDir = fileURL.deletingLastPathComponent()
            if parentDir != destDir && !fm.fileExists(atPath: parentDir.path) {
                try fm.createDirectory(at: parentDir, withIntermediateDirectories: true)
            }
        }

        // Initialize progress
        var progress = DownloadProgress(
            catalogId: entry.id,
            displayName: entry.displayName,
            currentFile: "",
            filesCompleted: 0,
            filesTotal: entry.files.count,
            bytesDownloaded: 0,
            bytesTotal: entry.totalSizeBytes,
            status: .downloading
        )

        // Check which files already exist
        var completedBytes: UInt64 = 0
        var filesToDownload: [ModelFile] = []
        for file in entry.files {
            let destURL = destDir.appendingPathComponent(file.filename)
            if fm.fileExists(atPath: destURL.path) {
                // Skip already downloaded files
                progress.filesCompleted += 1
                completedBytes += file.sizeBytes
            } else {
                filesToDownload.append(file)
            }
        }
        progress.bytesDownloaded = completedBytes
        await updateProgress(entry.id, progress)

        if filesToDownload.isEmpty {
            progress.status = .complete
            await updateProgress(entry.id, progress)
            return
        }

        // Download files sequentially
        let session = URLSession.shared
        currentTask = nil

        for file in filesToDownload {
            // Check cancellation
            guard await activeDownloads[entry.id]?.status == .downloading else { break }

            progress.currentFile = file.filename
            await updateProgress(entry.id, progress)

            let url = huggingFaceURL(repoId: entry.repoId, filename: file.filename)
            let destURL = destDir.appendingPathComponent(file.filename)

            do {
                // Use bytes(for:) for streaming download with progress
                var request = URLRequest(url: url)
                request.setValue("PrivateAgent/1.0", forHTTPHeaderField: "User-Agent")

                let (asyncBytes, response) = try await session.bytes(for: request)

                guard let httpResponse = response as? HTTPURLResponse,
                      (200...299).contains(httpResponse.statusCode) else {
                    let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                    throw DownloadError.httpError(filename: file.filename, statusCode: code)
                }

                // Stream to file
                let fileHandle = try FileHandle(forWritingTo: {
                    fm.createFile(atPath: destURL.path, contents: nil)
                    return destURL
                }())

                var fileBytes: UInt64 = 0
                let updateInterval: UInt64 = 512 * 1024  // Update UI every 512KB

                for try await byte in asyncBytes {
                    try Task.checkCancellation()
                    fileHandle.write(Data([byte]))
                    fileBytes += 1

                    if fileBytes % updateInterval == 0 {
                        progress.bytesDownloaded = completedBytes + fileBytes
                        await updateProgress(entry.id, progress)
                    }
                }
                try fileHandle.close()

                // Verify checksum if available
                if let sha = file.sha256, !sha.isEmpty {
                    let valid = try DownloadManager.verifySHA256(fileURL: destURL, expected: sha)
                    if !valid {
                        try? fm.removeItem(at: destURL)
                        throw DownloadError.sha256Mismatch(filename: file.filename)
                    }
                }

                completedBytes += fileBytes
                progress.filesCompleted += 1
                progress.bytesDownloaded = completedBytes
                await updateProgress(entry.id, progress)

            } catch is CancellationError {
                progress.status = .paused
                await updateProgress(entry.id, progress)
                return
            } catch {
                progress.status = .failed
                progress.currentFile = "Error: \(error.localizedDescription)"
                await updateProgress(entry.id, progress)
                throw error
            }
        }

        // Done
        progress.status = .complete
        progress.currentFile = ""
        await updateProgress(entry.id, progress)
    }

    /// Cancel download for a catalog entry.
    public func cancel(catalogId: String) {
        currentTask?.cancel()
        currentTask = nil
        activeDownloads.removeValue(forKey: catalogId)
    }

    /// Check free space.
    public func checkFreeSpace(for entry: CatalogEntry) async throws -> Bool {
        let storage = ModelStorage()
        let available = try await storage.availableSpaceBytes()
        return available >= entry.totalSizeBytes + 1_073_741_824
    }

    /// Verify SHA-256.
    public static func verifySHA256(fileURL: URL, expected: String) throws -> Bool {
        guard !expected.isEmpty else { return true }
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        let digest = SHA256.hash(data: data)
        let hex = digest.compactMap { String(format: "%02x", $0) }.joined()
        return hex.lowercased() == expected.lowercased()
    }

    // MARK: Private

    private var currentTask: Task<Void, Never>?

    @MainActor
    private func updateProgress(_ catalogId: String, _ progress: DownloadProgress) {
        activeDownloads[catalogId] = progress
    }

    private func huggingFaceURL(repoId: String, filename: String) -> URL {
        URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filename)")!
    }
}

// MARK: - DownloadError

public enum DownloadError: Error, LocalizedError {
    case insufficientDiskSpace
    case sha256Mismatch(filename: String)
    case httpError(filename: String, statusCode: Int)

    public var errorDescription: String? {
        switch self {
        case .insufficientDiskSpace:
            return "Not enough free disk space (requires 1 GB headroom)."
        case .sha256Mismatch(let filename):
            return "SHA-256 verification failed for \(filename)."
        case .httpError(let filename, let code):
            return "HTTP \(code) downloading \(filename)."
        }
    }
}
