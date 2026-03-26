// DownloadManager.swift
// ModelHub

import CryptoKit
import Foundation
import Observation

@Observable
public final class DownloadManager: NSObject, @unchecked Sendable {

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
        super.init()
        session = URLSession(
            configuration: .default,
            delegate: self,
            delegateQueue: nil
        )
    }

    // MARK: Public

    public private(set) var activeDownloads: [String: DownloadProgress] = [:]

    /// Download all files for a catalog entry sequentially.
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
            let parentDir = destDir.appendingPathComponent(file.filename).deletingLastPathComponent()
            if parentDir != destDir && !fm.fileExists(atPath: parentDir.path) {
                try fm.createDirectory(at: parentDir, withIntermediateDirectories: true)
            }
        }

        // Track which files need downloading
        var completedBytes: UInt64 = 0
        var filesToDownload: [ModelFile] = []

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

        for file in entry.files {
            let destURL = destDir.appendingPathComponent(file.filename)
            if fm.fileExists(atPath: destURL.path),
               file.sizeBytes == 0 || (try? fm.attributesOfItem(atPath: destURL.path)[.size] as? UInt64) == file.sizeBytes {
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

        // Download each file sequentially
        for file in filesToDownload {
            guard await activeDownloads[entry.id]?.status == .downloading else { break }

            progress.currentFile = file.filename
            await updateProgress(entry.id, progress)

            let url = huggingFaceURL(repoId: entry.repoId, filename: file.filename)
            let destURL = destDir.appendingPathComponent(file.filename)

            do {
                currentFileCompletedBytes = completedBytes
                currentCatalogId = entry.id

                let tempURL = try await downloadFileWithRetry(
                    url: url,
                    filename: file.filename,
                    maxRetries: 5
                )

                // Move temp file to final destination
                if fm.fileExists(atPath: destURL.path) {
                    try fm.removeItem(at: destURL)
                }
                try fm.moveItem(at: tempURL, to: destURL)

                // Get actual downloaded size
                let actualSize: UInt64
                if let attrs = try? fm.attributesOfItem(atPath: destURL.path),
                   let size = attrs[.size] as? UInt64 {
                    actualSize = size
                } else {
                    actualSize = file.sizeBytes
                }

                completedBytes += actualSize
                progress.filesCompleted += 1
                progress.bytesDownloaded = completedBytes
                await updateProgress(entry.id, progress)

            } catch is CancellationError {
                progress.status = .paused
                await updateProgress(entry.id, progress)
                return
            } catch {
                progress.status = .failed
                progress.currentFile = "\(file.filename): \(error.localizedDescription)"
                await updateProgress(entry.id, progress)
                throw error
            }
        }

        progress.status = .complete
        progress.currentFile = "Download complete"
        await updateProgress(entry.id, progress)
    }

    public func cancel(catalogId: String) {
        session.getAllTasks { tasks in
            tasks.forEach { $0.cancel() }
        }
        activeDownloads.removeValue(forKey: catalogId)
    }

    public func checkFreeSpace(for entry: CatalogEntry) async throws -> Bool {
        let storage = ModelStorage()
        let available = try await storage.availableSpaceBytes()
        return available >= entry.totalSizeBytes + 1_073_741_824
    }

    public static func verifySHA256(fileURL: URL, expected: String) throws -> Bool {
        guard !expected.isEmpty else { return true }
        let data = try Data(contentsOf: fileURL, options: .mappedIfSafe)
        let digest = SHA256.hash(data: data)
        let hex = digest.compactMap { String(format: "%02x", $0) }.joined()
        return hex.lowercased() == expected.lowercased()
    }

    // MARK: Private

    private var session: URLSession!
    private var currentFileCompletedBytes: UInt64 = 0
    private var currentCatalogId: String?

    /// Download a single file with automatic retry + resume on network errors.
    private func downloadFileWithRetry(url: URL, filename: String, maxRetries: Int) async throws -> URL {
        var request = URLRequest(url: url)
        request.setValue("PrivateAgent/1.0", forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = 300  // 5 min timeout per request

        var resumeData: Data?
        var lastError: Error?

        for attempt in 0...maxRetries {
            do {
                let (tempURL, response): (URL, URLResponse)

                if let data = resumeData {
                    // Resume from where we left off
                    (tempURL, response) = try await session.download(resumeFrom: data)
                } else {
                    (tempURL, response) = try await session.download(for: request)
                }

                guard let http = response as? HTTPURLResponse,
                      (200...299).contains(http.statusCode) else {
                    let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                    throw DownloadError.httpError(filename: filename, statusCode: code)
                }

                return tempURL

            } catch let error as NSError where error.code == -1005 || error.code == -1001 || error.code == -1009 {
                // -1005: connection lost, -1001: timeout, -1009: no internet
                lastError = error

                // Extract resume data if available
                if let data = error.userInfo[NSURLSessionDownloadTaskResumeData] as? Data {
                    resumeData = data
                }

                if attempt < maxRetries {
                    // Wait before retry: 2, 4, 8, 16, 32 seconds
                    let delay = UInt64(pow(2.0, Double(attempt + 1)))
                    try await Task.sleep(for: .seconds(delay))
                    continue
                }
            } catch {
                throw error
            }
        }

        throw lastError ?? DownloadError.httpError(filename: filename, statusCode: 0)
    }

    @MainActor
    private func updateProgress(_ catalogId: String, _ progress: DownloadProgress) {
        activeDownloads[catalogId] = progress
    }

    private func huggingFaceURL(repoId: String, filename: String) -> URL {
        URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(filename)")!
    }
}

// MARK: - URLSessionDownloadDelegate (for per-file progress)

extension DownloadManager: URLSessionDownloadDelegate {
    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Handled by the await session.download(for:) return value
    }

    public func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let catalogId = currentCatalogId else { return }
        let totalNow = currentFileCompletedBytes + UInt64(max(0, totalBytesWritten))

        DispatchQueue.main.async {
            if var progress = self.activeDownloads[catalogId] {
                progress.bytesDownloaded = totalNow
                self.activeDownloads[catalogId] = progress
            }
        }
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
