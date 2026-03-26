// ModelStorage.swift
// ModelHub

import Foundation

// MARK: - ModelStorage

public actor ModelStorage {

    // MARK: Lifecycle

    public init(storageDir: URL? = nil) {
        if let dir = storageDir {
            self.storageDir = dir
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first!
            self.storageDir = appSupport
                .appendingPathComponent("PrivateAgent", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
    }

    // MARK: Public

    public let storageDir: URL

    /// Returns directories under storageDir that contain a config.json file.
    public func listModels() throws -> [URL] {
        let fm = FileManager.default

        guard fm.fileExists(atPath: storageDir.path) else {
            return []
        }

        let contents = try fm.contentsOfDirectory(
            at: storageDir,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        return contents.filter { url in
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
                return false
            }
            let configURL = url.appendingPathComponent("config.json")
            return fm.fileExists(atPath: configURL.path)
        }
    }

    /// Returns the directory URL for a specific catalog entry ID.
    public func modelDir(for catalogId: String) -> URL {
        storageDir.appendingPathComponent(catalogId, isDirectory: true)
    }

    /// Returns true when every expected file exists and has the correct size.
    public func isModelComplete(catalogId: String, expectedFiles: [ModelFile]) -> Bool {
        let fm = FileManager.default
        let dir = modelDir(for: catalogId)

        for file in expectedFiles {
            let fileURL = dir.appendingPathComponent(file.filename)
            guard fm.fileExists(atPath: fileURL.path) else { return false }

            // Skip size check for placeholder files (sizeBytes == 0)
            guard file.sizeBytes > 0 else { continue }

            guard
                let attrs = try? fm.attributesOfItem(atPath: fileURL.path),
                let size = attrs[.size] as? UInt64,
                size == file.sizeBytes
            else {
                return false
            }
        }
        return true
    }

    /// Removes the directory for a catalog entry, including all downloaded files.
    public func deleteModel(catalogId: String) throws {
        let dir = modelDir(for: catalogId)
        guard FileManager.default.fileExists(atPath: dir.path) else { return }
        try FileManager.default.removeItem(at: dir)
    }

    /// Available disk space on the volume containing storageDir, in bytes.
    public func availableSpaceBytes() throws -> UInt64 {
        // Use a path that's guaranteed to exist for attributesOfFileSystem.
        // storageDir might not exist yet on first launch.
        let queryPath: String
        if FileManager.default.fileExists(atPath: storageDir.path) {
            queryPath = storageDir.path
        } else {
            // Fall back to home directory (always accessible in sandbox)
            queryPath = NSHomeDirectory()
        }
        let attrs = try FileManager.default.attributesOfFileSystem(forPath: queryPath)
        return (attrs[.systemFreeSize] as? UInt64) ?? 0
    }

    /// Sum of sizes of all files inside every downloaded model directory.
    public func totalDownloadedBytes() throws -> UInt64 {
        let fm = FileManager.default
        let dirs = try listModels()
        var total: UInt64 = 0

        for dir in dirs {
            guard
                let enumerator = fm.enumerator(
                    at: dir,
                    includingPropertiesForKeys: [.fileSizeKey],
                    options: [.skipsHiddenFiles]
                )
            else { continue }

            for case let fileURL as URL in enumerator {
                guard let values = try? fileURL.resourceValues(forKeys: [.fileSizeKey]),
                      let size = values.fileSize
                else { continue }
                total += UInt64(max(0, size))
            }
        }
        return total
    }

    // MARK: Private

    private func ensureStorageDirExists() throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: storageDir.path) {
            try fm.createDirectory(at: storageDir, withIntermediateDirectories: true)
        }
    }
}
