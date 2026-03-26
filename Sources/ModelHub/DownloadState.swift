// DownloadState.swift
// ModelHub

import Foundation

// MARK: - DownloadState

public struct DownloadState: Codable, Sendable {
    public let catalogId: String
    public var totalFiles: Int
    public var completedFiles: Int
    public var totalBytes: UInt64
    public var downloadedBytes: UInt64
    public var status: Status
    public var lastUpdated: Date

    public enum Status: String, Codable, Sendable {
        case pending, downloading, paused, verifying, complete, failed
    }

    public var progress: Double {
        guard totalBytes > 0 else { return 0 }
        return Double(downloadedBytes) / Double(totalBytes)
    }
}

// MARK: - DownloadStateStore

actor DownloadStateStore {
    private let stateFileURL: URL
    private var states: [String: DownloadState] = [:]

    init() {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("PrivateAgent", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.stateFileURL = dir.appendingPathComponent("download_states.json")
        // Load persisted states synchronously during init (safe: actor not yet shared)
        if let data = try? Data(contentsOf: dir.appendingPathComponent("download_states.json")),
           let decoded = try? JSONDecoder().decode([String: DownloadState].self, from: data) {
            self.states = decoded
        }
    }

    func get(_ catalogId: String) -> DownloadState? { states[catalogId] }
    func set(_ state: DownloadState) { states[state.catalogId] = state; save() }
    func remove(_ catalogId: String) { states.removeValue(forKey: catalogId); save() }
    func all() -> [DownloadState] { Array(states.values) }

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
