// CatalogEntry.swift
// ModelHub

import Foundation

// MARK: - ModelFile

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

// MARK: - CatalogEntry

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
        Double(totalSizeBytes) / 1_073_741_824.0
    }

    public init(
        id: String,
        displayName: String,
        repoId: String,
        description: String,
        totalSizeBytes: UInt64,
        quantization: String,
        expertLayers: Int,
        files: [ModelFile]
    ) {
        self.id = id
        self.displayName = displayName
        self.repoId = repoId
        self.description = description
        self.totalSizeBytes = totalSizeBytes
        self.quantization = quantization
        self.expertLayers = expertLayers
        self.files = files
    }
}
