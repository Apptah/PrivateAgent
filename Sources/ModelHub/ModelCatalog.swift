// ModelCatalog.swift
// ModelHub

import Foundation

// MARK: - ModelCatalog

public actor ModelCatalog {

    // MARK: Public

    public static let shared = ModelCatalog()

    public private(set) var entries: [CatalogEntry] = ModelCatalog.bundledCatalog

    public func refresh() async throws {
        // TODO: fetch from remote catalog endpoint and merge with bundled entries
    }

    public func entry(id: String) -> CatalogEntry? {
        entries.first { $0.id == id }
    }

    // MARK: Private

    private init() {}

    // MARK: - Bundled Catalog

    private static let bundledCatalog: [CatalogEntry] = [
        makeQ4Entry(),
        makeTieredEntry(),
    ]

    // MARK: Entry Factories

    /// Full Q4 — ~13.5 GB, all experts at 4-bit
    private static func makeQ4Entry() -> CatalogEntry {
        var files: [ModelFile] = [
            ModelFile(filename: "config.json", sizeBytes: 4_096),
            ModelFile(filename: "model_weights.json", sizeBytes: 200_000),
            ModelFile(filename: "model_weights.bin", sizeBytes: 1_800_000_000),
            ModelFile(filename: "vocab.bin", sizeBytes: 5_800_000),
            ModelFile(filename: "tokenizer.json", sizeBytes: 33_400_000),
            ModelFile(filename: "tokenizer.bin", sizeBytes: 13_100_000),
        ]
        for i in 0..<30 {
            files.append(ModelFile(
                filename: String(format: "packed_experts/layer_%02d.bin", i),
                sizeBytes: 381_681_664
            ))
        }

        return CatalogEntry(
            id: "gemma4-26b-a4b-q4",
            displayName: "Gemma 4 26B-A4B",
            repoId: "alexintosh/Gemma-4-26B-A4B-Q4-FlashMoE",
            description: "Full 4-bit quantization. Best quality. ~13.5 GB.",
            totalSizeBytes: 13_504_000_000,
            quantization: "4-bit",
            expertLayers: 30,
            files: files
        )
    }

    /// Tiered — ~9.5 GB, hot experts 4-bit, cold experts 2-bit
    private static func makeTieredEntry() -> CatalogEntry {
        var files: [ModelFile] = [
            ModelFile(filename: "config.json", sizeBytes: 4_096),
            ModelFile(filename: "model_weights.json", sizeBytes: 200_000),
            ModelFile(filename: "model_weights.bin", sizeBytes: 1_800_000_000),
            ModelFile(filename: "vocab.bin", sizeBytes: 5_800_000),
            ModelFile(filename: "tokenizer.json", sizeBytes: 33_400_000),
            ModelFile(filename: "tokenizer.bin", sizeBytes: 13_100_000),
            ModelFile(filename: "packed_experts_tiered/tiered_manifest.json", sizeBytes: 800_000),
        ]
        // Variable-size tiered layers (estimated for Gemma 4 26B-A4B)
        let layerSizes: [UInt64] = [
            268_435_456, 272_629_760, 264_241_152, 260_046_848, 255_852_544,
            247_463_936, 251_658_240, 243_269_632, 239_075_328, 247_463_936,
            255_852_544, 260_046_848, 251_658_240, 247_463_936, 239_075_328,
            243_269_632, 247_463_936, 251_658_240, 239_075_328, 247_463_936,
            243_269_632, 251_658_240, 255_852_544, 260_046_848, 251_658_240,
            247_463_936, 243_269_632, 251_658_240, 255_852_544, 260_046_848,
        ]
        for (i, size) in layerSizes.enumerated() {
            files.append(ModelFile(
                filename: String(format: "packed_experts_tiered/layer_%02d.bin", i),
                sizeBytes: size
            ))
        }

        return CatalogEntry(
            id: "gemma4-26b-a4b-tiered",
            displayName: "Gemma 4 26B-A4B Tiered",
            repoId: "alexintosh/Gemma-4-26B-A4B-Q4-Tiered-FlashMoE",
            description: "Tiered: hot experts 4-bit, cold 2-bit. Faster on iPhone. ~9.5 GB.",
            totalSizeBytes: 9_500_000_000,
            quantization: "Tiered (4-bit/2-bit)",
            expertLayers: 30,
            files: files
        )
    }
}
