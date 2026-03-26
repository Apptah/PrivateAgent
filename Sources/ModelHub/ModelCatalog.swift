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

    /// Full Q4 — ~19.5 GB, all experts at 4-bit
    private static func makeQ4Entry() -> CatalogEntry {
        var files: [ModelFile] = [
            ModelFile(filename: "config.json", sizeBytes: 3_809),
            ModelFile(filename: "model_weights.json", sizeBytes: 251_539),
            ModelFile(filename: "model_weights.bin", sizeBytes: 1_378_869_376),
            ModelFile(filename: "vocab.bin", sizeBytes: 3_360_287),
            ModelFile(filename: "tokenizer.json", sizeBytes: 19_989_343),
            ModelFile(filename: "tokenizer.bin", sizeBytes: 8_201_040),
        ]
        for i in 0..<40 {
            files.append(ModelFile(
                filename: String(format: "packed_experts/layer_%02d.bin", i),
                sizeBytes: 452_984_832
            ))
        }

        return CatalogEntry(
            id: "qwen3.5-35b-a3b-q4",
            displayName: "Qwen 3.5 35B-A3B",
            repoId: "alexintosh/Qwen3.5-35B-A3B-Q4-FlashMoE",
            description: "Full 4-bit quantization. Best quality. ~19.5 GB.",
            totalSizeBytes: 19_500_000_000,
            quantization: "4-bit",
            expertLayers: 40,
            files: files
        )
    }

    /// Tiered — ~13.4 GB, hot experts 4-bit, cold experts 2-bit
    private static func makeTieredEntry() -> CatalogEntry {
        var files: [ModelFile] = [
            ModelFile(filename: "config.json", sizeBytes: 3_809),
            ModelFile(filename: "model_weights.json", sizeBytes: 251_539),
            ModelFile(filename: "model_weights.bin", sizeBytes: 1_378_869_376),
            ModelFile(filename: "vocab.bin", sizeBytes: 3_360_287),
            ModelFile(filename: "tokenizer.json", sizeBytes: 19_989_343),
            ModelFile(filename: "tokenizer.bin", sizeBytes: 8_201_040),
            ModelFile(filename: "packed_experts_tiered/tiered_manifest.json", sizeBytes: 1_005_120),
        ]
        // Variable-size tiered layers (from flash-moe iOS port)
        let layerSizes: [UInt64] = [
            337_379_328, 349_175_808, 342_097_920, 331_087_872, 320_077_824,
            301_989_888, 301_989_888, 289_406_976, 285_474_816, 294_125_568,
            305_922_048, 306_708_480, 297_271_296, 293_339_136, 282_329_088,
            288_620_544, 287_834_112, 292_552_704, 280_756_224, 287_834_112,
            282_329_088, 283_115_520, 301_989_888, 305_135_616, 294_125_568,
            294_125_568, 281_542_656, 292_552_704, 296_484_864, 298_844_160,
            289_406_976, 291_766_272, 301_989_888, 302_776_320, 305_135_616,
            300_417_024, 298_057_728, 304_349_184, 301_989_888, 309_854_208,
        ]
        for (i, size) in layerSizes.enumerated() {
            files.append(ModelFile(
                filename: String(format: "packed_experts_tiered/layer_%02d.bin", i),
                sizeBytes: size
            ))
        }

        return CatalogEntry(
            id: "qwen3.5-35b-a3b-tiered",
            displayName: "Qwen 3.5 35B-A3B Tiered",
            repoId: "alexintosh/Qwen3.5-35B-A3B-Q4-Tiered-FlashMoE",
            description: "Tiered: hot experts 4-bit, cold 2-bit. Faster on iPhone. ~13.4 GB.",
            totalSizeBytes: 13_424_643_082,
            quantization: "Tiered (4-bit/2-bit)",
            expertLayers: 40,
            files: files
        )
    }
}
