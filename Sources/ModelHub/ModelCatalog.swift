// ModelCatalog.swift
// ModelHub

import Foundation

// MARK: - ModelCatalog

public actor ModelCatalog {

    // MARK: Public

    public static let shared = ModelCatalog()

    public private(set) var entries: [CatalogEntry] = ModelCatalog.bundledCatalog

    /// Placeholder for future remote catalog fetch.
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

    // MARK: Helpers

    private static func configFiles() -> [ModelFile] {
        [
            ModelFile(filename: "config.json",         sizeBytes: 4_096),
            ModelFile(filename: "model_weights.json",  sizeBytes: 512_000),
            ModelFile(filename: "model_weights.bin",   sizeBytes: 0),    // placeholder; actual size in entry total
            ModelFile(filename: "vocab.bin",           sizeBytes: 2_097_152),
            ModelFile(filename: "tokenizer.json",      sizeBytes: 1_048_576),
            ModelFile(filename: "tokenizer.bin",       sizeBytes: 1_048_576),
        ]
    }

    private static func expertFiles(count: Int, bytesEach: UInt64) -> [ModelFile] {
        (0 ..< count).map { layer in
            ModelFile(
                filename: String(format: "expert_%03d.bin", layer),
                sizeBytes: bytesEach
            )
        }
    }

    // MARK: Entry Factories

    /// Q4 quantisation — 19.5 GB total, 40 expert layers @ 452 MB each
    private static func makeQ4Entry() -> CatalogEntry {
        let expertSizeBytes: UInt64 = 452 * 1_048_576    // 452 MB
        let expertLayerCount = 40

        var files = configFiles()
        files += expertFiles(count: expertLayerCount, bytesEach: expertSizeBytes)

        // Total: 19.5 GB
        let totalSizeBytes: UInt64 = 19_947_524_096      // ≈ 19.5 GiB

        return CatalogEntry(
            id: "qwen3.5-35b-a3b-q4",
            displayName: "Qwen 3.5 35B-A3B (Q4)",
            repoId: "Qwen/Qwen3.5-35B-A3B-Q4_K_M-GGUF",
            description: "Qwen 3.5 35B MoE model with Q4_K_M quantisation. 40 expert layers at 452 MB each. Best quality/size tradeoff.",
            totalSizeBytes: totalSizeBytes,
            quantization: "Q4_K_M",
            expertLayers: expertLayerCount,
            files: files
        )
    }

    /// Tiered quantisation — 13.4 GB total, 40 expert layers @ 300 MB each
    private static func makeTieredEntry() -> CatalogEntry {
        let expertSizeBytes: UInt64 = 300 * 1_048_576    // 300 MB
        let expertLayerCount = 40

        var files = configFiles()
        files += expertFiles(count: expertLayerCount, bytesEach: expertSizeBytes)

        // Total: 13.4 GB
        let totalSizeBytes: UInt64 = 14_390_108_160      // ≈ 13.4 GiB

        return CatalogEntry(
            id: "qwen3.5-35b-a3b-tiered",
            displayName: "Qwen 3.5 35B-A3B (Tiered)",
            repoId: "Qwen/Qwen3.5-35B-A3B-Tiered-GGUF",
            description: "Qwen 3.5 35B MoE model with tiered quantisation. 40 expert layers at 300 MB each. Smaller footprint for memory-constrained devices.",
            totalSizeBytes: totalSizeBytes,
            quantization: "Tiered",
            expertLayers: expertLayerCount,
            files: files
        )
    }
}
