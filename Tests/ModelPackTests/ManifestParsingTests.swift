import Testing
import Foundation
import FlashMoECore
@testable import ModelPack

@Suite("Manifest Parsing Tests")
struct ManifestParsingTests {
    private var fixturesDir: URL {
        Bundle.module.resourceURL!.appendingPathComponent("Fixtures")
    }

    @Test("Parse config.json + privateagent-manifest.json")
    func parseFullManifest() throws {
        let manifest = try ModelManifest(modelDir: fixturesDir)
        #expect(manifest.hfConfig.hiddenSize == 2048)
        #expect(manifest.hfConfig.numHiddenLayers == 60)
        #expect(manifest.hfConfig.vocabSize == 151936)
        #expect(manifest.hfConfig.numExperts == 128)
        #expect(manifest.manifestVersion == 1)
        #expect(manifest.paManifest != nil)
        #expect(manifest.paManifest?.expertLayout?.quantBits == 4)
        #expect(manifest.paManifest?.turboQuantDefaults?.keyBitsX2 == 7)
    }

    @Test("Layer types default to every-4th-full-attn")
    func layerTypesDefault() throws {
        let manifest = try ModelManifest(modelDir: fixturesDir)
        let types = manifest.layerTypes
        #expect(types.count == 60)
        #expect(types[0] == "gdn")
        #expect(types[3] == "full_attn")
        #expect(types[7] == "full_attn")
    }

    @Test("PAModelDescBridge produces valid PA_ModelDesc")
    func bridgeProducesValidDesc() throws {
        let manifest = try ModelManifest(modelDir: fixturesDir)
        var desc = PAModelDescBridge.makeModelDesc(from: manifest)
        #expect(pa_model_desc_validate(&desc) == PA_STATUS_OK.rawValue)
        #expect(desc.num_layers == 60)
        #expect(desc.hidden_dim == 2048)
        #expect(desc.num_kv_heads == 2)
        #expect(desc.vocab_size == 151936)
        #expect(desc.num_experts == 128)
        #expect(desc.active_experts_k == 8)
        #expect(desc.expert_quant_bits == 4)
        #expect(desc.default_key_bits_x2 == 7)
    }

    @Test("PAModelDescBridge sets layer types correctly")
    func bridgeSetsLayerTypes() throws {
        let manifest = try ModelManifest(modelDir: fixturesDir)
        var desc = PAModelDescBridge.makeModelDesc(from: manifest)
        #expect(pa_model_desc_full_attn_count(&desc) == 15)
        #expect(pa_model_desc_gdn_count(&desc) == 45)
    }

    @Test("Missing config.json throws")
    func missingConfig() {
        let emptyDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try! FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDir) }
        #expect(throws: (any Error).self) {
            _ = try ModelManifest(modelDir: emptyDir)
        }
    }
}
