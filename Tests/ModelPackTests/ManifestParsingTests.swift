import Testing
import Foundation
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
