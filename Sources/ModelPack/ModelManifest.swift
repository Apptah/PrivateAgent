import Foundation

public struct ModelManifest: Sendable {
    public let modelDir: URL
    public let hfConfig: HFConfig
    public let paManifest: PAManifest?
    public let manifestVersion: UInt32

    public init(modelDir: URL) throws {
        self.modelDir = modelDir
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        self.hfConfig = try JSONDecoder().decode(HFConfig.self, from: configData)

        let manifestURL = modelDir.appendingPathComponent("privateagent-manifest.json")
        if let manifestData = try? Data(contentsOf: manifestURL) {
            self.paManifest = try JSONDecoder().decode(PAManifest.self, from: manifestData)
            self.manifestVersion = UInt32(self.paManifest?.manifestVersion ?? 1)
        } else {
            self.paManifest = nil
            self.manifestVersion = 1
        }
    }

    public var layerTypes: [String] {
        // 1. PA manifest layer types (highest priority)
        if let types = paManifest?.layerTypes, types.count == hfConfig.numHiddenLayers {
            return types
        }
        // 2. HF config layer types (from text_config.layer_types, already mapped to gdn/full_attn)
        if let types = hfConfig.layerTypes, types.count == hfConfig.numHiddenLayers {
            return types
        }
        // 3. Default: every 6th layer is full_attn (matches Gemma 4 sliding-window pattern)
        return (0..<hfConfig.numHiddenLayers).map { i in
            i % 6 == 5 ? "full_attn" : "gdn"
        }
    }
}
