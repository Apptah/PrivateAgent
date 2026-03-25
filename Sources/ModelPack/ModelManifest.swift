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
        if let types = paManifest?.layerTypes, types.count == hfConfig.numHiddenLayers {
            return types
        }
        return (0..<hfConfig.numHiddenLayers).map { i in
            i % 4 == 3 ? "full_attn" : "gdn"
        }
    }
}
