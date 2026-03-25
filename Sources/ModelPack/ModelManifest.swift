import Foundation

/// Parsed model manifest from config.json + privateagent-manifest.json.
/// Responsible for reading model metadata and producing a PA_ModelDesc
/// for the C runtime.
public struct ModelManifest: Sendable {
    public let modelDir: URL
    public let manifestVersion: UInt32

    // Placeholder — full parsing added in Plan 2.
    public init(modelDir: URL) {
        self.modelDir = modelDir
        self.manifestVersion = 1
    }
}
