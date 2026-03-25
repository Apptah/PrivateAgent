import Foundation
import FlashMoECore

/// Converts a Swift ModelManifest into a C PA_ModelDesc struct
/// for passing into the FlashMoERuntime.
public enum PAModelDescBridge {

    /// Build a PA_ModelDesc from a ModelManifest.
    /// Full implementation in Plan 2.
    public static func makeModelDesc(from manifest: ModelManifest) -> PA_ModelDesc {
        var desc = PA_ModelDesc()
        desc.manifest_version = manifest.manifestVersion
        return desc
    }
}
