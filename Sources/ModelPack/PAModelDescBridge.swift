import Foundation
import FlashMoECore

/// Converts a Swift ModelManifest into a C PA_ModelDesc struct
/// for passing into the FlashMoERuntime.
public enum PAModelDescBridge {

    // MARK: - Public API

    /// Build a PA_ModelDesc from a ModelManifest.
    public static func makeModelDesc(from manifest: ModelManifest) -> PA_ModelDesc {
        var desc = PA_ModelDesc()

        // Paths
        let modelPath = manifest.modelDir.path
        let weightsPath = manifest.modelDir.appendingPathComponent("model_weights.bin").path
        let tokenizerPath = manifest.modelDir.appendingPathComponent("tokenizer.bin").path
        setCString(&desc.model_dir, modelPath)
        setCString(&desc.weights_path, weightsPath)
        setCString(&desc.tokenizer_path, tokenizerPath)

        // Architecture from HFConfig
        let cfg = manifest.hfConfig
        desc.num_layers              = UInt32(cfg.numHiddenLayers)
        desc.num_experts             = UInt32(cfg.numExperts ?? 0)
        desc.active_experts_k        = UInt32(cfg.numExpertsPerTok ?? 0)
        desc.hidden_dim              = UInt32(cfg.hiddenSize)
        desc.vocab_size              = UInt32(cfg.vocabSize)
        desc.num_attn_heads          = UInt32(cfg.numAttentionHeads)
        desc.num_kv_heads            = UInt32(cfg.numKeyValueHeads)
        desc.head_dim                = UInt32(cfg.headDim ?? (cfg.hiddenSize / cfg.numAttentionHeads))
        desc.moe_intermediate        = UInt32(cfg.moeIntermediateSize ?? 0)
        desc.max_position_embeddings = UInt32(cfg.maxPositionEmbeddings ?? 0)
        desc.rms_norm_eps            = cfg.rmsNormEps ?? 0.0

        // Layer types array (C fixed array imported as Swift tuple)
        let layerTypes = manifest.layerTypes
        withUnsafeMutablePointer(to: &desc.layer_types) { ptr in
            let raw = UnsafeMutableRawPointer(ptr)
            let bound = raw.bindMemory(to: PA_LayerType.self, capacity: Int(PA_MAX_LAYERS))
            for (i, typeName) in layerTypes.prefix(Int(PA_MAX_LAYERS)).enumerated() {
                bound[i] = typeName == "full_attn" ? PA_LAYER_FULL_ATTN : PA_LAYER_GDN
            }
        }

        // Expert layout from PAManifest
        if let layout = manifest.paManifest?.expertLayout {
            desc.expert_quant_bits = UInt32(layout.quantBits)
            desc.dense_quant_bits  = UInt32(layout.denseQuantBits)
            desc.expert_size_each  = layout.expertSizeEach
        }

        // TurboQuant defaults from PAManifest
        if let tq = manifest.paManifest?.turboQuantDefaults {
            desc.default_key_bits_x2     = UInt16(tq.keyBitsX2)
            desc.default_value_bits_x2   = UInt16(tq.valueBitsX2)
            desc.default_tq_block_size   = UInt32(tq.blockSize)
            desc.default_transform_kind  = transformKindValue(tq.transformKind)
            desc.default_transform_seed  = tq.transformSeed
        }

        // Manifest version
        desc.manifest_version = manifest.manifestVersion

        return desc
    }

    // MARK: - Helpers

    /// Copy a Swift String into a C fixed-size char array (imported as a Swift tuple).
    private static func setCString<T>(_ dest: inout T, _ value: String) {
        withUnsafeMutablePointer(to: &dest) { ptr in
            let raw = UnsafeMutableRawPointer(ptr)
            let bound = raw.bindMemory(to: CChar.self, capacity: MemoryLayout<T>.size)
            value.withCString { src in
                strncpy(bound, src, MemoryLayout<T>.size - 1)
                bound[MemoryLayout<T>.size - 1] = 0
            }
        }
    }

    /// Map transform_kind string to PA_TransformKind raw value.
    private static func transformKindValue(_ kind: String) -> UInt32 {
        switch kind {
        case "structured_rotation": return PA_TRANSFORM_STRUCTURED_ROTATION.rawValue
        case "hadamard":            return PA_TRANSFORM_HADAMARD.rawValue
        default:                    return PA_TRANSFORM_NONE.rawValue
        }
    }
}
