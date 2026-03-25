import Foundation

public struct HFConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int?
    public let vocabSize: Int
    public let numExperts: Int?
    public let numExpertsPerTok: Int?
    public let moeIntermediateSize: Int?
    public let maxPositionEmbeddings: Int?
    public let rmsNormEps: Float?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
    }
}

public struct PAManifest: Codable, Sendable {
    public let manifestVersion: Int
    public let layerTypes: [String]?
    public let expertLayout: ExpertLayout?
    public let turboQuantDefaults: TQDefaults?
    public let checksums: [String: String]?

    enum CodingKeys: String, CodingKey {
        case manifestVersion = "manifest_version"
        case layerTypes = "layer_types"
        case expertLayout = "expert_layout"
        case turboQuantDefaults = "turboquant_defaults"
        case checksums
    }
}

public struct ExpertLayout: Codable, Sendable {
    public let quantBits: Int
    public let denseQuantBits: Int
    public let expertSizeEach: UInt64
    public let expertLayers: Int

    enum CodingKeys: String, CodingKey {
        case quantBits = "quant_bits"
        case denseQuantBits = "dense_quant_bits"
        case expertSizeEach = "expert_size_each"
        case expertLayers = "expert_layers"
    }
}

public struct TQDefaults: Codable, Sendable {
    public let keyBitsX2: Int
    public let valueBitsX2: Int
    public let blockSize: Int
    public let transformKind: String
    public let transformSeed: UInt64

    enum CodingKeys: String, CodingKey {
        case keyBitsX2 = "key_bits_x2"
        case valueBitsX2 = "value_bits_x2"
        case blockSize = "block_size"
        case transformKind = "transform_kind"
        case transformSeed = "transform_seed"
    }
}
