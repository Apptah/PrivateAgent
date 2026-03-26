import Foundation

/// Model architecture config — supports both flat HF format and nested Qwen3.5 format
/// (where fields live under `text_config`).
public struct HFConfig: Sendable {
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
    public let layerTypes: [String]?
}

extension HFConfig: Codable {
    // Internal flat struct matching standard HF keys
    private struct Flat: Codable {
        let hiddenSize: Int?
        let numHiddenLayers: Int?
        let numAttentionHeads: Int?
        let numKeyValueHeads: Int?
        let headDim: Int?
        let vocabSize: Int?
        let numExperts: Int?
        let numExpertsPerTok: Int?
        let moeIntermediateSize: Int?
        let maxPositionEmbeddings: Int?
        let rmsNormEps: Float?
        let layerTypes: [String]?

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
            case layerTypes = "layer_types"
        }
    }

    // Wrapper for nested Qwen3.5 format
    private struct Nested: Codable {
        let textConfig: Flat?
        enum CodingKeys: String, CodingKey {
            case textConfig = "text_config"
        }
    }

    public init(from decoder: Decoder) throws {
        // Try flat format first
        let flat = try Flat(from: decoder)

        // If hidden_size is at top level, use flat
        if let hs = flat.hiddenSize, let nhl = flat.numHiddenLayers, let vs = flat.vocabSize {
            self.hiddenSize = hs
            self.numHiddenLayers = nhl
            self.numAttentionHeads = flat.numAttentionHeads ?? 16
            self.numKeyValueHeads = flat.numKeyValueHeads ?? 2
            self.headDim = flat.headDim
            self.vocabSize = vs
            self.numExperts = flat.numExperts
            self.numExpertsPerTok = flat.numExpertsPerTok
            self.moeIntermediateSize = flat.moeIntermediateSize
            self.maxPositionEmbeddings = flat.maxPositionEmbeddings
            self.rmsNormEps = flat.rmsNormEps
            self.layerTypes = flat.layerTypes
            return
        }

        // Try nested text_config
        let nested = try Nested(from: decoder)
        guard let tc = nested.textConfig else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: [], debugDescription: "No hidden_size at top level and no text_config found"
            ))
        }

        self.hiddenSize = tc.hiddenSize ?? 2048
        self.numHiddenLayers = tc.numHiddenLayers ?? 40
        self.numAttentionHeads = tc.numAttentionHeads ?? 16
        self.numKeyValueHeads = tc.numKeyValueHeads ?? 2
        self.headDim = tc.headDim
        self.vocabSize = tc.vocabSize ?? 248320
        self.numExperts = tc.numExperts
        self.numExpertsPerTok = tc.numExpertsPerTok
        self.moeIntermediateSize = tc.moeIntermediateSize
        self.maxPositionEmbeddings = tc.maxPositionEmbeddings
        self.rmsNormEps = tc.rmsNormEps
        // Map "linear_attention" → "gdn", "full_attention" → "full_attn"
        self.layerTypes = tc.layerTypes?.map { type in
            switch type {
            case "linear_attention": return "gdn"
            case "full_attention": return "full_attn"
            default: return type
            }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: Flat.CodingKeys.self)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encodeIfPresent(headDim, forKey: .headDim)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encodeIfPresent(numExperts, forKey: .numExperts)
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
