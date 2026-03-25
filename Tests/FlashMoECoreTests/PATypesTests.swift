import Testing
import Foundation
@testable import FlashMoECore

@Suite("PA_Types C ABI Tests")
struct PATypesTests {

    @Test("PA_TensorRef has expected size and layout")
    func tensorRefLayout() {
        #expect(MemoryLayout<PA_TensorRef>.size > 0)
        var ref = PA_TensorRef()
        ref.rank = 2
        ref.shape.0 = 4
        ref.shape.1 = 8
        ref.dtype = 0  // f32
        ref.storage_kind = UInt32(PA_STORAGE_CPU.rawValue)
        ref.quant_scheme = UInt32(PA_QUANT_NONE.rawValue)
        #expect(pa_tensor_ref_validate(&ref) == PA_STATUS_OK.rawValue)
    }

    @Test("PA_TensorRef validation rejects rank 0")
    func tensorRefRejectsRank0() {
        var ref = PA_TensorRef()
        ref.rank = 0
        #expect(pa_tensor_ref_validate(&ref) != PA_STATUS_OK.rawValue)
    }

    @Test("PA_TensorRef validation rejects rank > 4")
    func tensorRefRejectsRank5() {
        var ref = PA_TensorRef()
        ref.rank = 5
        #expect(pa_tensor_ref_validate(&ref) != PA_STATUS_OK.rawValue)
    }

    @Test("PA_ModelDesc validation passes for valid desc")
    func modelDescValid() {
        var desc = PA_ModelDesc()
        withUnsafeMutablePointer(to: &desc.model_dir) { ptr in
            let raw = UnsafeMutableRawPointer(ptr)
            let bound = raw.bindMemory(to: CChar.self, capacity: Int(PA_MAX_PATH))
            "/tmp/model".withCString { src in
                _ = strcpy(bound, src)
            }
        }
        desc.num_layers = 60
        desc.hidden_dim = 2048
        desc.vocab_size = 151936
        desc.num_kv_heads = 2
        desc.head_dim = 256
        desc.manifest_version = 1
        #expect(pa_model_desc_validate(&desc) == PA_STATUS_OK.rawValue)
    }

    @Test("PA_ModelDesc validation rejects 0 layers")
    func modelDescRejectsZeroLayers() {
        var desc = PA_ModelDesc()
        desc.num_layers = 0
        #expect(pa_model_desc_validate(&desc) != PA_STATUS_OK.rawValue)
    }

    @Test("PA_ModelDesc validation rejects > PA_MAX_LAYERS")
    func modelDescRejectsTooManyLayers() {
        var desc = PA_ModelDesc()
        desc.num_layers = UInt32(PA_MAX_LAYERS) + 1
        #expect(pa_model_desc_validate(&desc) != PA_STATUS_OK.rawValue)
    }

    @Test("pa_model_desc_full_attn_count counts correctly")
    func fullAttnCount() {
        var desc = PA_ModelDesc()
        desc.num_layers = 5
        withUnsafeMutablePointer(to: &desc.layer_types) { ptr in
            let base = UnsafeMutableRawPointer(ptr).bindMemory(to: PA_LayerType.self, capacity: Int(PA_MAX_LAYERS))
            base[0] = PA_LAYER_GDN
            base[1] = PA_LAYER_GDN
            base[2] = PA_LAYER_GDN
            base[3] = PA_LAYER_FULL_ATTN
            base[4] = PA_LAYER_FULL_ATTN
        }
        #expect(pa_model_desc_full_attn_count(&desc) == 2)
        #expect(pa_model_desc_gdn_count(&desc) == 3)
    }

    @Test("pa_bits_from_x2 converts correctly")
    func bitsFromX2() {
        #expect(pa_bits_from_x2(7) == 3.5)
        #expect(pa_bits_from_x2(6) == 3.0)
        #expect(pa_bits_from_x2(8) == 4.0)
    }

    @Test("PA_QuantizedKVDesc key/value bits encoding")
    func kvDescBitsEncoding() {
        var kv = PA_QuantizedKVDesc()
        kv.key_bits_x2 = 7    // 3.5 bits
        kv.value_bits_x2 = 8  // 4.0 bits
        #expect(pa_bits_from_x2(kv.key_bits_x2) == 3.5)
        #expect(pa_bits_from_x2(kv.value_bits_x2) == 4.0)
    }

    @Test("pa_status_string returns non-null for all codes")
    func statusStrings() {
        let codes: [PA_Status] = [
            PA_STATUS_OK, PA_STATUS_ERROR_GENERIC, PA_STATUS_ERROR_OOM,
            PA_STATUS_ERROR_IO, PA_STATUS_ERROR_INVALID_MODEL,
            PA_STATUS_ERROR_METAL_INIT, PA_STATUS_ERROR_LOAD_FAILED,
            PA_STATUS_CONTEXT_EXHAUSTED, PA_STATUS_CANCELLED, PA_STATUS_THROTTLED,
        ]
        for code in codes {
            let str = pa_status_string(code)
            #expect(str != nil)
            #expect(String(cString: str!) != "Unknown status")
        }
    }
}
