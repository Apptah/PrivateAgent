import Testing
import Foundation
@testable import FlashMoECore

// MARK: - Helpers

/// Build a Gemma-4-26B-A4B-style PA_ModelDesc with:
///   30 layers, 25 GDN + 5 full_attn, hidden_dim=2816, vocab=262144,
///   num_attn_heads=16, num_kv_heads=8, head_dim=256,
///   expert_size≈3MB, active_k=8, num_experts=128
///   TQ 3.5-bit: key_bits_x2=7, value_bits_x2=7
private func makeGemma4Desc(keyBitsX2: UInt16 = 7, valueBitsX2: UInt16 = 7) -> PA_ModelDesc {
    var desc = PA_ModelDesc()

    // Set model_dir via unsafe pointer (C char array imports as Swift tuple)
    withUnsafeMutablePointer(to: &desc.model_dir) { ptr in
        let raw = UnsafeMutableRawPointer(ptr)
        let bound = raw.bindMemory(to: CChar.self, capacity: Int(PA_MAX_PATH))
        "/tmp/gemma4-model".withCString { src in
            _ = strcpy(bound, src)
        }
    }

    desc.manifest_version       = 1
    desc.num_layers             = 30
    desc.num_experts            = 128
    desc.active_experts_k       = 8
    desc.hidden_dim             = 2816
    desc.vocab_size             = 262144
    desc.num_attn_heads         = 16
    desc.num_kv_heads           = 8
    desc.head_dim               = 256
    desc.moe_intermediate       = 704
    desc.max_position_embeddings = 262144
    desc.rms_norm_eps           = 1e-6
    desc.expert_quant_bits      = 4
    desc.dense_quant_bits       = 4
    desc.expert_size_each       = 2975232  // ~2.8 MB per expert

    // TurboQuant settings
    desc.default_key_bits_x2   = keyBitsX2
    desc.default_value_bits_x2 = valueBitsX2
    desc.default_tq_block_size  = 64
    desc.default_transform_kind = UInt32(PA_TRANSFORM_HADAMARD.rawValue)
    desc.default_transform_seed = 42

    // 25 GDN + 5 full_attn layers (every 6th layer is full_attn)
    withUnsafeMutablePointer(to: &desc.layer_types) { ptr in
        let base = UnsafeMutableRawPointer(ptr).bindMemory(to: PA_LayerType.self, capacity: Int(PA_MAX_LAYERS))
        for i in 0..<30 {
            base[i] = (i % 6 == 5) ? PA_LAYER_FULL_ATTN : PA_LAYER_GDN
        }
    }

    return desc
}

// MARK: - Tests

@Suite("Memory Planner Tests")
struct MemoryPlannerTests {

    @Test("6GB available: produces valid budget for Gemma-4-26B-A4B with TQ 3.5-bit")
    func budgetGemma4_6GB_TQ() {
        var desc = makeGemma4Desc(keyBitsX2: 7, valueBitsX2: 7)
        let available: UInt64 = 6 * 1024 * 1024 * 1024  // 6 GB

        var budget = PA_MemoryBudget()
        let status = pa_compute_memory_budget(&desc, available, &budget)

        #expect(status == PA_STATUS_OK.rawValue)

        // Struct fields sanity
        #expect(budget.metal_buffers_bytes == 200 * 1024 * 1024)
        #expect(budget.scratch_bytes == 5 * 1024 * 1024)

        // expert_buffers = active_k * 2 * expert_size_each = 8 * 2 * 2975232 ≈ 45MB
        let expectedExpert: UInt64 = 8 * 2 * 2975232
        #expect(budget.expert_buffers_bytes == expectedExpert)

        // gdn_state = 25 * 16 * 256 * 256 * 4 bytes
        let expectedGDN: UInt64 = 25 * 16 * 256 * 256 * 4
        #expect(budget.gdn_state_bytes == expectedGDN)

        // mmap_weights = hidden_dim * vocab_size / 2
        let expectedMmap: UInt64 = 2816 * 262144 / 2
        #expect(budget.mmap_weights_bytes == expectedMmap)

        // full_attn_layer_count: 30 layers, every 6th is full_attn → 5
        #expect(budget.full_attn_layer_count == 5)

        // max_context must be a power of 2 in [512, 8192]
        let ctx = budget.max_context_length
        #expect(ctx >= 512)
        #expect(ctx <= 8192)
        #expect(ctx & (ctx - 1) == 0)  // power of 2

        // kv_cache_bytes must be positive
        #expect(budget.kv_cache_bytes > 0)

        // total_resident = total_dirty + kv_cache
        #expect(budget.total_resident_bytes == budget.total_dirty_bytes + budget.kv_cache_bytes)

        // total_dirty = metal + expert + gdn + scratch
        let expectedDirty = budget.metal_buffers_bytes + budget.expert_buffers_bytes
                          + budget.gdn_state_bytes + budget.scratch_bytes
        #expect(budget.total_dirty_bytes == expectedDirty)
    }

    @Test("100MB available: OOM rejection")
    func budgetOOM_100MB() {
        var desc = makeGemma4Desc()
        let available: UInt64 = 100 * 1024 * 1024  // 100 MB

        var budget = PA_MemoryBudget()
        let status = pa_compute_memory_budget(&desc, available, &budget)

        #expect(status == PA_STATUS_ERROR_OOM.rawValue)
    }

    @Test("TurboQuant key_bits_x2=7 yields >= context length than BF16 (key_bits_x2=0)")
    func tqGivesMoreContextThanBF16() {
        let available: UInt64 = 6 * 1024 * 1024 * 1024  // 6 GB

        var descTQ  = makeGemma4Desc(keyBitsX2: 7, valueBitsX2: 7)
        var descBF16 = makeGemma4Desc(keyBitsX2: 0, valueBitsX2: 0)

        var budgetTQ   = PA_MemoryBudget()
        var budgetBF16 = PA_MemoryBudget()

        let statusTQ   = pa_compute_memory_budget(&descTQ,   available, &budgetTQ)
        let statusBF16 = pa_compute_memory_budget(&descBF16, available, &budgetBF16)

        #expect(statusTQ   == PA_STATUS_OK.rawValue)
        #expect(statusBF16 == PA_STATUS_OK.rawValue)

        // TurboQuant at 3.5 bits should allow >= context than 16-bit BF16
        #expect(budgetTQ.max_context_length >= budgetBF16.max_context_length)
    }

    @Test("Null pointer inputs return generic error")
    func nullInputsReturnError() {
        var desc   = makeGemma4Desc()
        var budget = PA_MemoryBudget()

        #expect(pa_compute_memory_budget(nil, 6 * 1024 * 1024 * 1024, &budget) != PA_STATUS_OK.rawValue)
        #expect(pa_compute_memory_budget(&desc, 6 * 1024 * 1024 * 1024, nil)   != PA_STATUS_OK.rawValue)
    }
}
