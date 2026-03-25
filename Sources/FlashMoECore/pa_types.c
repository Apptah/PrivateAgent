#include "pa_types.h"
#include "pa_status.h"
#include <string.h>
#include <stdint.h>

int pa_model_desc_validate(const PA_ModelDesc *desc) {
    if (!desc) return PA_STATUS_ERROR_GENERIC;
    if (desc->num_layers == 0 || desc->num_layers > PA_MAX_LAYERS)
        return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->hidden_dim == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->vocab_size == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->num_kv_heads == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->head_dim == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->model_dir[0] == '\0') return PA_STATUS_ERROR_INVALID_MODEL;
    if (desc->manifest_version == 0) return PA_STATUS_ERROR_INVALID_MODEL;
    return PA_STATUS_OK;
}

int pa_tensor_ref_validate(const PA_TensorRef *ref) {
    if (!ref) return PA_STATUS_ERROR_GENERIC;
    if (ref->rank == 0 || ref->rank > 4) return PA_STATUS_ERROR_GENERIC;
    for (uint32_t i = 0; i < ref->rank; i++) {
        if (ref->shape[i] == 0) return PA_STATUS_ERROR_GENERIC;
    }
    return PA_STATUS_OK;
}

uint32_t pa_model_desc_full_attn_count(const PA_ModelDesc *desc) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < desc->num_layers; i++) {
        if (desc->layer_types[i] == PA_LAYER_FULL_ATTN) count++;
    }
    return count;
}

uint32_t pa_model_desc_gdn_count(const PA_ModelDesc *desc) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < desc->num_layers; i++) {
        if (desc->layer_types[i] == PA_LAYER_GDN) count++;
    }
    return count;
}

int pa_compute_memory_budget(const PA_ModelDesc *desc, uint64_t available_bytes, PA_MemoryBudget *out_budget) {
    if (!desc || !out_budget) return PA_STATUS_ERROR_GENERIC;

    memset(out_budget, 0, sizeof(PA_MemoryBudget));

    // Fixed estimates
    const uint64_t metal_buffers = 200ULL * 1024 * 1024;   // 200 MB
    const uint64_t scratch       = 5ULL * 1024 * 1024;     // 5 MB

    // Expert buffers: active_k * 2 (key+value sides) * bytes_per_expert
    uint64_t expert_buffers = (uint64_t)desc->active_experts_k * 2ULL * desc->expert_size_each;

    // GDN state: gdn_layers * num_attn_heads * head_dim * head_dim * sizeof(float)
    uint32_t gdn_layers = pa_model_desc_gdn_count(desc);
    uint64_t gdn_state  = (uint64_t)gdn_layers
                        * (uint64_t)desc->num_attn_heads
                        * (uint64_t)desc->head_dim
                        * (uint64_t)desc->head_dim
                        * sizeof(float);

    // mmap weights: hidden_dim * vocab_size / 2  (4-bit estimate)
    uint64_t mmap_weights = (uint64_t)desc->hidden_dim * (uint64_t)desc->vocab_size / 2ULL;

    // Total dirty (resident non-mmap)
    uint64_t total_dirty = metal_buffers + expert_buffers + gdn_state + scratch;

    // Sanity: need at least the dirty bytes
    if (available_bytes < total_dirty + 1024 * 1024) {
        return PA_STATUS_ERROR_OOM;
    }

    // KV budget: 70% of available minus dirty
    uint64_t kv_budget = (uint64_t)((double)available_bytes * 0.70) - total_dirty;
    if (kv_budget > available_bytes) kv_budget = 0; // underflow guard

    // Bytes per token for KV cache
    // Each token needs: full_attn_layers * num_kv_heads * head_dim * (key_bits + value_bits) / 8
    uint32_t full_attn_layers = pa_model_desc_full_attn_count(desc);
    uint64_t bytes_per_token;
    if (desc->default_key_bits_x2 > 0 && desc->default_value_bits_x2 > 0) {
        // TurboQuant path: bits_x2 / 2 gives actual bits, /8 converts to bytes
        // key: num_kv_heads * head_dim * key_bits / 8
        // value: num_kv_heads * head_dim * value_bits / 8
        // Combined as integer arithmetic: (bits_x2 * heads * head_dim) / 16
        uint64_t key_bytes_per_layer   = ((uint64_t)desc->default_key_bits_x2
                                          * (uint64_t)desc->num_kv_heads
                                          * (uint64_t)desc->head_dim + 15) / 16;
        uint64_t value_bytes_per_layer = ((uint64_t)desc->default_value_bits_x2
                                          * (uint64_t)desc->num_kv_heads
                                          * (uint64_t)desc->head_dim + 15) / 16;
        bytes_per_token = (uint64_t)full_attn_layers * (key_bytes_per_layer + value_bytes_per_layer);
    } else {
        // BF16 fallback: 2 bytes per element, key + value
        uint64_t kv_bytes_per_layer = 2ULL * (uint64_t)desc->num_kv_heads * (uint64_t)desc->head_dim * 2ULL;
        bytes_per_token = (uint64_t)full_attn_layers * kv_bytes_per_layer;
    }

    // Compute max_context as power-of-2 clamped to [512, 8192]
    uint32_t max_ctx = 0;
    if (bytes_per_token > 0) {
        uint64_t raw = kv_budget / bytes_per_token;
        // Round down to power of 2
        uint32_t p2 = 512;
        while (p2 * 2 <= raw && p2 < 8192) p2 *= 2;
        if (raw < 512) p2 = 0; // insufficient even for minimum
        max_ctx = p2;
    }

    if (max_ctx == 0) return PA_STATUS_ERROR_OOM;

    // Actual KV bytes used for the chosen context
    uint64_t kv_cache_bytes = (uint64_t)max_ctx * bytes_per_token;

    // Fill output
    out_budget->metal_buffers_bytes  = metal_buffers;
    out_budget->gdn_state_bytes      = gdn_state;
    out_budget->expert_buffers_bytes = expert_buffers;
    out_budget->kv_cache_bytes       = kv_cache_bytes;
    out_budget->scratch_bytes        = scratch;
    out_budget->total_dirty_bytes    = total_dirty;
    out_budget->mmap_weights_bytes   = mmap_weights;
    out_budget->total_resident_bytes = total_dirty + kv_cache_bytes;
    out_budget->max_context_length   = max_ctx;
    out_budget->full_attn_layer_count = full_attn_layers;

    return PA_STATUS_OK;
}
