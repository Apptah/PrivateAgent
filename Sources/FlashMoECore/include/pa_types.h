#ifndef PA_TYPES_H
#define PA_TYPES_H

#include <stdint.h>
#include <stddef.h>

// ── Storage kinds ──
typedef enum {
    PA_STORAGE_CPU = 0,
    PA_STORAGE_METAL_BUFFER = 1,
    PA_STORAGE_QUANTIZED_KV = 2,
} PA_StorageKind;

// ── Quantization schemes ──
typedef enum {
    PA_QUANT_NONE = 0,
    PA_QUANT_Q2 = 2,
    PA_QUANT_Q3 = 3,
    PA_QUANT_Q4 = 4,
    PA_QUANT_TQ3 = 103,    // TurboQuant 3-bit
    PA_QUANT_TQ3_5 = 107,  // TurboQuant 3.5-bit (custom)
    PA_QUANT_TQ4 = 104,    // TurboQuant 4-bit
} PA_QuantScheme;

// ── Tensor reference ──
typedef struct {
    void *data;
    uint64_t byte_offset;
    uint32_t dtype;          // 0=f32, 1=f16, 2=bf16, 3=i32, 4=u8
    uint32_t rank;
    uint32_t shape[4];
    uint32_t stride[4];
    uint32_t storage_kind;   // PA_StorageKind
    uint32_t quant_scheme;   // PA_QuantScheme
} PA_TensorRef;

// ── TurboQuant KV descriptor ──
typedef enum {
    PA_AUX_NONE = 0,
    PA_AUX_GLOBAL = 1,
    PA_AUX_PER_BLOCK = 2,
} PA_AuxParamsKind;

typedef enum {
    PA_TRANSFORM_NONE = 0,
    PA_TRANSFORM_STRUCTURED_ROTATION = 1,
    PA_TRANSFORM_HADAMARD = 2,
} PA_TransformKind;

typedef struct {
    uint16_t key_bits_x2;        // bits * 2: 7 = 3.5 bits, 6 = 3 bits, 8 = 4 bits
    uint16_t value_bits_x2;      // separate from key_bits; same encoding
    uint32_t block_size;
    uint32_t transform_kind;     // PA_TransformKind
    uint64_t transform_seed;     // deterministic, reproducible
    uint32_t residual_bits;      // QJL bits (typically 1)
    uint64_t main_codes_offset;  // byte offset into KV buffer
    uint64_t aux_params_offset;  // 0 if aux_params_kind == PA_AUX_NONE
    uint64_t qjl_bits_offset;
    uint32_t aux_params_kind;    // PA_AuxParamsKind
    uint8_t  v_uses_qjl;        // 0 = V uses MSE-only (default), 1 = V uses full TurboQuant
    uint8_t  graph_side_rotation; // 0 = rotate in compress/decompress, 1 = rotate in graph
    uint8_t  _reserved[2];      // padding for alignment
} PA_QuantizedKVDesc;

// ── Layer type ──
typedef enum {
    PA_LAYER_GDN = 0,           // GatedDeltaNet (linear attention, O(1) state)
    PA_LAYER_FULL_ATTN = 1,     // Full attention (needs KV cache)
} PA_LayerType;

// ── Model descriptor (C ABI, populated by Swift ModelPack) ──
#define PA_MAX_LAYERS 128
#define PA_MAX_PATH 1024

typedef struct {
    // Paths
    char model_dir[PA_MAX_PATH];         // root model directory
    char weights_path[PA_MAX_PATH];      // model_weights.bin
    char tokenizer_path[PA_MAX_PATH];    // tokenizer.bin

    // Architecture
    uint32_t num_layers;
    uint32_t num_experts;
    uint32_t active_experts_k;           // top-K active per token
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t num_attn_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t moe_intermediate;
    uint32_t max_position_embeddings;
    float rms_norm_eps;
    PA_LayerType layer_types[PA_MAX_LAYERS];

    // Expert layout
    uint32_t expert_quant_bits;          // 2, 3, or 4
    uint32_t dense_quant_bits;
    uint64_t expert_size_each;           // bytes per expert shard

    // TurboQuant defaults
    uint16_t default_key_bits_x2;        // 0 = TQ disabled
    uint16_t default_value_bits_x2;
    uint32_t default_tq_block_size;
    uint32_t default_transform_kind;
    uint64_t default_transform_seed;

    // Manifest version
    uint32_t manifest_version;           // 1 for v1
} PA_ModelDesc;

/// Validate a PA_ModelDesc. Returns PA_STATUS_OK if valid.
int pa_model_desc_validate(const PA_ModelDesc *desc);

/// Validate a PA_TensorRef. Returns PA_STATUS_OK if valid.
int pa_tensor_ref_validate(const PA_TensorRef *ref);

/// Count full-attention layers in a PA_ModelDesc.
uint32_t pa_model_desc_full_attn_count(const PA_ModelDesc *desc);

/// Count GDN layers in a PA_ModelDesc.
uint32_t pa_model_desc_gdn_count(const PA_ModelDesc *desc);

/// Get effective bits as float from x2 encoding. E.g. 7 → 3.5f
static inline float pa_bits_from_x2(uint16_t bits_x2) {
    return (float)bits_x2 / 2.0f;
}

// ── Memory budget ──
typedef struct {
    uint64_t metal_buffers_bytes;
    uint64_t gdn_state_bytes;
    uint64_t expert_buffers_bytes;
    uint64_t kv_cache_bytes;
    uint64_t scratch_bytes;
    uint64_t total_dirty_bytes;
    uint64_t mmap_weights_bytes;
    uint64_t total_resident_bytes;
    uint32_t max_context_length;
    uint32_t full_attn_layer_count;
} PA_MemoryBudget;

/// Compute a memory budget for the given model descriptor and available memory.
/// Fills out_budget with breakdown and max_context_length.
/// Returns PA_STATUS_OK on success, PA_STATUS_ERROR_OOM if memory is insufficient.
int pa_compute_memory_budget(const PA_ModelDesc *desc, uint64_t available_bytes, PA_MemoryBudget *out_budget);

#endif // PA_TYPES_H
