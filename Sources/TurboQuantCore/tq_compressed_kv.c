#include "TurboQuantCore.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ── Compressed KV layout (CPU reference) ─────────────────────────────────────
//
// Buffer layout for a single compressed K or V vector of `dim` elements:
//
//   [codes:     dim bytes         ]  1 byte per element (unpacked scalar quant)
//   [scale:     num_blocks * 4B   ]  float32 per block
//   [zero:      num_blocks * 4B   ]  float32 per block (block min)
//   [qjl_bits:  ceil(dim/8) bytes ]  packed 1-bit residual signs
//
// Total = dim + 2 * num_blocks * 4 + ceil(dim/8) bytes.
//
// tq_compress_kv  – compress one K or V vector
// tq_compressed_dot – dot(q_transformed, dequant(k_compressed)) block by block
// tq_decompress_v_tile – decompress V back to float, inverse-rotate to original space

// ── Layout helpers ────────────────────────────────────────────────────────────

static inline uint32_t num_blocks_for(uint32_t dim, uint32_t block_size) {
    return (dim + block_size - 1) / block_size;
}

// Byte offsets into the compressed buffer
static inline uint32_t offset_codes(void)                              { return 0; }
static inline uint32_t offset_scale(uint32_t dim)                     { return dim; }
static inline uint32_t offset_zero(uint32_t dim, uint32_t num_blocks) { return dim + num_blocks * 4; }
static inline uint32_t offset_qjl (uint32_t dim, uint32_t num_blocks) { return dim + num_blocks * 8; }

// Total bytes needed for a compressed vector
static uint32_t compressed_bytes(uint32_t dim, uint32_t block_size) {
    uint32_t nb  = num_blocks_for(dim, block_size);
    uint32_t qjl = (dim + 7) / 8;
    return dim + nb * 8 + qjl;
}

// ── tq_compress_kv ────────────────────────────────────────────────────────────

uint32_t tq_compress_kv(
    const float              *kv_input,
    uint32_t                  dim,
    uint8_t                  *compressed_out,
    const PA_QuantizedKVDesc *desc
) {
    if (!kv_input || !compressed_out || dim == 0 || !desc) return 0;

    uint32_t block_size = desc->block_size;
    if (block_size == 0) return 0;

    uint32_t nb      = num_blocks_for(dim, block_size);
    uint32_t qjl_bytes = (dim + 7) / 8;

    // Pointers into output buffer
    uint8_t *codes  = compressed_out + offset_codes();
    float   *scale  = (float *)(compressed_out + offset_scale(dim));
    float   *zero   = (float *)(compressed_out + offset_zero(dim, nb));
    uint8_t *qjl    = compressed_out + offset_qjl(dim, nb);

    // 1. Rotate a local copy of the input in-place
    float *rotated = (float *)malloc(dim * sizeof(float));
    if (!rotated) return 0;
    memcpy(rotated, kv_input, dim * sizeof(float));

    if (desc->transform_kind == PA_TRANSFORM_STRUCTURED_ROTATION ||
        desc->transform_kind == PA_TRANSFORM_HADAMARD) {
        tq_rotate_inplace(rotated, dim, desc->transform_seed);
    }

    // 2. Quantize (scalar, per-block min/max)
    tq_quantize_scalar(rotated, dim, codes, scale, zero, block_size, desc->key_bits_x2);

    // 3. Dequantize to compute residual for QJL
    float *dequant = (float *)malloc(dim * sizeof(float));
    if (!dequant) { free(rotated); return 0; }
    tq_dequantize_scalar(codes, scale, zero, dequant, dim, block_size, desc->key_bits_x2);

    // 4. Encode QJL residual bits
    tq_qjl_encode(rotated, dequant, qjl, dim, desc->transform_seed);

    free(dequant);
    free(rotated);

    return compressed_bytes(dim, block_size);
}

// ── tq_compressed_dot ─────────────────────────────────────────────────────────
//
// q_transformed is already rotated (caller called tq_rotate_query).
// We compute dot(q_transformed, dequant(codes)) block by block without
// materializing the full dequantized vector.

float tq_compressed_dot(
    const float              *q_transformed,
    uint32_t                  dim,
    const uint8_t            *k_compressed,
    const PA_QuantizedKVDesc *desc
) {
    if (!q_transformed || !k_compressed || dim == 0 || !desc) return 0.0f;

    uint32_t block_size = desc->block_size;
    if (block_size == 0) return 0.0f;

    uint32_t nb = num_blocks_for(dim, block_size);

    const uint8_t *codes = k_compressed + offset_codes();
    const float   *scale = (const float *)(k_compressed + offset_scale(dim));
    const float   *zero  = (const float *)(k_compressed + offset_zero(dim, nb));

    float dot = 0.0f;

    for (uint32_t b = 0; b < nb; b++) {
        uint32_t start = b * block_size;
        uint32_t end   = start + block_size;
        if (end > dim) end = dim;

        float s = scale[b];
        float z = zero[b];

        for (uint32_t i = start; i < end; i++) {
            // dequant inline: code * scale + zero
            float k_val = (float)codes[i] * s + z;
            dot += q_transformed[i] * k_val;
        }
    }

    return dot;
}

// ── tq_decompress_v_tile ──────────────────────────────────────────────────────
//
// Decompress a compressed V vector:
//   1. Dequantize codes
//   2. Inverse-rotate to get back to original space

void tq_decompress_v_tile(
    const uint8_t            *v_compressed,
    float                    *v_output,
    uint32_t                  dim,
    const PA_QuantizedKVDesc *desc
) {
    if (!v_compressed || !v_output || dim == 0 || !desc) return;

    uint32_t block_size = desc->block_size;
    if (block_size == 0) return;

    uint32_t nb = num_blocks_for(dim, block_size);

    const uint8_t *codes = v_compressed + offset_codes();
    const float   *scale = (const float *)(v_compressed + offset_scale(dim));
    const float   *zero  = (const float *)(v_compressed + offset_zero(dim, nb));

    // 1. Dequantize into output (reuse value_bits_x2 for V; fall back to key_bits_x2)
    uint16_t bits = desc->value_bits_x2 > 0 ? desc->value_bits_x2 : desc->key_bits_x2;
    tq_dequantize_scalar(codes, scale, zero, v_output, dim, block_size, bits);

    // 2. Inverse-rotate back to original space
    if (desc->transform_kind == PA_TRANSFORM_STRUCTURED_ROTATION ||
        desc->transform_kind == PA_TRANSFORM_HADAMARD) {
        tq_rotate_inverse_inplace(v_output, dim, desc->transform_seed);
    }
}
