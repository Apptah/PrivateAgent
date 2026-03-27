#include "TurboQuantCore.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ── Compressed KV layout (TurboQuant_prod, Algorithm 2) ──────────────────────
//
// Per-vector buffer layout:
//   [norm:     float32, 4 bytes        ]  ‖x‖₂
//   [codes:    dim bytes               ]  Lloyd-Max index, 1 byte per coordinate
//   [gamma:    float32, 4 bytes        ]  ‖residual‖₂  (γ)
//   [qjl_bits: ceil(dim/8) bytes       ]  sign(S @ residual), 1 bit per projection
//
// Total: 8 + dim + ceil(dim/8) bytes.

uint32_t tq_compressed_size(uint32_t dim) {
    return 4u                   // norm  (float32)
         + dim                  // codes (1 byte each)
         + 4u                   // gamma (float32)
         + (dim + 7u) / 8u;    // qjl_bits
}

// ── tq_compress_kv ────────────────────────────────────────────────────────────

uint32_t tq_compress_kv(const float *kv_input, uint32_t dim,
                         uint8_t *compressed_out,
                         const PA_QuantizedKVDesc *desc) {
    if (!kv_input || !compressed_out || !desc || dim == 0) return 0;

    // For K cache: reserve 1 bit for QJL, rest for MSE.
    // Use bits_x2 - 2 as the MSE bits_x2 (subtract 1 bit = subtract 2 in x2 encoding)
    uint16_t mse_bits_x2 = (desc->key_bits_x2 >= 4) ? desc->key_bits_x2 - 2 : 6;
    uint8_t mse_bits_int = (uint8_t)(mse_bits_x2 / 2);
    if (mse_bits_int < 1) mse_bits_int = 3;

    // Ensure QJL matrix is ready (use a distinct seed to avoid correlation)
    tq_qjl_init(dim, desc->transform_seed ^ 0xDEADBEEFCAFEBABEULL);

    // Allocate working buffers
    float *unit         = malloc(dim * sizeof(float));
    float *rotated      = malloc(dim * sizeof(float));
    float *dequantized  = malloc(dim * sizeof(float));
    float *reconstructed = malloc(dim * sizeof(float));
    float *residual     = malloc(dim * sizeof(float));
    if (!unit || !rotated || !dequantized || !reconstructed || !residual) {
        free(unit); free(rotated); free(dequantized);
        free(reconstructed); free(residual);
        return 0;
    }

    // 1. Compute ‖x‖₂ and normalise
    float norm2 = 0.0f;
    for (uint32_t i = 0; i < dim; i++) norm2 += kv_input[i] * kv_input[i];
    float norm    = sqrtf(norm2);
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (uint32_t i = 0; i < dim; i++) unit[i] = kv_input[i] * inv_norm;

    // 2. Rotate: y = R @ x_unit (dispatched by transform_kind)
    tq_dispatch_rotate(unit, rotated, dim, desc->transform_kind, desc->transform_seed);

    // Layout pointers into the output buffer
    float   *out_norm   = (float *)compressed_out;
    uint8_t *out_codes  = compressed_out + 4;
    float   *out_gamma  = (float *)(compressed_out + 4 + dim);
    uint8_t *out_qjl    = compressed_out + 4 + dim + 4;

    // 3. Store norm and quantise with split Lloyd-Max (supports fractional bit rates)
    *out_norm = norm;
    tq_quantize_lloydmax_split(rotated, dim, out_codes, mse_bits_x2);

    // 4. Reconstruct MSE approximation to get residual
    //    x_hat = norm * R^T @ dequant(codes)
    tq_dequantize_lloydmax_split(out_codes, dequantized, dim, mse_bits_x2);
    tq_dispatch_rotate_inverse(dequantized, reconstructed, dim,
                                desc->transform_kind, desc->transform_seed);
    for (uint32_t i = 0; i < dim; i++) reconstructed[i] *= norm;

    // 5. Residual: r = x - x_hat
    for (uint32_t i = 0; i < dim; i++) residual[i] = kv_input[i] - reconstructed[i];

    // 6. QJL-encode residual: qjl_bits = sign(S @ r),  gamma = ‖r‖₂
    tq_qjl_encode(residual, dim, out_qjl, out_gamma);

    free(unit); free(rotated); free(dequantized);
    free(reconstructed); free(residual);

    return tq_compressed_size(dim);
}

// ── tq_decompress_v ───────────────────────────────────────────────────────────
// Implements Algorithm 2 dequantization:
//   x_mse = norm * Π^T @ dequant(codes)
//   x_qjl = (sqrt(π/2) / d) * gamma * S^T @ sign_bits
//   output = x_mse + x_qjl

void tq_decompress_v(const uint8_t *v_compressed, float *v_output,
                      uint32_t dim, const PA_QuantizedKVDesc *desc) {
    if (!v_compressed || !v_output || dim == 0 || !desc) return;

    // MSE bits_x2: key_bits_x2 - 2 (reserve 1 bit for QJL)
    uint16_t mse_bits_x2 = (desc->key_bits_x2 >= 4) ? desc->key_bits_x2 - 2 : 6;

    // Ensure QJL matrix is ready
    tq_qjl_init(dim, desc->transform_seed ^ 0xDEADBEEFCAFEBABEULL);

    const float   *p_norm   = (const float *)v_compressed;
    const uint8_t *codes    = v_compressed + 4;
    const float   *p_gamma  = (const float *)(v_compressed + 4 + dim);
    const uint8_t *qjl_bits = v_compressed + 4 + dim + 4;

    float norm  = *p_norm;
    float gamma = *p_gamma;

    float *rotated         = malloc(dim * sizeof(float));
    float *qjl_contrib     = malloc(dim * sizeof(float));
    if (!rotated || !qjl_contrib) {
        free(rotated); free(qjl_contrib);
        return;
    }

    // 1. MSE part: v_output = norm * R^T @ dequant(codes)
    tq_dequantize_lloydmax_split(codes, rotated, dim, mse_bits_x2);
    tq_dispatch_rotate_inverse(rotated, v_output, dim,
                                desc->transform_kind, desc->transform_seed);
    for (uint32_t i = 0; i < dim; i++) v_output[i] *= norm;

    // 2. QJL correction: qjl_contrib = (sqrt(π/2)/d) * gamma * S^T @ sign_bits
    tq_qjl_decode(qjl_bits, gamma, qjl_contrib, dim);
    for (uint32_t i = 0; i < dim; i++) v_output[i] += qjl_contrib[i];

    free(rotated);
    free(qjl_contrib);
}

// ── tq_decompress_v_tile — backward-compatible alias ─────────────────────────

void tq_decompress_v_tile(const uint8_t *v_compressed, float *v_output,
                           uint32_t dim, const PA_QuantizedKVDesc *desc) {
    tq_decompress_v(v_compressed, v_output, dim, desc);
}

// ── V-specific MSE-only compression (no QJL) ────────────────────────────────
// Layout: [norm: f32] [codes: dim bytes]

uint32_t tq_compressed_size_v(uint32_t dim) {
    return 4u + dim;   // norm + codes only
}

uint32_t tq_compress_v(const float *v_input, uint32_t dim,
                        uint8_t *compressed_out,
                        const PA_QuantizedKVDesc *desc) {
    if (!v_input || !compressed_out || !desc || dim == 0) return 0;

    // V uses all bits for MSE — no 1-bit QJL reservation
    uint16_t v_bits_x2 = desc->value_bits_x2;
    if (v_bits_x2 < 2) v_bits_x2 = 8;

    float *unit    = malloc(dim * sizeof(float));
    float *rotated = malloc(dim * sizeof(float));
    if (!unit || !rotated) { free(unit); free(rotated); return 0; }

    // 1. Compute norm and normalise
    float norm2 = 0.0f;
    for (uint32_t i = 0; i < dim; i++) norm2 += v_input[i] * v_input[i];
    float norm = sqrtf(norm2);
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (uint32_t i = 0; i < dim; i++) unit[i] = v_input[i] * inv_norm;

    // 2. Rotate
    tq_dispatch_rotate(unit, rotated, dim, desc->transform_kind, desc->transform_seed);

    // 3. Store norm + quantise (supports fractional bit rates)
    float *out_norm  = (float *)compressed_out;
    uint8_t *out_codes = compressed_out + 4;
    *out_norm = norm;
    tq_quantize_lloydmax_split(rotated, dim, out_codes, v_bits_x2);

    free(unit); free(rotated);
    return tq_compressed_size_v(dim);
}

void tq_decompress_v_mse(const uint8_t *v_compressed, float *v_output,
                           uint32_t dim, const PA_QuantizedKVDesc *desc) {
    if (!v_compressed || !v_output || dim == 0 || !desc) return;

    uint16_t v_bits_x2 = desc->value_bits_x2;
    if (v_bits_x2 < 2) v_bits_x2 = 8;

    const float *p_norm = (const float *)v_compressed;
    const uint8_t *codes = v_compressed + 4;
    float norm = *p_norm;

    float *rotated = malloc(dim * sizeof(float));
    if (!rotated) return;

    tq_dequantize_lloydmax_split(codes, rotated, dim, v_bits_x2);
    tq_dispatch_rotate_inverse(rotated, v_output, dim,
                                desc->transform_kind, desc->transform_seed);
    for (uint32_t i = 0; i < dim; i++) v_output[i] *= norm;

    free(rotated);
}

// ── tq_compressed_dot ─────────────────────────────────────────────────────────
// Full dequant + dot product (paper does not compress the dot; it dequants K).
// query is in the original (unrotated) space.

float tq_compressed_dot(const float *query, uint32_t dim,
                         const uint8_t *k_compressed,
                         const PA_QuantizedKVDesc *desc) {
    if (!query || !k_compressed || dim == 0 || !desc) return 0.0f;

    float *k_decompressed = malloc(dim * sizeof(float));
    if (!k_decompressed) return 0.0f;

    tq_decompress_v(k_compressed, k_decompressed, dim, desc);

    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dot += query[i] * k_decompressed[i];
    }

    free(k_decompressed);
    return dot;
}

// ══════════════════════════════════════════════════════════════════════════════
// Graph-side rotation — prerotated variants
// ══════════════════════════════════════════════════════════════════════════════
// Input is already rotated in the compute graph. Skip rotation in compress/decompress.

uint32_t tq_compress_kv_prerotated(const float *kv_rotated, uint32_t dim,
                                     uint8_t *compressed_out,
                                     const PA_QuantizedKVDesc *desc) {
    if (!kv_rotated || !compressed_out || !desc || dim == 0) return 0;

    uint16_t mse_bits_x2 = (desc->key_bits_x2 >= 4) ? desc->key_bits_x2 - 2 : 6;

    tq_qjl_init(dim, desc->transform_seed ^ 0xDEADBEEFCAFEBABEULL);

    float *unit         = malloc(dim * sizeof(float));
    float *dequantized  = malloc(dim * sizeof(float));
    float *reconstructed = malloc(dim * sizeof(float));
    float *residual     = malloc(dim * sizeof(float));
    if (!unit || !dequantized || !reconstructed || !residual) {
        free(unit); free(dequantized); free(reconstructed); free(residual);
        return 0;
    }

    // 1. Compute norm and normalise (input is already rotated)
    float norm2 = 0.0f;
    for (uint32_t i = 0; i < dim; i++) norm2 += kv_rotated[i] * kv_rotated[i];
    float norm = sqrtf(norm2);
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (uint32_t i = 0; i < dim; i++) unit[i] = kv_rotated[i] * inv_norm;

    // 2. NO rotation — input is already in rotated space

    float *out_norm  = (float *)compressed_out;
    uint8_t *out_codes = compressed_out + 4;
    float *out_gamma = (float *)(compressed_out + 4 + dim);
    uint8_t *out_qjl = compressed_out + 4 + dim + 4;

    // 3. Quantise
    *out_norm = norm;
    tq_quantize_lloydmax_split(unit, dim, out_codes, mse_bits_x2);

    // 4. Reconstruct in rotated space to get residual
    tq_dequantize_lloydmax_split(out_codes, dequantized, dim, mse_bits_x2);
    for (uint32_t i = 0; i < dim; i++)
        reconstructed[i] = dequantized[i] * norm;

    // 5. Residual in rotated space
    for (uint32_t i = 0; i < dim; i++)
        residual[i] = kv_rotated[i] - reconstructed[i];

    // 6. QJL encode
    tq_qjl_encode(residual, dim, out_qjl, out_gamma);

    free(unit); free(dequantized); free(reconstructed); free(residual);
    return tq_compressed_size(dim);
}

uint32_t tq_compress_v_prerotated(const float *v_rotated, uint32_t dim,
                                    uint8_t *compressed_out,
                                    const PA_QuantizedKVDesc *desc) {
    if (!v_rotated || !compressed_out || !desc || dim == 0) return 0;

    uint16_t v_bits_x2 = desc->value_bits_x2;
    if (v_bits_x2 < 2) v_bits_x2 = 8;

    float *unit = malloc(dim * sizeof(float));
    if (!unit) return 0;

    float norm2 = 0.0f;
    for (uint32_t i = 0; i < dim; i++) norm2 += v_rotated[i] * v_rotated[i];
    float norm = sqrtf(norm2);
    float inv_norm = (norm > 1e-10f) ? 1.0f / norm : 0.0f;
    for (uint32_t i = 0; i < dim; i++) unit[i] = v_rotated[i] * inv_norm;

    float *out_norm  = (float *)compressed_out;
    uint8_t *out_codes = compressed_out + 4;
    *out_norm = norm;
    tq_quantize_lloydmax_split(unit, dim, out_codes, v_bits_x2);

    free(unit);
    return tq_compressed_size_v(dim);
}

void tq_decompress_v_mse_prerotated(const uint8_t *v_compressed, float *v_output,
                                      uint32_t dim, const PA_QuantizedKVDesc *desc) {
    if (!v_compressed || !v_output || dim == 0 || !desc) return;

    uint16_t v_bits_x2 = desc->value_bits_x2;
    if (v_bits_x2 < 2) v_bits_x2 = 8;

    const float *p_norm = (const float *)v_compressed;
    const uint8_t *codes = v_compressed + 4;
    float norm = *p_norm;

    // Dequantise directly — no inverse rotation (output stays in rotated space)
    tq_dequantize_lloydmax_split(codes, v_output, dim, v_bits_x2);
    for (uint32_t i = 0; i < dim; i++) v_output[i] *= norm;
}

float tq_compressed_dot_prerotated(const float *q_rotated, uint32_t dim,
                                     const uint8_t *k_compressed,
                                     const PA_QuantizedKVDesc *desc) {
    if (!q_rotated || !k_compressed || dim == 0 || !desc) return 0.0f;

    // Decompress K without inverse rotation (stays in rotated space)
    uint16_t mse_bits_x2 = (desc->key_bits_x2 >= 4) ? desc->key_bits_x2 - 2 : 6;

    tq_qjl_init(dim, desc->transform_seed ^ 0xDEADBEEFCAFEBABEULL);

    const float *p_norm   = (const float *)k_compressed;
    const uint8_t *codes  = k_compressed + 4;
    const float *p_gamma  = (const float *)(k_compressed + 4 + dim);
    const uint8_t *qjl_bits = k_compressed + 4 + dim + 4;

    float norm  = *p_norm;
    float gamma = *p_gamma;

    float *k_decompressed = malloc(dim * sizeof(float));
    float *qjl_contrib    = malloc(dim * sizeof(float));
    if (!k_decompressed || !qjl_contrib) {
        free(k_decompressed); free(qjl_contrib);
        return 0.0f;
    }

    // MSE part in rotated space (no inverse rotation)
    tq_dequantize_lloydmax_split(codes, k_decompressed, dim, mse_bits_x2);
    for (uint32_t i = 0; i < dim; i++) k_decompressed[i] *= norm;

    // QJL correction in rotated space
    tq_qjl_decode(qjl_bits, gamma, qjl_contrib, dim);
    for (uint32_t i = 0; i < dim; i++) k_decompressed[i] += qjl_contrib[i];

    // Dot product in rotated space (orthogonal transform preserves inner product)
    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; i++) dot += q_rotated[i] * k_decompressed[i];

    free(k_decompressed); free(qjl_contrib);
    return dot;
}
