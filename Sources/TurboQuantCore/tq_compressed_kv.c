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

    uint8_t total_bits = (uint8_t)(desc->key_bits_x2 / 2);
    if (total_bits < 2) total_bits = 4;
    uint8_t mse_bits = total_bits - 1;   // paper: (b-1) bits for MSE
    if (mse_bits < 1) mse_bits = 1;

    // Ensure rotation matrix is ready
    tq_rotation_init(dim, desc->transform_seed);
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

    // 2. Rotate: y = Π @ x_unit
    tq_rotate(unit, rotated, dim);

    // Layout pointers into the output buffer
    float   *out_norm   = (float *)compressed_out;
    uint8_t *out_codes  = compressed_out + 4;
    float   *out_gamma  = (float *)(compressed_out + 4 + dim);
    uint8_t *out_qjl    = compressed_out + 4 + dim + 4;

    // 3. Store norm and quantise with Lloyd-Max (mse_bits bits)
    *out_norm = norm;
    tq_quantize_lloydmax(rotated, dim, out_codes, mse_bits);

    // 4. Reconstruct MSE approximation to get residual
    //    x_hat = norm * Π^T @ dequant(codes)
    tq_dequantize_lloydmax(out_codes, dequantized, dim, mse_bits);
    tq_rotate_inverse(dequantized, reconstructed, dim);
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

    uint8_t total_bits = (uint8_t)(desc->key_bits_x2 / 2);
    if (total_bits < 2) total_bits = 4;
    uint8_t mse_bits = total_bits - 1;

    // Ensure matrices are ready
    tq_rotation_init(dim, desc->transform_seed);
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

    // 1. MSE part: v_output = norm * Π^T @ dequant(codes)
    tq_dequantize_lloydmax(codes, rotated, dim, mse_bits);
    tq_rotate_inverse(rotated, v_output, dim);
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
