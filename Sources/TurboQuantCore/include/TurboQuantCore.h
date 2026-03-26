#ifndef TURBOQUANT_CORE_H
#define TURBOQUANT_CORE_H

#include "FlashMoECore.h"

#include <stdint.h>
#include <stddef.h>

// TurboQuant KV cache compression — CPU reference implementations.
// Added in Plan 5.

#ifdef __cplusplus
extern "C" {
#endif

// ── Plan 5 Task 1: Seed-driven structured rotation ──

/// Apply structured rotation in-place: R @ x  where R = H @ diag(signs(seed)).
/// dim must be a power of 2.
void tq_rotate_inplace(float *x, uint32_t dim, uint64_t seed);

/// Apply inverse rotation in-place: R^T @ x  (= diag(signs(seed)) @ H @ x).
/// dim must be a power of 2.
void tq_rotate_inverse_inplace(float *x, uint32_t dim, uint64_t seed);

/// Rotate a query vector: q_out = R^{-1} @ q_in  (so that dot(R@k, q_out) == dot(k, q_in)).
/// q_in and q_out may not alias. dim must be a power of 2.
void tq_rotate_query(const float *q_in, float *q_out, uint32_t dim, uint64_t seed);

// ── Plan 5 Task 2: Scalar quantization + QJL (declared, implemented later) ──

/// Quantize input vector. Returns number of bytes written to codes.
uint32_t tq_quantize_scalar(const float *input, uint32_t dim,
                             uint8_t *codes, float *scale, float *zero,
                             uint32_t block_size, uint16_t bits_x2);

/// Dequantize codes back to float.
void tq_dequantize_scalar(const uint8_t *codes, const float *scale, const float *zero,
                           float *output, uint32_t dim,
                           uint32_t block_size, uint16_t bits_x2);

/// Encode QJL residual bits from the difference between input and its dequantized approximation.
void tq_qjl_encode(const float *input, const float *dequantized,
                   uint8_t *residual_bits, uint32_t dim, uint64_t qjl_seed);

/// Correct attention scores using QJL residual bits from query and key vectors.
void tq_qjl_correct_scores(float *scores, uint32_t num_tokens,
                            const uint8_t *q_residual_bits, const uint8_t *k_residual_bits,
                            uint32_t dim, uint64_t qjl_seed);

// ── Plan 5 Task 3: Compressed KV ops (declared, implemented later) ──

/// Compress a KV vector according to desc. Returns bytes written to compressed_out.
uint32_t tq_compress_kv(const float *kv_input, uint32_t dim,
                         uint8_t *compressed_out, const PA_QuantizedKVDesc *desc);

/// Compute dot product between a transformed query and a compressed key.
float tq_compressed_dot(const float *q_transformed, uint32_t dim,
                         const uint8_t *k_compressed, const PA_QuantizedKVDesc *desc);

/// Decompress a compressed value tile into float output.
void tq_decompress_v_tile(const uint8_t *v_compressed, float *v_output,
                           uint32_t dim, const PA_QuantizedKVDesc *desc);

#ifdef __cplusplus
}
#endif

#endif // TURBOQUANT_CORE_H
