#ifndef TURBOQUANT_CORE_H
#define TURBOQUANT_CORE_H

#include "FlashMoECore.h"

#include <stdint.h>
#include <stddef.h>

// TurboQuant KV cache compression — CPU reference implementation.
// Implements Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod)
// from Google's paper arXiv:2504.19874 (ICLR 2026).

#ifdef __cplusplus
extern "C" {
#endif

// ── Random orthogonal rotation ────────────────────────────────────────────────

/// Generate and cache a random orthogonal matrix for the given dim+seed.
/// Uses QR decomposition (modified Gram-Schmidt) of a random normal matrix.
/// Must be called once before using tq_rotate / tq_rotate_inverse.
/// Returns 0 on success, -1 on allocation failure.
int tq_rotation_init(uint32_t dim, uint64_t seed);

/// Free cached rotation matrix.
void tq_rotation_cleanup(void);

/// Apply rotation: y = Π @ x. dim must match tq_rotation_init.
void tq_rotate(const float *x, float *y, uint32_t dim);

/// Apply inverse rotation: x = Π^T @ y.
void tq_rotate_inverse(const float *y, float *x, uint32_t dim);

// ── Backward-compatible wrappers (old inplace API) ────────────────────────────

/// Apply forward rotation in-place. Calls tq_rotation_init if not yet done.
void tq_rotate_inplace(float *x, uint32_t dim, uint64_t seed);

/// Apply inverse rotation in-place. Calls tq_rotation_init if not yet done.
void tq_rotate_inverse_inplace(float *x, uint32_t dim, uint64_t seed);

/// Rotate a query vector: q_out = Π @ q_in. q_in and q_out must not alias.
void tq_rotate_query(const float *q_in, float *q_out, uint32_t dim, uint64_t seed);

// ── Fast Walsh-Hadamard Transform (WHT) ─────────────────────────────────────
// O(d log d) structured rotation: D₂ @ H @ D₁.
// dim must be a power of 2.

/// Initialize WHT sign diagonals for the given dim+seed.
/// Returns 0 on success, -1 if dim is not a power of 2 or on allocation failure.
int tq_wht_init(uint32_t dim, uint64_t seed);

/// Free WHT cached state.
void tq_wht_cleanup(void);

/// Forward WHT rotation: y = D₂ @ H @ D₁ @ x.
void tq_wht_rotate(const float *x, float *y, uint32_t dim);

/// Inverse WHT rotation: x = D₁ @ H @ D₂ @ y.
void tq_wht_rotate_inverse(const float *y, float *x, uint32_t dim);

// ── Transform dispatch (routes by PA_TransformKind) ─────────────────────────

/// Apply forward rotation using the transform specified by transform_kind.
void tq_dispatch_rotate(const float *x, float *y, uint32_t dim,
                         uint32_t transform_kind, uint64_t seed);

/// Apply inverse rotation using the transform specified by transform_kind.
void tq_dispatch_rotate_inverse(const float *y, float *x, uint32_t dim,
                                  uint32_t transform_kind, uint64_t seed);

// ── Lloyd-Max quantization ────────────────────────────────────────────────────

/// Quantize a unit-norm rotated vector using Lloyd-Max codebook.
/// codes: output indices (1 byte per element, values 0..2^bits-1)
/// bits: quantization bits per coordinate (1-4)
/// Returns number of codes written (== dim on success, 0 on error).
uint32_t tq_quantize_lloydmax(const float *rotated_unit, uint32_t dim,
                               uint8_t *codes, uint8_t bits);

/// Dequantize codes back to rotated unit vector using Lloyd-Max codebook.
void tq_dequantize_lloydmax(const uint8_t *codes, float *rotated_unit,
                              uint32_t dim, uint8_t bits);

/// Get Lloyd-Max codebook for given bit width.
/// Returns pointer to static array of 2^bits centroids (N(0,1) scale).
const float *tq_lloydmax_codebook(uint8_t bits);

/// Split quantise for fractional bit rates (outlier channel strategy).
/// bits_x2 encoding: 7 = 3.5-bit (first dim/2 at 4-bit, rest at 3-bit).
/// Even bits_x2 values delegate to standard tq_quantize_lloydmax.
uint32_t tq_quantize_lloydmax_split(const float *rotated_unit, uint32_t dim,
                                      uint8_t *codes, uint16_t bits_x2);

/// Split dequantise (mirrors tq_quantize_lloydmax_split).
void tq_dequantize_lloydmax_split(const uint8_t *codes, float *rotated_unit,
                                    uint32_t dim, uint16_t bits_x2);

// ── Backward-compatible scalar quant wrappers ─────────────────────────────────

/// Deprecated: delegates to tq_quantize_lloydmax. scale/zero ignored on new path.
uint32_t tq_quantize_scalar(const float *input, uint32_t dim,
                             uint8_t *codes, float *scale, float *zero,
                             uint32_t block_size, uint16_t bits_x2);

/// Deprecated: delegates to tq_dequantize_lloydmax.
void tq_dequantize_scalar(const uint8_t *codes, const float *scale,
                           const float *zero, float *output, uint32_t dim,
                           uint32_t block_size, uint16_t bits_x2);

// ── QJL random projection ─────────────────────────────────────────────────────

/// Initialize QJL random projection matrix S (d×d, N(0,1) entries, row-major).
/// Returns 0 on success, -1 on allocation failure.
int tq_qjl_init(uint32_t dim, uint64_t seed);

/// Free QJL projection matrix.
void tq_qjl_cleanup(void);

/// Encode residual via QJL: qjl_bits = sign(S @ residual).
/// residual_norm_out: stores ‖residual‖₂ (γ).
void tq_qjl_encode(const float *residual, uint32_t dim,
                    uint8_t *qjl_bits, float *residual_norm_out);

/// Dequantize QJL correction: output = (sqrt(π/2) / d) * gamma * S^T @ sign_bits.
void tq_qjl_decode(const uint8_t *qjl_bits, float gamma,
                    float *output, uint32_t dim);

/// Deprecated no-op: residual correction now done via full dequant + dot.
void tq_qjl_correct_scores(float *scores, uint32_t num_tokens,
                            const uint8_t *q_residual_bits,
                            const uint8_t *k_residual_bits,
                            uint32_t dim, uint64_t qjl_seed);

// ── Compressed KV cache ───────────────────────────────────────────────────────

/// Compressed KV layout per vector (TurboQuant_prod, Algorithm 2):
///   [norm:     float32, 4 bytes         ]  ‖x‖₂
///   [codes:    dim bytes                ]  1 byte per coord, Lloyd-Max index
///   [gamma:    float32, 4 bytes         ]  ‖residual‖₂
///   [qjl_bits: ceil(dim/8) bytes        ]  1 bit per projection sign
/// Total: 8 + dim + ceil(dim/8) bytes

/// Get compressed size per vector in bytes.
uint32_t tq_compressed_size(uint32_t dim);

/// Compress a KV vector using the full TurboQuant_prod pipeline.
/// Uses (total_bits - 1) bits for MSE + 1 bit QJL residual.
/// Returns bytes written to compressed_out (== tq_compressed_size(dim)), or 0 on error.
uint32_t tq_compress_kv(const float *kv_input, uint32_t dim,
                         uint8_t *compressed_out,
                         const PA_QuantizedKVDesc *desc);

/// Compute approximate dot product between query and compressed key.
/// query is in the ORIGINAL (unrotated) space.
float tq_compressed_dot(const float *query, uint32_t dim,
                         const uint8_t *k_compressed,
                         const PA_QuantizedKVDesc *desc);

/// Decompress a value vector to float (MSE part + QJL correction).
void tq_decompress_v(const uint8_t *v_compressed, float *v_output,
                      uint32_t dim, const PA_QuantizedKVDesc *desc);

/// Backward-compatible alias for tq_decompress_v.
void tq_decompress_v_tile(const uint8_t *v_compressed, float *v_output,
                           uint32_t dim, const PA_QuantizedKVDesc *desc);

// ── V-specific MSE-only compression (no QJL) ────────────────────────────────
// V cache uses MSE reconstruction, not inner product → QJL is unnecessary.
// Layout: [norm: f32, 4 bytes] [codes: dim bytes]. Total: 4 + dim bytes.

/// Get compressed size for a V vector (MSE-only, no QJL).
uint32_t tq_compressed_size_v(uint32_t dim);

/// Compress a V vector using MSE-only quantization (no QJL residual).
/// Uses all bits from value_bits_x2 for MSE (no 1-bit QJL reservation).
/// Returns bytes written, or 0 on error.
uint32_t tq_compress_v(const float *v_input, uint32_t dim,
                        uint8_t *compressed_out,
                        const PA_QuantizedKVDesc *desc);

/// Decompress a V vector from MSE-only format.
void tq_decompress_v_mse(const uint8_t *v_compressed, float *v_output,
                           uint32_t dim, const PA_QuantizedKVDesc *desc);

// ── Graph-side rotation (prerotated variants) ───────────────────────────────
// When graph_side_rotation=1, the caller rotates K/V/Q once in the compute
// graph (after RoPE). Compress/decompress skip the rotation step entirely.
// This eliminates 2× O(d log d) per compress + decompress call.

/// Compress a KV vector that is ALREADY rotated (skip rotation step).
uint32_t tq_compress_kv_prerotated(const float *kv_rotated, uint32_t dim,
                                     uint8_t *compressed_out,
                                     const PA_QuantizedKVDesc *desc);

/// Compress a V vector (MSE-only) that is ALREADY rotated.
uint32_t tq_compress_v_prerotated(const float *v_rotated, uint32_t dim,
                                    uint8_t *compressed_out,
                                    const PA_QuantizedKVDesc *desc);

/// Decompress V (MSE-only) without inverse rotation — output stays in rotated space.
void tq_decompress_v_mse_prerotated(const uint8_t *v_compressed, float *v_output,
                                      uint32_t dim, const PA_QuantizedKVDesc *desc);

/// Compute dot(q_rotated, decompress_k) where q is ALREADY rotated.
/// K is decompressed without inverse rotation (stays in rotated space).
float tq_compressed_dot_prerotated(const float *q_rotated, uint32_t dim,
                                     const uint8_t *k_compressed,
                                     const PA_QuantizedKVDesc *desc);

#ifdef __cplusplus
}
#endif

#endif // TURBOQUANT_CORE_H
