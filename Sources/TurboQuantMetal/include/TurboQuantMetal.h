#ifndef TURBOQUANT_METAL_H
#define TURBOQUANT_METAL_H

#include "TurboQuantCore.h"

#ifdef __cplusplus
extern "C" {
#endif

int tq_metal_init(void *device);
void tq_metal_cleanup(void);

/// Upload WHT sign diagonals to GPU buffers.
int tq_metal_upload_wht_diagonals(const int8_t *d1, const int8_t *d2, uint32_t dim);

/// Check if device supports fast half-precision.
int tq_metal_supports_half(void);

// ── Standard (with rotation in compress/decompress) ─────────────────────────

int tq_metal_compress_kv(void *cmd_buffer, const float *kv_input, uint32_t dim,
    void *compressed_buffer, const PA_QuantizedKVDesc *desc);

int tq_metal_compressed_qk_score(void *cmd_buffer, const float *q_transformed,
    uint32_t dim, void *compressed_kv_buffer, uint32_t num_tokens,
    float *scores_out, const PA_QuantizedKVDesc *desc);

int tq_metal_v_accumulate(void *cmd_buffer, void *compressed_v_buffer,
    const float *weights, uint32_t num_tokens, uint32_t dim,
    float *output, const PA_QuantizedKVDesc *desc);

// ── Prerotated variants (graph-side rotation) ───────────────────────────────

int tq_metal_compress_kv_prerotated(void *cmd_buffer, const float *kv_rotated,
    uint32_t dim, void *compressed_buffer, const PA_QuantizedKVDesc *desc);

int tq_metal_compressed_qk_score_prerotated(void *cmd_buffer,
    const float *q_rotated, uint32_t dim, void *compressed_kv_buffer,
    uint32_t num_tokens, float *scores_out, const PA_QuantizedKVDesc *desc);

int tq_metal_v_accumulate_prerotated(void *cmd_buffer, void *compressed_v_buffer,
    const float *weights, uint32_t num_tokens, uint32_t dim,
    float *output, const PA_QuantizedKVDesc *desc);

#ifdef __cplusplus
}
#endif

#endif
