#import <Foundation/Foundation.h>
#include <stdlib.h>
#include <string.h>
#include "TurboQuantMetal.h"

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Compute the byte stride (size) of one compressed vector given dim and desc.
/// Layout per vector:
///   main codes  : dim bytes  (1 byte per element at 8-bit; actual packing handled by tq_compress_kv)
///   scale+zero  : 2 * num_blocks * sizeof(float) bytes
///   QJL bits    : (dim + 7) / 8 bytes
static inline uint32_t compressed_stride(uint32_t dim, const PA_QuantizedKVDesc *desc) {
    uint32_t block_size = desc->block_size > 0 ? desc->block_size : dim;
    uint32_t num_blocks = (dim + block_size - 1) / block_size;
    return dim
         + 2u * num_blocks * (uint32_t)sizeof(float)
         + (dim + 7u) / 8u;
}

// ── Public API ────────────────────────────────────────────────────────────────

int tq_metal_init(void *device) {
    (void)device;
    // CPU stub: no Metal device needed.
    return 0;
}

void tq_metal_cleanup(void) {
    // CPU stub: nothing to release.
}

int tq_metal_compress_kv(void *cmd_buffer, const float *kv_input, uint32_t dim,
                          void *compressed_buffer, const PA_QuantizedKVDesc *desc) {
    (void)cmd_buffer;
    uint32_t written = tq_compress_kv(kv_input, dim,
                                       (uint8_t *)compressed_buffer, desc);
    return (written > 0) ? 0 : -1;
}

int tq_metal_compressed_qk_score(void *cmd_buffer, const float *q_transformed,
                                   uint32_t dim, void *compressed_kv_buffer,
                                   uint32_t num_tokens, float *scores_out,
                                   const PA_QuantizedKVDesc *desc) {
    (void)cmd_buffer;
    uint32_t stride = compressed_stride(dim, desc);
    const uint8_t *base = (const uint8_t *)compressed_kv_buffer;
    for (uint32_t t = 0; t < num_tokens; t++) {
        scores_out[t] = tq_compressed_dot(q_transformed, dim,
                                           base + (size_t)t * stride, desc);
    }
    return 0;
}

int tq_metal_v_accumulate(void *cmd_buffer, void *compressed_v_buffer,
                            const float *weights, uint32_t num_tokens, uint32_t dim,
                            float *output, const PA_QuantizedKVDesc *desc) {
    (void)cmd_buffer;
    uint32_t stride = compressed_stride(dim, desc);
    const uint8_t *base = (const uint8_t *)compressed_v_buffer;

    float *v_tile = (float *)malloc(dim * sizeof(float));
    if (!v_tile) return -1;

    memset(output, 0, dim * sizeof(float));

    for (uint32_t t = 0; t < num_tokens; t++) {
        tq_decompress_v_tile(base + (size_t)t * stride, v_tile, dim, desc);
        float w = weights[t];
        for (uint32_t d = 0; d < dim; d++) {
            output[d] += w * v_tile[d];
        }
    }

    free(v_tile);
    return 0;
}
