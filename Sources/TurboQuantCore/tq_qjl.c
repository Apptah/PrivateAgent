#include "TurboQuantCore.h"
#include <string.h>
#include <math.h>

static inline uint64_t qjl_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void tq_qjl_encode(
    const float *input, const float *dequantized,
    uint8_t *residual_bits, uint32_t dim, uint64_t qjl_seed
) {
    if (!input || !dequantized || !residual_bits || dim == 0) return;

    uint32_t num_bytes = (dim + 7) / 8;
    memset(residual_bits, 0, num_bytes);

    uint64_t state = qjl_seed;

    for (uint32_t i = 0; i < dim; i++) {
        float error = input[i] - dequantized[i];
        float rnd_sign = (qjl_splitmix64(&state) & 1) ? 1.0f : -1.0f;
        float projected = error * rnd_sign;

        if (projected >= 0.0f) {
            residual_bits[i / 8] |= (1 << (i % 8));
        }
    }
}

void tq_qjl_correct_scores(
    float *scores, uint32_t num_tokens,
    const uint8_t *q_residual_bits,
    const uint8_t *k_residual_bits,
    uint32_t dim, uint64_t qjl_seed
) {
    if (!scores || !q_residual_bits || !k_residual_bits) return;
    (void)qjl_seed;

    uint32_t num_bytes = (dim + 7) / 8;

    for (uint32_t t = 0; t < num_tokens; t++) {
        const uint8_t *k_bits = k_residual_bits + t * num_bytes;
        float correction = 0.0f;

        for (uint32_t i = 0; i < dim; i++) {
            int q_bit = (q_residual_bits[i / 8] >> (i % 8)) & 1;
            int k_bit = (k_bits[i / 8] >> (i % 8)) & 1;
            correction += (q_bit == k_bit) ? 1.0f : -1.0f;
        }

        scores[t] += correction / sqrtf((float)dim);
    }
}
