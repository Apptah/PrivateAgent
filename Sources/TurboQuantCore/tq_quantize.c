#include "TurboQuantCore.h"
#include <math.h>
#include <string.h>
#include <float.h>

uint32_t tq_quantize_scalar(
    const float *input, uint32_t dim,
    uint8_t *codes, float *scale, float *zero,
    uint32_t block_size, uint16_t bits_x2
) {
    if (!input || !codes || !scale || !zero || dim == 0) return 0;

    float bits = (float)bits_x2 / 2.0f;
    uint32_t levels = (uint32_t)(powf(2.0f, bits));
    if (levels < 2) levels = 2;
    float max_code = (float)(levels - 1);

    uint32_t num_blocks = (dim + block_size - 1) / block_size;
    uint32_t code_idx = 0;

    for (uint32_t b = 0; b < num_blocks; b++) {
        uint32_t start = b * block_size;
        uint32_t end = start + block_size;
        if (end > dim) end = dim;

        float bmin = FLT_MAX, bmax = -FLT_MAX;
        for (uint32_t i = start; i < end; i++) {
            if (input[i] < bmin) bmin = input[i];
            if (input[i] > bmax) bmax = input[i];
        }

        float range = bmax - bmin;
        if (range < 1e-10f) range = 1e-10f;

        scale[b] = range / max_code;
        zero[b] = bmin;

        for (uint32_t i = start; i < end; i++) {
            float normalized = (input[i] - bmin) / range;
            uint8_t code = (uint8_t)roundf(normalized * max_code);
            if (code > (uint8_t)max_code) code = (uint8_t)max_code;
            codes[code_idx++] = code;
        }
    }

    return code_idx;
}

void tq_dequantize_scalar(
    const uint8_t *codes, const float *scale, const float *zero,
    float *output, uint32_t dim,
    uint32_t block_size, uint16_t bits_x2
) {
    if (!codes || !scale || !zero || !output || dim == 0) return;
    (void)bits_x2;

    uint32_t num_blocks = (dim + block_size - 1) / block_size;
    uint32_t code_idx = 0;

    for (uint32_t b = 0; b < num_blocks; b++) {
        uint32_t start = b * block_size;
        uint32_t end = start + block_size;
        if (end > dim) end = dim;

        for (uint32_t i = start; i < end; i++) {
            output[i] = zero[b] + codes[code_idx] * scale[b];
            code_idx++;
        }
    }
}
