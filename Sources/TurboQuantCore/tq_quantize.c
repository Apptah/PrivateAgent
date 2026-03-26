#include "TurboQuantCore.h"
#include <math.h>
#include <string.h>

// ── Scalar Quantization (CPU reference) ──
//
// bits_x2 encoding: levels = pow(2, bits_x2 / 2.0)
//   6 → 3-bit  →  8 levels
//   7 → 3.5-bit → ~11 levels (floor(pow(2, 3.5)) = 11)
//   8 → 4-bit  → 16 levels

static inline float levels_from_bits_x2(uint16_t bits_x2) {
    // Use actual bit count as float: e.g. 7 → 3.5
    double bits = (double)bits_x2 / 2.0;
    return (float)pow(2.0, bits);
}

void tq_quantize_scalar(
    const float    *in,
    uint8_t        *out_codes,
    float          *out_scales,
    float          *out_zeros,
    size_t          n,
    size_t          block_size,
    uint16_t        bits_x2
) {
    float levels = levels_from_bits_x2(bits_x2);
    float max_code = levels - 1.0f;

    size_t num_blocks = (n + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end   = start + block_size;
        if (end > n) end = n;

        // Compute block min/max
        float block_min = in[start];
        float block_max = in[start];
        for (size_t i = start + 1; i < end; i++) {
            if (in[i] < block_min) block_min = in[i];
            if (in[i] > block_max) block_max = in[i];
        }

        float range = block_max - block_min;
        // Avoid divide-by-zero for constant blocks
        float scale = (range > 1e-9f) ? (range / max_code) : 1.0f;
        float zero  = block_min;

        out_scales[b] = scale;
        out_zeros[b]  = zero;

        // Quantize each element
        for (size_t i = start; i < end; i++) {
            float normalized = (in[i] - zero) / scale;
            // Round to nearest, clamp to [0, max_code]
            float rounded = normalized + 0.5f;
            if (rounded < 0.0f) rounded = 0.0f;
            if (rounded > max_code) rounded = max_code;
            out_codes[i] = (uint8_t)(int)rounded;
        }
    }
}

void tq_dequantize_scalar(
    const uint8_t  *codes,
    const float    *scales,
    const float    *zeros,
    float          *out,
    size_t          n,
    size_t          block_size,
    uint16_t        bits_x2
) {
    (void)bits_x2; // levels not needed for dequantization

    size_t num_blocks = (n + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end   = start + block_size;
        if (end > n) end = n;

        float scale = scales[b];
        float zero  = zeros[b];

        for (size_t i = start; i < end; i++) {
            out[i] = (float)codes[i] * scale + zero;
        }
    }
}
