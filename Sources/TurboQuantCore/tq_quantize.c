#include "TurboQuantCore.h"
#include <math.h>
#include <string.h>
#include <float.h>

// ── Lloyd-Max centroids for N(0,1) ────────────────────────────────────────────
// Pre-computed optimal scalar quantisation centroids for the standard normal
// distribution.  These match the theoretical Lloyd-Max solution and are used
// after scaling the rotated coordinates from N(0, 1/d) → N(0,1).

// 1-bit  → 2 levels
static const float codebook_1bit[] = { -0.7979f, 0.7979f };

// 2-bit  → 4 levels
static const float codebook_2bit[] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };

// 3-bit  → 8 levels
static const float codebook_3bit[] = {
    -2.1519f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1519f
};

// 4-bit  → 16 levels
static const float codebook_4bit[] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};

const float *tq_lloydmax_codebook(uint8_t bits) {
    switch (bits) {
        case 1:  return codebook_1bit;
        case 2:  return codebook_2bit;
        case 3:  return codebook_3bit;
        case 4:  return codebook_4bit;
        default: return codebook_4bit;
    }
}

// ── Lloyd-Max quantise (paper Algorithm 1) ────────────────────────────────────
// After rotating a unit-norm vector with Π, each coordinate is approximately
// N(0, 1/d).  We scale by sqrt(d) to bring it to N(0,1), find the nearest
// Lloyd-Max centroid, and store the index.

uint32_t tq_quantize_lloydmax(const float *rotated_unit, uint32_t dim,
                               uint8_t *codes, uint8_t bits) {
    if (!rotated_unit || !codes || dim == 0 || bits == 0 || bits > 4) return 0;

    const float *cb     = tq_lloydmax_codebook(bits);
    uint32_t     levels = 1u << bits;
    float        scale  = sqrtf((float)dim);   // N(0,1/d) → N(0,1)

    for (uint32_t i = 0; i < dim; i++) {
        float val      = rotated_unit[i] * scale;
        float best_d2  = FLT_MAX;
        uint8_t best_k = 0;
        for (uint32_t k = 0; k < levels; k++) {
            float d = val - cb[k];
            float d2 = d * d;
            if (d2 < best_d2) { best_d2 = d2; best_k = (uint8_t)k; }
        }
        codes[i] = best_k;
    }
    return dim;
}

// ── Lloyd-Max dequantise ──────────────────────────────────────────────────────
// Inverse: centroid → scaled back to N(0, 1/d).

void tq_dequantize_lloydmax(const uint8_t *codes, float *rotated_unit,
                              uint32_t dim, uint8_t bits) {
    if (!codes || !rotated_unit || dim == 0 || bits == 0 || bits > 4) return;

    const float *cb      = tq_lloydmax_codebook(bits);
    float        inv_scale = 1.0f / sqrtf((float)dim);

    for (uint32_t i = 0; i < dim; i++) {
        rotated_unit[i] = cb[codes[i]] * inv_scale;
    }
}

// ── Split quantisation (outlier channel, non-integer bit rates) ──────────────
// For fractional bit rates (e.g. bits_x2=7 → 3.5-bit):
//   First dim/2 channels use (lo_bits+1), remaining use lo_bits.
//   Data-oblivious: fixed split, not activation-magnitude based.
//   After rotation, channels are randomised, so any fixed split is fair.

uint32_t tq_quantize_lloydmax_split(const float *rotated_unit, uint32_t dim,
                                      uint8_t *codes, uint16_t bits_x2) {
    if (!rotated_unit || !codes || dim == 0) return 0;

    // Integer bit rate → delegate to standard path
    if ((bits_x2 & 1) == 0) {
        uint8_t bits = (uint8_t)(bits_x2 / 2);
        if (bits == 0 || bits > 4) bits = 4;
        return tq_quantize_lloydmax(rotated_unit, dim, codes, bits);
    }

    // Fractional: e.g. bits_x2=7 → lo=3, hi=4, split=dim/2
    uint8_t lo_bits = (uint8_t)(bits_x2 / 2);
    uint8_t hi_bits = lo_bits + 1;
    if (hi_bits > 4) hi_bits = 4;
    if (lo_bits < 1) lo_bits = 1;
    uint32_t hi_count = dim / 2;  // first half gets higher bits

    const float *cb_hi = tq_lloydmax_codebook(hi_bits);
    uint32_t levels_hi = 1u << hi_bits;
    const float *cb_lo = tq_lloydmax_codebook(lo_bits);
    uint32_t levels_lo = 1u << lo_bits;
    float scale = sqrtf((float)dim);

    // Quantise first half with hi_bits
    for (uint32_t i = 0; i < hi_count; i++) {
        float val = rotated_unit[i] * scale;
        float best_d2 = FLT_MAX;
        uint8_t best_k = 0;
        for (uint32_t k = 0; k < levels_hi; k++) {
            float d = val - cb_hi[k];
            float d2 = d * d;
            if (d2 < best_d2) { best_d2 = d2; best_k = (uint8_t)k; }
        }
        codes[i] = best_k;
    }
    // Quantise second half with lo_bits
    for (uint32_t i = hi_count; i < dim; i++) {
        float val = rotated_unit[i] * scale;
        float best_d2 = FLT_MAX;
        uint8_t best_k = 0;
        for (uint32_t k = 0; k < levels_lo; k++) {
            float d = val - cb_lo[k];
            float d2 = d * d;
            if (d2 < best_d2) { best_d2 = d2; best_k = (uint8_t)k; }
        }
        codes[i] = best_k;
    }
    return dim;
}

void tq_dequantize_lloydmax_split(const uint8_t *codes, float *rotated_unit,
                                    uint32_t dim, uint16_t bits_x2) {
    if (!codes || !rotated_unit || dim == 0) return;

    if ((bits_x2 & 1) == 0) {
        uint8_t bits = (uint8_t)(bits_x2 / 2);
        if (bits == 0 || bits > 4) bits = 4;
        tq_dequantize_lloydmax(codes, rotated_unit, dim, bits);
        return;
    }

    uint8_t lo_bits = (uint8_t)(bits_x2 / 2);
    uint8_t hi_bits = lo_bits + 1;
    if (hi_bits > 4) hi_bits = 4;
    if (lo_bits < 1) lo_bits = 1;
    uint32_t hi_count = dim / 2;

    const float *cb_hi = tq_lloydmax_codebook(hi_bits);
    const float *cb_lo = tq_lloydmax_codebook(lo_bits);
    float inv_scale = 1.0f / sqrtf((float)dim);

    for (uint32_t i = 0; i < hi_count; i++) {
        rotated_unit[i] = cb_hi[codes[i]] * inv_scale;
    }
    for (uint32_t i = hi_count; i < dim; i++) {
        rotated_unit[i] = cb_lo[codes[i]] * inv_scale;
    }
}

// ── Backward-compatible wrappers ──────────────────────────────────────────────
// The old API carried per-block scale/zero for affine quant.  Those fields are
// ignored; we delegate to the Lloyd-Max path instead.  Callers that read back
// scale/zero for their own arithmetic will get stale values — acceptable for
// the deprecated path.

uint32_t tq_quantize_scalar(const float *input, uint32_t dim,
                             uint8_t *codes, float *scale, float *zero,
                             uint32_t block_size, uint16_t bits_x2) {
    (void)scale;
    (void)zero;
    (void)block_size;
    uint8_t bits = (uint8_t)(bits_x2 / 2);
    if (bits == 0) bits = 4;
    return tq_quantize_lloydmax(input, dim, codes, bits);
}

void tq_dequantize_scalar(const uint8_t *codes, const float *scale,
                           const float *zero, float *output, uint32_t dim,
                           uint32_t block_size, uint16_t bits_x2) {
    (void)scale;
    (void)zero;
    (void)block_size;
    uint8_t bits = (uint8_t)(bits_x2 / 2);
    if (bits == 0) bits = 4;
    tq_dequantize_lloydmax(codes, output, dim, bits);
}
