#include "TurboQuantCore.h"
#include <math.h>
#include <string.h>

// ── splitmix64 PRNG ──
// Same generator used by the rotation transform code.
// Produces a deterministic sequence from an integer seed.

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// ── QJL Encode ──
//
// For each element i:
//   1. Compute residual = input[i] - dequant[i]
//   2. Draw a random sign r_i ∈ {-1, +1} from splitmix64
//   3. bit_i = (residual * r_i >= 0) ? 1 : 0
//   4. Pack 8 bits per output byte, LSB first within each byte

void tq_qjl_encode(
    const float    *input,
    const float    *dequant,
    uint8_t        *out_bits,
    size_t          n,
    uint64_t        seed
) {
    size_t num_bytes = (n + 7) / 8;
    memset(out_bits, 0, num_bytes);

    uint64_t state = seed;

    for (size_t i = 0; i < n; i++) {
        float residual = input[i] - dequant[i];

        // Random sign: bit 63 of splitmix64 output → +1 or -1
        uint64_t rnd = splitmix64(&state);
        float sign = (rnd >> 63) ? -1.0f : 1.0f;

        int bit = (residual * sign >= 0.0f) ? 1 : 0;

        // Pack LSB-first into output bytes
        size_t byte_idx = i / 8;
        size_t bit_idx  = i % 8;
        out_bits[byte_idx] |= (uint8_t)(bit << bit_idx);
    }
}

// ── QJL Score Correction ──
//
// For each token t, computes the inner product between Q and K residual
// projections using popcount on XNORed packed bits:
//
//   agreement(q, k) = popcount(~(q_bits XOR k_bits)) − (dim/2)
//   correction = agreement * (1 / sqrt(dim))
//
// The XNOR popcount gives the number of bit positions where q and k agree,
// which is an unbiased estimator of sign(q_residual) · sign(k_residual).
// Subtracting dim/2 centers the estimator around zero.

static inline int popcount_byte(uint8_t b) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(b);
#else
    int c = 0;
    while (b) { c += b & 1; b >>= 1; }
    return c;
#endif
}

void tq_qjl_correct_scores(
    const uint8_t  *q_bits,
    const uint8_t  *k_bits,
    float          *scores,
    size_t          num_tokens,
    size_t          dim
) {
    size_t bits_per_token = (dim + 7) / 8;
    float inv_sqrt_dim = (dim > 0) ? (1.0f / sqrtf((float)dim)) : 0.0f;

    for (size_t t = 0; t < num_tokens; t++) {
        const uint8_t *k_tok = k_bits + t * bits_per_token;

        int agreement = 0;
        for (size_t b = 0; b < bits_per_token; b++) {
            // XNOR: 1 where bits agree
            uint8_t xnor = (uint8_t)(~(q_bits[b] ^ k_tok[b]));
            agreement += popcount_byte(xnor);
        }

        // Center: full agreement (all bits match) → dim, random → dim/2
        float centered = (float)agreement - (float)(bits_per_token * 8) / 2.0f;
        scores[t] += centered * inv_sqrt_dim;
    }
}
