#include "TurboQuantCore.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ── splitmix64 PRNG ──────────────────────────────────────────────────────────
// Fast, high-quality 64-bit PRNG used for deterministic sign generation.

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// ── Sign generation ──────────────────────────────────────────────────────────

// Fill `signs` with ±1.0f deterministically from `seed`.
// Each bit of the PRNG output controls one sign: 1 bit → +1.0f or -1.0f.
static void generate_signs(float *signs, uint32_t dim, uint64_t seed) {
    uint64_t state = seed;
    uint32_t i = 0;
    while (i < dim) {
        uint64_t bits = splitmix64(&state);
        // Extract 64 signs from one PRNG call
        uint32_t batch = dim - i;
        if (batch > 64) batch = 64;
        for (uint32_t b = 0; b < batch; b++, i++) {
            signs[i] = ((bits >> b) & 1ULL) ? 1.0f : -1.0f;
        }
    }
}

// ── Fast Walsh-Hadamard Transform ────────────────────────────────────────────
// In-place, normalized by 1/sqrt(n). dim must be a power of 2.

static void fwht_inplace(float *x, uint32_t n) {
    // Iterative Cooley-Tukey-style FWHT
    for (uint32_t len = 1; len < n; len <<= 1) {
        for (uint32_t i = 0; i < n; i += len << 1) {
            for (uint32_t j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
    // Normalize by 1/sqrt(n)
    float inv_sqrt_n = 1.0f / sqrtf((float)n);
    for (uint32_t i = 0; i < n; i++) {
        x[i] *= inv_sqrt_n;
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

// Forward rotation: R @ x  =  H @ diag(signs) @ x
// 1. Multiply x element-wise by signs
// 2. Apply FWHT
void tq_rotate_inplace(float *x, uint32_t dim, uint64_t seed) {
    float *signs = (float *)malloc(dim * sizeof(float));
    generate_signs(signs, dim, seed);

    // Step 1: diag(signs) @ x
    for (uint32_t i = 0; i < dim; i++) {
        x[i] *= signs[i];
    }

    // Step 2: H @ (signs*x)
    fwht_inplace(x, dim);

    free(signs);
}

// Inverse rotation: R^T @ x  =  (H @ diag(signs))^T @ x
//                             =  diag(signs)^T @ H^T @ x
//                             =  diag(signs) @ H @ x   (H is symmetric, signs are ±1 so diag^T = diag)
// 1. Apply FWHT
// 2. Multiply by signs
void tq_rotate_inverse_inplace(float *x, uint32_t dim, uint64_t seed) {
    float *signs = (float *)malloc(dim * sizeof(float));
    generate_signs(signs, dim, seed);

    // Step 1: H @ x
    fwht_inplace(x, dim);

    // Step 2: diag(signs) @ (H @ x)
    for (uint32_t i = 0; i < dim; i++) {
        x[i] *= signs[i];
    }

    free(signs);
}

// Rotate query: q_out = R @ q_in
// Since R is orthogonal (R^T R = I), dot(R@k, R@q) = k^T R^T R q = k^T q.
// Both key and query receive the same forward rotation so their dot product
// is preserved. The task spec phrase "Q @ R^T" refers to the row-vector
// convention; in column-vector convention this is R @ q.
void tq_rotate_query(const float *q_in, float *q_out, uint32_t dim, uint64_t seed) {
    memcpy(q_out, q_in, dim * sizeof(float));
    tq_rotate_inplace(q_out, dim, seed);
}
