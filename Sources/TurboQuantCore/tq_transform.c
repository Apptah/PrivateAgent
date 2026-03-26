#include "TurboQuantCore.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ── Cached rotation matrix (Π, dim×dim, column-major) ────────────────────────
// Column-major layout: Π[col * dim + row]
// Forward:  y[i] = sum_j Π[j*dim + i] * x[j]   →  y = Π @ x
// Inverse:  x[j] = sum_i Π[j*dim + i] * y[i]   →  x = Π^T @ y

static float    *g_rotation     = NULL;
static uint32_t  g_rotation_dim = 0;

// ── splitmix64 PRNG ───────────────────────────────────────────────────────────

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// Box-Muller transform → standard normal sample
static float randn(uint64_t *state) {
    // u1 ∈ (0, 1], u2 ∈ [0, 1)
    float u1 = (float)(splitmix64(state) >> 11) / (float)(1ULL << 53) + 1e-10f;
    float u2 = (float)(splitmix64(state) >> 11) / (float)(1ULL << 53);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// ── Modified Gram-Schmidt QR ──────────────────────────────────────────────────
// Orthogonalises the columns of A (column-major, dim×dim) in-place → Q.
// Output Q is written column-major into the pre-allocated buffer.

static void gram_schmidt_qr(float *A, float *Q, uint32_t dim) {
    memcpy(Q, A, (size_t)dim * dim * sizeof(float));

    for (uint32_t j = 0; j < dim; j++) {
        // Orthogonalise column j against all previous columns
        for (uint32_t k = 0; k < j; k++) {
            float dot = 0.0f;
            for (uint32_t i = 0; i < dim; i++) {
                dot += Q[k * dim + i] * Q[j * dim + i];
            }
            for (uint32_t i = 0; i < dim; i++) {
                Q[j * dim + i] -= dot * Q[k * dim + i];
            }
        }
        // Normalise column j
        float norm2 = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            norm2 += Q[j * dim + i] * Q[j * dim + i];
        }
        float inv_norm = (norm2 > 1e-10f) ? 1.0f / sqrtf(norm2) : 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            Q[j * dim + i] *= inv_norm;
        }
    }
}

// ── Public init / cleanup ─────────────────────────────────────────────────────

int tq_rotation_init(uint32_t dim, uint64_t seed) {
    tq_rotation_cleanup();

    size_t n = (size_t)dim * dim;
    float *A = malloc(n * sizeof(float));
    g_rotation = malloc(n * sizeof(float));
    if (!A || !g_rotation) {
        free(A);
        free(g_rotation);
        g_rotation = NULL;
        return -1;
    }

    // Fill with N(0,1) samples (column-major)
    uint64_t state = seed;
    for (size_t i = 0; i < n; i++) {
        A[i] = randn(&state);
    }

    gram_schmidt_qr(A, g_rotation, dim);
    free(A);

    g_rotation_dim = dim;
    return 0;
}

void tq_rotation_cleanup(void) {
    free(g_rotation);
    g_rotation     = NULL;
    g_rotation_dim = 0;
}

// ── Core rotate / inverse ─────────────────────────────────────────────────────

void tq_rotate(const float *x, float *y, uint32_t dim) {
    if (!g_rotation || dim != g_rotation_dim) return;
    // y[i] = Σ_j Π[j*dim + i] * x[j]  (column-major forward matmul)
    for (uint32_t i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < dim; j++) {
            sum += g_rotation[j * dim + i] * x[j];
        }
        y[i] = sum;
    }
}

void tq_rotate_inverse(const float *y, float *x, uint32_t dim) {
    if (!g_rotation || dim != g_rotation_dim) return;
    // x[j] = Σ_i Π[j*dim + i] * y[i]  (column-major transpose matmul = Π^T @ y)
    for (uint32_t j = 0; j < dim; j++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            sum += g_rotation[j * dim + i] * y[i];
        }
        x[j] = sum;
    }
}

// ── Backward-compatible wrappers ──────────────────────────────────────────────

void tq_rotate_inplace(float *x, uint32_t dim, uint64_t seed) {
    if (!g_rotation || dim != g_rotation_dim) {
        tq_rotation_init(dim, seed);
    }
    float *tmp = malloc(dim * sizeof(float));
    if (!tmp) return;
    tq_rotate(x, tmp, dim);
    memcpy(x, tmp, dim * sizeof(float));
    free(tmp);
}

void tq_rotate_inverse_inplace(float *x, uint32_t dim, uint64_t seed) {
    if (!g_rotation || dim != g_rotation_dim) {
        tq_rotation_init(dim, seed);
    }
    float *tmp = malloc(dim * sizeof(float));
    if (!tmp) return;
    tq_rotate_inverse(x, tmp, dim);
    memcpy(x, tmp, dim * sizeof(float));
    free(tmp);
}

void tq_rotate_query(const float *q_in, float *q_out, uint32_t dim, uint64_t seed) {
    if (!g_rotation || dim != g_rotation_dim) {
        tq_rotation_init(dim, seed);
    }
    tq_rotate(q_in, q_out, dim);
}
