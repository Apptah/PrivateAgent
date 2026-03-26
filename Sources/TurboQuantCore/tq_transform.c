#include "TurboQuantCore.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

// ── Cached rotation matrix (Π, dim×dim, column-major) ────────────────────────
// Column-major layout: Π[col * dim + row]
// Forward:  y[i] = sum_j Π[j*dim + i] * x[j]   →  y = Π @ x
// Inverse:  x[j] = sum_i Π[j*dim + i] * y[i]   →  x = Π^T @ y

static float    *g_rotation     = NULL;
static uint32_t  g_rotation_dim = 0;
static pthread_mutex_t g_rotation_mutex = PTHREAD_MUTEX_INITIALIZER;

// ── splitmix64 PRNG ───────────────────────────────────────────────────────────

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

// Box-Muller transform → standard normal sample (double precision to avoid inf)
static float randn(uint64_t *state) {
    // Use double to avoid precision loss with large integer → float conversion
    double u1 = ((double)(splitmix64(state) >> 11) + 1.0) / ((double)(1ULL << 53) + 1.0);
    double u2 = (double)(splitmix64(state) >> 11) / (double)(1ULL << 53);
    // Clamp u1 away from 0 to prevent log(0)
    if (u1 < 1e-15) u1 = 1e-15;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

// ── Modified Gram-Schmidt QR (double precision for numerical stability) ──────
// Orthogonalises the columns of A (column-major, dim×dim).
// Computation in double, output stored in float Q.
// This is a one-time init cost, so double precision is acceptable.

static void gram_schmidt_qr(float *A, float *Q, uint32_t dim) {
    size_t n = (size_t)dim * dim;

    // Work in double precision to maintain orthogonality for large dim
    double *Qd = (double *)malloc(n * sizeof(double));
    if (!Qd) return;

    for (size_t i = 0; i < n; i++) Qd[i] = (double)A[i];

    for (uint32_t j = 0; j < dim; j++) {
        // Orthogonalise column j against all previous columns
        for (uint32_t k = 0; k < j; k++) {
            double dot = 0.0;
            for (uint32_t i = 0; i < dim; i++) {
                dot += Qd[k * dim + i] * Qd[j * dim + i];
            }
            for (uint32_t i = 0; i < dim; i++) {
                Qd[j * dim + i] -= dot * Qd[k * dim + i];
            }
        }
        // Normalise column j
        double norm2 = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            norm2 += Qd[j * dim + i] * Qd[j * dim + i];
        }
        double inv_norm = (norm2 > 1e-20) ? 1.0 / sqrt(norm2) : 0.0;
        for (uint32_t i = 0; i < dim; i++) {
            Qd[j * dim + i] *= inv_norm;
        }
    }

    // Convert back to float
    for (size_t i = 0; i < n; i++) Q[i] = (float)Qd[i];
    free(Qd);
}

// ── Public init / cleanup ─────────────────────────────────────────────────────

int tq_rotation_init(uint32_t dim, uint64_t seed) {
    pthread_mutex_lock(&g_rotation_mutex);
    // Inline cleanup (avoid recursive lock)
    free(g_rotation);
    g_rotation = NULL;
    g_rotation_dim = 0;

    size_t n = (size_t)dim * dim;
    float *A = malloc(n * sizeof(float));
    g_rotation = malloc(n * sizeof(float));
    if (!A || !g_rotation) {
        free(A);
        free(g_rotation);
        g_rotation = NULL;
        pthread_mutex_unlock(&g_rotation_mutex);
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
    pthread_mutex_unlock(&g_rotation_mutex);
    return 0;
}

void tq_rotation_cleanup(void) {
    pthread_mutex_lock(&g_rotation_mutex);
    free(g_rotation);
    g_rotation     = NULL;
    g_rotation_dim = 0;
    pthread_mutex_unlock(&g_rotation_mutex);
}

// ── Core rotate / inverse ─────────────────────────────────────────────────────

// Internal helpers (caller must hold g_rotation_mutex)
static void rotate_locked(const float *x, float *y, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < dim; j++) {
            sum += g_rotation[j * dim + i] * x[j];
        }
        y[i] = sum;
    }
}

static void rotate_inverse_locked(const float *y, float *x, uint32_t dim) {
    for (uint32_t j = 0; j < dim; j++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            sum += g_rotation[j * dim + i] * y[i];
        }
        x[j] = sum;
    }
}

void tq_rotate(const float *x, float *y, uint32_t dim) {
    pthread_mutex_lock(&g_rotation_mutex);
    if (!g_rotation || dim != g_rotation_dim) {
        pthread_mutex_unlock(&g_rotation_mutex);
        return;
    }
    rotate_locked(x, y, dim);
    pthread_mutex_unlock(&g_rotation_mutex);
}

void tq_rotate_inverse(const float *y, float *x, uint32_t dim) {
    pthread_mutex_lock(&g_rotation_mutex);
    if (!g_rotation || dim != g_rotation_dim) {
        pthread_mutex_unlock(&g_rotation_mutex);
        return;
    }
    rotate_inverse_locked(y, x, dim);
    pthread_mutex_unlock(&g_rotation_mutex);
}

// ── Backward-compatible wrappers ──────────────────────────────────────────────

void tq_rotate_inplace(float *x, uint32_t dim, uint64_t seed) {
    pthread_mutex_lock(&g_rotation_mutex);
    if (!g_rotation || dim != g_rotation_dim) {
        pthread_mutex_unlock(&g_rotation_mutex);
        tq_rotation_init(dim, seed);
        pthread_mutex_lock(&g_rotation_mutex);
    }
    if (!g_rotation || dim != g_rotation_dim) {
        pthread_mutex_unlock(&g_rotation_mutex);
        return;
    }
    float *tmp = malloc(dim * sizeof(float));
    if (!tmp) { pthread_mutex_unlock(&g_rotation_mutex); return; }
    rotate_locked(x, tmp, dim);
    memcpy(x, tmp, dim * sizeof(float));
    free(tmp);
    pthread_mutex_unlock(&g_rotation_mutex);
}

void tq_rotate_inverse_inplace(float *x, uint32_t dim, uint64_t seed) {
    pthread_mutex_lock(&g_rotation_mutex);
    if (!g_rotation || dim != g_rotation_dim) {
        pthread_mutex_unlock(&g_rotation_mutex);
        tq_rotation_init(dim, seed);
        pthread_mutex_lock(&g_rotation_mutex);
    }
    if (!g_rotation || dim != g_rotation_dim) {
        pthread_mutex_unlock(&g_rotation_mutex);
        return;
    }
    float *tmp = malloc(dim * sizeof(float));
    if (!tmp) { pthread_mutex_unlock(&g_rotation_mutex); return; }
    rotate_inverse_locked(x, tmp, dim);
    memcpy(x, tmp, dim * sizeof(float));
    free(tmp);
    pthread_mutex_unlock(&g_rotation_mutex);
}

void tq_rotate_query(const float *q_in, float *q_out, uint32_t dim, uint64_t seed) {
    pthread_mutex_lock(&g_rotation_mutex);
    if (!g_rotation || dim != g_rotation_dim) {
        pthread_mutex_unlock(&g_rotation_mutex);
        tq_rotation_init(dim, seed);
        pthread_mutex_lock(&g_rotation_mutex);
    }
    if (!g_rotation || dim != g_rotation_dim) {
        pthread_mutex_unlock(&g_rotation_mutex);
        return;
    }
    rotate_locked(q_in, q_out, dim);
    pthread_mutex_unlock(&g_rotation_mutex);
}
