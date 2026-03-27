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

// ══════════════════════════════════════════════════════════════════════════════
// Fast Walsh-Hadamard Transform (WHT) — O(d log d) structured rotation
// ══════════════════════════════════════════════════════════════════════════════
//
// Replaces the dense QR rotation with D₂ @ H @ D₁ where:
//   D₁, D₂ = random sign-flip diagonals ∈ {-1, +1}
//   H       = Walsh-Hadamard matrix (orthogonal, self-inverse up to scaling)
//
// Complexity: O(d log d) vs O(d²) for dense matrix multiply.
// Memory:     2*dim bytes vs dim*dim*4 bytes.

static int8_t   *g_wht_d1    = NULL;
static int8_t   *g_wht_d2    = NULL;
static uint32_t  g_wht_dim   = 0;
static pthread_mutex_t g_wht_mutex = PTHREAD_MUTEX_INITIALIZER;

// Generate a random sign-flip diagonal: each entry is +1 or -1
static void generate_sign_diag(int8_t *diag, uint32_t dim, uint64_t seed) {
    uint64_t state = seed;
    for (uint32_t i = 0; i < dim; i++) {
        diag[i] = (splitmix64(&state) & 1) ? 1 : -1;
    }
}

// In-place element-wise sign flip: x[i] *= diag[i]
static void apply_sign_diag(float *x, const int8_t *diag, uint32_t dim) {
    for (uint32_t i = 0; i < dim; i++) {
        if (diag[i] < 0) x[i] = -x[i];
    }
}

// In-place iterative Fast Walsh-Hadamard Transform.
// dim must be a power of 2. Normalizes by 1/sqrt(dim).
static void fwht_inplace(float *x, uint32_t dim) {
    for (uint32_t len = 1; len < dim; len <<= 1) {
        for (uint32_t i = 0; i < dim; i += len << 1) {
            for (uint32_t j = 0; j < len; j++) {
                float a = x[i + j];
                float b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)dim);
    for (uint32_t i = 0; i < dim; i++) x[i] *= scale;
}

int tq_wht_init(uint32_t dim, uint64_t seed) {
    if ((dim & (dim - 1)) != 0 || dim == 0) return -1;  // must be power of 2

    pthread_mutex_lock(&g_wht_mutex);
    free(g_wht_d1);
    free(g_wht_d2);
    g_wht_d1 = NULL;
    g_wht_d2 = NULL;
    g_wht_dim = 0;

    g_wht_d1 = (int8_t *)malloc(dim);
    g_wht_d2 = (int8_t *)malloc(dim);
    if (!g_wht_d1 || !g_wht_d2) {
        free(g_wht_d1); free(g_wht_d2);
        g_wht_d1 = NULL; g_wht_d2 = NULL;
        pthread_mutex_unlock(&g_wht_mutex);
        return -1;
    }

    generate_sign_diag(g_wht_d1, dim, seed);
    generate_sign_diag(g_wht_d2, dim, seed ^ 0x1234567890ABCDEFULL);

    g_wht_dim = dim;
    pthread_mutex_unlock(&g_wht_mutex);
    return 0;
}

void tq_wht_cleanup(void) {
    pthread_mutex_lock(&g_wht_mutex);
    free(g_wht_d1); free(g_wht_d2);
    g_wht_d1 = NULL; g_wht_d2 = NULL;
    g_wht_dim = 0;
    pthread_mutex_unlock(&g_wht_mutex);
}

void tq_wht_rotate(const float *x, float *y, uint32_t dim) {
    pthread_mutex_lock(&g_wht_mutex);
    if (!g_wht_d1 || dim != g_wht_dim) {
        pthread_mutex_unlock(&g_wht_mutex);
        return;
    }
    memcpy(y, x, dim * sizeof(float));
    apply_sign_diag(y, g_wht_d1, dim);
    fwht_inplace(y, dim);
    apply_sign_diag(y, g_wht_d2, dim);
    pthread_mutex_unlock(&g_wht_mutex);
}

void tq_wht_rotate_inverse(const float *y, float *x, uint32_t dim) {
    pthread_mutex_lock(&g_wht_mutex);
    if (!g_wht_d1 || dim != g_wht_dim) {
        pthread_mutex_unlock(&g_wht_mutex);
        return;
    }
    // Inverse = D₁ @ H @ D₂ (reverse order; H is self-inverse up to scaling,
    // and D is self-inverse since diag entries are ±1)
    memcpy(x, y, dim * sizeof(float));
    apply_sign_diag(x, g_wht_d2, dim);
    fwht_inplace(x, dim);
    apply_sign_diag(x, g_wht_d1, dim);
    pthread_mutex_unlock(&g_wht_mutex);
}

// ── Dispatch layer: route by transform_kind ─────────────────────────────────

void tq_dispatch_rotate(const float *x, float *y, uint32_t dim,
                         uint32_t transform_kind, uint64_t seed) {
    if (transform_kind == PA_TRANSFORM_HADAMARD) {
        pthread_mutex_lock(&g_wht_mutex);
        int need_init = (!g_wht_d1 || dim != g_wht_dim);
        pthread_mutex_unlock(&g_wht_mutex);
        if (need_init) tq_wht_init(dim, seed);
        tq_wht_rotate(x, y, dim);
    } else {
        // Default: structured rotation (QR)
        tq_rotation_init(dim, seed);
        tq_rotate(x, y, dim);
    }
}

void tq_dispatch_rotate_inverse(const float *y, float *x, uint32_t dim,
                                  uint32_t transform_kind, uint64_t seed) {
    if (transform_kind == PA_TRANSFORM_HADAMARD) {
        pthread_mutex_lock(&g_wht_mutex);
        int need_init = (!g_wht_d1 || dim != g_wht_dim);
        pthread_mutex_unlock(&g_wht_mutex);
        if (need_init) tq_wht_init(dim, seed);
        tq_wht_rotate_inverse(y, x, dim);
    } else {
        tq_rotation_init(dim, seed);
        tq_rotate_inverse(y, x, dim);
    }
}
