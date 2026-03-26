#include "TurboQuantCore.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ── Cached QJL projection matrix S (d×d, N(0,1), row-major) ──────────────────
// Row-major: S[i*dim + j]
// Forward projection: proj[i] = Σ_j S[i*dim + j] * residual[j]
// Transpose:          out[j]  = Σ_i S[i*dim + j] * sign[i]

static float    *g_qjl_matrix = NULL;
static uint32_t  g_qjl_dim    = 0;

// ── splitmix64 + Box-Muller (local copies to avoid symbol conflict) ────────────

static inline uint64_t qjl_splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static float qjl_randn(uint64_t *state) {
    float u1 = (float)(qjl_splitmix64(state) >> 11) / (float)(1ULL << 53) + 1e-10f;
    float u2 = (float)(qjl_splitmix64(state) >> 11) / (float)(1ULL << 53);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// ── Init / cleanup ────────────────────────────────────────────────────────────

int tq_qjl_init(uint32_t dim, uint64_t seed) {
    tq_qjl_cleanup();
    size_t n = (size_t)dim * dim;
    g_qjl_matrix = malloc(n * sizeof(float));
    if (!g_qjl_matrix) return -1;

    uint64_t state = seed;
    for (size_t i = 0; i < n; i++) {
        g_qjl_matrix[i] = qjl_randn(&state);
    }
    g_qjl_dim = dim;
    return 0;
}

void tq_qjl_cleanup(void) {
    free(g_qjl_matrix);
    g_qjl_matrix = NULL;
    g_qjl_dim    = 0;
}

// ── Encode: qjl_bits = sign(S @ residual) ────────────────────────────────────

void tq_qjl_encode(const float *residual, uint32_t dim,
                    uint8_t *qjl_bits, float *residual_norm_out) {
    if (!residual || !qjl_bits || dim == 0) return;

    // Compute ‖residual‖₂  (γ)
    float norm2 = 0.0f;
    for (uint32_t i = 0; i < dim; i++) norm2 += residual[i] * residual[i];
    float norm = sqrtf(norm2);
    if (residual_norm_out) *residual_norm_out = norm;

    // Ensure QJL matrix is ready (lazy init with a fixed seed if caller forgot)
    if (!g_qjl_matrix || dim != g_qjl_dim) {
        tq_qjl_init(dim, 0xDEADBEEFCAFEBABEULL);
    }

    uint32_t num_bytes = (dim + 7) / 8;
    memset(qjl_bits, 0, num_bytes);

    if (!g_qjl_matrix) return;

    // Project: proj[i] = S[i,:] · residual;  store sign bit
    for (uint32_t i = 0; i < dim; i++) {
        float proj = 0.0f;
        const float *row = g_qjl_matrix + (size_t)i * dim;
        for (uint32_t j = 0; j < dim; j++) {
            proj += row[j] * residual[j];
        }
        if (proj >= 0.0f) {
            qjl_bits[i / 8] |= (uint8_t)(1u << (i % 8));
        }
    }
}

// ── Decode: output = (sqrt(π/2) / d) * γ * S^T @ sign_bits ──────────────────

void tq_qjl_decode(const uint8_t *qjl_bits, float gamma,
                    float *output, uint32_t dim) {
    if (!qjl_bits || !output || !g_qjl_matrix || g_qjl_dim != dim || dim == 0) return;

    // Coefficient from paper: sqrt(π/2) / d
    float coeff = sqrtf((float)M_PI / 2.0f) / (float)dim * gamma;

    // output[j] = coeff * Σ_i S[i,j] * sign_i
    for (uint32_t j = 0; j < dim; j++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            float sign_i = ((qjl_bits[i / 8] >> (i % 8)) & 1u) ? 1.0f : -1.0f;
            // S^T[j,i] = S[i,j] = g_qjl_matrix[i*dim + j]
            sum += g_qjl_matrix[(size_t)i * dim + j] * sign_i;
        }
        output[j] = coeff * sum;
    }
}

// ── Deprecated no-op ──────────────────────────────────────────────────────────

void tq_qjl_correct_scores(float *scores, uint32_t num_tokens,
                            const uint8_t *q_residual_bits,
                            const uint8_t *k_residual_bits,
                            uint32_t dim, uint64_t qjl_seed) {
    // Replaced by full dequant + dot in new design.
    (void)scores; (void)num_tokens;
    (void)q_residual_bits; (void)k_residual_bits;
    (void)dim; (void)qjl_seed;
}
