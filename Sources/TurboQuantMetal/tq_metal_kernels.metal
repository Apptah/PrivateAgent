#include <metal_stdlib>
using namespace metal;

// ── Lloyd-Max dequantize kernel ──
// Dequantizes codes using codebook, applies inverse scale (1/sqrt(dim))
kernel void tq_dequantize_lloydmax_kernel(
    device const uint8_t *codes [[buffer(0)]],
    device const float *codebook [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float inv_scale = rsqrt(float(dim));
    output[tid] = codebook[codes[tid]] * inv_scale;
}

// ── Matrix-vector multiply kernel (for Π^T @ y) ──
// Computes out[j] = sum_i matrix[j*dim + i] * vec[i]
// Used for inverse rotation
kernel void tq_matvec_transpose_kernel(
    device const float *matrix [[buffer(0)]],    // dim × dim, column-major
    device const float *vec [[buffer(1)]],        // dim
    device float *output [[buffer(2)]],           // dim
    constant uint &dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    float sum = 0.0f;
    for (uint i = 0; i < dim; i++) {
        sum += matrix[tid * dim + i] * vec[i];
    }
    output[tid] = sum;
}

// ── Scale + accumulate kernel ──
// output[i] += weight * input[i]
kernel void tq_scale_accumulate_kernel(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant float &weight [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    output[tid] += weight * input[tid];
}

// ── Dot product kernel (reduction) ──
// Partial sums per threadgroup, final reduction on CPU
kernel void tq_dot_product_kernel(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device float *partial_sums [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float local_sum = 0.0f;
    if (tid < dim) {
        local_sum = a[tid] * b[tid];
    }
    shared_sum[lid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partial_sums[tgid] = shared_sum[0];
    }
}

// ── QJL decode kernel ──
// output[j] = coeff * sum_i S[i*dim + j] * sign_bits[i]
kernel void tq_qjl_decode_kernel(
    device const float *s_matrix [[buffer(0)]],   // d×d row-major
    device const uint8_t *qjl_bits [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant float &coeff [[buffer(3)]],           // sqrt(π/2)/d * gamma
    constant uint &dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float sum = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float sign_i = ((qjl_bits[i / 8] >> (i % 8)) & 1u) ? 1.0f : -1.0f;
        sum += s_matrix[i * dim + tid] * sign_i;  // S^T[tid, i] = S[i, tid]
    }
    output[tid] = coeff * sum;
}
