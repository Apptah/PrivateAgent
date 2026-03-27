#include <metal_stdlib>
using namespace metal;

// ══════════════════════════════════════════════════════════════════════════════
// TurboQuant Metal GPU Kernels
// ══════════════════════════════════════════════════════════════════════════════

// ── Lloyd-Max dequantize (float32) ──
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

// ── Lloyd-Max dequantize (half4 SIMD) ──
// Processes 4 elements per thread for better throughput
kernel void tq_dequantize_lloydmax_half4_kernel(
    device const uint8_t *codes [[buffer(0)]],
    device const half *codebook [[buffer(1)]],
    device half4 *output [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint base = tid * 4;
    if (base >= dim) return;
    half inv_scale = rsqrt(half(dim));
    half4 vals = half4(
        codebook[codes[base]],
        codebook[codes[base + 1]],
        codebook[codes[base + 2]],
        codebook[codes[base + 3]]
    );
    output[tid] = vals * inv_scale;
}

// ── WHT butterfly kernel ──
// One stage of the Walsh-Hadamard transform (in-place butterfly).
// Host dispatches log2(dim) times with increasing stage values.
kernel void tq_wht_butterfly_kernel(
    device float *data [[buffer(0)]],
    constant uint &dim [[buffer(1)]],
    constant uint &half_len [[buffer(2)]],   // len = 1 << stage; half_len = len
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one butterfly pair
    uint block_size = half_len << 1;
    uint block_idx = tid / half_len;
    uint offset = tid % half_len;
    uint i = block_idx * block_size + offset;
    if (i + half_len >= dim) return;

    float a = data[i];
    float b = data[i + half_len];
    data[i]            = a + b;
    data[i + half_len] = a - b;
}

// ── Sign diagonal apply kernel ──
// x[i] *= (diag[i] < 0) ? -1 : 1
kernel void tq_sign_diag_kernel(
    device float *x [[buffer(0)]],
    device const char *diag [[buffer(1)]],
    constant uint &dim [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    if (diag[tid] < 0) x[tid] = -x[tid];
}

// ── WHT normalize kernel ──
// x[i] *= 1/sqrt(dim)
kernel void tq_wht_normalize_kernel(
    device float *x [[buffer(0)]],
    constant uint &dim [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;
    x[tid] *= rsqrt(float(dim));
}

// ── Matrix-vector multiply kernel (for legacy Π^T @ y) ──
kernel void tq_matvec_transpose_kernel(
    device const float *matrix [[buffer(0)]],
    device const float *vec [[buffer(1)]],
    device float *output [[buffer(2)]],
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

// ── Scale + accumulate (half4) ──
kernel void tq_scale_accumulate_half4_kernel(
    device const half4 *input [[buffer(0)]],
    device half4 *output [[buffer(1)]],
    constant half &weight [[buffer(2)]],
    constant uint &dim_div4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim_div4) return;
    output[tid] += weight * input[tid];
}

// ── Dot product kernel (reduction) ──
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

// ── Dot product (half4, more efficient for dim <= 256) ──
kernel void tq_dot_product_half4_kernel(
    device const half4 *a [[buffer(0)]],
    device const half4 *b [[buffer(1)]],
    device float *partial_sums [[buffer(2)]],
    constant uint &dim_div4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float local_sum = 0.0f;
    if (tid < dim_div4) {
        half4 prod = a[tid] * b[tid];
        local_sum = float(prod.x + prod.y + prod.z + prod.w);
    }
    shared_sum[lid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

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
kernel void tq_qjl_decode_kernel(
    device const float *s_matrix [[buffer(0)]],
    device const uint8_t *qjl_bits [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant float &coeff [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= dim) return;

    float sum = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float sign_i = ((qjl_bits[i / 8] >> (i % 8)) & 1u) ? 1.0f : -1.0f;
        sum += s_matrix[i * dim + tid] * sign_i;
    }
    output[tid] = coeff * sum;
}
