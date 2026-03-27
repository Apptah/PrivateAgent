#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "TurboQuantMetal.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

// ── Pipeline state cache ──────────────────────────────────────────────────────

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_queue = nil;

// Original kernels
static id<MTLComputePipelineState> g_dequant_pso = nil;
static id<MTLComputePipelineState> g_matvec_pso = nil;
static id<MTLComputePipelineState> g_scale_acc_pso = nil;
static id<MTLComputePipelineState> g_dot_pso = nil;
static id<MTLComputePipelineState> g_qjl_decode_pso = nil;

// New kernels (WHT + half4)
static id<MTLComputePipelineState> g_dequant_half4_pso = nil;
static id<MTLComputePipelineState> g_wht_butterfly_pso = nil;
static id<MTLComputePipelineState> g_sign_diag_pso = nil;
static id<MTLComputePipelineState> g_wht_normalize_pso = nil;
static id<MTLComputePipelineState> g_scale_acc_half4_pso = nil;
static id<MTLComputePipelineState> g_dot_half4_pso = nil;

// Cached GPU buffers
static id<MTLBuffer> g_rotation_buf = nil;
static id<MTLBuffer> g_qjl_buf = nil;
static id<MTLBuffer> g_codebook_buf = nil;
static id<MTLBuffer> g_codebook_half_buf = nil;
static id<MTLBuffer> g_d1_buf = nil;
static id<MTLBuffer> g_d2_buf = nil;
static uint32_t g_cached_dim = 0;

// ── Helper: create PSO from kernel name ─────────────────────────────────────

static id<MTLComputePipelineState> create_pso(NSString *name) {
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        NSLog(@"TurboQuantMetal: missing kernel: %@", name);
        return nil;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
        NSLog(@"TurboQuantMetal: PSO creation failed for %@: %@", name, error);
    }
    return pso;
}

// ── Init / Cleanup ────────────────────────────────────────────────────────────

int tq_metal_init(void *device) {
    if (!device) return -1;
    g_device = (__bridge id<MTLDevice>)device;

    // Try loading pre-compiled metallib from bundle
    g_library = [g_device newDefaultLibrary];

    if (!g_library) {
        // Fallback: compile from source (macOS dev)
        NSArray *searchPaths = @[
            @"Sources/TurboQuantMetal/tq_metal_kernels.metal",
            @"tq_metal_kernels.metal"
        ];
        NSString *shaderPath = nil;
        for (NSString *path in searchPaths) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                shaderPath = path;
                break;
            }
        }
        if (shaderPath) {
            NSError *error = nil;
            NSString *src = [NSString stringWithContentsOfFile:shaderPath
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
            if (src) {
                g_library = [g_device newLibraryWithSource:src options:nil error:&error];
            }
        }
    }

    if (!g_library) {
        NSLog(@"TurboQuantMetal: failed to load shader library");
        return -1;
    }

    // Create command queue
    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        NSLog(@"TurboQuantMetal: failed to create command queue");
        return -1;
    }

    // Create PSOs for all kernels
    g_dequant_pso       = create_pso(@"tq_dequantize_lloydmax_kernel");
    g_matvec_pso        = create_pso(@"tq_matvec_transpose_kernel");
    g_scale_acc_pso     = create_pso(@"tq_scale_accumulate_kernel");
    g_dot_pso           = create_pso(@"tq_dot_product_kernel");
    g_qjl_decode_pso    = create_pso(@"tq_qjl_decode_kernel");
    g_dequant_half4_pso = create_pso(@"tq_dequantize_lloydmax_half4_kernel");
    g_wht_butterfly_pso = create_pso(@"tq_wht_butterfly_kernel");
    g_sign_diag_pso     = create_pso(@"tq_sign_diag_kernel");
    g_wht_normalize_pso = create_pso(@"tq_wht_normalize_kernel");
    g_scale_acc_half4_pso = create_pso(@"tq_scale_accumulate_half4_kernel");
    g_dot_half4_pso     = create_pso(@"tq_dot_product_half4_kernel");

    // Core kernels must succeed
    if (!g_dequant_pso || !g_dot_pso || !g_scale_acc_pso) return -1;

    return 0;
}

void tq_metal_cleanup(void) {
    g_dequant_pso = nil;
    g_matvec_pso = nil;
    g_scale_acc_pso = nil;
    g_dot_pso = nil;
    g_qjl_decode_pso = nil;
    g_dequant_half4_pso = nil;
    g_wht_butterfly_pso = nil;
    g_sign_diag_pso = nil;
    g_wht_normalize_pso = nil;
    g_scale_acc_half4_pso = nil;
    g_dot_half4_pso = nil;
    g_library = nil;
    g_queue = nil;
    g_rotation_buf = nil;
    g_qjl_buf = nil;
    g_codebook_buf = nil;
    g_codebook_half_buf = nil;
    g_d1_buf = nil;
    g_d2_buf = nil;
    g_device = nil;
    g_cached_dim = 0;
}

// ── WHT diagonal upload ─────────────────────────────────────────────────────

int tq_metal_upload_wht_diagonals(const int8_t *d1, const int8_t *d2, uint32_t dim) {
    if (!g_device || !d1 || !d2 || dim == 0) return -1;

    g_d1_buf = [g_device newBufferWithBytes:d1
                                     length:dim
                                    options:MTLResourceStorageModeShared];
    g_d2_buf = [g_device newBufferWithBytes:d2
                                     length:dim
                                    options:MTLResourceStorageModeShared];
    if (!g_d1_buf || !g_d2_buf) return -1;
    g_cached_dim = dim;
    return 0;
}

int tq_metal_supports_half(void) {
    // All Apple GPU devices since A8 support half precision
    return (g_device != nil) ? 1 : 0;
}

// ── GPU dispatch helpers ────────────────────────────────────────────────────

static inline BOOL gpu_ready(void) {
    return g_device != nil && g_queue != nil && g_dequant_pso != nil;
}

// Synchronous GPU dequantize + inverse rotation via WHT → output float buffer.
// Returns 0 on success, -1 on failure (caller should fall back to CPU).
static int gpu_dequant_wht_inverse(const uint8_t *codes, uint32_t dim,
                                     uint8_t mse_bits, float norm,
                                     float *output) {
    if (!gpu_ready() || !g_wht_butterfly_pso || !g_sign_diag_pso ||
        !g_d1_buf || !g_d2_buf) return -1;

    // Upload codebook
    const float *cb = tq_lloydmax_codebook(mse_bits);
    uint32_t levels = 1u << mse_bits;
    id<MTLBuffer> cb_buf = [g_device newBufferWithBytes:cb
                                                 length:levels * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> codes_buf = [g_device newBufferWithBytes:codes
                                                    length:dim
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [g_device newBufferWithLength:dim * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    if (!cb_buf || !codes_buf || !out_buf) return -1;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    // 1. Dequantize: codes → rotated floats
    [enc setComputePipelineState:g_dequant_pso];
    [enc setBuffer:codes_buf offset:0 atIndex:0];
    [enc setBuffer:cb_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBytes:&dim length:sizeof(dim) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];

    // 2. WHT inverse: D2 sign → butterfly stages → normalize → D1 sign
    [enc setComputePipelineState:g_sign_diag_pso];
    [enc setBuffer:out_buf offset:0 atIndex:0];
    [enc setBuffer:g_d2_buf offset:0 atIndex:1];
    [enc setBytes:&dim length:sizeof(dim) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];

    // Butterfly stages (log2(dim) passes)
    for (uint32_t half_len = 1; half_len < dim; half_len <<= 1) {
        uint32_t num_pairs = dim / 2;
        [enc setComputePipelineState:g_wht_butterfly_pso];
        [enc setBuffer:out_buf offset:0 atIndex:0];
        [enc setBytes:&dim length:sizeof(dim) atIndex:1];
        [enc setBytes:&half_len length:sizeof(half_len) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(num_pairs, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(num_pairs, 256), 1, 1)];
    }

    // Normalize by 1/sqrt(dim)
    [enc setComputePipelineState:g_wht_normalize_pso];
    [enc setBuffer:out_buf offset:0 atIndex:0];
    [enc setBytes:&dim length:sizeof(dim) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];

    // D1 sign
    [enc setComputePipelineState:g_sign_diag_pso];
    [enc setBuffer:out_buf offset:0 atIndex:0];
    [enc setBuffer:g_d1_buf offset:0 atIndex:1];
    [enc setBytes:&dim length:sizeof(dim) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(dim, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256), 1, 1)];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // Scale by norm and copy out
    float *gpu_out = (float *)[out_buf contents];
    for (uint32_t i = 0; i < dim; i++) {
        output[i] = gpu_out[i] * norm;
    }

    return 0;
}

// ── Public API ────────────────────────────────────────────────────────────────
// GPU dispatch with CPU fallback when Metal is not initialised or WHT buffers
// are not uploaded.

int tq_metal_compress_kv(void *cmd_buffer, const float *kv_input, uint32_t dim,
                          void *compressed_buffer, const PA_QuantizedKVDesc *desc) {
    // Compress is CPU-bound (one-time per token, not hot path)
    (void)cmd_buffer;
    uint32_t written = tq_compress_kv(kv_input, dim,
                                       (uint8_t *)compressed_buffer, desc);
    return (written > 0) ? 0 : -1;
}

int tq_metal_compressed_qk_score(void *cmd_buffer, const float *q_transformed,
                                   uint32_t dim, void *compressed_kv_buffer,
                                   uint32_t num_tokens, float *scores_out,
                                   const PA_QuantizedKVDesc *desc) {
    (void)cmd_buffer;
    uint32_t stride = tq_compressed_size(dim);
    const uint8_t *base = (const uint8_t *)compressed_kv_buffer;

    // Try GPU path for WHT-based decompress + dot
    if (gpu_ready() && g_wht_butterfly_pso && g_d1_buf &&
        desc->transform_kind == PA_TRANSFORM_HADAMARD) {
        uint16_t mse_bits_x2 = (desc->key_bits_x2 >= 4) ? desc->key_bits_x2 - 2 : 6;
        uint8_t mse_bits = (uint8_t)(mse_bits_x2 / 2);
        if (mse_bits < 1) mse_bits = 3;

        float *k_decompressed = (float *)malloc(dim * sizeof(float));
        if (k_decompressed) {
            for (uint32_t t = 0; t < num_tokens; t++) {
                const uint8_t *token_buf = base + (size_t)t * stride;
                float norm = *(const float *)token_buf;
                const uint8_t *codes = token_buf + 4;

                if (gpu_dequant_wht_inverse(codes, dim, mse_bits, norm,
                                             k_decompressed) == 0) {
                    // Add QJL correction on CPU (QJL matrix is large, GPU transfer overhead)
                    const float *p_gamma = (const float *)(token_buf + 4 + dim);
                    const uint8_t *qjl_bits = token_buf + 4 + dim + 4;
                    float gamma = *p_gamma;

                    float *qjl_contrib = (float *)malloc(dim * sizeof(float));
                    if (qjl_contrib) {
                        tq_qjl_decode(qjl_bits, gamma, qjl_contrib, dim);
                        float dot = 0.0f;
                        for (uint32_t d = 0; d < dim; d++) {
                            dot += q_transformed[d] * (k_decompressed[d] + qjl_contrib[d]);
                        }
                        scores_out[t] = dot;
                        free(qjl_contrib);
                    } else {
                        scores_out[t] = tq_compressed_dot(q_transformed, dim,
                                                           token_buf, desc);
                    }
                } else {
                    scores_out[t] = tq_compressed_dot(q_transformed, dim,
                                                       token_buf, desc);
                }
            }
            free(k_decompressed);
            return 0;
        }
    }

    // CPU fallback
    for (uint32_t t = 0; t < num_tokens; t++) {
        scores_out[t] = tq_compressed_dot(q_transformed, dim,
                                           base + (size_t)t * stride, desc);
    }
    return 0;
}

int tq_metal_v_accumulate(void *cmd_buffer, void *compressed_v_buffer,
                            const float *weights, uint32_t num_tokens, uint32_t dim,
                            float *output, const PA_QuantizedKVDesc *desc) {
    (void)cmd_buffer;
    uint32_t stride;
    int use_mse_only = (desc->v_uses_qjl == 0);

    if (use_mse_only) {
        stride = tq_compressed_size_v(dim);
    } else {
        stride = tq_compressed_size(dim);
    }

    const uint8_t *base = (const uint8_t *)compressed_v_buffer;
    float *v_tile = (float *)malloc(dim * sizeof(float));
    if (!v_tile) return -1;

    memset(output, 0, dim * sizeof(float));

    for (uint32_t t = 0; t < num_tokens; t++) {
        if (use_mse_only) {
            tq_decompress_v_mse(base + (size_t)t * stride, v_tile, dim, desc);
        } else {
            tq_decompress_v(base + (size_t)t * stride, v_tile, dim, desc);
        }
        float w = weights[t];
        for (uint32_t d = 0; d < dim; d++) {
            output[d] += w * v_tile[d];
        }
    }

    free(v_tile);
    return 0;
}

// ── Prerotated variants ─────────────────────────────────────────────────────

int tq_metal_compress_kv_prerotated(void *cmd_buffer, const float *kv_rotated,
                                      uint32_t dim, void *compressed_buffer,
                                      const PA_QuantizedKVDesc *desc) {
    (void)cmd_buffer;
    uint32_t written = tq_compress_kv_prerotated(kv_rotated, dim,
                                                   (uint8_t *)compressed_buffer, desc);
    return (written > 0) ? 0 : -1;
}

int tq_metal_compressed_qk_score_prerotated(void *cmd_buffer,
                                              const float *q_rotated, uint32_t dim,
                                              void *compressed_kv_buffer,
                                              uint32_t num_tokens, float *scores_out,
                                              const PA_QuantizedKVDesc *desc) {
    (void)cmd_buffer;
    uint32_t stride = tq_compressed_size(dim);
    const uint8_t *base = (const uint8_t *)compressed_kv_buffer;
    for (uint32_t t = 0; t < num_tokens; t++) {
        scores_out[t] = tq_compressed_dot_prerotated(q_rotated, dim,
                                                       base + (size_t)t * stride, desc);
    }
    return 0;
}

int tq_metal_v_accumulate_prerotated(void *cmd_buffer, void *compressed_v_buffer,
                                       const float *weights, uint32_t num_tokens,
                                       uint32_t dim, float *output,
                                       const PA_QuantizedKVDesc *desc) {
    (void)cmd_buffer;
    uint32_t stride = tq_compressed_size_v(dim);
    const uint8_t *base = (const uint8_t *)compressed_v_buffer;

    float *v_tile = (float *)malloc(dim * sizeof(float));
    if (!v_tile) return -1;

    memset(output, 0, dim * sizeof(float));

    for (uint32_t t = 0; t < num_tokens; t++) {
        tq_decompress_v_mse_prerotated(base + (size_t)t * stride, v_tile, dim, desc);
        float w = weights[t];
        for (uint32_t d = 0; d < dim; d++) {
            output[d] += w * v_tile[d];
        }
    }

    free(v_tile);
    return 0;
}
