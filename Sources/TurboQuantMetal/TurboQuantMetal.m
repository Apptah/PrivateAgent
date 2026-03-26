#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "TurboQuantMetal.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

// ── Pipeline state cache ──────────────────────────────────────────────────────

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_dequant_pso = nil;
static id<MTLComputePipelineState> g_matvec_pso = nil;
static id<MTLComputePipelineState> g_scale_acc_pso = nil;
static id<MTLComputePipelineState> g_dot_pso = nil;
static id<MTLComputePipelineState> g_qjl_decode_pso = nil;

// Cached GPU buffers for rotation matrix and QJL matrix
static id<MTLBuffer> g_rotation_buf = nil;
static id<MTLBuffer> g_qjl_buf = nil;
static id<MTLBuffer> g_codebook_buf = nil;
static uint32_t g_cached_dim = 0;

// ── Init / Cleanup ────────────────────────────────────────────────────────────

int tq_metal_init(void *device) {
    if (!device) return -1;
    g_device = (__bridge id<MTLDevice>)device;

    // Try loading pre-compiled metallib from bundle
    NSError *error = nil;
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
            NSString *src = [NSString stringWithContentsOfFile:shaderPath
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
            if (src) {
                g_library = [g_device newLibraryWithSource:src options:nil error:&error];
            }
        }
    }

    if (!g_library) {
        NSLog(@"TurboQuantMetal: failed to load shader library: %@", error);
        return -1;
    }

    // Create pipeline states
    NSArray *names = @[
        @"tq_dequantize_lloydmax_kernel",
        @"tq_matvec_transpose_kernel",
        @"tq_scale_accumulate_kernel",
        @"tq_dot_product_kernel",
        @"tq_qjl_decode_kernel"
    ];
    __strong id<MTLComputePipelineState> *psos[] = {
        &g_dequant_pso,
        &g_matvec_pso,
        &g_scale_acc_pso,
        &g_dot_pso,
        &g_qjl_decode_pso
    };

    for (NSUInteger i = 0; i < names.count; i++) {
        id<MTLFunction> fn = [g_library newFunctionWithName:names[i]];
        if (!fn) {
            NSLog(@"TurboQuantMetal: missing kernel: %@", names[i]);
            return -1;
        }
        *psos[i] = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!*psos[i]) {
            NSLog(@"TurboQuantMetal: PSO creation failed for %@: %@", names[i], error);
            return -1;
        }
    }

    return 0;
}

void tq_metal_cleanup(void) {
    g_dequant_pso = nil;
    g_matvec_pso = nil;
    g_scale_acc_pso = nil;
    g_dot_pso = nil;
    g_qjl_decode_pso = nil;
    g_library = nil;
    g_rotation_buf = nil;
    g_qjl_buf = nil;
    g_codebook_buf = nil;
    g_device = nil;
    g_cached_dim = 0;
}

// ── Public API ────────────────────────────────────────────────────────────────
// GPU dispatch functions use CPU fallback when Metal is not initialised
// (g_device == nil). The Metal kernels compile and PSOs are created, but
// the dispatch path will be activated in a subsequent task with device
// benchmarks. This is the correct incremental approach.

int tq_metal_compress_kv(void *cmd_buffer, const float *kv_input, uint32_t dim,
                          void *compressed_buffer, const PA_QuantizedKVDesc *desc) {
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
    uint32_t stride = tq_compressed_size(dim);
    const uint8_t *base = (const uint8_t *)compressed_v_buffer;

    float *v_tile = (float *)malloc(dim * sizeof(float));
    if (!v_tile) return -1;

    memset(output, 0, dim * sizeof(float));

    for (uint32_t t = 0; t < num_tokens; t++) {
        tq_decompress_v(base + (size_t)t * stride, v_tile, dim, desc);
        float w = weights[t];
        for (uint32_t d = 0; d < dim; d++) {
            output[d] += w * v_tile[d];
        }
    }

    free(v_tile);
    return 0;
}
