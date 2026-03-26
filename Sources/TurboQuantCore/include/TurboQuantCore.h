#ifndef TURBOQUANT_CORE_H
#define TURBOQUANT_CORE_H

#include "FlashMoECore.h"

// TurboQuant KV cache compression — CPU reference implementations.
// Added in Plan 5.

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Scalar Quantization ──
//
// Quantizes `n` float32 values into 1-byte-per-code format using per-block
// min/max scaling. `bits_x2` follows the PA_QuantizedKVDesc encoding:
//   6 = 3-bit, 7 = 3.5-bit, 8 = 4-bit.
// Number of quantization levels = pow(2, bits_x2 / 2.0).
// Each block of `block_size` elements shares a single (scale, zero) pair
// written to `out_scales` and `out_zeros` (both arrays of length n/block_size).
// `out_codes` must be at least n bytes.

void tq_quantize_scalar(
    const float    *in,         // input float values
    uint8_t        *out_codes,  // 1 byte per element (unpacked)
    float          *out_scales, // scale per block
    float          *out_zeros,  // zero-point (block min) per block
    size_t          n,          // total number of elements
    size_t          block_size, // elements per quantization block
    uint16_t        bits_x2     // bit-width * 2: 6=3b, 7=3.5b, 8=4b
);

// Dequantizes codes back to float32.
// Mirrors tq_quantize_scalar: out[i] = code[i] * scale[block] + zero[block].

void tq_dequantize_scalar(
    const uint8_t  *codes,      // 1 byte per element
    const float    *scales,     // scale per block
    const float    *zeros,      // zero-point per block
    float          *out,        // reconstructed float values
    size_t          n,
    size_t          block_size,
    uint16_t        bits_x2
);

// ── QJL Residual ──
//
// Encodes the quantization residual (input - dequantized) using a random
// sign projection, producing 1 bit per element packed into uint8_t words.
// `seed` drives a splitmix64 PRNG that generates the random signs (+1/-1).
// `out_bits` must hold ceil(n / 8) bytes.

void tq_qjl_encode(
    const float    *input,      // original float values (pre-quantization)
    const float    *dequant,    // reconstructed values from scalar quantization
    uint8_t        *out_bits,   // packed 1-bit residual signs, ceil(n/8) bytes
    size_t          n,
    uint64_t        seed
);

// Applies QJL correction to attention scores.
// For each of the `num_tokens` tokens, computes the popcount agreement
// between the Q residual bits and K residual bits and adds a correction
// term scaled by `1/sqrt(dim)` to `scores`.
//
// `q_bits` and `k_bits` are each `ceil(dim/8)` bytes per token.
// `scores` is an array of `num_tokens` floats to be corrected in-place.

void tq_qjl_correct_scores(
    const uint8_t  *q_bits,     // Q residual bits, ceil(dim/8) bytes
    const uint8_t  *k_bits,     // K residual bits, num_tokens * ceil(dim/8) bytes
    float          *scores,     // attention scores to correct, length num_tokens
    size_t          num_tokens,
    size_t          dim
);

#ifdef __cplusplus
}
#endif

#endif // TURBOQUANT_CORE_H
