#include <metal_stdlib>
using namespace metal;

kernel void tq_placeholder(device float *output [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    output[tid] = 0.0f;
}
