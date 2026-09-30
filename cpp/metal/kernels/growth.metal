#include <metal_stdlib>

using namespace metal;

kernel void advance_growth(device float* lengths [[buffer(0)]],
                           device const float* growth_rates [[buffer(1)]],
                           constant float& dt [[buffer(2)]], constant uint& count [[buffer(3)]],
                           uint index [[thread_position_in_grid]]) {
  if (index >= count) {
    return;
  }

  lengths[index] += growth_rates[index] * lengths[index] * dt;
}
