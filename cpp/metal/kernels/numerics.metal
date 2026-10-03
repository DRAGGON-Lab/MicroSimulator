#include <metal_stdlib>
using namespace metal;

struct NumericsParameters {
  uint count;
  float scalar;
};

kernel void numerics_ilu(device const uint* offsets [[buffer(0)]],
                         device const uint* columns [[buffer(1)]],
                         device const float* values [[buffer(2)]],
                         device const uint* diagonal [[buffer(3)]],
                         device const float* rhs [[buffer(4)]], device float* x [[buffer(5)]],
                         constant NumericsParameters& p [[buffer(15)]],
                         uint index [[thread_position_in_grid]]) {
  if (index) {
    return;
  }

  for (uint i = 0; i < p.count; ++i) {
    float value = rhs[i];

    for (uint j = offsets[i]; j < diagonal[i]; ++j) {
      value -= values[j] * x[columns[j]];
    }

    x[i] = value;
  }

  for (uint i = p.count; i-- > 0;) {
    float value = x[i];

    for (uint j = diagonal[i] + 1; j < offsets[i + 1]; ++j) {
      value -= values[j] * x[columns[j]];
    }

    x[i] = value / values[diagonal[i]];
  }
}

kernel void numerics_fill(device float* y [[buffer(0)]],
                          constant NumericsParameters& p [[buffer(15)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    y[i] = p.scalar;
  }
}

kernel void numerics_copy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant NumericsParameters& p [[buffer(15)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    y[i] = x[i];
  }
}

kernel void numerics_axpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant NumericsParameters& p [[buffer(15)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    y[i] += p.scalar * x[i];
  }
}

kernel void numerics_jacobi(device const float* rhs [[buffer(0)]],
                            device const float* ax [[buffer(1)]],
                            device const float* inverse_diagonal [[buffer(2)]],
                            device float* x [[buffer(3)]],
                            constant NumericsParameters& p [[buffer(15)]],
                            uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    x[i] += p.scalar * inverse_diagonal[i] * (rhs[i] - ax[i]);
  }
}

kernel void numerics_dot(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                         device float* out [[buffer(2)]],
                         constant NumericsParameters& p [[buffer(15)]],
                         uint i [[thread_position_in_grid]],
                         uint local [[thread_index_in_threadgroup]],
                         uint group [[threadgroup_position_in_grid]]) {
  threadgroup float values[128];
  values[local] = i < p.count ? x[i] * y[i] : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = 64; s; s /= 2) {
    if (local < s) {
      values[local] += values[local + s];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (!local) {
    out[group] = values[0];
  }
}

kernel void numerics_apply(device const uint* offsets [[buffer(0)]],
                           device const uint* columns [[buffer(1)]],
                           device const float* values [[buffer(2)]],
                           device const float* x [[buffer(3)]], device float* y [[buffer(4)]],
                           constant NumericsParameters& p [[buffer(15)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  float total = 0, correction = 0;

  for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
    const float term = values[j] * x[columns[j]] - correction;
    const float next = total + term;
    correction = (next - total) - term;
    total = next;
  }

  y[i] = total;
}

kernel void nutrient_growth(device const uint* offsets [[buffer(0)]],
                            device const float* cells [[buffer(1)]],
                            device const float* requirements [[buffer(2)]],
                            device const float* concentration [[buffer(3)]],
                            device const float* uptake [[buffer(4)]],
                            device float* alpha [[buffer(5)]], device float* output [[buffer(6)]],
                            constant NumericsParameters& p [[buffer(15)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  const uint begin = offsets[i], end = offsets[i + 1];
  const float mass = cells[6 * i], mu = cells[6 * i + 1], area = cells[6 * i + 2];
  float limitation = 1.0f, extent = INFINITY, largest = 0.0f;

  for (uint j = begin; j < end; ++j) {
    const float c = concentration[j], k = requirements[2 * j], y = requirements[2 * j + 1];
    limitation = fmin(limitation, c / (k + c));
    extent = fmin(extent, uptake[j] * y);
    largest = fmax(largest, uptake[j] * y);
  }

  for (uint j = begin; j < end; ++j) {
    const float c = concentration[j], k = requirements[2 * j], y = requirements[2 * j + 1];
    alpha[j] = end - begin == 1 ? mu * mass / (y * area * (k + c))
                                : (c > 0 ? mu * mass * limitation / (y * area * c) : 0.0f);
  }

  output[5 * i] = extent;
  output[5 * i + 1] = extent / cells[6 * i + 3];
  output[5 * i + 2] = output[5 * i + 1] / cells[6 * i + 4];
  output[5 * i + 3] = extent / (mass * cells[6 * i + 5]);
  output[5 * i + 4] = largest > 0 ? (largest - extent) / largest : 0.0f;
}

kernel void conservative_apply(device const uint* offsets [[buffer(0)]],
                               device const uint* columns [[buffer(1)]],
                               device const float* values [[buffer(2)]],
                               device const float* x [[buffer(3)]], device float* y [[buffer(4)]],
                               device const float* row_sums [[buffer(5)]],
                               constant NumericsParameters& p [[buffer(15)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  float total = 0, correction = 0;

  for (unsigned j = offsets[i]; j < offsets[i + 1]; ++j) {
    const float term = values[j] * (x[columns[j]] - x[i]) - correction;
    const float next = total + term;
    correction = (next - total) - term;
    total = next;
  }

  y[i] = total + row_sums[i] * x[i];
}
