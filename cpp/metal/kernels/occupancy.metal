#include <metal_stdlib>
using namespace metal;

struct OccupancyParameters {
  uint count, auxiliary, subdivisions, absolute;
  float scalar, cutoff, hx, hy, hz;
};

// Compensated, ordered sums avoid nondeterministic floating-point atomics.
void occupancy_add(float value, thread float& total, thread float& correction) {
  float adjusted = value - correction;
  float next = total + adjusted;
  correction = (next - total) - adjusted;
  total = next;
}

kernel void occupancy_geometry(device const float* centers [[buffer(0)]],
                               device const float* cells [[buffer(1)]],
                               device const uint* walls [[buffer(2)]],
                               device float* out [[buffer(3)]],
                               constant OccupancyParameters& p [[buffer(15)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  if (walls[i]) {
    out[i] = 0;

    return;
  }

  uint occupied = 0;
  float3 center(centers[3 * i], centers[3 * i + 1], centers[3 * i + 2]);

  for (uint x = 0; x < p.subdivisions; ++x) {
    for (uint y = 0; y < p.subdivisions; ++y) {
      for (uint z = 0; z < p.subdivisions; ++z) {
        float3 sample =
            center + (float3(x, y, z) + 0.5f) / float(p.subdivisions) * float3(p.hx, p.hy, p.hz) -
            0.5f * float3(p.hx, p.hy, p.hz);

        for (uint cell = 0; cell < p.auxiliary; ++cell) {
          uint base = 8 * cell;
          float3 delta = sample - float3(cells[base], cells[base + 1], cells[base + 2]);
          float3 direction(cells[base + 4], cells[base + 5], cells[base + 6]);
          // Scaling first avoids overflow/underflow in direction normalization.
          direction /= max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
          direction /= sqrt(dot(direction, direction));
          float axial =
              clamp(dot(delta, direction), -0.5f * cells[base + 3], 0.5f * cells[base + 3]);
          float3 distance = delta - axial * direction;
          float radius = cells[base + 7];
          distance /= radius;

          if (dot(distance, distance) <= 1) {
            ++occupied;
            break;
          }
        }
      }
    }
  }

  uint samples = p.subdivisions * p.subdivisions * p.subdivisions;
  float epsilon = float(samples - occupied) / float(samples);
  out[i] = epsilon < p.cutoff ? 0 : epsilon;
}

kernel void occupancy_volumes(device const float* epsilon [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant OccupancyParameters& p [[buffer(15)]],
                              uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    out[i] = epsilon[i] < p.cutoff ? 0 : epsilon[i] * p.scalar;
  }
}

kernel void occupancy_concentration(device const float* amount [[buffer(0)]],
                                    device const float* volume [[buffer(1)]],
                                    device float* out [[buffer(2)]],
                                    constant OccupancyParameters& p [[buffer(15)]],
                                    uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    out[i] = volume[i] > 0 ? amount[i] / volume[i] : 0;
  }
}

kernel void occupancy_face(device const float* data [[buffer(0)]], device float* out [[buffer(1)]],
                           constant OccupancyParameters& p [[buffer(15)]],
                           uint i [[thread_position_in_grid]]) {
  if (i != 0) {
    return;
  }

  float aperture =
      min(data[0], data[1]) < p.cutoff ? 0 : 2 * data[0] * data[1] / (data[0] + data[1]);
  out[0] = data[2] * aperture * data[3] / data[4];
  out[1] = aperture * data[3] * data[5];
}

kernel void occupancy_product(device const float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]], device float* out [[buffer(2)]],
                              constant OccupancyParameters& p [[buffer(15)]],
                              uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    out[i] = a[i] * b[i];
  }
}

kernel void occupancy_sum(device const float* input [[buffer(0)]], device float* out [[buffer(1)]],
                          constant OccupancyParameters& p [[buffer(15)]],
                          uint i [[thread_position_in_grid]]) {
  if (i != 0) {
    return;
  }

  float total = 0, correction = 0;

  for (uint j = 0; j < p.count; ++j) {
    occupancy_add(p.absolute ? abs(input[j]) : input[j], total, correction);
  }

  out[0] = total;
}

kernel void occupancy_normalize(device const float* values [[buffer(0)]],
                                device const float* total [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant OccupancyParameters& p [[buffer(15)]],
                                uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    out[i] = values[i] / total[0];
  }
}

kernel void occupancy_labels_init(device const float* old [[buffer(0)]],
                                  device const float* next [[buffer(1)]],
                                  device uint* out [[buffer(2)]],
                                  constant OccupancyParameters& p [[buffer(15)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i < p.count) {
    out[i] = old[i] > 0 || next[i] > 0 ? i : 0xffffffffu;
  }
}

kernel void occupancy_labels_step(device const uint* offsets [[buffer(0)]],
                                  device const uint* indices [[buffer(1)]],
                                  device const uint* labels [[buffer(2)]],
                                  device uint* out [[buffer(3)]],
                                  constant OccupancyParameters& p [[buffer(15)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  uint label = labels[i];

  if (label != 0xffffffffu) {
    for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
      label = min(label, labels[indices[j]]);
    }
  }

  out[i] = label;
}

kernel void occupancy_component_sums(device const uint* offsets [[buffer(0)]],
                                     device const uint* indices [[buffer(1)]],
                                     device const float* amount [[buffer(2)]],
                                     device const float* volume [[buffer(3)]],
                                     device float* out [[buffer(4)]],
                                     constant OccupancyParameters& p [[buffer(15)]],
                                     uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  float expelled = 0, capacity = 0, ec = 0, vc = 0;

  for (uint entry = offsets[i]; entry < offsets[i + 1]; ++entry) {
    uint j = indices[entry];

    if (volume[j] == 0) {
      occupancy_add(amount[j], expelled, ec);
    } else {
      occupancy_add(volume[j], capacity, vc);
    }
  }

  out[2 * i] = expelled;
  out[2 * i + 1] = capacity;
}

kernel void occupancy_remap(device const float* amount [[buffer(0)]],
                            device const float* volume [[buffer(1)]],
                            device const uint* labels [[buffer(2)]],
                            device const float* totals [[buffer(3)]],
                            device float* out [[buffer(4)]],
                            constant OccupancyParameters& p [[buffer(15)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  if (volume[i] == 0) {
    out[i] = 0;

    return;
  }

  uint component = labels[i];
  out[i] = amount[i] + totals[2 * component] * (volume[i] / totals[2 * component + 1]);
}

kernel void occupancy_assemble(
    device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]],
    device const float* edges [[buffer(2)]], device const float* volume [[buffer(3)]],
    device const float* amount [[buffer(4)]], device const float* source [[buffer(5)]],
    device const float* loss [[buffer(6)]], device float* diagonal [[buffer(7)]],
    device float* rhs [[buffer(8)]], constant OccupancyParameters& p [[buffer(15)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  float d = volume[i] * (1 + p.scalar * loss[i]);
  float b = amount[i] + p.scalar * source[i];

  for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
    float g = edges[3 * j], q = edges[3 * j + 1];
    d += p.scalar * (g + max(q, 0.0f));

    if (indices[j] == 0xffffffffu) {
      b += p.scalar * (g + max(-q, 0.0f)) * edges[3 * j + 2];
    }
  }

  diagonal[i] = volume[i] > 0 ? d : 1;
  rhs[i] = b;
}

kernel void occupancy_jacobi(device const float* diagonal [[buffer(0)]],
                             device const float* current [[buffer(1)]],
                             device const float* residual [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant OccupancyParameters& p [[buffer(15)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  out[i] = current[i] - residual[i] / diagonal[i];
}

kernel void occupancy_residual(
    device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]],
    device const float* edges [[buffer(2)]], device const float* volume [[buffer(3)]],
    device const float* amount [[buffer(4)]], device const float* source [[buffer(5)]],
    device const float* loss [[buffer(6)]], device const float* current [[buffer(7)]],
    device float* out [[buffer(8)]], constant OccupancyParameters& p [[buffer(15)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  float c = current[i];
  float value = volume[i] * c - amount[i], correction = 0;
  occupancy_add(p.scalar * (loss[i] * volume[i] * c - source[i]), value, correction);

  for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
    float neighbor = indices[j] == 0xffffffffu ? edges[3 * j + 2] : current[indices[j]];
    float g = edges[3 * j], q = edges[3 * j + 1];
    occupancy_add(p.scalar * (g * (c - neighbor) + q * (q >= 0 ? c : neighbor)), value, correction);
  }

  out[i] = value;
}

kernel void occupancy_finish(
    device const uint* offsets [[buffer(0)]], device const uint* indices [[buffer(1)]],
    device const float* edges [[buffer(2)]], device const float* volume [[buffer(3)]],
    device const float* loss [[buffer(4)]], device const float* current [[buffer(5)]],
    device float* amount [[buffer(6)]], device float* reaction [[buffer(7)]],
    device float* boundary [[buffer(8)]], constant OccupancyParameters& p [[buffer(15)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.count) {
    return;
  }

  amount[i] = volume[i] * current[i];
  reaction[i] = -p.scalar * loss[i] * amount[i];
  float exchange = 0, correction = 0;

  for (uint j = offsets[i]; j < offsets[i + 1]; ++j) {
    if (indices[j] == 0xffffffffu) {
      float g = edges[3 * j], q = edges[3 * j + 1], reservoir = edges[3 * j + 2];
      occupancy_add(
          p.scalar * (g * (reservoir - current[i]) - q * (q >= 0 ? current[i] : reservoir)),
          exchange, correction);
    }
  }

  boundary[i] = exchange;
}
