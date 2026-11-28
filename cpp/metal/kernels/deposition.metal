// Biomass deposition and the flow closures over it.
//
// A deposit runs in five passes over two index spaces:
//
//   1. deposit_weights (per cell): the cell's box and the normalized weights
//      of its tent over the box, restricted to one connected fluid region.
//   2. deposit_count (per cell): how many cells reach each site.
//   3. scan_blocks and scan_add of scan.metal (per site): the inclusive
//      prefix sum of the counts, which gives each site its range of
//      entries.
//   4. deposit_fill (per cell): each cell enters itself in the range of every
//      site it reaches, in whatever order the cells arrive.
//   5. deposit_sites (per site): each site sorts its cells into ascending
//      order and sums their biomass and channels in that order, so a run
//      gives the same result every time.

// Integral of the unit tent, 1 - |u| on [-1, 1], from -1 up to `edge`.
static float tent_cdf(float edge) {
  if (edge <= -1.0f) {
    return 0.0f;
  }
  if (edge < 0.0f) {
    return 0.5f * (edge + 1.0f) * (edge + 1.0f);
  }
  if (edge < 1.0f) {
    return 1.0f - 0.5f * (1.0f - edge) * (1.0f - edge);
  }
  return 1.0f;
}

// The integral over site `index` of a tent of `radius` about `center`, along
// an axis whose sites are centered at origin + index * spacing.
static float axis_weight(uint index, float origin, float spacing, float center, float radius) {
  float lower = (origin + (float(index) - 0.5f) * spacing - center) / radius;
  float upper = (origin + (float(index) + 0.5f) * spacing - center) / radius;
  return tent_cdf(upper) - tent_cdf(lower);
}

// The conserved biomass of a cell whose center holds its length in w.
static float cell_biomass(float4 center, float radius) {
  return pi * radius * radius * (center.w + 2.0f * radius);
}

// The box of sites that the tent of a cell centered at `center` reaches; a
// cell outside the sites' extent has an empty box.
static DepositBox cell_box(constant DepositGrid& grid, float4 center) {
  float3 origin = grid.origin.xyz;
  float3 spacing = grid.spacing.xyz;
  float3 count = float3(grid.shape.xyz);
  float3 first = origin - 0.5f * spacing;
  float3 end = origin + (count - 0.5f) * spacing;
  bool inside = all(center.xyz >= first) && all(center.xyz < end);
  DepositBox box;
  box.lower = uint4(0u);
  box.extent = uint4(0u);
  if (inside) {
    float radius = grid.origin.w;
    int3 low = max(int3(floor((center.xyz - radius - origin) / spacing + 0.5f)), int3(0));
    int3 high = int3(ceil((center.xyz + radius - origin) / spacing + 0.5f));
    high = clamp(high, low, max(low, int3(grid.shape.xyz)));
    uint3 extent = min(uint3(high - low), grid.box.xyz);
    box.lower = uint4(uint3(low), 1u);
    box.extent = uint4(extent, extent.x * extent.y * extent.z);
  }
  return box;
}

// The flat index of the box site at `local`, which counts x-major and
// z-minor through the box as the grid counts its sites.
static uint box_site(constant DepositGrid& grid, DepositBox box, uint local) {
  uint iz = local % box.extent.z;
  uint iy = (local / box.extent.z) % box.extent.y;
  uint ix = local / (box.extent.y * box.extent.z);
  return ((box.lower.x + ix) * grid.shape.y + box.lower.y + iy) * grid.shape.z + box.lower.z + iz;
}

// The box and weights of one cell, which sum to one. Where the grid has
// obstacles, only the face-connected fluid region holding the strongest
// weight carries weight: corner contact is not fluid, and biomass does not
// reach through a wall. Status is one when no fluid site carries weight.
kernel void deposit_weights(device const float4* centers [[buffer(0)]],
                            device const uchar* solid [[buffer(1)]],
                            device DepositBox* boxes [[buffer(2)]],
                            device float* weights [[buffer(3)]], device uint* status [[buffer(4)]],
                            constant DepositGrid& grid [[buffer(5)]],
                            uint cell [[thread_position_in_grid]]) {
  if (cell >= grid.cell_count) {
    return;
  }
  float4 center = centers[cell];
  DepositBox box = cell_box(grid, center);
  boxes[cell] = box;
  status[cell] = 0u;
  if (box.lower.w == 0u) {
    return;
  }
  device float* block = weights + cell * grid.box.w;
  uint ex = box.extent.x;
  uint ey = box.extent.y;
  uint ez = box.extent.z;
  float radius = grid.origin.w;
  float strongest = 0.0f;
  uint seed = 0u;
  for (uint ix = 0u; ix < ex; ++ix) {
    float wx = axis_weight(box.lower.x + ix, grid.origin.x, grid.spacing.x, center.x, radius);
    for (uint iy = 0u; iy < ey; ++iy) {
      float wxy =
          wx * axis_weight(box.lower.y + iy, grid.origin.y, grid.spacing.y, center.y, radius);
      for (uint iz = 0u; iz < ez; ++iz) {
        uint local = (ix * ey + iy) * ez + iz;
        float weight =
            wxy * axis_weight(box.lower.z + iz, grid.origin.z, grid.spacing.z, center.z, radius);
        if (grid.has_obstacles != 0u && solid[box_site(grid, box, local)] != 0u) {
          weight = 0.0f;
        }
        block[local] = weight;
        if (weight > strongest) {
          strongest = weight;
          seed = local;
        }
      }
    }
  }
  if (!(strongest > 0.0f)) {
    status[cell] = 1u;
    return;
  }
  if (grid.has_obstacles != 0u) {
    // A negative weight marks a site reached from the seed. Sweeps repeat
    // until one reaches no new site.
    uint plane = ey * ez;
    block[seed] = -block[seed];
    bool changed = true;
    while (changed) {
      changed = false;
      for (uint local = 0u; local < box.extent.w; ++local) {
        if (!(block[local] > 0.0f)) {
          continue;
        }
        uint ix = local / plane;
        uint iy = (local / ez) % ey;
        uint iz = local % ez;
        bool reached =
            (ix > 0u && block[local - plane] < 0.0f) ||
            (ix + 1u < ex && block[local + plane] < 0.0f) ||
            (iy > 0u && block[local - ez] < 0.0f) || (iy + 1u < ey && block[local + ez] < 0.0f) ||
            (iz > 0u && block[local - 1u] < 0.0f) || (iz + 1u < ez && block[local + 1u] < 0.0f);
        if (reached) {
          block[local] = -block[local];
          changed = true;
        }
      }
    }
    for (uint local = 0u; local < box.extent.w; ++local) {
      block[local] = block[local] < 0.0f ? -block[local] : 0.0f;
    }
  }
  float total = 0.0f;
  for (uint local = 0u; local < box.extent.w; ++local) {
    total += block[local];
  }
  for (uint local = 0u; local < box.extent.w; ++local) {
    block[local] /= total;
  }
}

// Counts each cell once at every site where it carries weight.
kernel void deposit_count(device const DepositBox* boxes [[buffer(0)]],
                          device const float* weights [[buffer(1)]],
                          device atomic_uint* counts [[buffer(2)]],
                          constant DepositGrid& grid [[buffer(3)]],
                          uint cell [[thread_position_in_grid]]) {
  if (cell >= grid.cell_count) {
    return;
  }
  DepositBox box = boxes[cell];
  device const float* block = weights + cell * grid.box.w;
  for (uint local = 0u; local < box.extent.w; ++local) {
    if (block[local] > 0.0f) {
      atomic_fetch_add_explicit(&counts[box_site(grid, box, local)], 1u, memory_order_relaxed);
    }
  }
}

// Enters each cell in the range of every site where it carries weight.
// `ends` is the inclusive prefix sum of the counts; site s owns entries
// [ends[s - 1], ends[s]). Cursors start at zero.
kernel void deposit_fill(device const DepositBox* boxes [[buffer(0)]],
                         device const float* weights [[buffer(1)]],
                         device const uint* ends [[buffer(2)]],
                         device atomic_uint* cursors [[buffer(3)]],
                         device uint* entries [[buffer(4)]],
                         constant DepositGrid& grid [[buffer(5)]],
                         uint cell [[thread_position_in_grid]]) {
  if (cell >= grid.cell_count) {
    return;
  }
  DepositBox box = boxes[cell];
  device const float* block = weights + cell * grid.box.w;
  for (uint local = 0u; local < box.extent.w; ++local) {
    if (!(block[local] > 0.0f)) {
      continue;
    }
    uint site = box_site(grid, box, local);
    uint first = site == 0u ? 0u : ends[site - 1u];
    uint offset = atomic_fetch_add_explicit(&cursors[site], 1u, memory_order_relaxed);
    entries[first + offset] = cell;
  }
}

// Restores the max-heap property below `root` in a list of `count` cells.
static void sift_down(device uint* list, uint root, uint count) {
  uint value = list[root];
  for (;;) {
    uint child = 2u * root + 1u;
    if (child >= count) {
      break;
    }
    if (child + 1u < count && list[child + 1u] > list[child]) {
      ++child;
    }
    if (list[child] <= value) {
      break;
    }
    list[root] = list[child];
    root = child;
  }
  list[root] = value;
}

// The weight that `cell` carries at the site at grid coordinates `site`,
// which its box contains.
static float site_weight(constant DepositGrid& grid, device const DepositBox* boxes,
                         device const float* weights, uint cell, uint3 site) {
  DepositBox box = boxes[cell];
  uint3 offset = site - box.lower.xyz;
  uint local = (offset.x * box.extent.y + offset.y) * box.extent.z + offset.z;
  return weights[cell * grid.box.w + local];
}

// Each site's biomass per unit volume, and for each channel the channel's
// per-cell value times biomass per unit volume, over the site's cells in
// ascending order. Heapsort keeps a site that many cells reach at
// O(n log n) without scratch storage.
kernel void deposit_sites(
    device const uint* ends [[buffer(0)]], device uint* entries [[buffer(1)]],
    device const DepositBox* boxes [[buffer(2)]], device const float* weights [[buffer(3)]],
    device const float4* centers [[buffer(4)]], device const float* radii [[buffer(5)]],
    device const float* channels [[buffer(6)]], device float* density [[buffer(7)]],
    device float* channel_density [[buffer(8)]], constant DepositGrid& grid [[buffer(9)]],
    uint site [[thread_position_in_grid]]) {
  if (site >= grid.shape.w) {
    return;
  }
  uint first = site == 0u ? 0u : ends[site - 1u];
  uint count = ends[site] - first;
  device uint* list = entries + first;
  if (count > 1u) {
    for (uint root = count / 2u; root > 0u; --root) {
      sift_down(list, root - 1u, count);
    }
    for (uint last = count - 1u; last > 0u; --last) {
      uint largest = list[0];
      list[0] = list[last];
      list[last] = largest;
      sift_down(list, 0u, last);
    }
  }
  uint3 coordinates = uint3(site / (grid.shape.y * grid.shape.z),
                            (site / grid.shape.z) % grid.shape.y, site % grid.shape.z);
  float mass = 0.0f;
  for (uint entry = 0u; entry < count; ++entry) {
    uint cell = list[entry];
    mass += cell_biomass(centers[cell], radii[cell]) *
            site_weight(grid, boxes, weights, cell, coordinates);
  }
  density[site] = mass / grid.spacing.w;
  for (uint channel = 0u; channel < grid.channel_count; ++channel) {
    float amount = 0.0f;
    for (uint entry = 0u; entry < count; ++entry) {
      uint cell = list[entry];
      float deposited = cell_biomass(centers[cell], radii[cell]) *
                        site_weight(grid, boxes, weights, cell, coordinates);
      amount += channels[channel * grid.cell_count + cell] * deposited;
    }
    channel_density[channel * grid.shape.w + site] = amount / grid.spacing.w;
  }
}

// The Kozeny-Carman drag of biomass packed to a volume fraction.
static float kozeny_carman(float fraction, float coefficient) {
  float open = 1.0f - fraction;
  return coefficient * fraction * fraction / (open * open * open);
}

// The shallow-flow coefficients of one x/y column. The column's biomass
// fraction is its density averaged over its fluid sites and capped, which
// sets the drag d. Each fluid site of base mobility b takes mobility
// b / (1 + b d), and the column takes the body force d V, where V is the
// biomass-weighted velocity that channels 0 and 1 carry.
kernel void shallow_coupling(
    device const uchar* solid [[buffer(0)]], device const float* base [[buffer(1)]],
    device const float* density [[buffer(2)]], device const float* channels [[buffer(3)]],
    device float* mobility [[buffer(4)]], device float* force [[buffer(5)]],
    constant ShallowCouplingData& data [[buffer(6)]], uint column [[thread_position_in_grid]]) {
  if (column >= data.shape.x * data.shape.y) {
    return;
  }
  uint depth = data.shape.z;
  uint first = column * depth;
  uint fluid = 0u;
  float mass = 0.0f;
  for (uint z = 0u; z < depth; ++z) {
    fluid += solid[first + z] == 0u ? 1u : 0u;
    mass += density[first + z];
  }
  float fraction = min(mass / float(max(fluid, 1u)), data.max_volume_fraction);
  float drag = kozeny_carman(fraction, data.drag_coefficient);
  for (uint z = 0u; z < depth; ++z) {
    float b = data.uniform_base != 0u ? base[0] : base[first + z];
    mobility[first + z] = solid[first + z] != 0u ? 0.0f : b / (1.0f + b * drag);
  }
  float2 pull = float2(0.0f);
  if (data.has_velocity != 0u && mass > 0.0f) {
    float2 momentum = float2(0.0f);
    for (uint z = 0u; z < depth; ++z) {
      momentum += float2(channels[first + z], channels[data.shape.w + first + z]);
    }
    pull = momentum * (drag / mass);
  }
  force[2u * column] = pull.x;
  force[2u * column + 1u] = pull.y;
}

// The Brinkman drag at one site: the Kozeny-Carman drag of its biomass
// fraction, capped, and zero on a solid site.
kernel void brinkman_drag(device const uchar* solid [[buffer(0)]],
                          device const float* density [[buffer(1)]],
                          device float* drag [[buffer(2)]],
                          constant BrinkmanDragData& data [[buffer(3)]],
                          uint site [[thread_position_in_grid]]) {
  if (site >= data.shape.w) {
    return;
  }
  float fraction = min(density[site], data.max_volume_fraction);
  drag[site] = solid[site] != 0u ? 0.0f : kozeny_carman(fraction, data.drag_coefficient);
}
