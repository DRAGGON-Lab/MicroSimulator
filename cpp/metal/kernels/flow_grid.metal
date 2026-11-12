// Grid layout and indexing helpers shared by the flow kernel sources. A Metal
// library is compiled from source at runtime, with no include path, so this
// fragment is concatenated ahead of each flow kernel source at build time
// rather than included by it.

struct FaceCoordinate {
  uint component;
  uint3 coordinate;
  uint3 dimensions;
};

inline uint site_index(uint3 coordinate, constant FlowGridParameters& grid) {
  return (coordinate.x * grid.dimensions.y + coordinate.y) * grid.dimensions.z + coordinate.z;
}

inline uint3 site_coordinate(uint index, constant FlowGridParameters& grid) {
  const uint z = index % grid.dimensions.z;
  index /= grid.dimensions.z;
  const uint y = index % grid.dimensions.y;
  return uint3(index / grid.dimensions.y, y, z);
}

inline FaceCoordinate face_coordinate(uint index, constant FlowGridParameters& grid) {
  uint component = index < grid.face_offsets.y ? 0 : (index < grid.face_offsets.z ? 1 : 2);
  const uint offset = component == 0 ? grid.face_offsets.x
                                     : (component == 1 ? grid.face_offsets.y : grid.face_offsets.z);
  uint local = index - offset;
  uint3 dimensions = grid.dimensions.xyz;
  dimensions[component] += 1;
  const uint z = local % dimensions.z;
  local /= dimensions.z;
  const uint y = local % dimensions.y;
  return {component, uint3(local / dimensions.y, y, z), dimensions};
}

inline uint face_index(uint component, uint3 coordinate, constant FlowGridParameters& grid) {
  const uint offset = component == 0 ? grid.face_offsets.x
                                     : (component == 1 ? grid.face_offsets.y : grid.face_offsets.z);
  if (component == 0) {
    return offset + (coordinate.x * grid.dimensions.y + coordinate.y) * grid.dimensions.z +
           coordinate.z;
  }
  if (component == 1) {
    return offset + (coordinate.x * (grid.dimensions.y + 1) + coordinate.y) * grid.dimensions.z +
           coordinate.z;
  }
  return offset + (coordinate.x * grid.dimensions.y + coordinate.y) * (grid.dimensions.z + 1) +
         coordinate.z;
}

inline float harmonic_mean(float first, float second) {
  const float sum = first + second;
  return sum > 0.0f ? 2.0f * first * second / sum : 0.0f;
}
