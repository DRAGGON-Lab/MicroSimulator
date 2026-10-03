#pragma once
#include <map>

#include "cm/transport_geometry.hpp"

namespace cm {
struct TransportEdge {
  std::uint32_t first, second, port;
  double area{0}, q{0}, weight{0};
};

struct TransportMembrane {
  std::uint32_t group;
  std::uint64_t body;
  double area;
};

struct TransportGeometry::Impl {
  FluidGridSpec grid;
  std::vector<FlowPort> ports;
  double dt;
  std::size_t old_count, new_count, group_count;
  std::vector<std::uint32_t> group;
  std::vector<double> v0, v1, final_volumes;
  std::vector<TransportEdge> edges;
  std::vector<TransportMembrane> membranes;
  std::map<std::uint64_t, double> body_area;
  GeometricFluxReport report;
};
}  // namespace cm
