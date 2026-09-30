#pragma once

#include "cm/growth.hpp"
#include "cm/solute_transport.hpp"
#include "cm/world_state.hpp"

namespace cm {
struct CellSurfaceExchange {
  CellId body_id{0};
  std::uint32_t solute{0}, species{0};
  double uptake_velocity{0}, secretion_rate{0};
};

struct ReserveRequirement {
  std::uint32_t species{0};
  double amount_per_biomass{0};
};

struct CultureEvent {
  double time{0};  // model time, strictly increasing
  std::vector<FlowPort> ports;
  std::vector<ChemicalBoundary> reservoirs;
};

struct CultureConfiguration {
  FluidGridSpec grid;
  FluidProperties fluid;
  std::vector<FlowPort> ports;
  std::vector<Solute> solutes;
  std::vector<ChemicalBoundary> reservoirs;
  std::vector<CellSurfaceExchange> exchange;
  std::vector<ReserveRequirement> biomass_requirements;
  double biomass_per_geometric_volume{1};
  LinearSolveParameters solver;
  FluidBodyStepParameters stepping;
  FluidGeometryParameters geometry;
  std::uint32_t maximum_substeps{1024}, maximum_retries{16};
  std::vector<CellGrowthModel> growth;
  std::vector<CultureEvent> events;
  std::string authoring_json;  // closed, data-only authoring provenance
  double coupling_tolerance{2e-6};
  std::uint32_t maximum_coupling_iterations{64};
  void validate(std::size_t species_count) const;
};

struct CultureCellState {
  CapsuleBody body;
  double biochemical_volume{0};
  std::vector<double> species_amounts;
  std::vector<double> uptake_totals;
  double realized_specific_rate{0};
  double biomass_produced{0};
};

struct CultureReport {
  std::uint32_t substeps{0}, retries{0};
  FluidSolveReport flow;
  SoluteTransportReport transport;
};

struct CultureCheckpoint {
  CultureConfiguration configuration;
  std::vector<CultureCellState> cells;
  std::vector<double> extracellular_amounts;
  std::vector<ChemicalTransfer> reservoir_totals;
  CultureReport last_report;
  double time{0};
  std::uint32_t event_index{0};
  void validate(const WorldStateCheckpoint&) const;
};
}  // namespace cm
