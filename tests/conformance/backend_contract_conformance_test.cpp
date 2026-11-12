#include <array>
#include <cassert>
#include <cstdint>

#include "backend_devices.hpp"
#include "microsimulator/simulation.hpp"

namespace {

constexpr std::array required_features{
    microsimulator::BackendFeature::growth,
    microsimulator::BackendFeature::species,
    microsimulator::BackendFeature::cell_contacts,
    microsimulator::BackendFeature::cell_mechanics,
    microsimulator::BackendFeature::external_constraints,
    microsimulator::BackendFeature::signals,
    microsimulator::BackendFeature::coupled_rates,
    microsimulator::BackendFeature::depth_averaged_flow,
    microsimulator::BackendFeature::resolved_flow,
};

void require_complete_backend(microsimulator::BackendKind backend, std::uint32_t device_index) {
  microsimulator::Simulation simulation(backend, 0, 0, device_index);
  for (const auto feature : required_features) {
    assert(simulation.supports(feature));
  }
}

}  // namespace

int main() {
  microsimulator::test::for_each_backend_device(require_complete_backend);
  return 0;
}
