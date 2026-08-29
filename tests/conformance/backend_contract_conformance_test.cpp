#include <array>
#include <cassert>
#include <cstdint>

#include "backend_devices.hpp"
#include "cm/simulation.hpp"

namespace {

constexpr std::array required_features{
    cm::BackendFeature::growth,
    cm::BackendFeature::species,
    cm::BackendFeature::cell_contacts,
    cm::BackendFeature::cell_mechanics,
    cm::BackendFeature::external_constraints,
    cm::BackendFeature::signals,
    cm::BackendFeature::coupled_rates,
    cm::BackendFeature::depth_averaged_flow,
    cm::BackendFeature::resolved_flow,
};

void require_complete_backend(cm::BackendKind backend, std::uint32_t device_index) {
  cm::Simulation simulation(backend, 0, 0, device_index);
  for (const auto feature : required_features) {
    assert(simulation.supports(feature));
  }
}

}  // namespace

int main() {
  cm::test::for_each_backend_device(require_complete_backend);
  return 0;
}
