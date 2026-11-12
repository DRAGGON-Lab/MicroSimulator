#include "microsimulator/names.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace microsimulator {
namespace {

template <typename Enum, std::size_t Count>
using NameTable = std::array<std::pair<Enum, std::string_view>, Count>;

constexpr NameTable<BackendKind, 3> backend_kinds{{
    {BackendKind::cpu, "cpu"},
    {BackendKind::metal, "metal"},
    {BackendKind::cuda, "cuda"},
}};

constexpr NameTable<GridBoundaryKind, 3> grid_boundary_kinds{{
    {GridBoundaryKind::no_flux, "no_flux"},
    {GridBoundaryKind::periodic, "periodic"},
    {GridBoundaryKind::fixed, "fixed"},
}};

constexpr NameTable<ConstraintRegion, 2> constraint_regions{{
    {ConstraintRegion::outside, "outside"},
    {ConstraintRegion::inside, "inside"},
}};

constexpr NameTable<SignalIntegrationKind, 3> signal_integration_kinds{{
    {SignalIntegrationKind::forward_euler, "forward_euler"},
    {SignalIntegrationKind::crank_nicolson, "crank_nicolson"},
    {SignalIntegrationKind::backward_euler, "backward_euler"},
}};

constexpr auto rate_ops = std::to_array<std::pair<RateOp, std::string_view>>({
#define MICROSIMULATOR_RATE_OP(name, code) {RateOp::name, #name},
#include "microsimulator/rate_ops.def"
#undef MICROSIMULATOR_RATE_OP
});

template <typename Table, typename Enum>
[[nodiscard]] std::string_view name_of(const Table& table, Enum value) {
  for (const auto& [candidate, name] : table) {
    if (candidate == value) {
      return name;
    }
  }
  throw std::invalid_argument("enumerator has no document name");
}

template <typename Table>
[[nodiscard]] auto parse(const Table& table, std::string_view name) noexcept
    -> std::optional<typename Table::value_type::first_type> {
  for (const auto& [candidate, candidate_name] : table) {
    if (candidate_name == name) {
      return candidate;
    }
  }
  return std::nullopt;
}

}  // namespace

std::string_view backend_kind_name(BackendKind kind) { return name_of(backend_kinds, kind); }
std::optional<BackendKind> parse_backend_kind(std::string_view name) noexcept {
  return parse(backend_kinds, name);
}

std::string_view grid_boundary_kind_name(GridBoundaryKind kind) {
  return name_of(grid_boundary_kinds, kind);
}
std::optional<GridBoundaryKind> parse_grid_boundary_kind(std::string_view name) noexcept {
  return parse(grid_boundary_kinds, name);
}

std::string_view constraint_region_name(ConstraintRegion region) {
  return name_of(constraint_regions, region);
}
std::optional<ConstraintRegion> parse_constraint_region(std::string_view name) noexcept {
  return parse(constraint_regions, name);
}

std::string_view signal_integration_kind_name(SignalIntegrationKind kind) {
  return name_of(signal_integration_kinds, kind);
}
std::optional<SignalIntegrationKind> parse_signal_integration_kind(std::string_view name) noexcept {
  return parse(signal_integration_kinds, name);
}

std::string_view rate_op_name(RateOp operation) { return name_of(rate_ops, operation); }
std::optional<RateOp> parse_rate_op(std::string_view name) noexcept {
  return parse(rate_ops, name);
}

}  // namespace microsimulator
