#pragma once

#include <optional>
#include <string_view>

#include "microsimulator/constraints.hpp"
#include "microsimulator/signals.hpp"
#include "microsimulator/species.hpp"
#include "microsimulator/types.hpp"

// The names of enumerators as they appear in checkpoint and scene documents.
// A name is part of those formats and never changes. Each parse function
// returns an empty optional for an unknown name.
namespace microsimulator {

[[nodiscard]] std::string_view backend_kind_name(BackendKind kind);
[[nodiscard]] std::optional<BackendKind> parse_backend_kind(std::string_view name) noexcept;

[[nodiscard]] std::string_view grid_boundary_kind_name(GridBoundaryKind kind);
[[nodiscard]] std::optional<GridBoundaryKind> parse_grid_boundary_kind(
    std::string_view name) noexcept;

[[nodiscard]] std::string_view constraint_region_name(ConstraintRegion region);
[[nodiscard]] std::optional<ConstraintRegion> parse_constraint_region(
    std::string_view name) noexcept;

[[nodiscard]] std::string_view signal_integration_kind_name(SignalIntegrationKind kind);
[[nodiscard]] std::optional<SignalIntegrationKind> parse_signal_integration_kind(
    std::string_view name) noexcept;

[[nodiscard]] std::string_view rate_op_name(RateOp operation);
[[nodiscard]] std::optional<RateOp> parse_rate_op(std::string_view name) noexcept;

}  // namespace microsimulator
