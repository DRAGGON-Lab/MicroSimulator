#include "cm/growth.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "numerics_device.hpp"

namespace cm {
namespace {
void validate_growth_requirements(const CellGrowthModel& model, std::size_t count) {
  auto positive = [](double x) {
    return std::isfinite(x) && x > 0;
  };
  std::set<std::uint32_t> seen;

  for (const auto& r : model.requirements) {
    if (r.solute >= count || !seen.insert(r.solute).second || !positive(r.half_saturation) ||
        !positive(r.biomass_yield)) {
      throw std::invalid_argument("invalid growth requirement");
    }
  }
}

void evaluate_growth_device(detail::NumericsDevice& d, const std::vector<unsigned>& offsets,
                            const std::vector<float>& cells, const std::vector<float>& requirements,
                            const std::vector<float>& environment, const std::vector<float>& uptake,
                            std::vector<GrowthEvaluation>& result) {
  // Fail explicitly if the backend working precision cannot represent the model.
  for (float x : cells) {
    if (!std::isfinite(x)) {
      throw std::invalid_argument("growth value not representable");
    }
  }

  for (float x : requirements) {
    if (!(x > 0) || !std::isfinite(x)) {
      throw std::invalid_argument("growth parameter not representable");
    }
  }

  auto b0 = d.upload(offsets), b1 = d.upload(cells), b2 = d.upload(requirements),
       b3 = d.upload(environment), b4 = d.upload(uptake);
  auto b5 = d.allocate(environment.size() * sizeof(float)),
       b6 = d.allocate(result.size() * 5 * sizeof(float));
  d.dispatch(detail::NumericsKernel::growth, {static_cast<unsigned>(result.size()), 0},
             {b0, b1, b2, b3, b4, b5, b6});
  auto alpha = d.download<float>(b5, environment.size()),
       values = d.download<float>(b6, result.size() * 5);

  for (std::size_t i = 0; i < result.size(); ++i) {
    auto& out = result[i];
    out.uptake_velocities.assign(alpha.begin() + offsets[i], alpha.begin() + offsets[i + 1]);
    out.biomass_gain = values[5 * i];
    out.biochemical_volume_gain = values[5 * i + 1];
    out.geometric_volume_gain = values[5 * i + 2];
    out.specific_rate = values[5 * i + 3];
    out.stoichiometric_residual = values[5 * i + 4];
  }
}

GrowthEvaluation evaluate_growth_cpu(const CellGrowthModel& m, const GrowthInput& in, double dt,
                                     double limitation, double extent, double largest) {
  GrowthEvaluation out;
  const double mass = in.biochemical_volume * m.biomass_density;

  for (std::size_t j = 0; j < m.requirements.size(); ++j) {
    const auto& r = m.requirements[j];
    double c = in.concentrations[j];
    out.uptake_velocities.push_back(
        m.requirements.size() == 1
            ? m.mu_max * mass / (r.biomass_yield * in.surface_area * (r.half_saturation + c))
            : (c > 0 ? m.mu_max * mass * limitation / (r.biomass_yield * in.surface_area * c) : 0));
  }

  out.biomass_gain = extent;
  out.biochemical_volume_gain = extent / m.biomass_density;
  out.geometric_volume_gain = out.biochemical_volume_gain / m.volume_ratio;
  out.specific_rate = extent / (mass * dt);
  out.stoichiometric_residual = largest > 0 ? (largest - extent) / largest : 0;

  return out;
}

void validate_growth_results(const std::vector<GrowthEvaluation>& result) {
  for (const auto& out : result) {
    for (double x : out.uptake_velocities) {
      if (!std::isfinite(x) || x < 0) {
        throw std::runtime_error("invalid growth uptake coefficient");
      }
    }

    if (!std::isfinite(out.biochemical_volume_gain) || out.biochemical_volume_gain < 0 ||
        !std::isfinite(out.specific_rate)) {
      throw std::runtime_error("invalid realized growth");
    }
  }
}
}  // namespace

void CellGrowthModel::validate(std::size_t count) const {
  auto positive = [](double x) {
    return std::isfinite(x) && x > 0;
  };

  if (!cell_id || !std::isfinite(mu_max) || mu_max < 0 || !positive(biomass_density) ||
      !positive(volume_ratio) || requirements.empty() ||
      (kind != GrowthKind::monod && kind != GrowthKind::essential) ||
      (kind == GrowthKind::monod && requirements.size() != 1)) {
    throw std::invalid_argument("invalid cell growth model");
  }

  validate_growth_requirements(*this, count);
}

struct GrowthExecutor::Impl {
  std::unique_ptr<detail::NumericsDevice> device;

  Impl(BackendKind kind, std::uint32_t index) : device(detail::make_numerics_device(kind, index)) {}
};

GrowthExecutor::GrowthExecutor(BackendKind k, std::uint32_t i)
    : impl_(std::make_unique<Impl>(k, i)) {}

GrowthExecutor::~GrowthExecutor() = default;

namespace {
void validate_growth_input(const CellGrowthModel& m, const GrowthInput& in) {
  if (in.concentrations.size() != m.requirements.size() ||
      in.uptake.size() != m.requirements.size() || !(in.biochemical_volume > 0) ||
      !(in.surface_area > 0)) {
    throw std::invalid_argument("invalid growth input");
  }
}
}  // namespace

std::vector<GrowthEvaluation> GrowthExecutor::evaluate(const std::vector<CellGrowthModel>& models,
                                                       const std::vector<GrowthInput>& input,
                                                       double dt) {
  if (models.size() != input.size() || !std::isfinite(dt) || dt <= 0) {
    throw std::invalid_argument("invalid growth evaluation dimensions or interval");
  }

  std::vector<GrowthEvaluation> result(models.size());
  std::vector<unsigned> offsets{0};
  std::vector<float> cells, requirements, environment, uptake;

  for (std::size_t i = 0; i < models.size(); ++i) {
    const auto& m = models[i];
    const auto& in = input[i];
    m.validate(std::numeric_limits<std::uint32_t>::max());

    validate_growth_input(m, in);

    cells.insert(cells.end(), {float(in.biochemical_volume * m.biomass_density), float(m.mu_max),
                               float(in.surface_area), float(m.biomass_density),
                               float(m.volume_ratio), float(dt)});
    double limitation = 1, extent = INFINITY, largest = 0;

    for (std::size_t j = 0; j < m.requirements.size(); ++j) {
      const auto& r = m.requirements[j];
      double c = in.concentrations[j], u = in.uptake[j];

      if (!std::isfinite(c) || c < 0 || !std::isfinite(u) || u < 0) {
        throw std::invalid_argument("invalid growth concentration or uptake");
      }

      requirements.insert(requirements.end(), {float(r.half_saturation), float(r.biomass_yield)});
      environment.push_back(float(c));
      uptake.push_back(float(u));
      limitation = std::min(limitation, c / (r.half_saturation + c));
      extent = std::min(extent, u * r.biomass_yield);
      largest = std::max(largest, u * r.biomass_yield);
    }

    offsets.push_back(static_cast<unsigned>(environment.size()));

    if (!impl_->device) {
      result[i] = evaluate_growth_cpu(m, in, dt, limitation, extent, largest);
    }
  }

  if (impl_->device && !models.empty()) {
    evaluate_growth_device(*impl_->device, offsets, cells, requirements, environment, uptake,
                           result);
  }

  validate_growth_results(result);

  return result;
}
}  // namespace cm
