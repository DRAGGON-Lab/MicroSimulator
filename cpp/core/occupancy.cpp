#include "cm/occupancy.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include "occupancy_device.hpp"

namespace cm {
namespace {
using detail::OccupancyKernel;
using detail::OccupancyParameters;
constexpr auto closed = std::numeric_limits<std::uint32_t>::max();

std::uint32_t count32(std::size_t count) {
  if (count >= closed) {
    throw std::overflow_error("occupancy arrays exceed uint32 indexing");
  }

  return static_cast<std::uint32_t>(count);
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

void finite(float value, const char* name) {
  require(std::isfinite(value), name);
}

void vector_valid(const std::vector<float>& values, bool nonnegative = true) {
  count32(values.size());

  for (float value : values) {
    require(std::isfinite(value) && (!nonnegative || value >= 0),
            "occupancy vectors must be finite and nonnegative");
  }
}

void storage_valid(const std::vector<float>& amount, const std::vector<float>& volume) {
  vector_valid(amount);
  vector_valid(volume);
  require(amount.size() == volume.size(), "amount and volume size mismatch");

  for (std::size_t i = 0; i < amount.size(); ++i) {
    require(volume[i] > 0 || amount[i] == 0, "nonnegative amounts require accessible storage");
  }
}

double sum(const std::vector<float>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0);
}

void check_balance(const OccupancyBalance& balance) {
  const double scale = std::max(
      {1.0, std::abs(balance.before), std::abs(balance.after),
       std::abs(balance.source) + std::abs(balance.reaction) + std::abs(balance.boundary)});

  if (!std::isfinite(balance.residual()) || std::abs(balance.residual()) > 5e-6 * scale) {
    throw std::runtime_error("occupancy amount ledger exceeds conservation tolerance");
  }
}

std::vector<float> checked_result(detail::OccupancyDevice& device,
                                  const detail::OccupancyBufferPtr& buffer, std::size_t count) {
  auto result = device.download<float>(buffer, count);

  for (float value : result) {
    if (!std::isfinite(value) || value < 0) {
      throw std::invalid_argument(
          "occupancy operation produced invalid values; no clipping is permitted");
    }
  }

  return result;
}

struct Graph {
  std::vector<std::uint32_t> offsets, indices;

  explicit Graph(const std::vector<std::vector<std::uint32_t>>& rows) : offsets{0} {
    for (const auto& row : rows) {
      indices.insert(indices.end(), row.begin(), row.end());
      offsets.push_back(count32(indices.size()));
    }
  }
};
}  // namespace

OccupancySolver::OccupancySolver(BackendKind backend, std::uint32_t device_index, float cutoff)
    : cutoff_(cutoff) {
  require(std::isfinite(cutoff) && cutoff > 0 && cutoff <= 1,
          "epsilon cutoff must be finite and in (0, 1]");

  switch (backend) {
    case BackendKind::metal:
#ifdef CM_HAS_METAL
      device_ = detail::make_metal_occupancy_device(device_index);
      return;
#else
      throw std::runtime_error("Metal occupancy backend was not built");
#endif
    case BackendKind::cuda:
#ifdef CM_HAS_CUDA
      device_ = detail::make_cuda_occupancy_device(device_index);
      return;
#else
      throw std::runtime_error("CUDA occupancy backend was not built");
#endif
    default:
      static_cast<void>(device_index);
      throw std::invalid_argument(
          "native occupancy requires Metal or CUDA; use occupancy_reference for CPU");
  }
}

OccupancySolver::~OccupancySolver() = default;

namespace {
std::vector<float> pack_occupancy_capsules(const std::vector<OccupancyCapsule>& cells) {
  std::vector<float> packed_cells;

  for (const auto& cell : cells) {
    for (float value : cell.center) {
      finite(value, "capsule geometry must be finite");
    }

    for (float value : cell.direction) {
      finite(value, "capsule geometry must be finite");
    }

    require(std::isfinite(cell.length) && cell.length >= 0 && std::isfinite(cell.radius) &&
                cell.radius > 0 &&
                std::hypot(double(cell.direction[0]), double(cell.direction[1]),
                           double(cell.direction[2])) > 0,
            "capsule requires nonnegative length, positive radius and direction");
    packed_cells.insert(packed_cells.end(), cell.center.begin(), cell.center.end());
    packed_cells.push_back(cell.length);
    packed_cells.insert(packed_cells.end(), cell.direction.begin(), cell.direction.end());
    packed_cells.push_back(cell.radius);
  }

  return packed_cells;
}

void validate_occupancy_step_parameters(float dt, std::uint32_t max_iterations,
                                        float relative_tolerance) {
  require(std::isfinite(dt) && dt >= 0, "dt must be finite and nonnegative");
  require(max_iterations > 0 && std::isfinite(relative_tolerance) && relative_tolerance > 0 &&
              relative_tolerance <= 1e-5F,
          "invalid occupancy solver parameters");
}

void append_occupancy_faces(std::vector<std::vector<std::uint32_t>>& rows,
                            std::vector<std::vector<float>>& row_data,
                            const std::vector<float>& volume,
                            const std::vector<OccupancyFace>& faces) {
  const auto n = volume.size();
  auto append = [&](std::uint32_t i, std::uint32_t j, float g, float q, float c) {
    rows[i].push_back(j);
    row_data[i].insert(row_data[i].end(), {g, q, c});
  };

  for (const auto& face : faces) {
    auto i = face.first, j = face.second;
    require(i < n && j < n && i != j, "invalid transport face indices");
    require(
        std::isfinite(face.conductance) && face.conductance >= 0 && std::isfinite(face.volume_flux),
        "invalid transport coefficients");
    require((volume[i] > 0 && volume[j] > 0) || (face.conductance == 0 && face.volume_flux == 0),
            "closed storage cannot have an open face");
    append(i, j, face.conductance, face.volume_flux, 0);
    append(j, i, face.conductance, -face.volume_flux, 0);
  }
}

void append_occupancy_reservoirs(std::vector<std::vector<std::uint32_t>>& rows,
                                 std::vector<std::vector<float>>& row_data,
                                 const std::vector<float>& volume,
                                 const std::vector<OccupancyReservoir>& reservoirs) {
  const auto n = volume.size();
  auto append = [&](std::uint32_t i, std::uint32_t j, float g, float q, float c) {
    rows[i].push_back(j);
    row_data[i].insert(row_data[i].end(), {g, q, c});
  };

  for (const auto& face : reservoirs) {
    require(face.site < n && volume[face.site] > 0, "reservoir must connect accessible storage");
    require(std::isfinite(face.concentration) && face.concentration >= 0 &&
                std::isfinite(face.conductance) && face.conductance >= 0 &&
                std::isfinite(face.volume_flux),
            "invalid reservoir coefficients");
    append(face.site, closed, face.conductance, face.volume_flux, face.concentration);
  }
}

}  // namespace

std::vector<float> OccupancySolver::geometric_porosity(
    const std::vector<std::array<float, 3>>& centers, std::array<float, 3> spacing,
    const std::vector<OccupancyCapsule>& cells, std::uint32_t subdivisions,
    const std::vector<std::uint32_t>& walls) {
  // m^3 <= 2^24 keeps integer sample counts exactly representable in float32.
  require(subdivisions >= 1 && subdivisions <= 256, "subdivisions must lie in [1, 256]");
  count32(centers.size() * 3);
  count32(cells.size() * 8);

  for (float h : spacing) {
    require(std::isfinite(h) && h > 0, "spacing must be finite and positive");
  }

  for (const auto& center : centers) {
    for (float value : center) {
      finite(value, "centers must be finite");
    }
  }

  const auto packed_cells = pack_occupancy_capsules(cells);

  require(walls.empty() || walls.size() == centers.size(), "walls size mismatch");

  for (auto wall : walls) {
    require(wall <= 1, "walls must contain Booleans");
  }

  auto mask = walls.empty() ? std::vector<std::uint32_t>(centers.size()) : walls;
  OccupancyParameters p{.count = count32(centers.size()),
                        .auxiliary = count32(cells.size()),
                        .subdivisions = subdivisions,
                        .cutoff = cutoff_,
                        .hx = spacing[0],
                        .hy = spacing[1],
                        .hz = spacing[2]};
  auto out = device_->allocate(centers.size() * sizeof(float));
  device_->dispatch(
      OccupancyKernel::geometry, p,
      {device_->upload(centers), device_->upload(packed_cells), device_->upload(mask), out});

  return checked_result(*device_, out, centers.size());
}

std::vector<float> OccupancySolver::accessible_volumes(const std::vector<float>& porosity,
                                                       float voxel_volume) {
  vector_valid(porosity);

  for (float value : porosity) {
    require(value <= 1, "porosity must lie in [0, 1]");
  }

  require(std::isfinite(voxel_volume) && voxel_volume > 0,
          "voxel volume must be finite and positive");
  OccupancyParameters p{
      .count = count32(porosity.size()), .scalar = voxel_volume, .cutoff = cutoff_};
  auto out = device_->allocate(porosity.size() * sizeof(float));
  device_->dispatch(OccupancyKernel::volumes, p, {device_->upload(porosity), out});
  auto result = checked_result(*device_, out, porosity.size());

  for (std::size_t i = 0; i < result.size(); ++i) {
    require(porosity[i] < cutoff_ || result[i] > 0, "accessible volume underflow");
  }

  return result;
}

std::vector<float> OccupancySolver::concentration(const std::vector<float>& amount,
                                                  const std::vector<float>& volume) {
  storage_valid(amount, volume);
  OccupancyParameters p{.count = count32(amount.size())};
  auto out = device_->allocate(amount.size() * sizeof(float));
  device_->dispatch(OccupancyKernel::concentration, p,
                    {device_->upload(amount), device_->upload(volume), out});

  return checked_result(*device_, out, amount.size());
}

OccupancyFace OccupancySolver::porosity_face(std::uint32_t first, std::uint32_t second,
                                             float epsilon_first, float epsilon_second,
                                             float diffusion, float area, float distance,
                                             float velocity) {
  require(std::isfinite(epsilon_first) && epsilon_first >= 0 && epsilon_first <= 1 &&
              std::isfinite(epsilon_second) && epsilon_second >= 0 && epsilon_second <= 1,
          "face porosities must lie in [0, 1]");
  require(std::isfinite(diffusion) && diffusion >= 0 && std::isfinite(area) && area > 0 &&
              std::isfinite(distance) && distance > 0 && std::isfinite(velocity),
          "invalid face data");
  const std::vector<float> data{epsilon_first, epsilon_second, diffusion, area, distance, velocity};
  auto out = device_->allocate(2 * sizeof(float));
  device_->dispatch(OccupancyKernel::face, {.count = 1, .cutoff = cutoff_},
                    {device_->upload(data), out});
  auto result = device_->download<float>(out, 2);

  for (float value : result) {
    finite(value, "face coefficients overflow");
  }

  return {first, second, result[0], result[1]};
}

std::vector<float> OccupancySolver::exchange_weights(const std::vector<float>& kernel,
                                                     const std::vector<float>& volume) {
  vector_valid(kernel);
  vector_valid(volume);
  require(kernel.size() == volume.size(), "exchange kernel and volume size mismatch");
  require(!kernel.empty(), "cell has no valid accessible exchange support");
  OccupancyParameters p{.count = count32(kernel.size())};
  auto weights = device_->allocate(kernel.size() * sizeof(float));
  auto total = device_->allocate(sizeof(float));
  auto out = device_->allocate(kernel.size() * sizeof(float));
  device_->dispatch(OccupancyKernel::product, p,
                    {device_->upload(kernel), device_->upload(volume), weights});
  device_->dispatch(OccupancyKernel::sum, p, {weights, total});
  float capacity = device_->download<float>(total, 1)[0];
  require(std::isfinite(capacity) && capacity > 0, "cell has no valid accessible exchange support");
  device_->dispatch(OccupancyKernel::normalize, p, {weights, total, out});

  return checked_result(*device_, out, kernel.size());
}

std::vector<float> OccupancySolver::remap_amounts(
    const std::vector<float>& amount, const std::vector<float>& old_volume,
    const std::vector<float>& new_volume,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& neighbors) {
  storage_valid(amount, old_volume);
  vector_valid(new_volume);
  const auto n = count32(amount.size());
  require(new_volume.size() == n, "new volume size mismatch");
  std::vector<std::vector<std::uint32_t>> rows(n);

  for (auto [i, j] : neighbors) {
    require(i < n && j < n && i != j, "invalid neighbor edge");
    rows[i].push_back(j);
    rows[j].push_back(i);
  }

  if (!n) {
    return {};
  }

  const Graph graph(rows);
  auto old = device_->upload(old_volume), next = device_->upload(new_volume);
  auto offsets = device_->upload(graph.offsets), indices = device_->upload(graph.indices);
  auto labels = device_->allocate(n * sizeof(std::uint32_t));
  auto updated = device_->allocate(n * sizeof(std::uint32_t));
  OccupancyParameters p{.count = n};
  device_->dispatch(OccupancyKernel::labels_init, p, {old, next, labels});
  auto previous = device_->download<std::uint32_t>(labels, n);

  for (std::uint32_t iteration = 0; iteration < n; ++iteration) {
    device_->dispatch(OccupancyKernel::labels_step, p, {offsets, indices, labels, updated});
    auto current = device_->download<std::uint32_t>(updated, n);
    std::swap(labels, updated);

    if (current == previous) {
      previous = std::move(current);
      break;
    }

    previous = std::move(current);

    if (iteration == n - 1) {
      throw std::runtime_error("occupancy component labels did not converge");
    }
  }

  // Only topology is packed on the host. Device labels determine membership;
  // all amount/capacity arithmetic below runs on the selected GPU in site order.
  std::vector<std::vector<std::uint32_t>> components(n);

  for (std::uint32_t i = 0; i < n; ++i) {
    if (previous[i] != closed) {
      components[previous[i]].push_back(i);
    }
  }

  const Graph members(components);
  const auto total_count = count32(std::size_t{2} * n);
  auto totals = device_->allocate(total_count * sizeof(float));
  auto input = device_->upload(amount);
  device_->dispatch(
      OccupancyKernel::component_sums, p,
      {device_->upload(members.offsets), device_->upload(members.indices), input, next, totals});
  const auto values = device_->download<float>(totals, total_count);

  for (std::uint32_t i = 0; i < n; ++i) {
    require(std::isfinite(values[2 * i]) && std::isfinite(values[2 * i + 1]) &&
                (values[2 * i] == 0 || values[2 * i + 1] > 0),
            "closing component has solute but no accessible recipient (or volume overflow)");
  }

  auto out = device_->allocate(n * sizeof(float));
  device_->dispatch(OccupancyKernel::remap, p, {input, next, labels, totals, out});
  auto result = checked_result(*device_, out, n);
  check_balance({.before = sum(amount), .after = sum(result)});

  return result;
}

namespace {
void validate_occupancy_sources(const std::vector<float>& volume, const std::vector<float>& sources,
                                const std::vector<float>& losses) {
  const auto n = volume.size();
  vector_valid(sources, false);
  vector_valid(losses);
  require(sources.size() == n && losses.size() == n, "source or loss size mismatch");

  for (std::uint32_t i = 0; i < n; ++i) {
    require(volume[i] > 0 || sources[i] == 0, "sources require accessible storage");
  }
}
}  // namespace

OccupancyStep OccupancySolver::backward_euler(
    const std::vector<float>& amount, const std::vector<float>& volume,
    const std::vector<OccupancyFace>& faces, float dt, const std::vector<float>& source,
    const std::vector<float>& loss, const std::vector<OccupancyReservoir>& reservoirs,
    std::uint32_t max_iterations, float relative_tolerance) {
  storage_valid(amount, volume);
  validate_occupancy_step_parameters(dt, max_iterations, relative_tolerance);
  const auto n = count32(amount.size());
  const auto sources = source.empty() ? std::vector<float>(n) : source;
  const auto losses = loss.empty() ? std::vector<float>(n) : loss;
  validate_occupancy_sources(volume, sources, losses);

  std::vector<std::vector<std::uint32_t>> rows(n);
  std::vector<std::vector<float>> row_data(n);
  append_occupancy_faces(rows, row_data, volume, faces);

  append_occupancy_reservoirs(rows, row_data, volume, reservoirs);

  if (!n || dt == 0) {
    return {amount, {.before = sum(amount), .after = sum(amount)}};
  }

  const Graph graph(rows);
  std::vector<float> coefficients;

  for (const auto& row : row_data) {
    coefficients.insert(coefficients.end(), row.begin(), row.end());
  }

  count32(coefficients.size());
  const auto offsets = device_->upload(graph.offsets), indices = device_->upload(graph.indices);
  const auto edges = device_->upload(coefficients), v = device_->upload(volume);
  const auto a = device_->upload(amount), s = device_->upload(sources), k = device_->upload(losses);
  auto diagonal = device_->allocate(n * sizeof(float)), rhs = device_->allocate(n * sizeof(float));
  auto current = device_->allocate(n * sizeof(float)), next = device_->allocate(n * sizeof(float));
  auto residual = device_->allocate(n * sizeof(float)), total = device_->allocate(sizeof(float));
  OccupancyParameters p{.count = n, .scalar = dt};
  device_->dispatch(OccupancyKernel::assemble, p,
                    {offsets, indices, edges, v, a, s, k, diagonal, rhs});
  // Overflow/underflow must not turn into a false convergence decision.
  const auto diagonals = checked_result(*device_, diagonal, n);

  for (auto value : diagonals) {
    require(value > 0, "occupancy matrix diagonal underflow");
  }

  const auto right_hand_side = device_->download<float>(rhs, n);
  vector_valid(right_hand_side, false);
  double rhs_norm = 0;

  for (float value : right_hand_side) {
    rhs_norm += std::abs(double(value));
  }

  device_->dispatch(OccupancyKernel::concentration, p, {a, v, current});
  OccupancyStep result;

  for (std::uint32_t iteration = 0;; ++iteration) {
    // Evaluate in conservative flux form, avoiding cancellation between a
    // large diffusive diagonal and nearly equal neighboring concentrations.
    device_->dispatch(OccupancyKernel::residual, p,
                      {offsets, indices, edges, v, a, s, k, current, residual});

    if (iteration % 8 == 0 || iteration == max_iterations) {
      auto reduction = p;
      reduction.absolute = 1;
      device_->dispatch(OccupancyKernel::sum, reduction, {residual, total});
      const double error = device_->download<float>(total, 1)[0];

      if (!std::isfinite(error)) {
        throw std::runtime_error("non-finite occupancy solver residual");
      }

      result.relative_residual = rhs_norm == 0 ? error : error / rhs_norm;

      if (error <= double(relative_tolerance) * rhs_norm) {
        result.iterations = iteration;
        break;
      }

      if (iteration == max_iterations) {
        std::ostringstream message;
        message << "occupancy backward Euler did not converge; relative residual "
                << result.relative_residual << " exceeds " << relative_tolerance
                << "; candidate rejected";
        throw std::runtime_error(message.str());
      }
    }

    device_->dispatch(OccupancyKernel::jacobi, p, {diagonal, current, residual, next});
    std::swap(current, next);
  }

  auto out = device_->allocate(n * sizeof(float));
  auto reaction = device_->allocate(n * sizeof(float)),
       boundary = device_->allocate(n * sizeof(float));
  device_->dispatch(OccupancyKernel::finish, p,
                    {offsets, indices, edges, v, k, current, out, reaction, boundary});
  result.amount = checked_result(*device_, out, n);
  result.balance = {.before = sum(amount),
                    .after = sum(result.amount),
                    .source = double(dt) * sum(sources),
                    .reaction = sum(device_->download<float>(reaction, n)),
                    .boundary = sum(device_->download<float>(boundary, n))};
  check_balance(result.balance);

  return result;
}

}  // namespace cm
