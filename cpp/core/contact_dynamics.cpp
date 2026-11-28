#include "microsimulator/contact_dynamics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cell_contact_plan.hpp"
#include "core/contact_step_device.hpp"
#include "core/contact_stepper.hpp"
#include "core/external_contact_plan.hpp"
#include "core/flow_drift_plan.hpp"
#include "core/shared_constants.hpp"
#include "microsimulator/mechanics.hpp"

namespace microsimulator {
namespace {

// The device kernels select the growth load law by these codes.
static_assert(static_cast<std::uint32_t>(GrowthLoadLaw::none) == shared::growth_load_law_none);
static_assert(static_cast<std::uint32_t>(GrowthLoadLaw::exponential) ==
              shared::growth_load_law_exponential);
static_assert(static_cast<std::uint32_t>(GrowthLoadLaw::linear) == shared::growth_load_law_linear);

// The solve converges to this fraction of the overlap tolerance, which leaves
// the rest for the error of moving the cells along a linearized gap.
constexpr float solve_tolerance_fraction = 0.25F;
// Fraction of the load scale of the growth response that an accepted overlap
// may add to a contact force.
constexpr float load_resolution_fraction = 0.1F;
// Refinement rounds within which a substep counts as short enough to try a
// longer one next.
constexpr std::uint32_t easy_refinements = 2;
// A rotation this large in one substep is outside the linearization the
// substep's rows rest on.
constexpr float maximum_substep_rotation = 0.5F;
// The rotation a substep's length is planned for. The margin below the limit
// absorbs a turn rate that changes as the contacts do.
constexpr float planned_substep_rotation = 0.8F * maximum_substep_rotation;
// Solver iterations after which the fastest-turning cell's rotation is read
// to decide whether a substep of untried length is worth solving. A cell
// turns under the forces of its own neighbors, which settle long before the
// pressure across a colony does.
constexpr std::uint32_t rotation_probe_iterations = 128;
// The contacts a cell is first given room for, and the rows it may add in a
// round before its storage grows.
constexpr std::uint32_t candidates_per_cell = 8;
constexpr std::uint32_t walls_per_cell = 4;
constexpr std::uint32_t minimum_capacity = 64;

using detail::StepBuffer;
using detail::StepKernel;

// The overlap accepted at the end of a substep. An overlap of e that a jammed
// pair cannot recover stays as a contact force of e * stiffness. Where growth
// responds to load, the tolerance is therefore held to the overlap that a
// small fraction of the load scale of that response presses into a contact.
[[nodiscard]] float effective_overlap_tolerance(const ContactDynamicsParameters& parameters) {
  auto load_scale = 0.0F;
  if (parameters.growth.law == GrowthLoadLaw::linear) {
    load_scale = parameters.growth.stall_load - parameters.growth.free_load;
  } else if (parameters.growth.law == GrowthLoadLaw::exponential &&
             parameters.growth.sensitivity > 0.0F) {
    load_scale = 1.0F / parameters.growth.sensitivity;
  }
  if (load_scale <= 0.0F) {
    return parameters.overlap_tolerance;
  }
  return std::min(parameters.overlap_tolerance,
                  load_resolution_fraction * load_scale / parameters.contact_stiffness);
}

[[nodiscard]] bool finite(const Vec3& value) {
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

[[nodiscard]] float word_float(std::uint32_t word) {
  return std::bit_cast<float>(word);
}

[[nodiscard]] std::uint32_t grown(std::uint32_t needed) {
  return std::bit_ceil(std::max(minimum_capacity, needed + needed / 4));
}

// What an attempt at a substep came to. `rotation` is the largest angle any
// cell turned through in the last move the attempt tried, which is what the
// next attempt's length is planned from.
struct Attempt {
  bool accepted{false};
  float rotation{0.0F};
  std::uint32_t refinements{0};
  std::uint32_t rows{0};
  float max_overlap{0.0F};
  std::uint64_t iterations{0};
};

// A step of contact dynamics held on a backend's device: the base geometry
// of the step, the rows of the substep being solved, and the forces of the
// last accepted substep. Each refinement round is enqueued whole, and its
// status is read back once.
class ResidentStep {
 public:
  ResidentStep(ComputeBackend& backend, const WorldState& state, const ConstraintSet& constraints,
               const ContactDynamicsParameters& parameters,
               const ContactParameters& contact_parameters,
               const ConstraintContactParameters& constraint_parameters,
               const detail::ContactDrift& drift, float dt)
      : device_(&backend.contact_step_device()), parameters_(parameters) {
    const auto cells = detail::plan_cell_contacts(state, contact_parameters);
    auto wall_parameters = constraint_parameters;
    wall_parameters.activation_margin = parameters.candidate_margin;
    const auto walls = detail::plan_external_contacts(state, constraints, wall_parameters, true);
    cell_count_ = cells.data.cell_count;
    bucket_count_ = cells.data.bucket_count;
    constraint_count_ = walls.data.constraint_count;
    candidate_capacity_ = grown(candidates_per_cell * cell_count_);
    wall_capacity_ = constraint_count_ == 0 ? 1 : grown(walls_per_cell * cell_count_);
    row_capacity_ = grown(2 * touch_capacity() + cell_count_);
    rows_hint_ = 4 * cell_count_;

    params_.cell_count = cell_count_;
    params_.constraint_count = constraint_count_;
    params_.bucket_count = bucket_count_;
    params_.parallel_sine_threshold = contact_parameters.parallel_sine_threshold;
    params_.contact_degeneracy = contact_parameters.degeneracy_epsilon;
    params_.constraint_degeneracy = constraint_parameters.degeneracy_epsilon;
    params_.contact_compliance = 1.0F / parameters.contact_stiffness;
    params_.wall_compliance = 1.0F / parameters.wall_stiffness;
    params_.recovery = parameters.max_overlap_recovery;
    params_.tolerance = effective_overlap_tolerance(parameters);
    params_.max_rotation = maximum_substep_rotation;
    params_.step_dt = dt;
    params_.drag = parameters.drag;
    params_.max_iterations = parameters.max_iterations;
    params_.solve.tolerance = solve_tolerance_fraction * params_.tolerance;
    params_.solve.law = static_cast<std::uint32_t>(parameters.growth.law);
    params_.solve.sensitivity = parameters.growth.sensitivity;
    params_.solve.free_load = parameters.growth.free_load;
    params_.solve.stall_load = parameters.growth.stall_load;

    const auto geometry = state.geometry_state();
    const auto attributes = state.cell_attributes();
    std::vector<device::float4> rates(std::max<std::size_t>(cell_count_, 1));
    std::vector<std::uint8_t> fixed(rates.size(), 0);
    for (std::size_t slot = 0; slot < cell_count_; ++slot) {
      rates[slot] = {attributes.growth_rates[slot], geometry.lengths[slot], 0.0F, 0.0F};
      fixed[slot] = attributes.fixed[slot] != 0 ? 1 : 0;
    }
    initial_lengths_.assign(geometry.lengths.begin(), geometry.lengths.end());

    reserve_cells();
    reserve_detection();
    reserve_rows(false);
    upload(StepBuffer::base_centers, cells.centers);
    upload(StepBuffer::base_axes, cells.axes);
    upload(StepBuffer::trial_centers, cells.centers);
    upload(StepBuffer::trial_axes, cells.axes);
    upload(StepBuffer::ids, cells.ids);
    upload(StepBuffer::cell_rates, rates);
    upload(StepBuffer::fixed_cells, fixed);
    upload(StepBuffer::constraints, walls.constraints);
    if (drift.grid != nullptr) {
      const auto plan = detail::plan_flow_drift(state, *drift.grid, 0.0F, 0.0F);
      params_.has_drift = 1;
      params_.drift = plan.data;
      upload(StepBuffer::solid, plan.solid);
      upload(StepBuffer::x_faces, plan.field->x_faces);
      upload(StepBuffer::y_faces, plan.field->y_faces);
      upload(StepBuffer::z_faces, plan.field->z_faces);
    } else {
      const std::array<std::uint8_t, 1> none{};
      const std::array<float, 1> zero{};
      upload_bytes(StepBuffer::solid, none.data(), sizeof(none));
      upload_bytes(StepBuffer::x_faces, zero.data(), sizeof(zero));
      upload_bytes(StepBuffer::y_faces, zero.data(), sizeof(zero));
      upload_bytes(StepBuffer::z_faces, zero.data(), sizeof(zero));
    }
    const std::array<std::uint32_t, MS_STEP_WORDS> words{};
    upload_bytes(StepBuffer::status, words.data(), sizeof(words));
  }

  // The number of pieces the step's growth and drift bound it to.
  [[nodiscard]] float pieces(float dt, float travel_limit) {
    if (params_.has_drift != 0) {
      auto p = params_;
      p.drift.dt = 0.0F;
      run(StepKernel::drift, cell_count_, p);
    }
    run(StepKernel::begin_bounds, cell_count_, params_);
    const auto words = read_status();
    check_drift(words);
    const auto growth = word_float(words[MS_STEP_GROWTH]);
    auto pieces = std::max(1.0F, std::ceil(growth / parameters_.max_growth_fraction));
    if (params_.has_drift != 0 && travel_limit > 0.0F) {
      const auto speed = word_float(words[MS_STEP_DRIFT_SPEED]);
      pieces = std::max(pieces, std::ceil(speed * dt / travel_limit));
    }
    return pieces;
  }

  // One attempt at a substep of length `dt` from the base geometry. With
  // `probe`, an attempt whose cells turn too far is given up after the first
  // few solver iterations rather than solved to the end and then discarded.
  [[nodiscard]] Attempt attempt(float dt, bool probe) {
    auto p = params_;
    p.dt = dt;
    p.solve.dt = dt;
    p.drift.dt = dt;
    p.round = 0;
    const auto probing = probe && parameters_.max_iterations > rotation_probe_iterations;
    std::array<std::uint32_t, MS_STEP_WORDS> words{};
    // A first round that runs out of room runs again whole: nothing it
    // changed is an input to it.
    for (;;) {
      run(StepKernel::substep_begin, 1, p);
      run(StepKernel::round_begin, 1, p);
      if (p.has_drift != 0) {
        run(StepKernel::drift, cell_count_, p);
      }
      run(StepKernel::bodies, cell_count_, p);
      detect(p, 0, parameters_.candidate_margin, MS_STEP_STAGE_DETECT_BASE);
      run(StepKernel::rows_build, touch_capacity(), with_capacities(p));
      run(StepKernel::rows_commit, 1, p);
      index_rows(p, true);
      if (probing) {
        auto early = with_capacities(p);
        early.probe = 1;
        early.max_iterations = rotation_probe_iterations;
        solve(early);
        run(StepKernel::probe_rotation, cell_count_, early);
        run(StepKernel::probe_decide, 1, early);
      }
      words = finish_round(p);
      if (words[MS_STEP_OVERFLOW] == MS_STEP_STAGE_DETECT_BASE) {
        grow_detection(words);
        continue;
      }
      break;
    }
    Attempt result;
    for (std::uint32_t refinement = 0;; ++refinement) {
      result.iterations = words[MS_STEP_ITERATIONS];
      result.rotation = word_float(words[MS_STEP_ROTATION]);
      if (words[MS_STEP_FAILED] != 0) {
        return result;
      }
      rows_hint_ = words[MS_STEP_ROWS];
      if (words[MS_STEP_OVERLAPPING] == 0) {
        result.accepted = true;
        result.refinements = refinement;
        result.rows = words[MS_STEP_ROWS];
        result.max_overlap = word_float(words[MS_STEP_MAX_OVERLAP]);
        accept();
        return result;
      }
      if (refinement == parameters_.max_refinements) {
        return result;
      }
      // The next round appends a row for each overlapping contact and solves
      // again from the multipliers the last solve reached.
      p.round = refinement + 1;
      if (words[MS_STEP_ROWS] + words[MS_STEP_OVERLAPPING] > row_capacity_) {
        row_capacity_ = grown(words[MS_STEP_ROWS] + words[MS_STEP_OVERLAPPING]);
        reserve_rows(true);
      }
      run(StepKernel::append_check, 1, with_capacities(p));
      run(StepKernel::append, touch_capacity(), with_capacities(p));
      run(StepKernel::append_commit, 1, p);
      run(StepKernel::round_begin, 1, p);
      index_rows(p, false);
      words = finish_round(p);
    }
  }

  // Writes the base geometry back to the state and fills in the step's
  // result from the last accepted substep.
  void finish(WorldState& state, ContactDynamicsResult& result) {
    std::vector<device::float4> centers(cell_count_);
    std::vector<device::float4> axes(cell_count_);
    std::vector<device::ContactResponse> response(cell_count_);
    device_->download(StepBuffer::base_centers, centers.data(),
                      centers.size() * sizeof(centers[0]));
    device_->download(StepBuffer::base_axes, axes.data(), axes.size() * sizeof(axes[0]));
    device_->download(StepBuffer::accepted_response, response.data(),
                      response.size() * sizeof(response[0]));
    for (std::size_t slot = 0; slot < cell_count_; ++slot) {
      state.set_cell_geometry(static_cast<Slot>(slot),
                              {centers[slot].x, centers[slot].y, centers[slot].z},
                              {axes[slot].x, axes[slot].y, axes[slot].z}, centers[slot].w);
      const auto& linear = response[slot].linear_length;
      result.center_velocities[slot] = {linear.x, linear.y, linear.z};
      result.axial_loads[slot] = response[slot].rotation.w;
      result.length_increments[slot] = centers[slot].w - initial_lengths_[slot];
    }
  }

 private:
  [[nodiscard]] std::uint32_t touch_capacity() const {
    return 2 * candidate_capacity_ + wall_capacity_;
  }

  [[nodiscard]] device::ContactStepParams with_capacities(device::ContactStepParams p) const {
    p.candidate_capacity = candidate_capacity_;
    p.wall_capacity = wall_capacity_;
    p.touch_capacity = touch_capacity();
    p.row_capacity = row_capacity_;
    return p;
  }

  void run(StepKernel kernel, std::uint32_t count, const device::ContactStepParams& p) {
    if (count != 0) {
      device_->run(kernel, count, with_capacities(p));
    }
  }

  void scan(device::ContactStepParams p, std::uint32_t kind, std::uint32_t stage) {
    p = with_capacities(p);
    p.scan = kind;
    p.stage = stage;
    device_->scan(p);
  }

  void solve(const device::ContactStepParams& p) {
    device_->solve(with_capacities(p), rows_hint_);
    run(StepKernel::solve_finish, 1, p);
    run(StepKernel::solve_copy, row_capacity_, p);
  }

  // The contacts of the base (0) or trial (1) geometry within `margin`, into
  // the touches (0) or the found contacts (1) as `geometry` names.
  void detect(device::ContactStepParams p, std::uint32_t geometry, float margin,
              std::uint32_t stage) {
    p.geometry = geometry;
    p.output = geometry;
    p.margin = margin;
    run(StepKernel::detect_clear, std::max(bucket_count_, 1U), p);
    run(StepKernel::bounds, cell_count_, p);
    run(StepKernel::width, cell_count_, p);
    run(StepKernel::place, cell_count_, p);
    scan(p, MS_STEP_SCAN_BUCKETS, stage);
    run(StepKernel::grid_fill, cell_count_, p);
    run(StepKernel::candidate_count, cell_count_, p);
    scan(p, MS_STEP_SCAN_CANDIDATES, stage);
    run(StepKernel::candidate_fill, cell_count_, p);
    run(StepKernel::pair_count, candidate_capacity_, p);
    scan(p, MS_STEP_SCAN_PAIRS, stage);
    run(StepKernel::pair_fill, candidate_capacity_, p);
    if (constraint_count_ != 0) {
      run(StepKernel::wall_count, cell_count_ * constraint_count_, p);
      scan(p, MS_STEP_SCAN_WALLS, stage);
      run(StepKernel::wall_fill, cell_count_ * constraint_count_, p);
    }
    run(StepKernel::touches_total, 1, p);
    run(StepKernel::touches_fill, touch_capacity(), p);
  }

  // Files the rows by first cell and by the cells they touch, warm-starts
  // the first round's rows, and sets the solve's first step length.
  void index_rows(const device::ContactStepParams& p, bool warm_start) {
    run(StepKernel::rows_clear, cell_count_ + 1, p);
    run(StepKernel::keyed_count, row_capacity_, p);
    scan(p, MS_STEP_SCAN_KEYED, 0);
    run(StepKernel::keyed_fill, row_capacity_, p);
    run(StepKernel::keyed_sort, cell_count_, p);
    if (warm_start) {
      run(StepKernel::warm_start, row_capacity_, p);
    }
    run(StepKernel::incidence_count, row_capacity_, p);
    scan(p, MS_STEP_SCAN_INCIDENCE, 0);
    run(StepKernel::incidence_fill, row_capacity_, p);
    run(StepKernel::incidence_sort, cell_count_, p);
    run(StepKernel::row_bound, row_capacity_, p);
    run(StepKernel::solve_prepare, 1, p);
  }

  // Solves the round's rows, moves the cells, finds the contacts the move
  // leaves overlapping, and reads the round's status. A detection that runs
  // out of room runs again with more.
  [[nodiscard]] std::array<std::uint32_t, MS_STEP_WORDS> finish_round(
      const device::ContactStepParams& p) {
    solve(p);
    run(StepKernel::move, cell_count_, p);
    run(StepKernel::move_check, 1, p);
    for (;;) {
      detect(p, 1, 0.0F, MS_STEP_STAGE_DETECT_TRIAL);
      run(StepKernel::overlap_flags, touch_capacity(), p);
      scan(p, MS_STEP_SCAN_FOUND, 0);
      auto words = read_status();
      check_drift(words);
      if (words[MS_STEP_SOLVE_ERROR] == MS_STEP_SOLVE_NO_DIAGONAL) {
        throw std::invalid_argument("contact system has no positive diagonal");
      }
      if (words[MS_STEP_OVERFLOW] != MS_STEP_STAGE_DETECT_TRIAL) {
        return words;
      }
      grow_detection(words);
      const std::uint32_t clear = 0;
      upload_bytes(StepBuffer::status, &clear, sizeof(clear), MS_STEP_OVERFLOW * sizeof(clear));
    }
  }

  void accept() {
    run(StepKernel::accept, std::max(cell_count_ + 1, row_capacity_), params_);
    run(StepKernel::accept_commit, 1, params_);
  }

  void grow_detection(const std::array<std::uint32_t, MS_STEP_WORDS>& words) {
    if (words[MS_STEP_CANDIDATES] > candidate_capacity_) {
      candidate_capacity_ = grown(words[MS_STEP_CANDIDATES]);
    }
    if (words[MS_STEP_WALL_CONTACTS] > wall_capacity_) {
      wall_capacity_ = grown(words[MS_STEP_WALL_CONTACTS]);
    }
    reserve_detection();
    if (row_capacity_ < touch_capacity() + cell_count_) {
      row_capacity_ = grown(touch_capacity() + cell_count_);
      reserve_rows(true);
    }
  }

  static void check_drift(const std::array<std::uint32_t, MS_STEP_WORDS>& words) {
    if (words[MS_STEP_DRIFT_ERROR] != 0) {
      const std::array<std::uint32_t, 1> code{words[MS_STEP_DRIFT_ERROR]};
      detail::check_flow_drift_status(code);
    }
  }

  [[nodiscard]] std::array<std::uint32_t, MS_STEP_WORDS> read_status() {
    std::array<std::uint32_t, MS_STEP_WORDS> words{};
    device_->download(StepBuffer::status, words.data(), sizeof(words));
    return words;
  }

  template <typename T>
  void upload(StepBuffer buffer, const std::vector<T>& values) {
    upload_bytes(buffer, values.data(), values.size() * sizeof(T));
  }

  void upload_bytes(StepBuffer buffer, const void* data, std::size_t bytes,
                    std::size_t offset = 0) {
    device_->reserve(buffer, offset + bytes, true);
    device_->upload(buffer, data, bytes, offset);
  }

  template <typename T>
  void reserve(StepBuffer buffer, std::size_t count, bool preserve = false) {
    device_->reserve(buffer, std::max<std::size_t>(count, 1) * sizeof(T), preserve);
  }

  void reserve_cells() {
    const std::size_t cells = cell_count_;
    for (const auto buffer :
         {StepBuffer::base_centers, StepBuffer::base_axes, StepBuffer::trial_centers,
          StepBuffer::trial_axes, StepBuffer::cell_rates, StepBuffer::lower, StepBuffer::upper}) {
      reserve<device::float4>(buffer, cells);
    }
    reserve<std::uint8_t>(StepBuffer::fixed_cells, cells);
    reserve<std::uint64_t>(StepBuffer::ids, cells);
    reserve<device::ContactBodyData>(StepBuffer::bodies, cells);
    reserve<float>(StepBuffer::drift, 6 * cells);
    reserve<device::ContactResponse>(StepBuffer::response, cells);
    reserve<device::ContactResponse>(StepBuffer::accepted_response, cells);
    reserve<device::uint4>(StepBuffer::coordinates, cells);
    for (const auto buffer :
         {StepBuffer::members, StepBuffer::candidate_counts, StepBuffer::candidate_ends,
          StepBuffer::keyed_counts, StepBuffer::keyed_cursors, StepBuffer::incidence_counts,
          StepBuffer::incidence_cursors}) {
      reserve<std::uint32_t>(buffer, cells);
    }
    for (const auto buffer : {StepBuffer::keyed_offsets, StepBuffer::incidence_offsets,
                              StepBuffer::remembered_offsets}) {
      reserve<std::uint32_t>(buffer, cells + 1);
    }
    for (const auto buffer :
         {StepBuffer::bucket_counts, StepBuffer::bucket_ends, StepBuffer::bucket_cursors}) {
      reserve<std::uint32_t>(buffer, bucket_count_);
    }
    reserve<std::uint32_t>(StepBuffer::wall_counts,
                           static_cast<std::size_t>(cell_count_) * constraint_count_);
    reserve<std::uint32_t>(StepBuffer::wall_ends,
                           static_cast<std::size_t>(cell_count_) * constraint_count_);
    reserve<device::ContactSolveState>(StepBuffer::solve_state, 1);
    reserve<std::uint32_t>(StepBuffer::status, MS_STEP_WORDS);
  }

  void reserve_detection() {
    reserve<device::uint2>(StepBuffer::candidates, candidate_capacity_);
    reserve<std::uint32_t>(StepBuffer::pair_counts, candidate_capacity_);
    reserve<std::uint32_t>(StepBuffer::pair_ends, candidate_capacity_);
    reserve<device::CellContactRecord>(StepBuffer::cell_records, 2 * candidate_capacity_);
    reserve<device::ExternalContactRecord>(StepBuffer::wall_records, wall_capacity_);
    const auto touches = touch_capacity();
    reserve<device::ContactTouch>(StepBuffer::touches, touches);
    reserve<device::ContactTouch>(StepBuffer::found, touches);
    reserve<std::uint32_t>(StepBuffer::found_flags, touches);
    reserve<std::uint32_t>(StepBuffer::found_ends, touches);
  }

  void reserve_rows(bool preserve) {
    const std::size_t rows = row_capacity_;
    reserve<device::ContactRowData>(StepBuffer::rows, rows, preserve);
    reserve<device::uint4>(StepBuffer::keys, rows, preserve);
    reserve<float>(StepBuffer::multipliers, rows, preserve);
    reserve<float>(StepBuffer::multipliers_next, rows);
    reserve<float>(StepBuffer::gaps, rows);
    reserve<float>(StepBuffer::gaps_next, rows);
    reserve<device::float4>(StepBuffer::terms, rows);
    reserve<std::uint32_t>(StepBuffer::keyed_rows, rows);
    reserve<std::uint32_t>(StepBuffer::incidence_entries, 2 * rows);
    reserve<device::uint4>(StepBuffer::remembered_keys, rows, true);
    reserve<float>(StepBuffer::remembered_multipliers, rows, true);
    reserve<std::uint32_t>(StepBuffer::remembered_rows, rows, true);
  }

  detail::ContactStepDevice* device_;
  const ContactDynamicsParameters& parameters_;
  device::ContactStepParams params_{};
  std::vector<float> initial_lengths_;
  std::uint32_t cell_count_{0};
  std::uint32_t bucket_count_{0};
  std::uint32_t constraint_count_{0};
  std::uint32_t candidate_capacity_{0};
  std::uint32_t wall_capacity_{0};
  std::uint32_t row_capacity_{0};
  std::uint32_t rows_hint_{0};
};

}  // namespace

void validate_growth_load_response(const GrowthLoadResponse& response) {
  if (response.law > GrowthLoadLaw::linear) {
    throw std::invalid_argument("growth load law is invalid");
  }
  if (!std::isfinite(response.sensitivity) || response.sensitivity < 0.0F) {
    throw std::invalid_argument("growth load sensitivity must be finite and non-negative");
  }
  if (!std::isfinite(response.free_load) || !std::isfinite(response.stall_load) ||
      response.free_load < 0.0F || response.stall_load <= response.free_load) {
    throw std::invalid_argument(
        "growth stall load must be finite and exceed a finite, non-negative free load");
  }
}

float growth_load_factor(const GrowthLoadResponse& response, float load) {
  switch (response.law) {
    case GrowthLoadLaw::none:
      return 1.0F;
    case GrowthLoadLaw::exponential:
      return std::exp(-response.sensitivity * std::max(0.0F, load));
    case GrowthLoadLaw::linear:
      return std::clamp(
          1.0F - (load - response.free_load) / (response.stall_load - response.free_load), 0.0F,
          1.0F);
  }
  throw std::invalid_argument("growth load law is invalid");
}

void validate_contact_dynamics_parameters(const ContactDynamicsParameters& parameters) {
  const auto positive = [](float value) { return std::isfinite(value) && value > 0.0F; };
  if (!positive(parameters.drag)) {
    throw std::invalid_argument("contact dynamics drag must be finite and positive");
  }
  if (!positive(parameters.contact_stiffness) || !positive(parameters.wall_stiffness)) {
    throw std::invalid_argument("contact and wall stiffness must be finite and positive");
  }
  if (!positive(parameters.overlap_tolerance)) {
    throw std::invalid_argument("contact overlap tolerance must be finite and positive");
  }
  if (!std::isfinite(parameters.candidate_margin) || parameters.candidate_margin < 0.0F) {
    throw std::invalid_argument("contact candidate margin must be finite and non-negative");
  }
  if (!positive(parameters.max_overlap_recovery)) {
    throw std::invalid_argument("contact overlap recovery must be finite and positive");
  }
  if (!positive(parameters.max_growth_fraction)) {
    throw std::invalid_argument("contact growth fraction must be finite and positive");
  }
  if (parameters.max_iterations == 0) {
    throw std::invalid_argument("contact dynamics iteration limit must be positive");
  }
  validate_growth_load_response(parameters.growth);
}

void validate_contact_system(const ContactSystem& system) {
  if (!std::isfinite(system.dt) || system.dt <= 0.0F) {
    throw std::invalid_argument("contact system time step must be finite and positive");
  }
  validate_growth_load_response(system.growth);
  for (const auto& body : system.bodies) {
    if (!finite(body.axis) || !std::isfinite(body.translational_mobility) ||
        !std::isfinite(body.transverse_mobility) || !std::isfinite(body.axial_mobility) ||
        !std::isfinite(body.length_rate) ||
        !std::ranges::all_of(body.drift, [](float value) { return std::isfinite(value); }) ||
        body.translational_mobility < 0.0F || body.transverse_mobility < 0.0F ||
        body.axial_mobility < 0.0F) {
      throw std::invalid_argument("contact body is invalid");
    }
  }
  for (const auto& row : system.rows) {
    if (row.first_slot >= system.bodies.size() ||
        (row.second_slot != invalid_slot && row.second_slot >= system.bodies.size())) {
      throw std::invalid_argument("contact row slot is out of range");
    }
    auto valid = std::isfinite(row.offset) && std::isfinite(row.compliance) &&
                 row.compliance >= 0.0F && std::isfinite(row.first_length_lever) &&
                 std::isfinite(row.second_length_lever) && row.first_length_lever >= 0.0F &&
                 row.second_length_lever >= 0.0F;
    for (std::size_t index = 0; index < row.first_jacobian.size(); ++index) {
      valid = valid && std::isfinite(row.first_jacobian[index]) &&
              std::isfinite(row.second_jacobian[index]);
    }
    if (!valid) {
      throw std::invalid_argument("contact row is invalid");
    }
  }
}

namespace detail {

ContactDynamicsResult advance_contact_dynamics(
    WorldState& state, const ConstraintSet& constraints, ComputeBackend& backend, float dt,
    const ContactDynamicsParameters& parameters, const ContactParameters& contact_parameters,
    const ConstraintContactParameters& constraint_parameters, const ContactDrift& drift) {
  if (!std::isfinite(dt) || dt < 0.0F) {
    throw std::invalid_argument("time step must be finite and non-negative");
  }
  validate_contact_dynamics_parameters(parameters);
  validate_contact_parameters(contact_parameters);
  validate_constraint_contact_parameters(constraint_parameters);
  state.validate();

  ContactDynamicsResult result;
  result.center_velocities.resize(state.size());
  result.axial_loads.assign(state.size(), 0.0F);
  result.length_increments.assign(state.size(), 0.0F);
  if (state.empty() || dt == 0.0F) {
    return result;
  }

  // The state holds the step's starting geometry until the step succeeds.
  ResidentStep step(backend, state, constraints, parameters, contact_parameters,
                    constraint_parameters, drift, dt);
  // The first substep is as long as the growth and drift bounds allow.
  const auto pieces = step.pieces(dt, drift.travel_limit);
  if (!std::isfinite(pieces)) {
    throw std::runtime_error("contact dynamics growth per step is not finite");
  }
  const auto longest = dt / pieces;
  const auto shortest = std::ldexp(longest, -static_cast<int>(parameters.max_step_halvings));

  auto remaining = static_cast<double>(dt);
  auto length = longest;
  // The longest substep this step has taken. A longer one is untried.
  auto proven = 0.0F;
  while (remaining > 0.0) {
    // A final piece within rounding of the remainder takes all of it.
    if (static_cast<double>(length) >= remaining * (1.0 - 1.0e-6)) {
      length = static_cast<float>(remaining);
    }
    const auto attempt = step.attempt(length, length > proven);
    result.report.iterations += attempt.iterations;
    if (!attempt.accepted) {
      // Rotation is close to proportional to the length of a substep, so an
      // attempt that turned a cell too far is retried at the length that
      // brings the turn within the planned rotation, not merely at half.
      const auto excess = attempt.rotation / planned_substep_rotation;
      const auto halvings =
          excess > 2.0F ? static_cast<std::uint32_t>(std::ceil(std::log2(excess))) : 1U;
      const auto failed = length;
      length = std::ldexp(length, -static_cast<int>(std::min<std::uint32_t>(halvings, 64)));
      result.report.halvings += halvings;
      if (length < shortest) {
        throw std::runtime_error("contact dynamics did not resolve the contacts of a substep of " +
                                 std::to_string(failed));
      }
      continue;
    }
    remaining -= static_cast<double>(length);
    proven = std::max(proven, length);
    ++result.report.substeps;
    result.report.refinements += attempt.refinements;
    result.report.max_rows = std::max<std::size_t>(result.report.max_rows, attempt.rows);
    result.report.max_overlap = attempt.max_overlap;
    // A substep that settled in few rounds has room to lengthen, unless
    // doubling it would turn a cell further than planned.
    if (attempt.refinements <= easy_refinements &&
        2.0F * attempt.rotation <= planned_substep_rotation) {
      length = std::min(longest, length * 2.0F);
    }
  }
  step.finish(state, result);
  return result;
}

}  // namespace detail
}  // namespace microsimulator
