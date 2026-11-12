#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "microsimulator/types.hpp"
#include "microsimulator/world_state.hpp"

namespace microsimulator {

struct ContactParameters {
  float activation_margin{0.01F};
  float parallel_sine_threshold{0.1F};
  float degeneracy_epsilon{1.0e-6F};
};

struct CellContact {
  CellId first_id{invalid_cell_id};
  CellId second_id{invalid_cell_id};
  Slot first_slot{invalid_slot};
  Slot second_slot{invalid_slot};
  std::uint8_t ordinal{0};
  Vec3 point_on_first{};
  Vec3 normal{};
  float signed_separation{0.0F};
  float weight{1.0F};
};

struct ContactCandidate {
  Slot first_slot{invalid_slot};
  Slot second_slot{invalid_slot};
};

void validate_contact_parameters(const ContactParameters& parameters);

[[nodiscard]] std::vector<ContactCandidate> find_cell_contact_candidates(
    const WorldState& state, const ContactParameters& parameters = ContactParameters{});

class ContactGraph {
 public:
  ContactGraph() = default;
  ContactGraph(std::size_t cell_count, std::vector<CellContact> contacts);

  [[nodiscard]] std::size_t cell_count() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] std::span<const CellContact> contacts() const& noexcept;
  [[nodiscard]] std::span<const CellContact> contacts() && = delete;
  [[nodiscard]] std::span<const std::size_t> incident_contact_indices(Slot slot) const&;
  [[nodiscard]] std::span<const std::size_t> incident_contact_indices(Slot slot) && = delete;
  [[nodiscard]] std::span<const CellId> neighbor_ids(Slot slot) const&;
  [[nodiscard]] std::span<const CellId> neighbor_ids(Slot slot) && = delete;

 private:
  std::size_t cell_count_{0};
  std::vector<CellContact> contacts_;
  std::vector<std::size_t> incidence_offsets_{0};
  std::vector<std::size_t> incidence_contact_indices_;
  std::vector<std::size_t> neighbor_offsets_{0};
  std::vector<CellId> neighbor_ids_;
};

[[nodiscard]] ContactGraph find_cell_contacts_cpu(
    const WorldState& state, const ContactParameters& parameters = ContactParameters{});
[[nodiscard]] ContactGraph find_cell_contacts_cpu_exhaustive(
    const WorldState& state, const ContactParameters& parameters = ContactParameters{});

}  // namespace microsimulator
