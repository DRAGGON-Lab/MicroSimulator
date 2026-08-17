#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cm/simulation.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, module) {
  module.doc() = "MicroSimulator native simulation core";

  module.def("backend_device_count", &cm::backend_device_count, "backend"_a);
  module.def("backend_available", &cm::backend_available, "backend"_a, "device_index"_a = 0);

  nb::enum_<cm::BackendKind>(module, "BackendKind")
      .value("CPU", cm::BackendKind::cpu)
      .value("METAL", cm::BackendKind::metal)
      .value("CUDA", cm::BackendKind::cuda);

  nb::enum_<cm::BackendFeature>(module, "BackendFeature")
      .value("GROWTH", cm::BackendFeature::growth)
      .value("SPECIES", cm::BackendFeature::species)
      .value("CELL_CONTACTS", cm::BackendFeature::cell_contacts)
      .value("CELL_MECHANICS", cm::BackendFeature::cell_mechanics)
      .value("EXTERNAL_CONSTRAINTS", cm::BackendFeature::external_constraints)
      .value("SIGNALS", cm::BackendFeature::signals)
      .value("COUPLED_RATES", cm::BackendFeature::coupled_rates);

  nb::enum_<cm::GridBoundaryKind>(module, "GridBoundaryKind")
      .value("NO_FLUX", cm::GridBoundaryKind::no_flux)
      .value("PERIODIC", cm::GridBoundaryKind::periodic)
      .value("FIXED", cm::GridBoundaryKind::fixed);

  nb::enum_<cm::SignalIntegrationKind>(module, "SignalIntegrationKind")
      .value("FORWARD_EULER", cm::SignalIntegrationKind::forward_euler)
      .value("CRANK_NICOLSON", cm::SignalIntegrationKind::crank_nicolson);

  nb::enum_<cm::RateOp>(module, "RateOp")
      .value("CONSTANT", cm::RateOp::constant)
      .value("SPECIES", cm::RateOp::species)
      .value("POSITION_X", cm::RateOp::position_x)
      .value("POSITION_Y", cm::RateOp::position_y)
      .value("POSITION_Z", cm::RateOp::position_z)
      .value("CELL_LENGTH", cm::RateOp::cell_length)
      .value("CELL_RADIUS", cm::RateOp::cell_radius)
      .value("GROWTH_RATE", cm::RateOp::growth_rate)
      .value("CELL_TYPE", cm::RateOp::cell_type)
      .value("CELL_VOLUME", cm::RateOp::cell_volume)
      .value("CELL_VOLUME_CHANGE_RATE", cm::RateOp::cell_volume_change_rate)
      .value("CELL_SURFACE_AREA", cm::RateOp::cell_surface_area)
      .value("ADD", cm::RateOp::add)
      .value("SUBTRACT", cm::RateOp::subtract)
      .value("MULTIPLY", cm::RateOp::multiply)
      .value("DIVIDE", cm::RateOp::divide)
      .value("POWER", cm::RateOp::power)
      .value("MINIMUM", cm::RateOp::minimum)
      .value("MAXIMUM", cm::RateOp::maximum)
      .value("NEGATE", cm::RateOp::negate)
      .value("EXPONENTIAL", cm::RateOp::exponential)
      .value("LOGARITHM", cm::RateOp::logarithm)
      .value("LESS", cm::RateOp::less)
      .value("LESS_EQUAL", cm::RateOp::less_equal)
      .value("GREATER", cm::RateOp::greater)
      .value("GREATER_EQUAL", cm::RateOp::greater_equal)
      .value("EQUAL", cm::RateOp::equal)
      .value("SELECT", cm::RateOp::select)
      .value("SIGNAL", cm::RateOp::signal);

  nb::enum_<cm::ConstraintRegion>(module, "ConstraintRegion")
      .value("OUTSIDE", cm::ConstraintRegion::outside)
      .value("INSIDE", cm::ConstraintRegion::inside);
  module.attr("SphereRegion") = module.attr("ConstraintRegion");

  nb::enum_<cm::ExternalConstraintKind>(module, "ExternalConstraintKind")
      .value("PLANE", cm::ExternalConstraintKind::plane)
      .value("SPHERE", cm::ExternalConstraintKind::sphere)
      .value("BOX", cm::ExternalConstraintKind::box);

  nb::enum_<cm::RodEndpoint>(module, "RodEndpoint")
      .value("NEGATIVE", cm::RodEndpoint::negative)
      .value("POSITIVE", cm::RodEndpoint::positive);

  nb::enum_<cm::SolverStatus>(module, "SolverStatus")
      .value("CONVERGED", cm::SolverStatus::converged)
      .value("ITERATION_LIMIT", cm::SolverStatus::iteration_limit)
      .value("BREAKDOWN", cm::SolverStatus::breakdown);

  nb::enum_<cm::SolverBreakdown>(module, "SolverBreakdown")
      .value("NONE", cm::SolverBreakdown::none)
      .value("NON_FINITE_RESIDUAL", cm::SolverBreakdown::non_finite_residual)
      .value("NON_FINITE_CURVATURE", cm::SolverBreakdown::non_finite_curvature)
      .value("NON_POSITIVE_CURVATURE", cm::SolverBreakdown::non_positive_curvature);

  nb::class_<cm::Vec3>(module, "Vec3")
      .def(nb::init<float, float, float>(), "x"_a = 0.0F, "y"_a = 0.0F, "z"_a = 0.0F)
      .def_rw("x", &cm::Vec3::x)
      .def_rw("y", &cm::Vec3::y)
      .def_rw("z", &cm::Vec3::z);

  nb::class_<cm::BackendInfo>(module, "BackendInfo")
      .def_ro("kind", &cm::BackendInfo::kind)
      .def_ro("name", &cm::BackendInfo::name)
      .def_ro("device", &cm::BackendInfo::device)
      .def_ro("device_index", &cm::BackendInfo::device_index)
      .def_ro("native", &cm::BackendInfo::native);

  nb::class_<cm::GridBoundary>(module, "GridBoundary")
      .def(nb::init<>())
      .def_rw("kind", &cm::GridBoundary::kind)
      .def_rw("values", &cm::GridBoundary::values)
      .def("validate", &cm::GridBoundary::validate, "signal_count"_a);

  nb::class_<cm::GridShape>(module, "GridShape")
      .def(nb::init<>())
      .def_rw("x", &cm::GridShape::x)
      .def_rw("y", &cm::GridShape::y)
      .def_rw("z", &cm::GridShape::z);

  nb::class_<cm::SignalSolveParameters>(module, "SignalSolveParameters")
      .def(nb::init<>())
      .def_rw("max_iterations", &cm::SignalSolveParameters::max_iterations)
      .def_rw("absolute_tolerance", &cm::SignalSolveParameters::absolute_tolerance)
      .def_rw("relative_tolerance", &cm::SignalSolveParameters::relative_tolerance)
      .def("validate", &cm::SignalSolveParameters::validate);

  nb::class_<cm::SignalSolveReport>(module, "SignalSolveReport")
      .def_ro("converged", &cm::SignalSolveReport::converged)
      .def_ro("iterations", &cm::SignalSolveReport::iterations)
      .def_ro("residual_rms", &cm::SignalSolveReport::residual_rms);

  nb::class_<cm::SignalGridAffineReaction>(module, "SignalGridAffineReaction")
      .def(nb::init<>())
      .def_rw("source_rates", &cm::SignalGridAffineReaction::source_rates)
      .def_rw("loss_rates", &cm::SignalGridAffineReaction::loss_rates)
      .def("validate", &cm::SignalGridAffineReaction::validate, "level_count"_a);

  nb::class_<cm::SignalGridSpec>(module, "SignalGridSpec")
      .def(nb::init<>())
      .def_rw("signal_count", &cm::SignalGridSpec::signal_count)
      .def_rw("shape", &cm::SignalGridSpec::shape)
      .def_rw("origin", &cm::SignalGridSpec::origin)
      .def_rw("spacing", &cm::SignalGridSpec::spacing)
      .def_rw("diffusion", &cm::SignalGridSpec::diffusion)
      .def_rw("advection", &cm::SignalGridSpec::advection)
      .def_rw("reaction", &cm::SignalGridSpec::reaction)
      .def_rw("obstacles", &cm::SignalGridSpec::obstacles)
      .def_rw("integration", &cm::SignalGridSpec::integration)
      .def_rw("solver", &cm::SignalGridSpec::solver)
      .def_rw("x_lower", &cm::SignalGridSpec::x_lower)
      .def_rw("x_upper", &cm::SignalGridSpec::x_upper)
      .def_rw("y_lower", &cm::SignalGridSpec::y_lower)
      .def_rw("y_upper", &cm::SignalGridSpec::y_upper)
      .def_rw("z_lower", &cm::SignalGridSpec::z_lower)
      .def_rw("z_upper", &cm::SignalGridSpec::z_upper)
      .def_prop_ro("site_count", &cm::SignalGridSpec::site_count)
      .def_prop_ro("level_count", &cm::SignalGridSpec::level_count)
      .def_prop_ro("voxel_volume", &cm::SignalGridSpec::voxel_volume)
      .def("validate", &cm::SignalGridSpec::validate);

  nb::class_<cm::SignalGridCheckpoint>(module, "_SignalGridCheckpoint")
      .def(nb::init<>())
      .def_rw("spec", &cm::SignalGridCheckpoint::spec)
      .def_rw("levels", &cm::SignalGridCheckpoint::levels)
      .def("validate", &cm::SignalGridCheckpoint::validate);

  nb::class_<cm::CellInit>(module, "CellInit")
      .def(nb::init<>())
      .def_rw("position", &cm::CellInit::position)
      .def_rw("direction", &cm::CellInit::direction)
      .def_rw("length", &cm::CellInit::length)
      .def_rw("radius", &cm::CellInit::radius)
      .def_rw("growth_rate", &cm::CellInit::growth_rate)
      .def_rw("cell_type", &cm::CellInit::cell_type)
      .def_rw("fixed", &cm::CellInit::fixed)
      .def_rw("species", &cm::CellInit::species);

  nb::class_<cm::CellSnapshot>(module, "CellSnapshot")
      .def(nb::init<>())
      .def_rw("id", &cm::CellSnapshot::id)
      .def_rw("slot", &cm::CellSnapshot::slot)
      .def_rw("position", &cm::CellSnapshot::position)
      .def_rw("direction", &cm::CellSnapshot::direction)
      .def_rw("length", &cm::CellSnapshot::length)
      .def_rw("radius", &cm::CellSnapshot::radius)
      .def_rw("growth_rate", &cm::CellSnapshot::growth_rate)
      .def_rw("cell_type", &cm::CellSnapshot::cell_type)
      .def_rw("fixed", &cm::CellSnapshot::fixed)
      .def_rw("species", &cm::CellSnapshot::species);

  nb::class_<cm::LineageEntry>(module, "_LineageEntry")
      .def(nb::init<>())
      .def_rw("child", &cm::LineageEntry::child)
      .def_rw("parent", &cm::LineageEntry::parent);

  nb::class_<cm::WorldStateCheckpoint>(module, "_WorldStateCheckpoint")
      .def(nb::init<>())
      .def_rw("species_count", &cm::WorldStateCheckpoint::species_count)
      .def_rw("next_id", &cm::WorldStateCheckpoint::next_id)
      .def_rw("cells", &cm::WorldStateCheckpoint::cells)
      .def_rw("lineage", &cm::WorldStateCheckpoint::lineage)
      .def("validate", &cm::WorldStateCheckpoint::validate);

  nb::class_<cm::RateInstruction>(module, "RateInstruction")
      .def(nb::init<>())
      .def_rw("operation", &cm::RateInstruction::operation)
      .def_rw("first", &cm::RateInstruction::first)
      .def_rw("second", &cm::RateInstruction::second)
      .def_rw("third", &cm::RateInstruction::third)
      .def_rw("value", &cm::RateInstruction::value);

  nb::class_<cm::SpeciesRatePlan>(module, "SpeciesRatePlan")
      .def(nb::init<std::size_t, std::vector<cm::RateInstruction>, std::vector<std::uint32_t>>(),
           "species_count"_a, "instructions"_a, "outputs"_a)
      .def_static("zero", &cm::SpeciesRatePlan::zero, "species_count"_a)
      .def_prop_ro("species_count", &cm::SpeciesRatePlan::species_count)
      .def_prop_ro("instructions",
                   [](const cm::SpeciesRatePlan& plan) {
                     return std::vector<cm::RateInstruction>(plan.instructions().begin(),
                                                             plan.instructions().end());
                   })
      .def_prop_ro("outputs",
                   [](const cm::SpeciesRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.outputs().begin(),
                                                       plan.outputs().end());
                   })
      .def("validate", &cm::SpeciesRatePlan::validate);

  nb::class_<cm::CoupledRatePlan>(module, "CoupledRatePlan")
      .def(nb::init<std::size_t, std::size_t, std::vector<cm::RateInstruction>,
                    std::vector<std::uint32_t>, std::vector<std::uint32_t>>(),
           "species_count"_a, "signal_count"_a, "instructions"_a, "species_outputs"_a,
           "signal_outputs"_a)
      .def_prop_ro("species_count", &cm::CoupledRatePlan::species_count)
      .def_prop_ro("signal_count", &cm::CoupledRatePlan::signal_count)
      .def_prop_ro("instructions",
                   [](const cm::CoupledRatePlan& plan) {
                     return std::vector<cm::RateInstruction>(plan.instructions().begin(),
                                                             plan.instructions().end());
                   })
      .def_prop_ro("species_outputs",
                   [](const cm::CoupledRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.species_outputs().begin(),
                                                       plan.species_outputs().end());
                   })
      .def_prop_ro("signal_outputs",
                   [](const cm::CoupledRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.signal_outputs().begin(),
                                                       plan.signal_outputs().end());
                   })
      .def("validate", &cm::CoupledRatePlan::validate);

  nb::class_<cm::ContactParameters>(module, "ContactParameters")
      .def(nb::init<>())
      .def_rw("activation_margin", &cm::ContactParameters::activation_margin)
      .def_rw("parallel_sine_threshold", &cm::ContactParameters::parallel_sine_threshold)
      .def_rw("degeneracy_epsilon", &cm::ContactParameters::degeneracy_epsilon);

  nb::class_<cm::CellContact>(module, "CellContact")
      .def_ro("first_id", &cm::CellContact::first_id)
      .def_ro("second_id", &cm::CellContact::second_id)
      .def_ro("first_slot", &cm::CellContact::first_slot)
      .def_ro("second_slot", &cm::CellContact::second_slot)
      .def_ro("ordinal", &cm::CellContact::ordinal)
      .def_ro("point_on_first", &cm::CellContact::point_on_first)
      .def_ro("normal", &cm::CellContact::normal)
      .def_ro("signed_separation", &cm::CellContact::signed_separation)
      .def_ro("weight", &cm::CellContact::weight);

  nb::class_<cm::ContactGraph>(module, "ContactGraph")
      .def_prop_ro("cell_count", &cm::ContactGraph::cell_count)
      .def_prop_ro("empty", &cm::ContactGraph::empty)
      .def_prop_ro("contacts",
                   [](const cm::ContactGraph& graph) {
                     return std::vector<cm::CellContact>(graph.contacts().begin(),
                                                         graph.contacts().end());
                   })
      .def("__len__", &cm::ContactGraph::size)
      .def(
          "incident_contact_indices",
          [](const cm::ContactGraph& graph, cm::Slot slot) {
            const auto indices = graph.incident_contact_indices(slot);
            return std::vector<std::size_t>(indices.begin(), indices.end());
          },
          "slot"_a)
      .def(
          "neighbor_ids",
          [](const cm::ContactGraph& graph, cm::Slot slot) {
            const auto ids = graph.neighbor_ids(slot);
            return std::vector<cm::CellId>(ids.begin(), ids.end());
          },
          "slot"_a);

  nb::class_<cm::PlaneConstraintInit>(module, "PlaneConstraintInit")
      .def(nb::init<>())
      .def_rw("point", &cm::PlaneConstraintInit::point)
      .def_rw("inward_normal", &cm::PlaneConstraintInit::inward_normal)
      .def_rw("coefficient", &cm::PlaneConstraintInit::coefficient);

  nb::class_<cm::SphereConstraintInit>(module, "SphereConstraintInit")
      .def(nb::init<>())
      .def_rw("center", &cm::SphereConstraintInit::center)
      .def_rw("radius", &cm::SphereConstraintInit::radius)
      .def_rw("coefficient", &cm::SphereConstraintInit::coefficient)
      .def_rw("allowed_region", &cm::SphereConstraintInit::allowed_region);

  nb::class_<cm::BoxConstraintInit>(module, "BoxConstraintInit")
      .def(nb::init<>())
      .def_rw("center", &cm::BoxConstraintInit::center)
      .def_rw("half_extents", &cm::BoxConstraintInit::half_extents)
      .def_rw("coefficient", &cm::BoxConstraintInit::coefficient)
      .def_rw("allowed_region", &cm::BoxConstraintInit::allowed_region);

  nb::class_<cm::PlaneConstraint>(module, "_PlaneConstraint")
      .def(nb::init<>())
      .def_rw("id", &cm::PlaneConstraint::id)
      .def_rw("point", &cm::PlaneConstraint::point)
      .def_rw("inward_normal", &cm::PlaneConstraint::inward_normal)
      .def_rw("coefficient", &cm::PlaneConstraint::coefficient);

  nb::class_<cm::SphereConstraint>(module, "_SphereConstraint")
      .def(nb::init<>())
      .def_rw("id", &cm::SphereConstraint::id)
      .def_rw("center", &cm::SphereConstraint::center)
      .def_rw("radius", &cm::SphereConstraint::radius)
      .def_rw("coefficient", &cm::SphereConstraint::coefficient)
      .def_rw("allowed_region", &cm::SphereConstraint::allowed_region);

  nb::class_<cm::BoxConstraint>(module, "_BoxConstraint")
      .def(nb::init<>())
      .def_rw("id", &cm::BoxConstraint::id)
      .def_rw("center", &cm::BoxConstraint::center)
      .def_rw("half_extents", &cm::BoxConstraint::half_extents)
      .def_rw("coefficient", &cm::BoxConstraint::coefficient)
      .def_rw("allowed_region", &cm::BoxConstraint::allowed_region);

  nb::class_<cm::ConstraintSetCheckpoint>(module, "_ConstraintSetCheckpoint")
      .def(nb::init<>())
      .def_rw("next_id", &cm::ConstraintSetCheckpoint::next_id)
      .def_rw("planes", &cm::ConstraintSetCheckpoint::planes)
      .def_rw("spheres", &cm::ConstraintSetCheckpoint::spheres)
      .def_rw("boxes", &cm::ConstraintSetCheckpoint::boxes)
      .def("validate", &cm::ConstraintSetCheckpoint::validate);

  nb::class_<cm::SimulationCheckpoint>(module, "_SimulationCheckpoint")
      .def(nb::init<>())
      .def_rw("schema_version", &cm::SimulationCheckpoint::schema_version)
      .def_rw("time", &cm::SimulationCheckpoint::time)
      .def_rw("world", &cm::SimulationCheckpoint::world)
      .def_rw("constraints", &cm::SimulationCheckpoint::constraints)
      .def_rw("species_rate_plan", &cm::SimulationCheckpoint::species_rate_plan)
      .def_rw("signal_grid", &cm::SimulationCheckpoint::signal_grid)
      .def_rw("coupled_rate_plan", &cm::SimulationCheckpoint::coupled_rate_plan)
      .def("validate", &cm::SimulationCheckpoint::validate);

  nb::class_<cm::ConstraintContactParameters>(module, "ConstraintContactParameters")
      .def(nb::init<>())
      .def_rw("activation_margin", &cm::ConstraintContactParameters::activation_margin)
      .def_rw("degeneracy_epsilon", &cm::ConstraintContactParameters::degeneracy_epsilon);

  nb::class_<cm::ExternalContact>(module, "ExternalContact")
      .def_ro("cell_id", &cm::ExternalContact::cell_id)
      .def_ro("cell_slot", &cm::ExternalContact::cell_slot)
      .def_ro("constraint_id", &cm::ExternalContact::constraint_id)
      .def_ro("constraint_kind", &cm::ExternalContact::constraint_kind)
      .def_ro("endpoint", &cm::ExternalContact::endpoint)
      .def_ro("point_on_cell", &cm::ExternalContact::point_on_cell)
      .def_ro("normal", &cm::ExternalContact::normal)
      .def_ro("signed_separation", &cm::ExternalContact::signed_separation)
      .def_ro("weight", &cm::ExternalContact::weight);

  nb::class_<cm::ExternalContactGraph>(module, "ExternalContactGraph")
      .def_prop_ro("cell_count", &cm::ExternalContactGraph::cell_count)
      .def_prop_ro("empty", &cm::ExternalContactGraph::empty)
      .def_prop_ro("contacts",
                   [](const cm::ExternalContactGraph& graph) {
                     return std::vector<cm::ExternalContact>(graph.contacts().begin(),
                                                             graph.contacts().end());
                   })
      .def("__len__", &cm::ExternalContactGraph::size)
      .def(
          "incident_contact_indices",
          [](const cm::ExternalContactGraph& graph, cm::Slot slot) {
            const auto indices = graph.incident_contact_indices(slot);
            return std::vector<std::size_t>(indices.begin(), indices.end());
          },
          "slot"_a);

  nb::class_<cm::CellCorrection>(module, "CellCorrection")
      .def_ro("translation", &cm::CellCorrection::translation)
      .def_ro("rotation", &cm::CellCorrection::rotation)
      .def_ro("length", &cm::CellCorrection::length);

  nb::class_<cm::MechanicsParameters>(module, "MechanicsParameters")
      .def(nb::init<>())
      .def_rw("mu_a", &cm::MechanicsParameters::mu_a)
      .def_rw("gamma", &cm::MechanicsParameters::gamma)
      .def_rw("residual_rms_tolerance", &cm::MechanicsParameters::residual_rms_tolerance)
      .def_rw("max_iterations", &cm::MechanicsParameters::max_iterations);

  nb::class_<cm::MechanicsIntegrationParameters>(module, "MechanicsIntegrationParameters")
      .def(nb::init<>())
      .def_rw("max_rotation_radians", &cm::MechanicsIntegrationParameters::max_rotation_radians)
      .def_rw("require_convergence", &cm::MechanicsIntegrationParameters::require_convergence);

  nb::class_<cm::SolverReport>(module, "SolverReport")
      .def_ro("status", &cm::SolverReport::status)
      .def_ro("breakdown", &cm::SolverReport::breakdown)
      .def_ro("iterations", &cm::SolverReport::iterations)
      .def_ro("initial_residual_rms", &cm::SolverReport::initial_residual_rms)
      .def_ro("final_residual_rms", &cm::SolverReport::final_residual_rms);

  nb::class_<cm::MechanicsSolveResult>(module, "MechanicsSolveResult")
      .def_ro("corrections", &cm::MechanicsSolveResult::corrections)
      .def_ro("report", &cm::MechanicsSolveResult::report);

  nb::class_<cm::Simulation>(module, "Simulation")
      .def(nb::init<cm::BackendKind, std::size_t, std::size_t, std::uint32_t>(),
           "backend"_a = cm::BackendKind::cpu, "reserved_capacity"_a = 0, "species_count"_a = 0,
           "device_index"_a = 0)
      .def(nb::init<cm::BackendKind, const cm::SimulationCheckpoint&, std::uint32_t>(), "backend"_a,
           "checkpoint"_a, "device_index"_a = 0)
      .def_prop_ro("backend_info", &cm::Simulation::backend_info)
      .def("supports", &cm::Simulation::supports, "feature"_a)
      .def_prop_ro("time", &cm::Simulation::time)
      .def_prop_ro("cell_count", &cm::Simulation::cell_count)
      .def_prop_ro("species_count", &cm::Simulation::species_count)
      .def_prop_ro("signal_count", &cm::Simulation::signal_count)
      .def_prop_ro("has_signal_grid", &cm::Simulation::has_signal_grid)
      .def_prop_ro("last_signal_solve_report", &cm::Simulation::last_signal_solve_report)
      .def_prop_ro("has_coupled_rate_plan", &cm::Simulation::has_coupled_rate_plan)
      .def("add_cell", &cm::Simulation::add_cell, "cell"_a)
      .def("add_plane_constraint", &cm::Simulation::add_plane_constraint, "plane"_a)
      .def("add_sphere_constraint", &cm::Simulation::add_sphere_constraint, "sphere"_a)
      .def("add_box_constraint", &cm::Simulation::add_box_constraint, "box"_a)
      .def("set_cell_geometry", &cm::Simulation::set_cell_geometry, "id"_a, "position"_a,
           "direction"_a, "length"_a)
      .def("set_cell_attributes", &cm::Simulation::set_cell_attributes, "id"_a, "growth_rate"_a,
           "cell_type"_a)
      .def("set_cell_fixed", &cm::Simulation::set_cell_fixed, "id"_a, "fixed"_a)
      .def(
          "set_species",
          [](cm::Simulation& simulation, cm::CellId id, const std::vector<float>& levels) {
            simulation.set_species(id, levels);
          },
          "id"_a, "levels"_a)
      .def("set_species_rate_plan", &cm::Simulation::set_species_rate_plan, "plan"_a)
      .def("set_coupled_rate_plan", &cm::Simulation::set_coupled_rate_plan, "plan"_a)
      .def("clear_coupled_rate_plan", &cm::Simulation::clear_coupled_rate_plan)
      .def("configure_signal_grid", &cm::Simulation::configure_signal_grid, "spec"_a,
           "levels"_a = std::vector<float>{})
      .def(
          "set_signal_levels",
          [](cm::Simulation& simulation, const std::vector<float>& levels) {
            simulation.set_signal_levels(levels);
          },
          "levels"_a)
      .def("divide", &cm::Simulation::divide, "parent_id"_a, "first_fraction"_a)
      .def("divide_equal", &cm::Simulation::divide_equal, "parent_id"_a)
      .def("step", &cm::Simulation::step, "dt"_a)
      .def("find_cell_contacts", &cm::Simulation::find_cell_contacts,
           "parameters"_a = cm::ContactParameters{})
      .def("find_external_contacts", &cm::Simulation::find_external_contacts,
           "parameters"_a = cm::ConstraintContactParameters{})
      .def("solve_cell_mechanics", &cm::Simulation::solve_cell_mechanics,
           "mechanics_parameters"_a = cm::MechanicsParameters{},
           "contact_parameters"_a = cm::ContactParameters{},
           "constraint_parameters"_a = cm::ConstraintContactParameters{})
      .def("relax_cell_mechanics", &cm::Simulation::relax_cell_mechanics,
           "mechanics_parameters"_a = cm::MechanicsParameters{},
           "contact_parameters"_a = cm::ContactParameters{},
           "integration_parameters"_a = cm::MechanicsIntegrationParameters{},
           "constraint_parameters"_a = cm::ConstraintContactParameters{})
      .def("cell", &cm::Simulation::cell, "id"_a)
      .def("cells", &cm::Simulation::cells)
      .def("lineage_parent", &cm::Simulation::lineage_parent, "id"_a)
      .def_prop_ro("signal_levels", &cm::Simulation::signal_levels)
      .def("sample_signals", &cm::Simulation::sample_signals, "position"_a)
      .def("_checkpoint", &cm::Simulation::checkpoint)
      .def("validate", &cm::Simulation::validate);
}
