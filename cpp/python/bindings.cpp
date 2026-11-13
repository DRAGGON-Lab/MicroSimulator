#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "microsimulator/simulation.hpp"
#include "native_call.hpp"

namespace nb = nanobind;
using namespace nb::literals;
void bind_occupancy(nb::module_& module);
void bind_serialization(nb::module_& module);

NB_MODULE(_core, module) {
  module.doc() = "MicroSimulator native simulation core";

  module.def("backend_device_count", &microsimulator::backend_device_count, "backend"_a);
  module.def("backend_available", &microsimulator::backend_available, "backend"_a,
             "device_index"_a = 0);

  nb::enum_<microsimulator::BackendKind>(module, "BackendKind")
      .value("CPU", microsimulator::BackendKind::cpu)
      .value("METAL", microsimulator::BackendKind::metal)
      .value("CUDA", microsimulator::BackendKind::cuda);

  bind_occupancy(module);

  nb::enum_<microsimulator::BackendFeature>(module, "BackendFeature")
      .value("GROWTH", microsimulator::BackendFeature::growth)
      .value("SPECIES", microsimulator::BackendFeature::species)
      .value("CELL_CONTACTS", microsimulator::BackendFeature::cell_contacts)
      .value("CELL_MECHANICS", microsimulator::BackendFeature::cell_mechanics)
      .value("EXTERNAL_CONSTRAINTS", microsimulator::BackendFeature::external_constraints)
      .value("SIGNALS", microsimulator::BackendFeature::signals)
      .value("COUPLED_RATES", microsimulator::BackendFeature::coupled_rates)
      .value("DEPTH_AVERAGED_FLOW", microsimulator::BackendFeature::depth_averaged_flow)
      .value("RESOLVED_FLOW", microsimulator::BackendFeature::resolved_flow);

  nb::enum_<microsimulator::FlowAxis>(module, "FlowAxis")
      .value("X", microsimulator::FlowAxis::x)
      .value("Y", microsimulator::FlowAxis::y)
      .value("Z", microsimulator::FlowAxis::z);

  nb::enum_<microsimulator::GridBoundaryKind>(module, "GridBoundaryKind")
      .value("NO_FLUX", microsimulator::GridBoundaryKind::no_flux)
      .value("PERIODIC", microsimulator::GridBoundaryKind::periodic)
      .value("FIXED", microsimulator::GridBoundaryKind::fixed);
  nb::enum_<microsimulator::GridFace>(module, "GridFace")
      .value("X_LOWER", microsimulator::GridFace::x_lower)
      .value("X_UPPER", microsimulator::GridFace::x_upper)
      .value("Y_LOWER", microsimulator::GridFace::y_lower)
      .value("Y_UPPER", microsimulator::GridFace::y_upper)
      .value("Z_LOWER", microsimulator::GridFace::z_lower)
      .value("Z_UPPER", microsimulator::GridFace::z_upper);

  nb::enum_<microsimulator::SignalIntegrationKind>(module, "SignalIntegrationKind")
      .value("FORWARD_EULER", microsimulator::SignalIntegrationKind::forward_euler)
      .value("CRANK_NICOLSON", microsimulator::SignalIntegrationKind::crank_nicolson)
      .value("BACKWARD_EULER", microsimulator::SignalIntegrationKind::backward_euler);

  nb::enum_<microsimulator::RateOp>(module, "RateOp")
      .value("CONSTANT", microsimulator::RateOp::constant)
      .value("SPECIES", microsimulator::RateOp::species)
      .value("POSITION_X", microsimulator::RateOp::position_x)
      .value("POSITION_Y", microsimulator::RateOp::position_y)
      .value("POSITION_Z", microsimulator::RateOp::position_z)
      .value("CELL_LENGTH", microsimulator::RateOp::cell_length)
      .value("CELL_RADIUS", microsimulator::RateOp::cell_radius)
      .value("GROWTH_RATE", microsimulator::RateOp::growth_rate)
      .value("CELL_TYPE", microsimulator::RateOp::cell_type)
      .value("CELL_VOLUME", microsimulator::RateOp::cell_volume)
      .value("CELL_VOLUME_CHANGE_RATE", microsimulator::RateOp::cell_volume_change_rate)
      .value("CELL_SURFACE_AREA", microsimulator::RateOp::cell_surface_area)
      .value("ADD", microsimulator::RateOp::add)
      .value("SUBTRACT", microsimulator::RateOp::subtract)
      .value("MULTIPLY", microsimulator::RateOp::multiply)
      .value("DIVIDE", microsimulator::RateOp::divide)
      .value("POWER", microsimulator::RateOp::power)
      .value("MINIMUM", microsimulator::RateOp::minimum)
      .value("MAXIMUM", microsimulator::RateOp::maximum)
      .value("NEGATE", microsimulator::RateOp::negate)
      .value("EXPONENTIAL", microsimulator::RateOp::exponential)
      .value("LOGARITHM", microsimulator::RateOp::logarithm)
      .value("LESS", microsimulator::RateOp::less)
      .value("LESS_EQUAL", microsimulator::RateOp::less_equal)
      .value("GREATER", microsimulator::RateOp::greater)
      .value("GREATER_EQUAL", microsimulator::RateOp::greater_equal)
      .value("EQUAL", microsimulator::RateOp::equal)
      .value("SELECT", microsimulator::RateOp::select)
      .value("SIGNAL", microsimulator::RateOp::signal);

  nb::enum_<microsimulator::ConstraintRegion>(module, "ConstraintRegion")
      .value("OUTSIDE", microsimulator::ConstraintRegion::outside)
      .value("INSIDE", microsimulator::ConstraintRegion::inside);
  module.attr("SphereRegion") = module.attr("ConstraintRegion");

  nb::enum_<microsimulator::ExternalConstraintKind>(module, "ExternalConstraintKind")
      .value("PLANE", microsimulator::ExternalConstraintKind::plane)
      .value("SPHERE", microsimulator::ExternalConstraintKind::sphere)
      .value("BOX", microsimulator::ExternalConstraintKind::box)
      .value("CYLINDER", microsimulator::ExternalConstraintKind::cylinder);

  nb::enum_<microsimulator::RodContactLocation>(module, "RodContactLocation")
      .value("NEGATIVE", microsimulator::RodContactLocation::negative)
      .value("POSITIVE", microsimulator::RodContactLocation::positive)
      .value("INTERIOR", microsimulator::RodContactLocation::interior);
  module.attr("RodEndpoint") = module.attr("RodContactLocation");

  nb::enum_<microsimulator::SolverStatus>(module, "SolverStatus")
      .value("CONVERGED", microsimulator::SolverStatus::converged)
      .value("ITERATION_LIMIT", microsimulator::SolverStatus::iteration_limit)
      .value("BREAKDOWN", microsimulator::SolverStatus::breakdown);

  nb::enum_<microsimulator::SolverBreakdown>(module, "SolverBreakdown")
      .value("NONE", microsimulator::SolverBreakdown::none)
      .value("NON_FINITE_RESIDUAL", microsimulator::SolverBreakdown::non_finite_residual)
      .value("NON_FINITE_CURVATURE", microsimulator::SolverBreakdown::non_finite_curvature)
      .value("NON_POSITIVE_CURVATURE", microsimulator::SolverBreakdown::non_positive_curvature);

  nb::class_<microsimulator::Vec3>(module, "Vec3")
      .def(nb::init<float, float, float>(), "x"_a = 0.0F, "y"_a = 0.0F, "z"_a = 0.0F)
      .def_rw("x", &microsimulator::Vec3::x)
      .def_rw("y", &microsimulator::Vec3::y)
      .def_rw("z", &microsimulator::Vec3::z);

  nb::class_<microsimulator::BackendInfo>(module, "BackendInfo")
      .def_ro("kind", &microsimulator::BackendInfo::kind)
      .def_ro("name", &microsimulator::BackendInfo::name)
      .def_ro("device", &microsimulator::BackendInfo::device)
      .def_ro("device_index", &microsimulator::BackendInfo::device_index)
      .def_ro("native", &microsimulator::BackendInfo::native);

  nb::class_<microsimulator::GridBoundary>(module, "GridBoundary")
      .def(nb::init<>())
      .def_rw("kind", &microsimulator::GridBoundary::kind)
      .def_rw("values", &microsimulator::GridBoundary::values)
      .def("validate", &microsimulator::GridBoundary::validate, "signal_count"_a);

  nb::class_<microsimulator::GridShape>(module, "GridShape")
      .def(nb::init<>())
      .def_rw("x", &microsimulator::GridShape::x)
      .def_rw("y", &microsimulator::GridShape::y)
      .def_rw("z", &microsimulator::GridShape::z);

  nb::class_<microsimulator::SignalSolveParameters>(module, "SignalSolveParameters")
      .def(nb::init<>())
      .def_rw("max_iterations", &microsimulator::SignalSolveParameters::max_iterations)
      .def_rw("absolute_tolerance", &microsimulator::SignalSolveParameters::absolute_tolerance)
      .def_rw("relative_tolerance", &microsimulator::SignalSolveParameters::relative_tolerance)
      .def("validate", &microsimulator::SignalSolveParameters::validate);

  nb::class_<microsimulator::SignalSolveReport>(module, "SignalSolveReport")
      .def_ro("converged", &microsimulator::SignalSolveReport::converged)
      .def_ro("iterations", &microsimulator::SignalSolveReport::iterations)
      .def_ro("residual_rms", &microsimulator::SignalSolveReport::residual_rms);

  nb::class_<microsimulator::SignalGridAffineReaction>(module, "SignalGridAffineReaction")
      .def(nb::init<>())
      .def_rw("source_rates", &microsimulator::SignalGridAffineReaction::source_rates)
      .def_rw("loss_rates", &microsimulator::SignalGridAffineReaction::loss_rates)
      .def("validate", &microsimulator::SignalGridAffineReaction::validate, "level_count"_a);

  nb::class_<microsimulator::SignalGridVelocityField>(module, "SignalGridVelocityField")
      .def(nb::init<>())
      .def_rw("x_faces", &microsimulator::SignalGridVelocityField::x_faces)
      .def_rw("y_faces", &microsimulator::SignalGridVelocityField::y_faces)
      .def_rw("z_faces", &microsimulator::SignalGridVelocityField::z_faces);

  nb::class_<microsimulator::DepthAveragedFlowParameters>(module, "DepthAveragedFlowParameters")
      .def(nb::init<>())
      .def_rw("mean_inlet_speed", &microsimulator::DepthAveragedFlowParameters::mean_inlet_speed)
      .def_rw("axis", &microsimulator::DepthAveragedFlowParameters::axis)
      .def_rw("relative_tolerance",
              &microsimulator::DepthAveragedFlowParameters::relative_tolerance)
      .def_rw("max_iterations", &microsimulator::DepthAveragedFlowParameters::max_iterations)
      .def("validate", &microsimulator::DepthAveragedFlowParameters::validate);

  nb::class_<microsimulator::DepthAveragedFlowReport>(module, "DepthAveragedFlowReport")
      .def_ro("iterations", &microsimulator::DepthAveragedFlowReport::iterations)
      .def_prop_ro("residual",
                   [](const microsimulator::DepthAveragedFlowReport& report) {
                     return report.relative_residual;
                   })
      .def_ro("relative_residual", &microsimulator::DepthAveragedFlowReport::relative_residual)
      .def_ro("mean_inlet_speed", &microsimulator::DepthAveragedFlowReport::mean_inlet_speed)
      .def_ro("max_speed", &microsimulator::DepthAveragedFlowReport::max_speed);

  nb::class_<microsimulator::DepthAveragedFlowResult>(module, "DepthAveragedFlowResult")
      .def_ro("field", &microsimulator::DepthAveragedFlowResult::field)
      .def_ro("report", &microsimulator::DepthAveragedFlowResult::report);

  nb::class_<microsimulator::ResolvedFlowParameters>(module, "ResolvedFlowParameters")
      .def(nb::init<>())
      .def_rw("mean_inlet_speed", &microsimulator::ResolvedFlowParameters::mean_inlet_speed)
      .def_rw("axis", &microsimulator::ResolvedFlowParameters::axis)
      .def_rw("relative_tolerance", &microsimulator::ResolvedFlowParameters::relative_tolerance)
      .def_rw("max_outer_iterations", &microsimulator::ResolvedFlowParameters::max_outer_iterations)
      .def_rw("inner_relative_tolerance",
              &microsimulator::ResolvedFlowParameters::inner_relative_tolerance)
      .def_rw("max_inner_iterations", &microsimulator::ResolvedFlowParameters::max_inner_iterations)
      .def("validate", &microsimulator::ResolvedFlowParameters::validate);

  nb::class_<microsimulator::ResolvedFlowReport>(module, "ResolvedFlowReport")
      .def_ro("outer_iterations", &microsimulator::ResolvedFlowReport::outer_iterations)
      .def_ro("inner_iterations", &microsimulator::ResolvedFlowReport::inner_iterations)
      .def_ro("relative_residual", &microsimulator::ResolvedFlowReport::relative_residual)
      .def_ro("momentum_relative_residual",
              &microsimulator::ResolvedFlowReport::momentum_relative_residual)
      .def_ro("divergence_rms", &microsimulator::ResolvedFlowReport::divergence_rms)
      .def_ro("mean_inlet_speed", &microsimulator::ResolvedFlowReport::mean_inlet_speed)
      .def_ro("max_speed", &microsimulator::ResolvedFlowReport::max_speed)
      .def_ro("min_gap_voxels", &microsimulator::ResolvedFlowReport::min_gap_voxels);

  nb::class_<microsimulator::ResolvedFlowResult>(module, "ResolvedFlowResult")
      .def_ro("field", &microsimulator::ResolvedFlowResult::field)
      .def_ro("report", &microsimulator::ResolvedFlowResult::report);

  nb::class_<microsimulator::SignalGridSpec>(module, "SignalGridSpec")
      .def(nb::init<>())
      .def_rw("signal_count", &microsimulator::SignalGridSpec::signal_count)
      .def_rw("shape", &microsimulator::SignalGridSpec::shape)
      .def_rw("origin", &microsimulator::SignalGridSpec::origin)
      .def_rw("spacing", &microsimulator::SignalGridSpec::spacing)
      .def_rw("diffusion", &microsimulator::SignalGridSpec::diffusion)
      .def_rw("advection", &microsimulator::SignalGridSpec::advection)
      .def_rw("reaction", &microsimulator::SignalGridSpec::reaction)
      .def_rw("obstacles", &microsimulator::SignalGridSpec::obstacles)
      .def_rw("velocity_field", &microsimulator::SignalGridSpec::velocity_field)
      .def_rw("integration", &microsimulator::SignalGridSpec::integration)
      .def_rw("solver", &microsimulator::SignalGridSpec::solver)
      .def_rw("x_lower", &microsimulator::SignalGridSpec::x_lower)
      .def_rw("x_upper", &microsimulator::SignalGridSpec::x_upper)
      .def_rw("y_lower", &microsimulator::SignalGridSpec::y_lower)
      .def_rw("y_upper", &microsimulator::SignalGridSpec::y_upper)
      .def_rw("z_lower", &microsimulator::SignalGridSpec::z_lower)
      .def_rw("z_upper", &microsimulator::SignalGridSpec::z_upper)
      .def_prop_ro("site_count", &microsimulator::SignalGridSpec::site_count)
      .def_prop_ro("level_count", &microsimulator::SignalGridSpec::level_count)
      .def_prop_ro("voxel_volume", &microsimulator::SignalGridSpec::voxel_volume)
      .def("validate", &microsimulator::SignalGridSpec::validate);

  nb::class_<microsimulator::SignalGridCheckpoint>(module, "_SignalGridCheckpoint")
      .def(nb::init<>())
      .def_rw("spec", &microsimulator::SignalGridCheckpoint::spec)
      .def_rw("levels", &microsimulator::SignalGridCheckpoint::levels)
      .def("validate", &microsimulator::SignalGridCheckpoint::validate);

  nb::class_<microsimulator::CellInit>(module, "CellInit")
      .def(nb::init<>())
      .def_rw("position", &microsimulator::CellInit::position)
      .def_rw("direction", &microsimulator::CellInit::direction)
      .def_rw("length", &microsimulator::CellInit::length)
      .def_rw("radius", &microsimulator::CellInit::radius)
      .def_rw("growth_rate", &microsimulator::CellInit::growth_rate)
      .def_rw("cell_type", &microsimulator::CellInit::cell_type)
      .def_rw("fixed", &microsimulator::CellInit::fixed)
      .def_rw("species", &microsimulator::CellInit::species);

  nb::class_<microsimulator::CellSnapshot>(module, "CellSnapshot")
      .def(nb::init<>())
      .def_rw("id", &microsimulator::CellSnapshot::id)
      .def_rw("slot", &microsimulator::CellSnapshot::slot)
      .def_rw("position", &microsimulator::CellSnapshot::position)
      .def_rw("direction", &microsimulator::CellSnapshot::direction)
      .def_rw("length", &microsimulator::CellSnapshot::length)
      .def_rw("radius", &microsimulator::CellSnapshot::radius)
      .def_rw("growth_rate", &microsimulator::CellSnapshot::growth_rate)
      .def_rw("cell_type", &microsimulator::CellSnapshot::cell_type)
      .def_rw("fixed", &microsimulator::CellSnapshot::fixed)
      .def_rw("species", &microsimulator::CellSnapshot::species);

  nb::class_<microsimulator::LineageEntry>(module, "_LineageEntry")
      .def(nb::init<>())
      .def_rw("child", &microsimulator::LineageEntry::child)
      .def_rw("parent", &microsimulator::LineageEntry::parent);

  nb::class_<microsimulator::WorldStateCheckpoint>(module, "_WorldStateCheckpoint")
      .def(nb::init<>())
      .def_rw("species_count", &microsimulator::WorldStateCheckpoint::species_count)
      .def_rw("next_id", &microsimulator::WorldStateCheckpoint::next_id)
      .def_rw("cells", &microsimulator::WorldStateCheckpoint::cells)
      .def_rw("lineage", &microsimulator::WorldStateCheckpoint::lineage)
      .def("validate", &microsimulator::WorldStateCheckpoint::validate);

  nb::class_<microsimulator::RateInstruction>(module, "RateInstruction")
      .def(nb::init<>())
      .def_rw("operation", &microsimulator::RateInstruction::operation)
      .def_rw("first", &microsimulator::RateInstruction::first)
      .def_rw("second", &microsimulator::RateInstruction::second)
      .def_rw("third", &microsimulator::RateInstruction::third)
      .def_rw("value", &microsimulator::RateInstruction::value);

  nb::class_<microsimulator::SpeciesRatePlan>(module, "SpeciesRatePlan")
      .def(nb::init<std::size_t, std::vector<microsimulator::RateInstruction>,
                    std::vector<std::uint32_t>>(),
           "species_count"_a, "instructions"_a, "outputs"_a)
      .def_static("zero", &microsimulator::SpeciesRatePlan::zero, "species_count"_a)
      .def_prop_ro("species_count", &microsimulator::SpeciesRatePlan::species_count)
      .def_prop_ro("instructions",
                   [](const microsimulator::SpeciesRatePlan& plan) {
                     return std::vector<microsimulator::RateInstruction>(
                         plan.instructions().begin(), plan.instructions().end());
                   })
      .def_prop_ro("outputs",
                   [](const microsimulator::SpeciesRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.outputs().begin(),
                                                       plan.outputs().end());
                   })
      .def("validate", &microsimulator::SpeciesRatePlan::validate);

  nb::class_<microsimulator::CoupledRatePlan>(module, "CoupledRatePlan")
      .def(nb::init<std::size_t, std::size_t, std::vector<microsimulator::RateInstruction>,
                    std::vector<std::uint32_t>, std::vector<std::uint32_t>>(),
           "species_count"_a, "signal_count"_a, "instructions"_a, "species_outputs"_a,
           "signal_outputs"_a)
      .def_prop_ro("species_count", &microsimulator::CoupledRatePlan::species_count)
      .def_prop_ro("signal_count", &microsimulator::CoupledRatePlan::signal_count)
      .def_prop_ro("instructions",
                   [](const microsimulator::CoupledRatePlan& plan) {
                     return std::vector<microsimulator::RateInstruction>(
                         plan.instructions().begin(), plan.instructions().end());
                   })
      .def_prop_ro("species_outputs",
                   [](const microsimulator::CoupledRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.species_outputs().begin(),
                                                       plan.species_outputs().end());
                   })
      .def_prop_ro("signal_outputs",
                   [](const microsimulator::CoupledRatePlan& plan) {
                     return std::vector<std::uint32_t>(plan.signal_outputs().begin(),
                                                       plan.signal_outputs().end());
                   })
      .def("validate", &microsimulator::CoupledRatePlan::validate);

  nb::class_<microsimulator::ContactParameters>(module, "ContactParameters")
      .def(nb::init<>())
      .def_rw("activation_margin", &microsimulator::ContactParameters::activation_margin)
      .def_rw("parallel_sine_threshold",
              &microsimulator::ContactParameters::parallel_sine_threshold)
      .def_rw("degeneracy_epsilon", &microsimulator::ContactParameters::degeneracy_epsilon);

  nb::class_<microsimulator::CellContact>(module, "CellContact")
      .def_ro("first_id", &microsimulator::CellContact::first_id)
      .def_ro("second_id", &microsimulator::CellContact::second_id)
      .def_ro("first_slot", &microsimulator::CellContact::first_slot)
      .def_ro("second_slot", &microsimulator::CellContact::second_slot)
      .def_ro("ordinal", &microsimulator::CellContact::ordinal)
      .def_ro("point_on_first", &microsimulator::CellContact::point_on_first)
      .def_ro("normal", &microsimulator::CellContact::normal)
      .def_ro("signed_separation", &microsimulator::CellContact::signed_separation)
      .def_ro("weight", &microsimulator::CellContact::weight);

  nb::class_<microsimulator::ContactGraph>(module, "ContactGraph")
      .def_prop_ro("cell_count", &microsimulator::ContactGraph::cell_count)
      .def_prop_ro("empty", &microsimulator::ContactGraph::empty)
      .def_prop_ro("contacts",
                   [](const microsimulator::ContactGraph& graph) {
                     return std::vector<microsimulator::CellContact>(graph.contacts().begin(),
                                                                     graph.contacts().end());
                   })
      .def("__len__", &microsimulator::ContactGraph::size)
      .def(
          "incident_contact_indices",
          [](const microsimulator::ContactGraph& graph, microsimulator::Slot slot) {
            const auto indices = graph.incident_contact_indices(slot);
            return std::vector<std::size_t>(indices.begin(), indices.end());
          },
          "slot"_a)
      .def(
          "neighbor_ids",
          [](const microsimulator::ContactGraph& graph, microsimulator::Slot slot) {
            const auto ids = graph.neighbor_ids(slot);
            return std::vector<microsimulator::CellId>(ids.begin(), ids.end());
          },
          "slot"_a);

  nb::class_<microsimulator::PlaneConstraintInit>(module, "PlaneConstraintInit")
      .def(nb::init<>())
      .def_rw("point", &microsimulator::PlaneConstraintInit::point)
      .def_rw("inward_normal", &microsimulator::PlaneConstraintInit::inward_normal)
      .def_rw("coefficient", &microsimulator::PlaneConstraintInit::coefficient);

  nb::class_<microsimulator::SphereConstraintInit>(module, "SphereConstraintInit")
      .def(nb::init<>())
      .def_rw("center", &microsimulator::SphereConstraintInit::center)
      .def_rw("radius", &microsimulator::SphereConstraintInit::radius)
      .def_rw("coefficient", &microsimulator::SphereConstraintInit::coefficient)
      .def_rw("allowed_region", &microsimulator::SphereConstraintInit::allowed_region);

  nb::class_<microsimulator::BoxConstraintInit>(module, "BoxConstraintInit")
      .def(nb::init<>())
      .def_rw("center", &microsimulator::BoxConstraintInit::center)
      .def_rw("half_extents", &microsimulator::BoxConstraintInit::half_extents)
      .def_rw("coefficient", &microsimulator::BoxConstraintInit::coefficient)
      .def_rw("allowed_region", &microsimulator::BoxConstraintInit::allowed_region);

  nb::class_<microsimulator::CylinderConstraintInit>(module, "CylinderConstraintInit")
      .def(nb::init<>())
      .def_rw("center", &microsimulator::CylinderConstraintInit::center)
      .def_rw("radius", &microsimulator::CylinderConstraintInit::radius)
      .def_rw("half_height", &microsimulator::CylinderConstraintInit::half_height)
      .def_rw("coefficient", &microsimulator::CylinderConstraintInit::coefficient)
      .def_rw("allowed_region", &microsimulator::CylinderConstraintInit::allowed_region);

  nb::class_<microsimulator::PlaneConstraint>(module, "_PlaneConstraint")
      .def(nb::init<>())
      .def_rw("id", &microsimulator::PlaneConstraint::id)
      .def_rw("point", &microsimulator::PlaneConstraint::point)
      .def_rw("inward_normal", &microsimulator::PlaneConstraint::inward_normal)
      .def_rw("coefficient", &microsimulator::PlaneConstraint::coefficient);

  nb::class_<microsimulator::SphereConstraint>(module, "_SphereConstraint")
      .def(nb::init<>())
      .def_rw("id", &microsimulator::SphereConstraint::id)
      .def_rw("center", &microsimulator::SphereConstraint::center)
      .def_rw("radius", &microsimulator::SphereConstraint::radius)
      .def_rw("coefficient", &microsimulator::SphereConstraint::coefficient)
      .def_rw("allowed_region", &microsimulator::SphereConstraint::allowed_region);

  nb::class_<microsimulator::BoxConstraint>(module, "_BoxConstraint")
      .def(nb::init<>())
      .def_rw("id", &microsimulator::BoxConstraint::id)
      .def_rw("center", &microsimulator::BoxConstraint::center)
      .def_rw("half_extents", &microsimulator::BoxConstraint::half_extents)
      .def_rw("coefficient", &microsimulator::BoxConstraint::coefficient)
      .def_rw("allowed_region", &microsimulator::BoxConstraint::allowed_region);

  nb::class_<microsimulator::CylinderConstraint>(module, "_CylinderConstraint")
      .def(nb::init<>())
      .def_rw("id", &microsimulator::CylinderConstraint::id)
      .def_rw("center", &microsimulator::CylinderConstraint::center)
      .def_rw("radius", &microsimulator::CylinderConstraint::radius)
      .def_rw("half_height", &microsimulator::CylinderConstraint::half_height)
      .def_rw("coefficient", &microsimulator::CylinderConstraint::coefficient)
      .def_rw("allowed_region", &microsimulator::CylinderConstraint::allowed_region);

  nb::class_<microsimulator::ConstraintSetCheckpoint>(module, "_ConstraintSetCheckpoint")
      .def(nb::init<>())
      .def_rw("next_id", &microsimulator::ConstraintSetCheckpoint::next_id)
      .def_rw("planes", &microsimulator::ConstraintSetCheckpoint::planes)
      .def_rw("spheres", &microsimulator::ConstraintSetCheckpoint::spheres)
      .def_rw("boxes", &microsimulator::ConstraintSetCheckpoint::boxes)
      .def_rw("cylinders", &microsimulator::ConstraintSetCheckpoint::cylinders)
      .def("validate", &microsimulator::ConstraintSetCheckpoint::validate);

  nb::class_<microsimulator::SimulationCheckpoint>(module, "_SimulationCheckpoint")
      .def(nb::init<>())
      .def_rw("schema_version", &microsimulator::SimulationCheckpoint::schema_version)
      .def_rw("time", &microsimulator::SimulationCheckpoint::time)
      .def_rw("world", &microsimulator::SimulationCheckpoint::world)
      .def_rw("constraints", &microsimulator::SimulationCheckpoint::constraints)
      .def_rw("species_rate_plan", &microsimulator::SimulationCheckpoint::species_rate_plan)
      .def_rw("signal_grid", &microsimulator::SimulationCheckpoint::signal_grid)
      .def_rw("coupled_rate_plan", &microsimulator::SimulationCheckpoint::coupled_rate_plan)
      .def("validate", &microsimulator::SimulationCheckpoint::validate);

  nb::class_<microsimulator::ConstraintContactParameters>(module, "ConstraintContactParameters")
      .def(nb::init<>())
      .def_rw("activation_margin", &microsimulator::ConstraintContactParameters::activation_margin)
      .def_rw("degeneracy_epsilon",
              &microsimulator::ConstraintContactParameters::degeneracy_epsilon);

  nb::class_<microsimulator::ExternalContact>(module, "ExternalContact")
      .def_ro("cell_id", &microsimulator::ExternalContact::cell_id)
      .def_ro("cell_slot", &microsimulator::ExternalContact::cell_slot)
      .def_ro("constraint_id", &microsimulator::ExternalContact::constraint_id)
      .def_ro("constraint_kind", &microsimulator::ExternalContact::constraint_kind)
      .def_ro("location", &microsimulator::ExternalContact::location)
      .def_prop_ro("endpoint",
                   [](const microsimulator::ExternalContact& contact) { return contact.location; })
      .def_ro("point_on_cell", &microsimulator::ExternalContact::point_on_cell)
      .def_ro("normal", &microsimulator::ExternalContact::normal)
      .def_ro("signed_separation", &microsimulator::ExternalContact::signed_separation)
      .def_ro("weight", &microsimulator::ExternalContact::weight);

  nb::class_<microsimulator::ExternalContactGraph>(module, "ExternalContactGraph")
      .def_prop_ro("cell_count", &microsimulator::ExternalContactGraph::cell_count)
      .def_prop_ro("empty", &microsimulator::ExternalContactGraph::empty)
      .def_prop_ro("contacts",
                   [](const microsimulator::ExternalContactGraph& graph) {
                     return std::vector<microsimulator::ExternalContact>(graph.contacts().begin(),
                                                                         graph.contacts().end());
                   })
      .def("__len__", &microsimulator::ExternalContactGraph::size)
      .def(
          "incident_contact_indices",
          [](const microsimulator::ExternalContactGraph& graph, microsimulator::Slot slot) {
            const auto indices = graph.incident_contact_indices(slot);
            return std::vector<std::size_t>(indices.begin(), indices.end());
          },
          "slot"_a);

  nb::class_<microsimulator::CellCorrection>(module, "CellCorrection")
      .def_ro("translation", &microsimulator::CellCorrection::translation)
      .def_ro("rotation", &microsimulator::CellCorrection::rotation)
      .def_ro("length", &microsimulator::CellCorrection::length);

  nb::class_<microsimulator::MechanicsParameters>(module, "MechanicsParameters")
      .def(nb::init<>())
      .def_rw("mu_a", &microsimulator::MechanicsParameters::mu_a)
      .def_rw("gamma", &microsimulator::MechanicsParameters::gamma)
      .def_rw("residual_rms_tolerance",
              &microsimulator::MechanicsParameters::residual_rms_tolerance)
      .def_rw("max_iterations", &microsimulator::MechanicsParameters::max_iterations);

  nb::class_<microsimulator::MechanicsIntegrationParameters>(module,
                                                             "MechanicsIntegrationParameters")
      .def(nb::init<>())
      .def_rw("max_rotation_radians",
              &microsimulator::MechanicsIntegrationParameters::max_rotation_radians)
      .def_rw("require_convergence",
              &microsimulator::MechanicsIntegrationParameters::require_convergence);

  nb::class_<microsimulator::SolverReport>(module, "SolverReport")
      .def_ro("status", &microsimulator::SolverReport::status)
      .def_ro("breakdown", &microsimulator::SolverReport::breakdown)
      .def_ro("iterations", &microsimulator::SolverReport::iterations)
      .def_ro("initial_residual_rms", &microsimulator::SolverReport::initial_residual_rms)
      .def_ro("final_residual_rms", &microsimulator::SolverReport::final_residual_rms);

  nb::class_<microsimulator::MechanicsSolveResult>(module, "MechanicsSolveResult")
      .def_ro("corrections", &microsimulator::MechanicsSolveResult::corrections)
      .def_ro("report", &microsimulator::MechanicsSolveResult::report);

  nb::class_<microsimulator::Simulation>(module, "Simulation")
      .def(nb::init<microsimulator::BackendKind, std::size_t, std::size_t, std::uint32_t>(),
           "backend"_a = microsimulator::BackendKind::cpu, "reserved_capacity"_a = 0,
           "species_count"_a = 0, "device_index"_a = 0)
      .def(nb::init<microsimulator::BackendKind, const microsimulator::SimulationCheckpoint&,
                    std::uint32_t>(),
           "backend"_a, "checkpoint"_a, "device_index"_a = 0)
      .def_prop_ro("backend_info",
                   microsimulator::python::exclusive(&microsimulator::Simulation::backend_info))
      .def("supports", microsimulator::python::exclusive(&microsimulator::Simulation::supports),
           "feature"_a)
      .def_prop_ro("time", microsimulator::python::exclusive(&microsimulator::Simulation::time))
      .def_prop_ro("cell_count",
                   microsimulator::python::exclusive(&microsimulator::Simulation::cell_count))
      .def_prop_ro("species_count",
                   microsimulator::python::exclusive(&microsimulator::Simulation::species_count))
      .def_prop_ro("signal_count",
                   microsimulator::python::exclusive(&microsimulator::Simulation::signal_count))
      .def_prop_ro("has_signal_grid",
                   microsimulator::python::exclusive(&microsimulator::Simulation::has_signal_grid))
      .def_prop_ro(
          "last_signal_solve_report",
          microsimulator::python::exclusive(&microsimulator::Simulation::last_signal_solve_report))
      .def_prop_ro("has_coupled_rate_plan", microsimulator::python::exclusive(
                                                &microsimulator::Simulation::has_coupled_rate_plan))
      .def("add_cell", microsimulator::python::exclusive(&microsimulator::Simulation::add_cell),
           "cell"_a)
      .def("remove_cell",
           microsimulator::python::exclusive(&microsimulator::Simulation::remove_cell), "id"_a)
      .def("apply_flow_drift",
           microsimulator::python::released(&microsimulator::Simulation::apply_flow_drift), "dt"_a,
           "integration"_a = microsimulator::MechanicsIntegrationParameters{})
      .def("add_plane_constraint",
           microsimulator::python::exclusive(&microsimulator::Simulation::add_plane_constraint),
           "plane"_a)
      .def("add_sphere_constraint",
           microsimulator::python::exclusive(&microsimulator::Simulation::add_sphere_constraint),
           "sphere"_a)
      .def("add_box_constraint",
           microsimulator::python::exclusive(&microsimulator::Simulation::add_box_constraint),
           "box"_a)
      .def("add_cylinder_constraint",
           microsimulator::python::exclusive(&microsimulator::Simulation::add_cylinder_constraint),
           "cylinder"_a)
      .def("set_cell_geometry",
           microsimulator::python::exclusive(&microsimulator::Simulation::set_cell_geometry),
           "id"_a, "position"_a, "direction"_a, "length"_a)
      .def("set_cell_attributes",
           microsimulator::python::exclusive(&microsimulator::Simulation::set_cell_attributes),
           "id"_a, "growth_rate"_a, "cell_type"_a)
      .def("set_cell_fixed",
           microsimulator::python::exclusive(&microsimulator::Simulation::set_cell_fixed), "id"_a,
           "fixed"_a)
      .def(
          "set_species",
          [](microsimulator::Simulation& simulation, microsimulator::CellId id,
             const std::vector<float>& levels) {
            microsimulator::python::BusyObjects::require_idle(&simulation);
            simulation.set_species(id, levels);
          },
          "id"_a, "levels"_a)
      .def("set_species_rate_plan",
           microsimulator::python::exclusive(&microsimulator::Simulation::set_species_rate_plan),
           "plan"_a)
      .def("set_coupled_rate_plan",
           microsimulator::python::exclusive(&microsimulator::Simulation::set_coupled_rate_plan),
           "plan"_a)
      .def("clear_coupled_rate_plan",
           microsimulator::python::exclusive(&microsimulator::Simulation::clear_coupled_rate_plan))
      .def("configure_signal_grid",
           microsimulator::python::exclusive(&microsimulator::Simulation::configure_signal_grid),
           "spec"_a, "levels"_a = std::vector<float>{})
      .def(
          "set_signal_levels",
          [](microsimulator::Simulation& simulation, const std::vector<float>& levels) {
            microsimulator::python::BusyObjects::require_idle(&simulation);
            simulation.set_signal_levels(levels);
          },
          "levels"_a)
      .def("set_velocity_field",
           microsimulator::python::exclusive(&microsimulator::Simulation::set_velocity_field),
           "field"_a.none())
      .def("set_signal_reaction",
           microsimulator::python::exclusive(&microsimulator::Simulation::set_signal_reaction),
           "reaction"_a.none())
      .def(
          "set_signal_boundary_values",
          [](microsimulator::Simulation& simulation, microsimulator::GridFace face,
             const std::vector<float>& values) {
            microsimulator::python::BusyObjects::require_idle(&simulation);
            simulation.set_signal_boundary_values(face, values);
          },
          "face"_a, "values"_a)
      .def("divide", microsimulator::python::exclusive(&microsimulator::Simulation::divide),
           "parent_id"_a, "first_fraction"_a)
      .def("divide_equal",
           microsimulator::python::exclusive(&microsimulator::Simulation::divide_equal),
           "parent_id"_a)
      .def("step", microsimulator::python::released(&microsimulator::Simulation::step), "dt"_a)
      .def("find_cell_contacts",
           microsimulator::python::released(&microsimulator::Simulation::find_cell_contacts),
           "parameters"_a = microsimulator::ContactParameters{})
      .def("find_external_contacts",
           microsimulator::python::released(&microsimulator::Simulation::find_external_contacts),
           "parameters"_a = microsimulator::ConstraintContactParameters{})
      .def("solve_cell_mechanics",
           microsimulator::python::released(&microsimulator::Simulation::solve_cell_mechanics),
           "mechanics_parameters"_a = microsimulator::MechanicsParameters{},
           "contact_parameters"_a = microsimulator::ContactParameters{},
           "constraint_parameters"_a = microsimulator::ConstraintContactParameters{})
      .def("relax_cell_mechanics",
           microsimulator::python::released(&microsimulator::Simulation::relax_cell_mechanics),
           "mechanics_parameters"_a = microsimulator::MechanicsParameters{},
           "contact_parameters"_a = microsimulator::ContactParameters{},
           "integration_parameters"_a = microsimulator::MechanicsIntegrationParameters{},
           "constraint_parameters"_a = microsimulator::ConstraintContactParameters{})
      .def(
          "solve_depth_averaged_flow",
          [](microsimulator::Simulation& simulation, const microsimulator::SignalGridSpec& spec,
             const std::vector<float>& mobility,
             const microsimulator::DepthAveragedFlowParameters& parameters) {
            const microsimulator::python::ReleasedCall call(&simulation);
            return simulation.solve_depth_averaged_flow(spec, mobility, parameters);
          },
          "spec"_a, "mobility"_a = std::vector<float>{},
          "parameters"_a = microsimulator::DepthAveragedFlowParameters{})
      .def(
          "solve_resolved_flow",
          [](microsimulator::Simulation& simulation, const microsimulator::SignalGridSpec& spec,
             const std::vector<float>& drag,
             const microsimulator::ResolvedFlowParameters& parameters) {
            const microsimulator::python::ReleasedCall call(&simulation);
            return simulation.solve_resolved_flow(spec, drag, parameters);
          },
          "spec"_a, "drag"_a = std::vector<float>{},
          "parameters"_a = microsimulator::ResolvedFlowParameters{})
      .def("cell", microsimulator::python::exclusive(&microsimulator::Simulation::cell), "id"_a)
      .def("cells", microsimulator::python::exclusive(&microsimulator::Simulation::cells))
      .def("lineage_parent",
           microsimulator::python::exclusive(&microsimulator::Simulation::lineage_parent), "id"_a)
      .def_prop_ro("signal_levels",
                   microsimulator::python::exclusive(&microsimulator::Simulation::signal_levels))
      .def("sample_signals",
           microsimulator::python::exclusive(&microsimulator::Simulation::sample_signals),
           "position"_a)
      .def("_checkpoint",
           microsimulator::python::exclusive(&microsimulator::Simulation::checkpoint))
      .def("validate", microsimulator::python::exclusive(&microsimulator::Simulation::validate));

  bind_serialization(module);
}
