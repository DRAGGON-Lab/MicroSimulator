#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "cm/culture_simulation.hpp"
#include "cm/stokes_flow.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {
template <std::size_t N>
nb::tuple array_tuple(const std::array<double, N>& value) {
  if constexpr (N == 3) {
    return nb::make_tuple(value[0], value[1], value[2]);
  } else {
    return nb::make_tuple(value[0], value[1], value[2], value[3]);
  }
}
}  // namespace

namespace {

void bind_ChemicalBoundaryKind(nb::module_& module) {
  nb::enum_<cm::ChemicalBoundaryKind>(module, "ChemicalBoundaryKind")
      .value("RESERVOIR_CONTACT", cm::ChemicalBoundaryKind::reservoir_contact)
      .value("ADVECTIVE", cm::ChemicalBoundaryKind::advective)
      .value("OUTFLOW", cm::ChemicalBoundaryKind::outflow);
}

void bind_GrowthKind(nb::module_& module) {
  nb::enum_<cm::GrowthKind>(module, "GrowthKind")
      .value("MONOD", cm::GrowthKind::monod)
      .value("ESSENTIAL", cm::GrowthKind::essential);
}

void bind_GrowthRequirement(nb::module_& module) {
  nb::class_<cm::GrowthRequirement>(module, "GrowthRequirement")
      .def(nb::init<>())
      .def_rw("solute", &cm::GrowthRequirement::solute)
      .def_rw("half_saturation", &cm::GrowthRequirement::half_saturation)
      .def_rw("biomass_yield", &cm::GrowthRequirement::biomass_yield);
}

void bind_CellGrowthModel(nb::module_& module) {
  nb::class_<cm::CellGrowthModel>(module, "CellGrowthModel")
      .def(nb::init<>())
      .def_rw("cell_id", &cm::CellGrowthModel::cell_id)
      .def_rw("kind", &cm::CellGrowthModel::kind)
      .def_rw("mu_max", &cm::CellGrowthModel::mu_max)
      .def_rw("biomass_density", &cm::CellGrowthModel::biomass_density)
      .def_rw("volume_ratio", &cm::CellGrowthModel::volume_ratio)
      .def_rw("requirements", &cm::CellGrowthModel::requirements);
}

void bind_GrowthInput(nb::module_& module) {
  nb::class_<cm::GrowthInput>(module, "GrowthInput")
      .def(nb::init<>())
      .def_rw("biochemical_volume", &cm::GrowthInput::biochemical_volume)
      .def_rw("surface_area", &cm::GrowthInput::surface_area)
      .def_rw("concentrations", &cm::GrowthInput::concentrations)
      .def_rw("uptake", &cm::GrowthInput::uptake);
}

void bind_GrowthEvaluation(nb::module_& module) {
  nb::class_<cm::GrowthEvaluation>(module, "GrowthEvaluation")
      .def_ro("uptake_velocities", &cm::GrowthEvaluation::uptake_velocities)
      .def_ro("biomass_gain", &cm::GrowthEvaluation::biomass_gain)
      .def_ro("biochemical_volume_gain", &cm::GrowthEvaluation::biochemical_volume_gain)
      .def_ro("geometric_volume_gain", &cm::GrowthEvaluation::geometric_volume_gain)
      .def_ro("specific_rate", &cm::GrowthEvaluation::specific_rate)
      .def_ro("stoichiometric_residual", &cm::GrowthEvaluation::stoichiometric_residual);
}

void bind_GrowthExecutor(nb::module_& module) {
  nb::class_<cm::GrowthExecutor>(module, "GrowthExecutor")
      .def(nb::init<cm::BackendKind, std::uint32_t>(), "backend"_a = cm::BackendKind::cpu,
           "device_index"_a = 0)
      .def("evaluate", &cm::GrowthExecutor::evaluate, "models"_a, "inputs"_a, "dt"_a);
}

void bind_CultureEvent(nb::module_& module) {
  nb::class_<cm::CultureEvent>(module, "CultureEvent")
      .def(nb::init<>())
      .def_rw("time", &cm::CultureEvent::time)
      .def_rw("ports", &cm::CultureEvent::ports)
      .def_rw("reservoirs", &cm::CultureEvent::reservoirs);
}

void bind_FluidGridSpec(nb::module_& module) {
  nb::class_<cm::FluidGridSpec>(module, "FluidGridSpec")
      .def(nb::init<>())
      .def_rw("shape", &cm::FluidGridSpec::shape)
      .def_rw("origin", &cm::FluidGridSpec::origin)
      .def_rw("spacing", &cm::FluidGridSpec::spacing)
      .def_rw("length_unit_m", &cm::FluidGridSpec::length_unit_m)
      .def_rw("time_unit_s", &cm::FluidGridSpec::time_unit_s)
      .def_rw("obstacles", &cm::FluidGridSpec::obstacles)
      .def("validate", &cm::FluidGridSpec::validate)
      .def_prop_ro("site_count", &cm::FluidGridSpec::site_count);
}

void bind_FluidProperties(nb::module_& module) {
  nb::class_<cm::FluidProperties>(module, "FluidProperties")
      .def(nb::init<>())
      .def_rw("viscosity_pa_s", &cm::FluidProperties::viscosity_pa_s)
      .def_rw("density_kg_m3", &cm::FluidProperties::density_kg_m3)
      .def("validate", &cm::FluidProperties::validate);
}

void bind_FlowPortKind(nb::module_& module) {
  nb::enum_<cm::FlowPortKind>(module, "FlowPortKind")
      .value("PRESSURE", cm::FlowPortKind::pressure)
      .value("FLOW_RATE", cm::FlowPortKind::flow_rate);
}

void bind_FlowPort(nb::module_& module) {
  nb::class_<cm::FlowPort>(module, "FlowPort")
      .def(nb::init<>())
      .def_rw("name", &cm::FlowPort::name)
      .def_rw("axis", &cm::FlowPort::axis)
      .def_rw("upper", &cm::FlowPort::upper)
      .def_rw("kind", &cm::FlowPort::kind)
      .def_rw("value", &cm::FlowPort::value)
      .def_rw("sites", &cm::FlowPort::sites);
}

void bind_LinearSolveParameters(nb::module_& module) {
  nb::class_<cm::LinearSolveParameters>(module, "LinearSolveParameters")
      .def(nb::init<>())
      .def_rw("relative_tolerance", &cm::LinearSolveParameters::relative_tolerance)
      .def_rw("absolute_tolerance", &cm::LinearSolveParameters::absolute_tolerance)
      .def_rw("max_iterations", &cm::LinearSolveParameters::max_iterations)
      .def_rw("memory_limit_bytes", &cm::LinearSolveParameters::memory_limit_bytes)
      .def("validate", &cm::LinearSolveParameters::validate);
}

void bind_FlowPortResult(nb::module_& module) {
  nb::class_<cm::FlowPortResult>(module, "FlowPortResult")
      .def_ro("name", &cm::FlowPortResult::name)
      .def_ro("pressure_pa", &cm::FlowPortResult::pressure_pa)
      .def_ro("flow_rate_m3_s", &cm::FlowPortResult::flow_rate_m3_s)
      .def_ro("area_m2", &cm::FlowPortResult::area_m2);
}

void bind_CapsuleBody(nb::module_& module) {
  nb::class_<cm::CapsuleBody>(module, "CapsuleBody")
      .def(nb::init<>())
      .def_rw("id", &cm::CapsuleBody::id)
      .def_prop_rw(
          "position",
          [](const cm::CapsuleBody& body) {
            return array_tuple(body.position);
          },
          [](cm::CapsuleBody& body, std::array<double, 3> value) {
            body.position = value;
          })
      .def_prop_rw(
          "orientation",
          [](const cm::CapsuleBody& body) {
            return array_tuple(body.orientation);
          },
          [](cm::CapsuleBody& body, std::array<double, 4> value) {
            body.orientation = value;
          })
      .def_rw("length", &cm::CapsuleBody::length)
      .def_rw("length_rate", &cm::CapsuleBody::length_rate)
      .def_rw("radius", &cm::CapsuleBody::radius)
      .def_rw("fixed", &cm::CapsuleBody::fixed)
      .def_prop_rw(
          "force_n",
          [](const cm::CapsuleBody& body) {
            return array_tuple(body.force_n);
          },
          [](cm::CapsuleBody& body, std::array<double, 3> value) {
            body.force_n = value;
          })
      .def_prop_rw(
          "torque_nm",
          [](const cm::CapsuleBody& body) {
            return array_tuple(body.torque_nm);
          },
          [](cm::CapsuleBody& body, std::array<double, 3> value) {
            body.torque_nm = value;
          })
      .def("validate", &cm::CapsuleBody::validate)
      .def_prop_ro("geometric_volume", &cm::CapsuleBody::geometric_volume);
}

void bind_FluidBodyResult(nb::module_& module) {
  nb::class_<cm::FluidBodyResult>(module, "FluidBodyResult")
      .def_ro("id", &cm::FluidBodyResult::id)
      .def_prop_ro("velocity",
                   [](const cm::FluidBodyResult& value) {
                     return array_tuple(value.velocity);
                   })
      .def_prop_ro("angular_velocity",
                   [](const cm::FluidBodyResult& value) {
                     return array_tuple(value.angular_velocity);
                   })
      .def_prop_ro("hydrodynamic_force_n",
                   [](const cm::FluidBodyResult& value) {
                     return array_tuple(value.hydrodynamic_force_n);
                   })
      .def_prop_ro("hydrodynamic_torque_nm",
                   [](const cm::FluidBodyResult& value) {
                     return array_tuple(value.hydrodynamic_torque_nm);
                   })
      .def_ro("no_slip_rms_m_s", &cm::FluidBodyResult::no_slip_rms_m_s)
      .def_ro("volume_change_rate_m3_s", &cm::FluidBodyResult::volume_change_rate_m3_s)
      .def_ro("marker_count", &cm::FluidBodyResult::marker_count);
}

void bind_FluidSolveReport(nb::module_& module) {
  nb::class_<cm::FluidSolveReport>(module, "FluidSolveReport")
      .def(nb::init<>())
      .def_rw("iterations", &cm::FluidSolveReport::iterations)
      .def_rw("relative_residual", &cm::FluidSolveReport::relative_residual)
      .def_rw("absolute_residual", &cm::FluidSolveReport::absolute_residual)
      .def_rw("divergence_rms_per_s", &cm::FluidSolveReport::divergence_rms_per_s)
      .def_rw("continuity_rms_per_s", &cm::FluidSolveReport::continuity_rms_per_s)
      .def_rw("source_volume_rate_m3_s", &cm::FluidSolveReport::source_volume_rate_m3_s)
      .def_rw("max_speed_m_s", &cm::FluidSolveReport::max_speed_m_s)
      .def_rw("reynolds_number", &cm::FluidSolveReport::reynolds_number)
      .def_rw("viscous_relaxation_time_s", &cm::FluidSolveReport::viscous_relaxation_time_s)
      .def_rw("net_flow_rate_m3_s", &cm::FluidSolveReport::net_flow_rate_m3_s)
      .def_rw("estimated_memory_bytes", &cm::FluidSolveReport::estimated_memory_bytes);
}

void bind_FluidFlowResult(nb::module_& module) {
  nb::class_<cm::FluidFlowResult>(module, "FluidFlowResult")
      .def_ro("field", &cm::FluidFlowResult::field)
      .def_ro("pressure_pa", &cm::FluidFlowResult::pressure_pa)
      .def_ro("ports", &cm::FluidFlowResult::ports)
      .def_ro("bodies", &cm::FluidFlowResult::bodies)
      .def_ro("report", &cm::FluidFlowResult::report);
}

void bind_FluidBodyStepParameters(nb::module_& module) {
  nb::class_<cm::FluidBodyStepParameters>(module, "FluidBodyStepParameters")
      .def(nb::init<>())
      .def_rw("minimum_gap_m", &cm::FluidBodyStepParameters::minimum_gap_m)
      .def_rw("maximum_displacement_fraction",
              &cm::FluidBodyStepParameters::maximum_displacement_fraction)
      .def_rw("max_halvings", &cm::FluidBodyStepParameters::max_halvings)
      .def_rw("max_contact_iterations", &cm::FluidBodyStepParameters::max_contact_iterations)
      .def("validate", &cm::FluidBodyStepParameters::validate);
}

void bind_FluidContactResult(nb::module_& module) {
  nb::class_<cm::FluidContactResult>(module, "FluidContactResult")
      .def_ro("first_id", &cm::FluidContactResult::first_id)
      .def_ro("second_id", &cm::FluidContactResult::second_id)
      .def_prop_ro("normal",
                   [](const cm::FluidContactResult& value) {
                     return array_tuple(value.normal);
                   })
      .def_prop_ro("point_on_first",
                   [](const cm::FluidContactResult& value) {
                     return array_tuple(value.point_on_first);
                   })
      .def_ro("initial_gap_m", &cm::FluidContactResult::initial_gap_m)
      .def_ro("normal_force_n", &cm::FluidContactResult::normal_force_n);
}

void bind_FluidBodyStep(nb::module_& module) {
  nb::class_<cm::FluidBodyStep>(module, "FluidBodyStep")
      .def_ro("accepted_dt", &cm::FluidBodyStep::accepted_dt)
      .def_ro("bodies", &cm::FluidBodyStep::bodies)
      .def_ro("flow", &cm::FluidBodyStep::flow)
      .def_ro("contacts", &cm::FluidBodyStep::contacts)
      .def_ro("halvings", &cm::FluidBodyStep::halvings)
      .def_ro("contact_iterations", &cm::FluidBodyStep::contact_iterations);
}

void bind_StokesFlowSolver(nb::module_& module) {
  nb::class_<cm::StokesFlowSolver>(module, "StokesFlowSolver")
      .def(nb::init<cm::BackendKind, std::uint32_t>(), "backend"_a = cm::BackendKind::cpu,
           "device_index"_a = 0)
      .def("solve", &cm::StokesFlowSolver::solve, "grid"_a, "fluid"_a, "ports"_a,
           "parameters"_a = cm::LinearSolveParameters{})
      .def("solve_bodies", &cm::StokesFlowSolver::solve_bodies, "grid"_a, "fluid"_a, "ports"_a,
           "bodies"_a, "parameters"_a = cm::LinearSolveParameters{})
      .def("propose_body_step", &cm::StokesFlowSolver::propose_body_step, "grid"_a, "fluid"_a,
           "ports"_a, "bodies"_a, "maximum_dt"_a,
           "solve_parameters"_a = cm::LinearSolveParameters{},
           "step_parameters"_a = cm::FluidBodyStepParameters{});
}

void bind_FluidGeometryParameters(nb::module_& module) {
  nb::class_<cm::FluidGeometryParameters>(module, "FluidGeometryParameters")
      .def(nb::init<>())
      .def_rw("surface_resolution", &cm::FluidGeometryParameters::surface_resolution)
      .def_rw("maximum_surface_error_fraction",
              &cm::FluidGeometryParameters::maximum_surface_error_fraction)
      .def_rw("memory_limit_bytes", &cm::FluidGeometryParameters::memory_limit_bytes)
      .def("validate", &cm::FluidGeometryParameters::validate);
}

void bind_FluidFragment(nb::module_& module) {
  nb::class_<cm::FluidFragment>(module, "FluidFragment")
      .def_ro("site", &cm::FluidFragment::site)
      .def_ro("component", &cm::FluidFragment::component)
      .def_ro("volume", &cm::FluidFragment::volume)
      .def_prop_ro("centroid", [](const cm::FluidFragment& f) {
        return array_tuple(f.centroid);
      });
}

void bind_FluidFace(nb::module_& module) {
  nb::class_<cm::FluidFace>(module, "FluidFace")
      .def_ro("first", &cm::FluidFace::first)
      .def_ro("second", &cm::FluidFace::second)
      .def_ro("area", &cm::FluidFace::area)
      .def_ro("body_id", &cm::FluidFace::body_id)
      .def_ro("axis", &cm::FluidFace::axis)
      .def_ro("grid_face", &cm::FluidFace::grid_face)
      .def_prop_ro("centroid",
                   [](const cm::FluidFace& f) {
                     return array_tuple(f.centroid);
                   })
      .def_prop_ro("normal", [](const cm::FluidFace& f) {
        return array_tuple(f.normal);
      });
}

void bind_FluidGeometryReport(nb::module_& module) {
  nb::class_<cm::FluidGeometryReport>(module, "FluidGeometryReport")
      .def_ro("fluid_volume", &cm::FluidGeometryReport::fluid_volume)
      .def_ro("expected_fluid_volume", &cm::FluidGeometryReport::expected_fluid_volume)
      .def_ro("volume_error", &cm::FluidGeometryReport::volume_error)
      .def_ro("maximum_surface_error", &cm::FluidGeometryReport::maximum_surface_error)
      .def_ro("component_count", &cm::FluidGeometryReport::component_count)
      .def_ro("estimated_memory_bytes", &cm::FluidGeometryReport::estimated_memory_bytes);
}

void bind_FluidOverlap(nb::module_& module) {
  nb::class_<cm::FluidOverlap>(module, "FluidOverlap")
      .def_ro("first", &cm::FluidOverlap::first)
      .def_ro("second", &cm::FluidOverlap::second)
      .def_ro("volume", &cm::FluidOverlap::volume);
}

void bind_FluidGeometry(nb::module_& module) {
  nb::class_<cm::FluidGeometry>(module, "FluidGeometry")
      .def(nb::init<const cm::FluidGridSpec&, const std::vector<cm::CapsuleBody>&,
                    const cm::FluidGeometryParameters&>(),
           "grid"_a, "bodies"_a, "parameters"_a = cm::FluidGeometryParameters{})
      .def_prop_ro("grid",
                   [](const cm::FluidGeometry& g) {
                     return g.grid();
                   })
      .def_prop_ro("bodies",
                   [](const cm::FluidGeometry& g) {
                     return g.bodies();
                   })
      .def_prop_ro("fragments",
                   [](const cm::FluidGeometry& g) {
                     return g.fragments();
                   })
      .def_prop_ro("faces",
                   [](const cm::FluidGeometry& g) {
                     return g.faces();
                   })
      .def_prop_ro("report",
                   [](const cm::FluidGeometry& g) {
                     return g.report();
                   })
      .def("overlaps", &cm::FluidGeometry::overlaps, "other"_a);
}

void bind_Solute(nb::module_& module) {
  nb::class_<cm::Solute>(module, "Solute")
      .def(nb::init<>())
      .def_rw("name", &cm::Solute::name)
      .def_rw("diffusion", &cm::Solute::diffusion)
      .def_rw("amount_unit", &cm::Solute::amount_unit);
}

void bind_ChemicalBoundary(nb::module_& module) {
  nb::class_<cm::ChemicalBoundary>(module, "ChemicalBoundary")
      .def(nb::init<>())
      .def_rw("port", &cm::ChemicalBoundary::port)
      .def_rw("concentrations", &cm::ChemicalBoundary::concentrations)
      .def_rw("kind", &cm::ChemicalBoundary::kind)
      .def_rw("allow_backflow", &cm::ChemicalBoundary::allow_backflow);
}

void bind_SurfaceTransferLaw(nb::module_& module) {
  nb::class_<cm::SurfaceTransferLaw>(module, "SurfaceTransferLaw")
      .def(nb::init<>())
      .def_rw("body_id", &cm::SurfaceTransferLaw::body_id)
      .def_rw("solute", &cm::SurfaceTransferLaw::solute)
      .def_rw("uptake_velocity", &cm::SurfaceTransferLaw::uptake_velocity)
      .def_rw("secretion_rate", &cm::SurfaceTransferLaw::secretion_rate);
}

void bind_ChemicalTransfer(nb::module_& module) {
  nb::class_<cm::ChemicalTransfer>(module, "ChemicalTransfer")
      .def(nb::init<>())
      .def_rw("port", &cm::ChemicalTransfer::port)
      .def_rw("body_id", &cm::ChemicalTransfer::body_id)
      .def_rw("amounts", &cm::ChemicalTransfer::amounts);
}

void bind_SoluteTransportReport(nb::module_& module) {
  nb::class_<cm::SoluteTransportReport>(module, "SoluteTransportReport")
      .def(nb::init<>())
      .def_rw("projection_iterations", &cm::SoluteTransportReport::projection_iterations)
      .def_rw("transport_iterations", &cm::SoluteTransportReport::transport_iterations)
      .def_rw("maximum_volume_residual", &cm::SoluteTransportReport::maximum_volume_residual)
      .def_rw("mass_balance_error", &cm::SoluteTransportReport::mass_balance_error);
}

void bind_SurfaceEnvironment(nb::module_& module) {
  nb::class_<cm::SurfaceEnvironment>(module, "SurfaceEnvironment")
      .def_ro("body_id", &cm::SurfaceEnvironment::body_id)
      .def_ro("area", &cm::SurfaceEnvironment::area)
      .def_ro("concentrations", &cm::SurfaceEnvironment::concentrations);
}

void bind_SoluteTransportResult(nb::module_& module) {
  nb::class_<cm::SoluteTransportResult>(module, "SoluteTransportResult")
      .def_ro("amounts", &cm::SoluteTransportResult::amounts)
      .def_ro("concentrations", &cm::SoluteTransportResult::concentrations)
      .def_ro("reservoirs", &cm::SoluteTransportResult::reservoirs)
      .def_ro("cells", &cm::SoluteTransportResult::cells)
      .def_ro("report", &cm::SoluteTransportResult::report)
      .def_ro("surfaces", &cm::SoluteTransportResult::surfaces);
}

void bind_GeometricFluxReport(nb::module_& module) {
  nb::class_<cm::GeometricFluxReport>(module, "GeometricFluxReport")
      .def_ro("projection_iterations", &cm::GeometricFluxReport::projection_iterations)
      .def_ro("maximum_volume_residual", &cm::GeometricFluxReport::maximum_volume_residual);
}

void bind_TransportGeometry(nb::module_& module) {
  nb::class_<cm::TransportGeometry>(module, "TransportGeometry")
      .def(nb::init<const cm::FluidGeometry&, const cm::FluidGeometry&, const cm::MacVelocityField&,
                    const std::vector<cm::FlowPort>&, double, cm::BackendKind, std::uint32_t,
                    const cm::LinearSolveParameters&>(),
           "before"_a, "after"_a, "velocity"_a, "ports"_a, "dt"_a,
           "backend"_a = cm::BackendKind::cpu, "device_index"_a = 0,
           "parameters"_a = cm::LinearSolveParameters{})
      .def_prop_ro("report", &cm::TransportGeometry::report, nb::rv_policy::copy);
}

void bind_SoluteTransportSolver(nb::module_& module) {
  nb::class_<cm::SoluteTransportSolver>(module, "SoluteTransportSolver")
      .def(nb::init<cm::BackendKind, std::uint32_t>(), "backend"_a = cm::BackendKind::cpu,
           "device_index"_a = 0)
      .def("propose", &cm::SoluteTransportSolver::propose, "geometry"_a, "solutes"_a,
           "boundaries"_a, "amounts"_a, "exchange"_a = std::vector<cm::SurfaceTransferLaw>{},
           "parameters"_a = cm::LinearSolveParameters{})
      .def("step", &cm::SoluteTransportSolver::step, "before"_a, "after"_a, "velocity"_a, "ports"_a,
           "solutes"_a, "reservoirs"_a, "amounts"_a, "dt"_a,
           "exchange"_a = std::vector<cm::SurfaceTransferLaw>{},
           "parameters"_a = cm::LinearSolveParameters{});
}

void bind_CellSurfaceExchange(nb::module_& module) {
  nb::class_<cm::CellSurfaceExchange>(module, "CellSurfaceExchange")
      .def(nb::init<>())
      .def_rw("body_id", &cm::CellSurfaceExchange::body_id)
      .def_rw("solute", &cm::CellSurfaceExchange::solute)
      .def_rw("species", &cm::CellSurfaceExchange::species)
      .def_rw("uptake_velocity", &cm::CellSurfaceExchange::uptake_velocity)
      .def_rw("secretion_rate", &cm::CellSurfaceExchange::secretion_rate);
}

void bind_ReserveRequirement(nb::module_& module) {
  nb::class_<cm::ReserveRequirement>(module, "ReserveRequirement")
      .def(nb::init<>())
      .def_rw("species", &cm::ReserveRequirement::species)
      .def_rw("amount_per_biomass", &cm::ReserveRequirement::amount_per_biomass);
}

void bind_CultureConfiguration(nb::module_& module) {
  nb::class_<cm::CultureConfiguration>(module, "CultureConfiguration")
      .def(nb::init<>())
      .def_rw("grid", &cm::CultureConfiguration::grid)
      .def_rw("fluid", &cm::CultureConfiguration::fluid)
      .def_rw("ports", &cm::CultureConfiguration::ports)
      .def_rw("solutes", &cm::CultureConfiguration::solutes)
      .def_rw("reservoirs", &cm::CultureConfiguration::reservoirs)
      .def_rw("exchange", &cm::CultureConfiguration::exchange)
      .def_rw("biomass_requirements", &cm::CultureConfiguration::biomass_requirements)
      .def_rw("biomass_per_geometric_volume",
              &cm::CultureConfiguration::biomass_per_geometric_volume)
      .def_rw("solver", &cm::CultureConfiguration::solver)
      .def_rw("stepping", &cm::CultureConfiguration::stepping)
      .def_rw("geometry", &cm::CultureConfiguration::geometry)
      .def_rw("maximum_substeps", &cm::CultureConfiguration::maximum_substeps)
      .def_rw("maximum_retries", &cm::CultureConfiguration::maximum_retries)
      .def("validate", &cm::CultureConfiguration::validate, "species_count"_a)
      .def_rw("growth", &cm::CultureConfiguration::growth)
      .def_rw("events", &cm::CultureConfiguration::events)
      .def_rw("authoring_json", &cm::CultureConfiguration::authoring_json)
      .def_rw("coupling_tolerance", &cm::CultureConfiguration::coupling_tolerance)
      .def_rw("maximum_coupling_iterations",
              &cm::CultureConfiguration::maximum_coupling_iterations);
}

void bind_CultureCellState(nb::module_& module) {
  nb::class_<cm::CultureCellState>(module, "CultureCellState")
      .def(nb::init<>())
      .def_rw("body", &cm::CultureCellState::body)
      .def_rw("biochemical_volume", &cm::CultureCellState::biochemical_volume)
      .def_rw("species_amounts", &cm::CultureCellState::species_amounts)
      .def_rw("uptake_totals", &cm::CultureCellState::uptake_totals)
      .def_rw("realized_specific_rate", &cm::CultureCellState::realized_specific_rate)
      .def_rw("biomass_produced", &cm::CultureCellState::biomass_produced);
}

void bind_CultureReport(nb::module_& module) {
  nb::class_<cm::CultureReport>(module, "CultureReport")
      .def(nb::init<>())
      .def_rw("substeps", &cm::CultureReport::substeps)
      .def_rw("retries", &cm::CultureReport::retries)
      .def_rw("flow", &cm::CultureReport::flow)
      .def_rw("transport", &cm::CultureReport::transport);
}

void bind_CultureCheckpoint(nb::module_& module) {
  nb::class_<cm::CultureCheckpoint>(module, "CultureCheckpoint")
      .def(nb::init<>())
      .def_rw("configuration", &cm::CultureCheckpoint::configuration)
      .def_rw("cells", &cm::CultureCheckpoint::cells)
      .def_rw("extracellular_amounts", &cm::CultureCheckpoint::extracellular_amounts)
      .def_rw("reservoir_totals", &cm::CultureCheckpoint::reservoir_totals)
      .def_rw("last_report", &cm::CultureCheckpoint::last_report)
      .def("validate", &cm::CultureCheckpoint::validate, "world"_a)
      .def_rw("time", &cm::CultureCheckpoint::time)
      .def_rw("event_index", &cm::CultureCheckpoint::event_index);
}

}  // namespace

void bind_culture(nb::module_& module) {
  bind_ChemicalBoundaryKind(module);
  bind_GrowthKind(module);
  bind_GrowthRequirement(module);
  bind_CellGrowthModel(module);
  bind_GrowthInput(module);
  bind_GrowthEvaluation(module);
  bind_GrowthExecutor(module);
  bind_CultureEvent(module);
  bind_FluidGridSpec(module);
  bind_FluidProperties(module);
  bind_FlowPortKind(module);
  bind_FlowPort(module);
  bind_LinearSolveParameters(module);
  bind_FlowPortResult(module);
  bind_CapsuleBody(module);
  bind_FluidBodyResult(module);
  bind_FluidSolveReport(module);
  bind_FluidFlowResult(module);
  bind_FluidBodyStepParameters(module);
  bind_FluidContactResult(module);
  bind_FluidBodyStep(module);
  bind_StokesFlowSolver(module);
  bind_FluidGeometryParameters(module);
  bind_FluidFragment(module);
  bind_FluidFace(module);
  bind_FluidGeometryReport(module);
  bind_FluidOverlap(module);
  bind_FluidGeometry(module);
  bind_Solute(module);
  bind_ChemicalBoundary(module);
  bind_SurfaceTransferLaw(module);
  bind_ChemicalTransfer(module);
  bind_SoluteTransportReport(module);
  bind_SurfaceEnvironment(module);
  bind_SoluteTransportResult(module);
  bind_GeometricFluxReport(module);
  bind_TransportGeometry(module);
  bind_SoluteTransportSolver(module);
  bind_CellSurfaceExchange(module);
  bind_ReserveRequirement(module);
  bind_CultureConfiguration(module);
  bind_CultureCellState(module);
  bind_CultureReport(module);
  bind_CultureCheckpoint(module);
}
