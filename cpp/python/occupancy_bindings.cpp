#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "cm/occupancy.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void bind_occupancy(nb::module_& module) {
  nb::class_<cm::OccupancyCapsule>(module, "OccupancyCapsule")
      .def(nb::init<std::array<float, 3>, std::array<float, 3>, float, float>());
  nb::class_<cm::OccupancyFace>(module, "OccupancyFace")
      .def(nb::init<std::uint32_t, std::uint32_t, float, float>())
      .def_ro("first", &cm::OccupancyFace::first)
      .def_ro("second", &cm::OccupancyFace::second)
      .def_ro("conductance", &cm::OccupancyFace::conductance)
      .def_ro("volume_flux", &cm::OccupancyFace::volume_flux);
  nb::class_<cm::OccupancyReservoir>(module, "OccupancyReservoir")
      .def(nb::init<std::uint32_t, float, float, float>());
  nb::class_<cm::OccupancyBalance>(module, "OccupancyBalance")
      .def_ro("before", &cm::OccupancyBalance::before)
      .def_ro("after", &cm::OccupancyBalance::after)
      .def_ro("source", &cm::OccupancyBalance::source)
      .def_ro("reaction", &cm::OccupancyBalance::reaction)
      .def_ro("boundary", &cm::OccupancyBalance::boundary);
  nb::class_<cm::OccupancyStep>(module, "OccupancyStep")
      .def_ro("amount", &cm::OccupancyStep::amount)
      .def_ro("balance", &cm::OccupancyStep::balance)
      .def_ro("iterations", &cm::OccupancyStep::iterations)
      .def_ro("relative_residual", &cm::OccupancyStep::relative_residual);
  nb::class_<cm::OccupancySolver>(module, "OccupancySolver")
      .def(nb::init<cm::BackendKind, std::uint32_t, float>())
      .def("geometric_porosity", &cm::OccupancySolver::geometric_porosity)
      .def("accessible_volumes", &cm::OccupancySolver::accessible_volumes)
      .def("concentration", &cm::OccupancySolver::concentration)
      .def("porosity_face", &cm::OccupancySolver::porosity_face)
      .def("remap_amounts", &cm::OccupancySolver::remap_amounts)
      .def("exchange_weights", &cm::OccupancySolver::exchange_weights)
      .def("backward_euler", &cm::OccupancySolver::backward_euler);
}
