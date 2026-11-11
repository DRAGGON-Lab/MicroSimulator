#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "microsimulator/occupancy.hpp"
#include "native_call.hpp"

namespace nb = nanobind;
using namespace nb::literals;

void bind_occupancy(nb::module_& module) {
  nb::class_<microsimulator::OccupancyCapsule>(module, "OccupancyCapsule")
      .def(nb::init<std::array<float, 3>, std::array<float, 3>, float, float>());
  nb::class_<microsimulator::OccupancyFace>(module, "OccupancyFace")
      .def(nb::init<std::uint32_t, std::uint32_t, float, float>())
      .def_ro("first", &microsimulator::OccupancyFace::first)
      .def_ro("second", &microsimulator::OccupancyFace::second)
      .def_ro("conductance", &microsimulator::OccupancyFace::conductance)
      .def_ro("volume_flux", &microsimulator::OccupancyFace::volume_flux);
  nb::class_<microsimulator::OccupancyReservoir>(module, "OccupancyReservoir")
      .def(nb::init<std::uint32_t, float, float, float>());
  nb::class_<microsimulator::OccupancyBalance>(module, "OccupancyBalance")
      .def_ro("before", &microsimulator::OccupancyBalance::before)
      .def_ro("after", &microsimulator::OccupancyBalance::after)
      .def_ro("source", &microsimulator::OccupancyBalance::source)
      .def_ro("reaction", &microsimulator::OccupancyBalance::reaction)
      .def_ro("boundary", &microsimulator::OccupancyBalance::boundary);
  nb::class_<microsimulator::OccupancyStep>(module, "OccupancyStep")
      .def_ro("amount", &microsimulator::OccupancyStep::amount)
      .def_ro("balance", &microsimulator::OccupancyStep::balance)
      .def_ro("iterations", &microsimulator::OccupancyStep::iterations)
      .def_ro("relative_residual", &microsimulator::OccupancyStep::relative_residual);
  nb::class_<microsimulator::OccupancySolver>(module, "OccupancySolver")
      .def(nb::init<microsimulator::BackendKind, std::uint32_t, float>())
      .def("geometric_porosity",
           microsimulator::python::released(&microsimulator::OccupancySolver::geometric_porosity))
      .def("accessible_volumes",
           microsimulator::python::released(&microsimulator::OccupancySolver::accessible_volumes))
      .def("concentration",
           microsimulator::python::released(&microsimulator::OccupancySolver::concentration))
      .def("porosity_face",
           microsimulator::python::released(&microsimulator::OccupancySolver::porosity_face))
      .def("remap_amounts",
           microsimulator::python::released(&microsimulator::OccupancySolver::remap_amounts))
      .def("exchange_weights",
           microsimulator::python::released(&microsimulator::OccupancySolver::exchange_weights))
      .def("backward_euler",
           microsimulator::python::released(&microsimulator::OccupancySolver::backward_euler));
}
