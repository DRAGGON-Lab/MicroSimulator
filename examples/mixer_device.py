"""A whole mixer chip: three liquids metered by channel length, joined and mixed.

The device is the design tool's example mixer, read from the scene of that
design. Three inlets feed three serpentine resistors of different lengths, the
three streams join in a collector, and the joined stream runs through a mixing
serpentine to the outlet. The scene's path is the model's ``scene`` parameter;
the design tool writes one with

    python -c "import json; from microfluidics.project import Project; \
        print(json.dumps(Project('designs/mixer.py').scene()))" > mixer.scene.json

The window takes in every channel and cuts the four access pads through their
middles, where tubing reaches the chip, so each pad is a port in a side of the
window. The design drives the three inlets from one pressure source, and so
does this model: the inlets hold one pressure, the outlet carries the total
flow the design's solution gives it, and how that flow divides among the three
liquids is the flow solver's own answer for the drawn channels.

A viewer shows one signal at a time, so signal 0 marks all three inlets at
once: it enters the first inlet at 1.0, the second at 0.6 and the third at 0.3,
and shows each stream by its shade and their blend where they have mixed.
Signals 1, 2 and 3 are the liquids entering the third, second and first inlets,
each alone.

Lengths are micrometers and time is seconds, so the design's microliters per
minute become cubic micrometers per second. The diffusivity is that of a small
molecule in water. The liquids run side by side out of the junction and mix
only as they diffuse into one another along the serpentine; the design tool,
which solves a network, takes them as mixed where they join. First-order
transport adds numerical mixing in the serpentine's bends, so the mixing seen
here is an upper bound until checked on a finer grid.

The model holds no cells. Its geometry and total flow come from the scene; the
grid spacing, time step and diffusivity are model choices.
"""

from __future__ import annotations

from pathlib import Path

from microsimulator import (
    ChannelMetadata,
    GridFace,
    ModelContext,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    Vec3,
)
from microsimulator.checkpoint import JSONValue
from microsimulator.design import DesignScene, load_design_scene
from microsimulator.microfluidics import LayoutDevice, LayoutPort

# The design's access pads: the inlets from the collector's closed end toward
# its outlet, and the outlet.
INLETS = ("third_in", "second_in", "first_in")
OUTLET = "outlet"
# What the marker signal carries into each inlet, in that order.
MARKS = (0.3, 0.6, 1.0)
CHANNELS = ChannelMetadata(
    signals=(
        "All inlets (first 1.0, second 0.6, third 0.3)",
        "Third liquid",
        "Second liquid",
        "First liquid",
    )
)
SIGNALS = 1 + len(INLETS)
# Wall left around the channels on the window's two sides that cut no pad.
MARGIN = 300.0
SPACING = (50.0, 50.0, 50.0)
DIFFUSIVITY = 500.0
# One pressure for every inlet, in the flow solver's units. Only its being the
# same at each inlet matters: the outlet's stated flow sets the scale.
INLET_PRESSURE = 1.0
# The flow solver's relative residual. The channels are hundreds of voxels
# long, and at the default the fluxes entering and leaving differ by a third of
# a percent; at this they differ by a fiftieth of one.
FLOW_TOLERANCE = 2.0e-7


def _corners(scene: DesignScene, path: str) -> tuple[tuple[float, float], tuple[float, float]]:
    instance = scene.instance(path)
    if instance.low is None or instance.high is None:
        raise ValueError(f"the design draws nothing at {path!r}")
    return instance.low, instance.high


def device(scene: DesignScene) -> LayoutDevice:
    """The whole chip as a device, with a port where the window cuts each access pad."""

    drawn = [
        (instance.low, instance.high)
        for instance in scene.instances
        if instance.low is not None and instance.high is not None
    ]
    inlet_low, inlet_high = _corners(scene, INLETS[0])
    outlet_low, outlet_high = _corners(scene, OUTLET)
    # The inlet pads stand in a row along the chip's lower side and the outlet
    # pad at its right: the window's lower side passes through the middles of
    # the first and its right side through the middle of the second.
    low = (min(corner[0] for corner, _ in drawn) - MARGIN, (inlet_low[1] + inlet_high[1]) * 0.5)
    high = ((outlet_low[0] + outlet_high[0]) * 0.5, max(far[1] for _, far in drawn) + MARGIN)
    window = scene.window(low, high)

    inlets = sorted(
        (crossing for crossing in window.crossings if crossing.side == GridFace.Y_LOWER),
        key=lambda crossing: crossing.low,
    )
    (outlet,) = (crossing for crossing in window.crossings if crossing.side == GridFace.X_UPPER)
    if len(inlets) != len(INLETS):
        raise ValueError("the window does not cut one pad for each inlet")
    ports = [
        LayoutPort(
            crossing.side,
            crossing.low,
            crossing.high,
            pressure=INLET_PRESSURE,
            values=(MARKS[inlet], *(float(other == inlet) for other in range(len(INLETS)))),
        )
        for inlet, crossing in enumerate(inlets)
    ]
    # Microliters per minute to cubic micrometers per second, over the cut pad.
    (leaving,) = scene.flows_of(OUTLET)
    flux = abs(leaving.flow_ul_min) * 1.0e9 / 60.0
    speed = flux / ((outlet.high - outlet.low) * outlet.height)
    ports.append(LayoutPort(outlet.side, outlet.low, outlet.high, speed=-speed))
    return LayoutDevice(window.boxes, window.size, ports=tuple(ports), wall_thickness=50.0)


def grid_for(chip: LayoutDevice) -> SignalGridSpec:
    """A signal grid over the chip, one layer of fluid deep: the marker and each liquid."""

    origin, shape = chip.lattice(SPACING)
    grid = SignalGridSpec()
    grid.signal_count = SIGNALS
    grid.shape, grid.origin, grid.spacing = shape, origin, Vec3(*SPACING)
    grid.diffusion = [DIFFUSIVITY] * SIGNALS
    grid.integration = SignalIntegrationKind.BACKWARD_EULER
    grid.solver.absolute_tolerance = 1.0e-5
    return grid


class _Transport:
    """A simulation that only carries its signals: there are no cells to regulate."""

    channel_metadata = CHANNELS

    def __init__(self, simulation: Simulation) -> None:
        self._simulation = simulation

    @property
    def simulation(self) -> Simulation:
        return self._simulation

    def step(self, dt: float) -> None:
        self._simulation.step(dt)

    def controller_state(self) -> JSONValue:
        return {"scope": "mixer-device"}


def build(context: ModelContext) -> _Transport:
    scene_path = context.parameters.get("scene")
    if not isinstance(scene_path, str):
        raise ValueError("the mixer needs a scene parameter: the path of a design scene")
    chip = device(load_design_scene(Path(scene_path)))
    simulation = context.simulation()
    grid = grid_for(chip)
    empty = [0.0] * SIGNALS
    chip.apply_to_grid(grid, empty, empty, simulation=simulation, tolerance=FLOW_TOLERANCE)
    shape = grid.shape
    simulation.configure_signal_grid(grid, [0.0] * (SIGNALS * shape.x * shape.y * shape.z))
    chip.add_constraints(simulation)
    return _Transport(simulation)
