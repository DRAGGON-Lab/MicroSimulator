from __future__ import annotations

import math
import random
from collections.abc import Callable
from pathlib import Path

import pytest
from microsimulator import (
    BackendFeature,
    BackendKind,
    BoxConstraintInit,
    CellInit,
    ConstraintRegion,
    ContactDynamicsConfig,
    ContactDynamicsParameters,
    ControllerStateError,
    ControllerStep,
    DivisionEvent,
    DivisionRequest,
    GridShape,
    GrowthLoadLaw,
    GrowthLoadResponse,
    MechanicsConfig,
    NativeController,
    PlaneConstraintInit,
    RatePlanBuilder,
    SignalGridSpec,
    Simulation,
    StepPlan,
    Vec3,
    backend_available,
    load_checkpoint_bundle,
    save_checkpoint,
)
from microsimulator.biomass import biomass_volume

BACKENDS = [pytest.param(backend, id=backend.name.lower()) for backend in BackendKind]


def _simulation(backend: BackendKind, **arguments: int) -> Simulation:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    simulation = Simulation(backend, **arguments)
    if not simulation.supports(BackendFeature.CONTACT_DYNAMICS):
        pytest.skip("backend does not implement contact dynamics")
    return simulation


def _between_walls(simulation: Simulation, half_width: float) -> None:
    for sign in (-1.0, 1.0):
        wall = PlaneConstraintInit()
        wall.point = Vec3(sign * half_width, 0.0, 0.0)
        wall.inward_normal = Vec3(-sign, 0.0, 0.0)
        simulation.add_plane_constraint(wall)


def _stalling(stall_load: float) -> ContactDynamicsParameters:
    parameters = ContactDynamicsParameters()
    growth = parameters.growth
    growth.law = GrowthLoadLaw.LINEAR
    growth.stall_load = stall_load
    parameters.growth = growth
    return parameters


def _divide_at(length: float) -> Callable[[ControllerStep], StepPlan]:
    """Return a regulation callback that divides cells reaching ``length``."""

    def regulate(step: ControllerStep) -> StepPlan:
        return StepPlan(
            divisions=tuple(
                DivisionRequest(cell.id, 0.5) for cell in step.cells if cell.length >= length
            )
        )

    return regulate


def _tilt_daughters(step: ControllerStep, event: DivisionEvent) -> None:
    """Turn each daughter by a small angle in the plane so a line of cells can buckle."""

    for daughter in (event.first, event.second):
        angle = step.rng.gauss(0.0, 0.01)
        x, y = daughter.direction.x, daughter.direction.y
        step.simulation.set_cell_geometry(
            daughter.id,
            daughter.position,
            Vec3(
                x * math.cos(angle) - y * math.sin(angle),
                x * math.sin(angle) + y * math.cos(angle),
                daughter.direction.z,
            ),
            daughter.length,
        )


def _chamber(backend: BackendKind, config: ContactDynamicsConfig) -> NativeController:
    simulation = _simulation(backend)
    chamber = BoxConstraintInit()
    chamber.half_extents = Vec3(2.0, 2.0, 0.6)
    chamber.allowed_region = ConstraintRegion.INSIDE
    simulation.add_box_constraint(chamber)
    founder = CellInit()
    founder.length = 1.0
    founder.radius = 0.4
    simulation.add_cell(founder)
    return NativeController(
        simulation,
        model_id="contact-dynamics-chamber-test",
        model_version=1,
        rng=random.Random(3),
        regulate=_divide_at(2.0),
        on_division=_tilt_daughters,
        mechanics=MechanicsConfig(contact_dynamics=config),
    )


def _footprint(simulation: Simulation) -> float:
    return sum(
        math.pi * cell.radius**2 + 2.0 * cell.radius * cell.length for cell in simulation.cells()
    )


def _deepest_overlap(simulation: Simulation) -> float:
    contacts = simulation.find_cell_contacts().contacts
    walls = simulation.find_external_contacts().contacts
    return max([0.0, *(-contact.signed_separation for contact in (*contacts, *walls))])


def test_growth_load_response_factor() -> None:
    response = GrowthLoadResponse()
    assert response.factor(50.0) == 1.0
    response.law = GrowthLoadLaw.EXPONENTIAL
    response.sensitivity = 0.5
    assert math.isclose(response.factor(2.0), math.exp(-1.0), rel_tol=1e-6)
    response.law = GrowthLoadLaw.LINEAR
    response.free_load, response.stall_load = 1.0, 3.0
    assert response.factor(1.0) == 1.0
    assert math.isclose(response.factor(2.0), 0.5, rel_tol=1e-6)
    assert response.factor(9.0) == 0.0
    response.stall_load = 0.5
    with pytest.raises(ValueError, match="stall load"):
        response.factor(1.0)


def test_contact_dynamics_config_round_trips_and_validates() -> None:
    config = ContactDynamicsConfig(
        growth_law="linear", growth_free_load=1.0, growth_stall_load=8.0, contact_stiffness=2000.0
    )
    mechanics = MechanicsConfig(flow_drift=True, contact_dynamics=config)
    assert MechanicsConfig.from_json(mechanics.to_json()) == mechanics

    parameters = config.native_parameters()
    assert parameters.growth.law == GrowthLoadLaw.LINEAR
    assert parameters.growth.stall_load == 8.0
    assert parameters.contact_stiffness == 2000.0

    # A payload written without the entry selects relaxation passes.
    legacy = MechanicsConfig(passes=2).to_json()
    del legacy["contact_dynamics"]
    assert MechanicsConfig.from_json(legacy) == MechanicsConfig(passes=2)

    with pytest.raises(ControllerStateError, match="growth_law"):
        ContactDynamicsConfig(growth_law="quadratic")
    with pytest.raises(ControllerStateError, match="stall load"):
        ContactDynamicsConfig(growth_law="linear", growth_free_load=2.0, growth_stall_load=2.0)
    with pytest.raises(ControllerStateError, match="contact_stiffness"):
        ContactDynamicsConfig(contact_stiffness=0.0)
    invalid = mechanics.to_json()
    invalid["contact_dynamics"] = {"drag": 1.0}
    with pytest.raises(ControllerStateError, match="contact dynamics configuration"):
        MechanicsConfig.from_json(invalid)
    with pytest.raises(ControllerStateError, match="ContactDynamicsConfig"):
        MechanicsConfig(contact_dynamics=object())  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", BACKENDS)
def test_unloaded_cells_grow_exactly_as_without_contact(backend: BackendKind) -> None:
    simulation = _simulation(backend)
    reference = Simulation(backend)
    cell = CellInit()
    cell.length = 3.0
    cell.growth_rate = 0.8
    for target in (simulation, reference):
        target.add_cell(cell)
    for dt in (0.0, 0.05, 0.4, 1.0):
        result = simulation.step(dt, ContactDynamicsParameters())
        reference.step(dt)
        # Substeps add the same elongation in pieces, to rounding.
        assert math.isclose(simulation.cell(1).length, reference.cell(1).length, rel_tol=1e-6)
        assert simulation.time == reference.time
        assert result.report.max_rows == 0
        assert result.axial_loads == [0.0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_cell_between_walls_stalls_and_stops_consuming(backend: BackendKind) -> None:
    simulation = _simulation(backend, species_count=1)
    spec = SignalGridSpec()
    shape = GridShape()
    shape.x = shape.y = shape.z = 1
    spec.shape = shape
    spec.origin = Vec3(-4.0, -4.0, -4.0)
    spec.spacing = Vec3(8.0, 8.0, 8.0)
    spec.signal_count, spec.diffusion, spec.advection = 1, [0], [Vec3()]
    simulation.configure_signal_grid(spec, [10.0])
    rates = RatePlanBuilder()
    simulation.set_coupled_rate_plan(
        rates.coupled_plan(1, 1, (rates.constant(0),), (-rates.cell_volume_change_rate() / 0.4,))
    )
    _between_walls(simulation, 2.0)
    cell = CellInit()
    cell.length, cell.radius, cell.species = 2.0, 0.5, [1.0]
    cid = simulation.add_cell(cell)
    initial = biomass_volume(cell.length, cell.radius)

    parameters = _stalling(4.0)
    result = simulation.step(0.0, parameters)
    for _ in range(200):
        result = simulation.step(0.05, parameters)
        current = simulation.cell(cid)
        consumed = (10.0 - simulation.signal_levels[0]) * spec.voxel_volume
        grown = biomass_volume(current.length, current.radius) - initial
        # Uptake is charged for the elongation that happened, not the nominal rate.
        assert math.isclose(0.4 * consumed, grown, abs_tol=2e-4)

    # The walls leave room for a cylinder of length 3.
    filled = simulation.cell(cid)
    assert 3.0 - 2e-3 <= filled.length <= 3.0 + 2e-3
    assert math.isclose(result.axial_loads[0], 4.0, rel_tol=0.05)
    assert result.length_increments[0] == 0.0
    level = simulation.signal_levels[0]
    for _ in range(20):
        simulation.step(0.05, parameters)
    assert simulation.signal_levels[0] == level
    assert math.isclose(simulation.cell(cid).length, filled.length, abs_tol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_closed_chamber_fills_and_holds(backend: BackendKind) -> None:
    controller = _chamber(backend, ContactDynamicsConfig(growth_law="linear", growth_stall_load=2.0))
    for _ in range(300):
        controller.step(0.05)
    simulation = controller.simulation
    count, footprint = simulation.cell_count, _footprint(simulation)
    for _ in range(300):
        controller.step(0.05)

    assert simulation.cell_count == count
    assert math.isclose(_footprint(simulation), footprint, abs_tol=0.02)
    assert 0.6 < _footprint(simulation) / 16.0 < 1.0
    assert _deepest_overlap(simulation) < 0.01
    result = controller.last_contact_dynamics
    assert result is not None
    assert max(result.length_increments) < 1e-5
    assert controller.last_mechanics_reports == ()
    for cell in simulation.cells():
        assert abs(cell.position.x) <= 2.0 and abs(cell.position.y) <= 2.0
        assert abs(cell.position.z) < 1e-3


@pytest.mark.parametrize("backend", BACKENDS)
def test_unconfined_colony_grows_in_full_without_overlap(backend: BackendKind) -> None:
    simulation = _simulation(backend, reserved_capacity=512)
    founder = CellInit()
    founder.length = 2.0
    simulation.add_cell(founder)
    controller = NativeController(
        simulation,
        model_id="contact-dynamics-colony-test",
        model_version=1,
        rng=random.Random(5),
        regulate=_divide_at(4.0),
        on_division=_tilt_daughters,
        mechanics=MechanicsConfig(contact_dynamics=ContactDynamicsConfig()),
    )
    # The same lineage with growth and division alone.
    reference = Simulation(BackendKind.CPU, reserved_capacity=512)
    reference.add_cell(founder)
    worst = 0.0
    for _ in range(130):
        for cell in reference.cells():
            if cell.length >= 4.0:
                reference.divide(cell.id, 0.5)
        reference.step(0.05)
        controller.step(0.05)
        result = controller.last_contact_dynamics
        assert result is not None
        worst = max(worst, result.report.max_overlap)

    assert simulation.cell_count == reference.cell_count >= 64
    length = sum(cell.length for cell in simulation.cells())
    assert math.isclose(length, sum(cell.length for cell in reference.cells()), rel_tol=1e-4)
    assert worst < 0.01
    assert all(abs(cell.position.z) < 1e-3 for cell in simulation.cells())
    # A colony that buckled is far wider than its cells are long, and no
    # longer a single line.
    xs = [cell.position.x for cell in simulation.cells()]
    ys = [cell.position.y for cell in simulation.cells()]
    assert max(ys) - min(ys) > 4.0
    assert max(xs) - min(xs) < 0.5 * length


@pytest.mark.parametrize("backend", BACKENDS)
def test_resume_reproduces_the_uninterrupted_trajectory(
    backend: BackendKind, tmp_path: Path
) -> None:
    config = ContactDynamicsConfig(growth_law="linear", growth_stall_load=2.0)
    uninterrupted = _chamber(backend, config)
    for _ in range(120):
        uninterrupted.step(0.05)

    split = _chamber(backend, config)
    for _ in range(70):
        split.step(0.05)
    midpoint = tmp_path / "midpoint.json"
    save_checkpoint(split.simulation, midpoint, controller=split.controller_state())
    resumed = NativeController.from_checkpoint(
        load_checkpoint_bundle(midpoint, backend=backend),
        model_id="contact-dynamics-chamber-test",
        model_version=1,
        regulate=_divide_at(2.0),
        on_division=_tilt_daughters,
    )
    for _ in range(50):
        resumed.step(0.05)

    assert resumed.controller_state() == uninterrupted.controller_state()
    assert resumed.simulation.cell_count == uninterrupted.simulation.cell_count > 4
    for actual, expected in zip(
        resumed.simulation.cells(), uninterrupted.simulation.cells(), strict=True
    ):
        assert actual.id == expected.id
        assert actual.length == expected.length
        assert (actual.position.x, actual.position.y) == (expected.position.x, expected.position.y)


@pytest.mark.parametrize("backend", BACKENDS)
def test_free_colony_matches_the_continuum_pressure_profile(backend: BackendKind) -> None:
    """A colony growing without load response against its drag on the substrate.

    Treating the colony as an incompressible fluid that grows at the areal
    rate g and is dragged at zeta per unit area gives the radial pressure
    p(r) = zeta * g * (R^2 - r^2) / 4. Drag per unit cell length becomes
    zeta = drag * packing / diameter per unit area, and the axial load on a
    cell is the pressure across its diameter.
    """

    simulation = _simulation(backend, reserved_capacity=2048)
    founder = CellInit()
    founder.length = 2.0
    simulation.add_cell(founder)
    rng = random.Random(1)
    thresholds: dict[int, float] = {}
    parameters = ContactDynamicsParameters()
    areas: list[float] = []
    step_length = 0.05
    result = simulation.step(0.0, parameters)
    for _ in range(200):
        for cell in simulation.cells():
            threshold = thresholds.setdefault(cell.id, 4.0 * (1.0 + 0.2 * (rng.random() - 0.5)))
            if cell.length < threshold:
                continue
            for daughter_id in simulation.divide(cell.id, 0.5):
                daughter = simulation.cell(daughter_id)
                angle = rng.gauss(0.0, 0.02)
                x, y = daughter.direction.x, daughter.direction.y
                simulation.set_cell_geometry(
                    daughter_id,
                    daughter.position,
                    Vec3(
                        x * math.cos(angle) - y * math.sin(angle),
                        x * math.sin(angle) + y * math.cos(angle),
                        0.0,
                    ),
                    daughter.length,
                )
        result = simulation.step(step_length, parameters)
        areas.append(_footprint(simulation))

    cells = simulation.cells()
    assert len(cells) > 800
    center_x = sum(cell.position.x for cell in cells) / len(cells)
    center_y = sum(cell.position.y for cell in cells) / len(cells)
    radii = [math.hypot(cell.position.x - center_x, cell.position.y - center_y) for cell in cells]
    colony_radius = sorted(radii)[int(0.98 * len(radii))]
    packing = areas[-1] / (math.pi * colony_radius**2)
    areal_rate = math.log(areas[-1] / areas[-21]) / (20 * step_length)
    # Rigid cells pack to a little under a full layer.
    assert 0.8 < packing < 1.0

    bin_count = 6
    bins: list[list[float]] = [[] for _ in range(bin_count)]
    for radius, cell in zip(radii, cells, strict=True):
        bins[min(bin_count - 1, int(radius / colony_radius * bin_count))].append(
            result.axial_loads[cell.slot]
        )
    means = [sum(loads) / len(loads) for loads in bins]
    central = parameters.drag * packing * areal_rate * colony_radius**2 / 4.0
    for index, mean in enumerate(means):
        position = (index + 0.5) / bin_count
        predicted = central * (1.0 - position**2)
        assert math.isclose(mean, predicted, rel_tol=0.35, abs_tol=0.05 * central)
    # The few cells of the innermost bin are noisy; the fall toward the edge is not.
    assert means[1:] == sorted(means[1:], reverse=True)
    assert means[0] > 3.0 * means[-1]


def test_backend_without_the_feature_rejects_the_step() -> None:
    for backend in BackendKind:
        if not backend_available(backend):
            continue
        simulation = Simulation(backend)
        if simulation.supports(BackendFeature.CONTACT_DYNAMICS):
            continue
        simulation.add_cell(CellInit())
        with pytest.raises(RuntimeError, match="contact dynamics"):
            simulation.step(0.1, ContactDynamicsParameters())
        assert simulation.time == 0.0
