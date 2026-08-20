from __future__ import annotations

import copy
import json
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    CellUpdate,
    ControllerPlanError,
    ControllerStateError,
    ControllerStep,
    DivisionEvent,
    DivisionRequest,
    GridBoundaryKind,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    SignalGridSpec,
    SignalGridVelocityField,
    Simulation,
    SimulationController,
    StepPlan,
    Vec3,
    backend_available,
    build_model,
    capture_random_state,
    load_checkpoint_bundle,
    restore_random_state,
    run_simulation,
    save_checkpoint,
)
from microsimulator.checkpoint import JSONValue


def _one_cell(backend: BackendKind = BackendKind.CPU) -> Simulation:
    simulation = Simulation(backend, species_count=1)
    cell = CellInit()
    cell.length = 4.0
    cell.radius = 0.5
    cell.species = [2.0]
    simulation.add_cell(cell)
    return simulation


def _simulation_payload(path: Path) -> object:
    document = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    return document["simulation"]


def _one_cell_in_uniform_flow() -> Simulation:
    sites = 9
    shape = GridShape()
    shape.x, shape.y, shape.z = sites, 1, 1
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.spacing = Vec3(1.0, 1.0, 1.0)
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    grid.x_lower.kind = GridBoundaryKind.FIXED
    grid.x_lower.values = [0.0]
    grid.x_upper.kind = GridBoundaryKind.FIXED
    grid.x_upper.values = [0.0]
    velocity = SignalGridVelocityField()
    velocity.x_faces = [0.5] * (sites + 1)
    velocity.y_faces = [0.0] * (2 * sites)
    velocity.z_faces = [0.0] * (2 * sites)
    grid.velocity_field = velocity

    simulation = Simulation(BackendKind.CPU)
    simulation.configure_signal_grid(grid)
    cell = CellInit()
    cell.position = Vec3(3.0, 0.0, 0.0)
    cell.direction = Vec3(1.0, 0.0, 0.0)
    cell.length = 1.0
    cell.radius = 0.25
    simulation.add_cell(cell)
    return simulation


def test_random_stream_round_trip_preserves_uniform_and_gaussian_draws() -> None:
    stream = random.Random(1729)
    stream.random()
    stream.gauss(0.0, 1.0)
    state = capture_random_state(stream)
    restored = restore_random_state(state)

    assert [restored.random() for _ in range(8)] == [stream.random() for _ in range(8)]
    assert [restored.gauss(0.0, 1.0) for _ in range(8)] == [
        stream.gauss(0.0, 1.0) for _ in range(8)
    ]


def test_random_stream_rejects_malformed_state() -> None:
    def change_version(value: dict[str, Any]) -> None:
        value["version"] = 2

    def add_field(value: dict[str, Any]) -> None:
        value["extra"] = None

    def shorten_vector(value: dict[str, Any]) -> None:
        cast(list[Any], value["state"]).pop()

    def invalidate_word(value: dict[str, Any]) -> None:
        cast(list[Any], value["state"])[0] = -1

    def invalidate_gaussian(value: dict[str, Any]) -> None:
        value["gauss_next"] = float("nan")

    mutations: list[Callable[[dict[str, Any]], None]] = [
        change_version,
        add_field,
        shorten_vector,
        invalidate_word,
        invalidate_gaussian,
    ]
    for mutation in mutations:
        value = cast(dict[str, Any], copy.deepcopy(capture_random_state(random.Random(1))))
        mutation(value)
        with pytest.raises(ControllerStateError, match="random state"):
            restore_random_state(value)


def test_mechanics_config_round_trip_preserves_flow_drift() -> None:
    configuration = MechanicsConfig(passes=2, flow_drift=True)

    assert MechanicsConfig.from_json(configuration.to_json()) == configuration

    invalid = configuration.to_json()
    invalid["flow_drift"] = 1
    with pytest.raises(
        ControllerStateError,
        match=r"mechanics\.flow_drift must be Boolean",
    ):
        MechanicsConfig.from_json(invalid)


def test_native_controller_resume_preserves_flow_drift_trajectory(tmp_path: Path) -> None:
    def build() -> NativeController:
        return NativeController(
            _one_cell_in_uniform_flow(),
            model_id="flow-drift-resume-test",
            model_version=1,
            rng=random.Random(7),
            mechanics=MechanicsConfig(flow_drift=True),
        )

    uninterrupted = build()
    for _ in range(4):
        uninterrupted.step(0.25)

    split = build()
    for _ in range(2):
        split.step(0.25)
    midpoint = tmp_path / "flow-midpoint.cm2.json"
    save_checkpoint(split.simulation, midpoint, controller=split.controller_state())
    resumed = NativeController.from_checkpoint(
        load_checkpoint_bundle(midpoint),
        model_id="flow-drift-resume-test",
        model_version=1,
    )
    for _ in range(2):
        resumed.step(0.25)

    expected = tmp_path / "flow-expected.cm2.json"
    actual = tmp_path / "flow-actual.cm2.json"
    save_checkpoint(
        uninterrupted.simulation,
        expected,
        controller=uninterrupted.controller_state(),
    )
    save_checkpoint(resumed.simulation, actual, controller=resumed.controller_state())

    assert _simulation_payload(actual) == _simulation_payload(expected)
    assert resumed.controller_state() == uninterrupted.controller_state()
    assert math.isclose(resumed.simulation.cell(1).position.x, 3.5, abs_tol=1.0e-6)


@pytest.mark.parametrize("backend", list(BackendKind))
def test_native_controller_composes_regulation_division_and_mechanics(
    backend: BackendKind,
) -> None:
    if not backend_available(backend):
        pytest.skip("native backend is not built")
    simulation = _one_cell(backend)

    def regulate(step: ControllerStep) -> StepPlan:
        parent = step.cells[0]
        step.state["regulated"] = True
        return StepPlan(
            updates=(
                CellUpdate(
                    parent.id,
                    growth_rate=0.2,
                    cell_type=7,
                    species=(3.0,),
                ),
            ),
            divisions=(DivisionRequest(parent.id, 0.25),),
        )

    def divided(step: ControllerStep, event: DivisionEvent) -> None:
        step.state["daughter_ids"] = [event.first.id, event.second.id]
        step.simulation.set_species(event.first.id, [4.0])
        step.simulation.set_species(event.second.id, [5.0])

    controller = NativeController(
        simulation,
        model_id="controller-integration-test",
        model_version=1,
        rng=random.Random(17),
        regulate=regulate,
        on_division=divided,
        mechanics=MechanicsConfig(passes=2),
    )
    assert isinstance(controller, SimulationController)

    controller.step(0.1)

    assert controller.completed_steps == 1
    assert controller.state["regulated"] is True
    daughter_ids = cast(list[JSONValue], controller.state["daughter_ids"])
    assert len(daughter_ids) == 2
    assert len(controller.last_mechanics_reports) == 2
    for daughter_id in daughter_ids:
        assert isinstance(daughter_id, int)
        daughter = simulation.cell(daughter_id)
        assert simulation.lineage_parent(daughter.id) == 1
        assert daughter.cell_type == 7
        assert math.isclose(daughter.growth_rate, 0.2, rel_tol=1.0e-6)


@pytest.mark.parametrize(
    "controller_kind", ["microsimulator-native-controller", "cellmodeller2-native-controller"]
)
def test_native_controller_resumes_model_state_rng_and_mechanics(
    tmp_path: Path, controller_kind: str
) -> None:
    def regulate(step: ControllerStep) -> StepPlan:
        draws = step.state.get("draws", 0)
        if not isinstance(draws, int):
            raise AssertionError("invalid test state")
        step.state["draws"] = draws + 1
        updates = tuple(
            CellUpdate(cell.id, growth_rate=0.05 + 0.1 * step.rng.random()) for cell in step.cells
        )
        divisions = (
            (DivisionRequest(step.cells[0].id),)
            if step.completed_steps == 1 and len(step.cells) == 1
            else ()
        )
        return StepPlan(updates=updates, divisions=divisions)

    def build() -> NativeController:
        return NativeController(
            _one_cell(),
            model_id="exact-resume-test",
            model_version=3,
            rng=random.Random(99),
            regulate=regulate,
            mechanics=MechanicsConfig(passes=1),
            state={"draws": 0},
        )

    uninterrupted = build()
    for _ in range(5):
        uninterrupted.step(0.125)

    split = build()
    for _ in range(2):
        split.step(0.125)
    midpoint = tmp_path / "midpoint.cm2.json"
    controller_state = split.controller_state()
    controller_state["kind"] = controller_kind
    save_checkpoint(
        split.simulation,
        midpoint,
        controller=controller_state,
    )
    resumed = NativeController.from_checkpoint(
        load_checkpoint_bundle(midpoint),
        model_id="exact-resume-test",
        model_version=3,
        regulate=regulate,
    )
    for _ in range(3):
        resumed.step(0.125)

    expected = tmp_path / "expected.cm2.json"
    actual = tmp_path / "actual.cm2.json"
    save_checkpoint(
        uninterrupted.simulation,
        expected,
        controller=uninterrupted.controller_state(),
    )
    save_checkpoint(
        resumed.simulation,
        actual,
        controller=resumed.controller_state(),
    )
    assert _simulation_payload(actual) == _simulation_payload(expected)
    assert resumed.controller_state() == uninterrupted.controller_state()
    assert resumed.completed_steps == 5
    assert resumed.state == {"draws": 5}


def test_native_controller_validates_complete_plan_before_mutation() -> None:
    simulation = _one_cell()

    def invalid(step: ControllerStep) -> StepPlan:
        cell_id = step.cells[0].id
        return StepPlan(
            updates=(
                CellUpdate(cell_id, growth_rate=0.1),
                CellUpdate(cell_id, growth_rate=0.2),
            )
        )

    controller = NativeController(
        simulation,
        model_id="invalid-plan-test",
        model_version=1,
        rng=random.Random(0),
        regulate=invalid,
    )
    before = simulation.cell(1)
    with pytest.raises(ControllerPlanError, match="duplicate"):
        controller.step(0.1)
    after = simulation.cell(1)
    assert after.growth_rate == before.growth_rate
    assert simulation.time == 0.0
    assert controller.completed_steps == 0


def test_native_controller_rejects_wrong_model_identity(tmp_path: Path) -> None:
    controller = NativeController(
        _one_cell(),
        model_id="identity-test",
        model_version=1,
        rng=random.Random(0),
    )
    path = tmp_path / "identity.cm2.json"
    save_checkpoint(
        controller.simulation,
        path,
        controller=controller.controller_state(),
    )
    bundle = load_checkpoint_bundle(path)
    with pytest.raises(ControllerStateError, match="identity does not match"):
        NativeController.from_checkpoint(
            bundle,
            model_id="identity-test",
            model_version=2,
        )


def test_native_controller_example_builds_and_resumes(tmp_path: Path) -> None:
    model = Path(__file__).resolve().parents[2] / "examples" / "native_controller.py"
    parameters: dict[str, JSONValue] = {
        "initial_length": 3.5,
        "division_length": 4.0,
    }
    controller, provenance = build_model(
        model,
        ModelContext(BackendKind.CPU, 0, seed=31, parameters=parameters),
    )
    assert isinstance(controller, NativeController)
    first = tmp_path / "first.cm2.json"
    run_simulation(
        controller,
        steps=3,
        dt=0.25,
        output=first,
        provenance=provenance,
    )

    resumed, resumed_provenance = build_model(
        model,
        ModelContext(BackendKind.CPU, 0, seed=31, parameters=parameters),
        checkpoint=load_checkpoint_bundle(first),
    )
    assert isinstance(resumed, NativeController)
    output = tmp_path / "resumed.cm2.json"
    run_simulation(
        resumed,
        steps=2,
        dt=0.25,
        output=output,
        provenance=resumed_provenance,
    )
    assert load_checkpoint_bundle(output).controller is not None
    assert resumed.completed_steps == 5

def test_step_plan_removals_delete_cells_after_divisions() -> None:
    simulation = Simulation(BackendKind.CPU, species_count=0)
    first = CellInit()
    first_id = simulation.add_cell(first)
    second = CellInit()
    second.position = Vec3(5.0, 0.0, 0.0)
    second_id = simulation.add_cell(second)

    def regulate(step: ControllerStep) -> StepPlan:
        del step
        return StepPlan(removals=(second_id,))

    controller = NativeController(
        simulation,
        model_id="removal-test",
        model_version=1,
        rng=random.Random(3),
        regulate=regulate,
    )
    controller.step(0.05)

    identifiers = [cell.id for cell in simulation.cells()]
    assert identifiers == [first_id]

    def bad_regulate(step: ControllerStep) -> StepPlan:
        del step
        return StepPlan(removals=(second_id,))

    bad_controller = NativeController(
        simulation,
        model_id="removal-test",
        model_version=1,
        rng=random.Random(3),
        regulate=bad_regulate,
    )
    with pytest.raises(ControllerPlanError, match="unknown or duplicate"):
        bad_controller.step(0.05)

    def conflicted(step: ControllerStep) -> StepPlan:
        del step
        return StepPlan(
            divisions=(DivisionRequest(first_id),),
            removals=(first_id,),
        )

    conflict_controller = NativeController(
        simulation,
        model_id="removal-test",
        model_version=1,
        rng=random.Random(3),
        regulate=conflicted,
    )
    with pytest.raises(ControllerPlanError, match="removes a dividing cell"):
        conflict_controller.step(0.05)
