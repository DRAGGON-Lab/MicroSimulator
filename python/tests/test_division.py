from __future__ import annotations

import random
from typing import cast

import pytest
from microsimulator import (
    BackendKind,
    CellInit,
    ControllerStateError,
    ControllerStep,
    NativeController,
    Simulation,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from microsimulator.checkpoint import JSONValue


def test_uniform_length_division_tracks_native_identities() -> None:
    simulation = Simulation(BackendKind.CPU)
    founder = CellInit()
    founder.length = 4.0
    founder_id = simulation.add_cell(founder)
    stream = random.Random(7)
    state: dict[str, JSONValue] = {}
    policy = UniformLengthDivision(2.0, 2.5, jitter_z=False)
    policy.initialize(state, stream, (founder_id,))

    def regulate(step: ControllerStep) -> StepPlan:
        return StepPlan(divisions=policy.requests(step))

    controller = NativeController(
        simulation,
        model_id="division-policy-test",
        model_version=1,
        rng=stream,
        regulate=regulate,
        on_division=policy.on_division,
        state=state,
    )
    controller.step(0.0)

    targets_state = cast(dict[str, JSONValue], controller.state["length_division"])
    targets = cast(dict[str, JSONValue], targets_state["targets"])
    daughters = simulation.cells()
    assert set(targets) == {str(cell.id) for cell in daughters}
    assert {simulation.lineage_parent(cell.id) for cell in daughters} == {founder_id}
    assert all(cell.direction.z == 0.0 for cell in daughters)


def test_division_and_removal_in_the_same_plan_coexist() -> None:
    # Plan removals apply after divisions, so a forgotten (about-to-be-removed)
    # cell is still active while division callbacks run.
    simulation = Simulation(BackendKind.CPU)
    divider = CellInit()
    divider.length = 4.0
    divider.position = Vec3(0.0, 5.0, 0.0)
    divider_id = simulation.add_cell(divider)
    leaver = CellInit()
    leaver.length = 1.0
    leaver_id = simulation.add_cell(leaver)
    stream = random.Random(11)
    state: dict[str, JSONValue] = {}
    policy = UniformLengthDivision(2.0, 2.5, jitter_z=False)
    policy.initialize(state, stream, (divider_id, leaver_id))

    def regulate(step: ControllerStep) -> StepPlan:
        divisions = policy.requests(step)
        policy.forget(step, (leaver_id,))
        return StepPlan(divisions=divisions, removals=(leaver_id,))

    controller = NativeController(
        simulation,
        model_id="division-with-removal-test",
        model_version=1,
        rng=stream,
        regulate=regulate,
        on_division=policy.on_division,
        state=state,
    )
    controller.step(0.0)

    targets_state = cast(dict[str, JSONValue], controller.state["length_division"])
    targets = cast(dict[str, JSONValue], targets_state["targets"])
    daughters = simulation.cells()
    assert len(daughters) == 2
    assert set(targets) == {str(cell.id) for cell in daughters}
    assert all(simulation.lineage_parent(cell.id) == divider_id for cell in daughters)


def test_uniform_length_division_rejects_missing_target_state() -> None:
    simulation = Simulation()
    simulation.add_cell(CellInit())
    policy = UniformLengthDivision(2.0, 2.5)
    controller = NativeController(
        simulation,
        model_id="missing-target-test",
        model_version=1,
        rng=random.Random(0),
        regulate=lambda step: StepPlan(divisions=policy.requests(step)),
    )
    with pytest.raises(ControllerStateError, match="length_division"):
        controller.step(0.1)


def test_founder_initialization_caps_native_precision_without_resampling() -> None:
    from microsimulator import capped_founder_length

    stream = random.Random(71)
    expected = random.Random(71)
    policy = UniformLengthDivision(2.5, 3.0)
    simulation = Simulation(BackendKind.CPU, species_count=2)
    founders: list[CellInit] = []
    for index, length in enumerate((3.5, 1.0, 3.5, 2.75)):
        founder = CellInit()
        founder.position = Vec3(index * 10.0, 2.0, 3.0)
        founder.direction = Vec3(0.0, 1.0, 0.0)
        founder.length = length
        founder.radius = 0.4
        founder.cell_type = index
        founder.species = [2.0, 3.0]
        founders.append(founder)
    state: dict[str, JSONValue] = {}
    ids = policy.initialize_founders(simulation, state, stream, tuple(founders))
    target_state = cast(dict[str, JSONValue], state[policy.state_key])
    targets = cast(dict[str, float], target_state["targets"])
    for index, (cell_id, requested) in enumerate(zip(ids, (3.5, 1.0, 3.5, 2.75), strict=True)):
        target = expected.uniform(2.5, 3.0)
        cell = simulation.cell(cell_id)
        assert targets[str(cell_id)] == target
        assert cell.length <= target
        assert cell.length == capped_founder_length(requested, target)
        assert (cell.position.x, cell.position.y, cell.position.z) == (index * 10.0, 2.0, 3.0)
        assert (cell.direction.x, cell.direction.y, cell.direction.z) == (0.0, 1.0, 0.0)
        assert abs(cell.radius - 0.4) < 1.0e-7
        assert cell.cell_type == index
        assert cell.species == [2.0, 3.0]
    assert stream.getstate() == expected.getstate()
    with pytest.raises(ControllerStateError, match="already contains"):
        policy.initialize_founders(simulation, state, stream, ())


def test_capped_founder_length_rounds_down_when_nearest_float_exceeds_target() -> None:
    from microsimulator import capped_founder_length

    target = 2.99999999
    founder = CellInit()
    founder.length = target
    assert founder.length > target  # nearest native float rounds up
    founder.length = capped_founder_length(3.5, target)
    assert founder.length <= target
    assert capped_founder_length(1.5, target) == 1.5
    assert capped_founder_length(0.0, 0.0) == 0.0
    for invalid in (-1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            capped_founder_length(invalid, 3.0)
        with pytest.raises(ValueError):
            capped_founder_length(3.0, invalid)
