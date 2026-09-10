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
