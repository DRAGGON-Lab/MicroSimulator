from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
from microsimulator import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    BackendKind,
    BoxConstraintInit,
    CellInit,
    CheckpointBundle,
    CheckpointError,
    ConstraintRegion,
    CoupledRatePlan,
    GridShape,
    PlaneConstraintInit,
    RateInstruction,
    RateOp,
    SignalGridAffineReaction,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    SpeciesRatePlan,
    SphereConstraintInit,
    SphereRegion,
    Vec3,
    load_checkpoint,
    load_checkpoint_bundle,
    save_checkpoint,
)
from microsimulator.checkpoint import JSONValue


def _instruction(
    operation: RateOp,
    *,
    first: int = 0,
    second: int = 0,
    third: int = 0,
    value: float = 0.0,
) -> RateInstruction:
    instruction = RateInstruction()
    instruction.operation = operation
    instruction.first = first
    instruction.second = second
    instruction.third = third
    instruction.value = value
    return instruction


def _make_simulation() -> tuple[Simulation, int, int]:
    simulation = Simulation(BackendKind.CPU, species_count=2)
    shape = GridShape()
    shape.x = 3
    shape.y = 1
    shape.z = 1
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.diffusion = [0.5]
    grid.advection = [Vec3()]
    simulation.configure_signal_grid(grid, [0.0, 1.0, 0.0])
    simulation.set_species_rate_plan(
        SpeciesRatePlan(
            2,
            [
                _instruction(RateOp.SPECIES, first=0),
                _instruction(RateOp.CONSTANT, value=0.125),
                _instruction(RateOp.ADD, first=0, second=1),
                _instruction(RateOp.SPECIES, first=1),
                _instruction(RateOp.NEGATE, first=3),
            ],
            [2, 4],
        )
    )
    first = CellInit()
    first.position = Vec3(1.25, -2.5, 0.75)
    first.direction = Vec3(1.0, 2.0, 3.0)
    first.length = 4.5
    first.radius = 0.4
    first.growth_rate = 0.2
    first.cell_type = 7
    first.fixed = True
    first.species = [3.0, 1.5]
    first_id = simulation.add_cell(first)

    second = CellInit()
    second.position = Vec3(-0.5, 1.0, 2.0)
    second.length = 2.75
    second.radius = 0.3
    second.growth_rate = 0.05
    second.cell_type = -2
    second.species = [0.25, 4.0]
    simulation.add_cell(second)

    plane = PlaneConstraintInit()
    plane.point = Vec3(0.0, -3.0, 0.0)
    plane.inward_normal = Vec3(0.0, 2.0, 0.0)
    plane.coefficient = 1.25
    assert simulation.add_plane_constraint(plane) == 1

    sphere = SphereConstraintInit()
    sphere.center = Vec3(1.0, 2.0, 3.0)
    sphere.radius = 8.0
    sphere.coefficient = 0.75
    sphere.allowed_region = SphereRegion.INSIDE
    assert simulation.add_sphere_constraint(sphere) == 2

    box = BoxConstraintInit()
    box.center = Vec3(50.0, 50.0, 50.0)
    box.half_extents = Vec3(4.0, 2.0, 1.0)
    box.coefficient = 0.5
    box.allowed_region = ConstraintRegion.OUTSIDE
    assert simulation.add_box_constraint(box) == 3

    simulation.step(0.125)
    daughter_a, daughter_b = simulation.divide_equal(first_id)
    simulation.step(0.03125)
    return simulation, daughter_a, daughter_b


def _assert_cells_exact(
    actual: Simulation, expected: Simulation, *, compare_fixed: bool = True
) -> None:
    assert actual.time == expected.time
    assert actual.species_count == expected.species_count
    assert actual.signal_count == expected.signal_count
    assert actual.has_signal_grid == expected.has_signal_grid
    assert actual.has_coupled_rate_plan == expected.has_coupled_rate_plan
    if actual.has_signal_grid:
        assert actual.signal_levels == expected.signal_levels
    actual_cells = actual.cells()
    expected_cells = expected.cells()
    assert len(actual_cells) == len(expected_cells)
    for left, right in zip(actual_cells, expected_cells, strict=True):
        assert left.id == right.id
        assert left.slot == right.slot
        assert (left.position.x, left.position.y, left.position.z) == (
            right.position.x,
            right.position.y,
            right.position.z,
        )
        assert (left.direction.x, left.direction.y, left.direction.z) == (
            right.direction.x,
            right.direction.y,
            right.direction.z,
        )
        assert left.length == right.length
        assert left.radius == right.radius
        assert left.growth_rate == right.growth_rate
        assert left.cell_type == right.cell_type
        if compare_fixed:
            assert left.fixed == right.fixed
        assert left.species == right.species


def _document(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _rewrite_with_state_digest(path: Path, document: dict[str, Any]) -> None:
    simulation = document["simulation"]
    canonical = json.dumps(
        simulation,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    document["integrity"]["simulation"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")


def _remove_fixed_fields(document: dict[str, Any]) -> None:
    for cell in document["simulation"]["world"]["cells"]:
        del cell["fixed"]


def _remove_affine_reaction(document: dict[str, Any]) -> None:
    grid = document["simulation"]["signal_grid"]
    if grid is not None:
        del grid["spec"]["reaction"]


def _remove_constraint_boxes(document: dict[str, Any]) -> None:
    del document["simulation"]["constraints"]["boxes"]
    del document["simulation"]["constraints"]["cylinders"]


def _remove_grid_obstacles(document: dict[str, Any]) -> None:
    grid = document["simulation"].get("signal_grid")
    if grid is not None:
        del grid["spec"]["obstacles"]
        del grid["spec"]["velocity_field"]


@pytest.mark.parametrize("format_name", [CHECKPOINT_FORMAT, "cellmodeller2-checkpoint"])
def test_checkpoint_round_trip_resumes_exactly(tmp_path: Path, format_name: str) -> None:
    original, daughter_a, daughter_b = _make_simulation()
    path = tmp_path / "colony.cm2.json"
    save_checkpoint(
        original,
        path,
        provenance={"model": "two-species-test", "parameters": {"seed": 19}},
    )

    document = _document(path)
    assert document["format"] == CHECKPOINT_FORMAT
    assert document["version"] == CHECKPOINT_VERSION
    assert document["provenance"]["model"] == "two-species-test"
    assert document["source_backend"]["kind"] == "cpu"
    assert document["source_backend"]["device_index"] == 0
    assert "module_source" not in path.read_text(encoding="utf-8")
    assert list(tmp_path.glob(".*.tmp")) == []

    document["format"] = format_name
    path.write_text(json.dumps(document), encoding="utf-8")
    restored = load_checkpoint(path)
    _assert_cells_exact(restored, original)
    assert restored.lineage_parent(daughter_a) == 1
    assert restored.lineage_parent(daughter_b) == 1
    assert restored.signal_levels == original.signal_levels

    added = CellInit()
    added.species = [0.5, 0.75]
    assert restored.add_cell(added) == original.add_cell(added)
    plane = PlaneConstraintInit()
    assert restored.add_plane_constraint(plane) == original.add_plane_constraint(plane)
    restored.step(0.0625)
    original.step(0.0625)
    _assert_cells_exact(restored, original)


def test_checkpoint_bundle_reports_validated_source_metadata(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "source.cm2.json"
    save_checkpoint(simulation, path)

    bundle = load_checkpoint_bundle(path)

    assert bundle.schema_version == CHECKPOINT_VERSION
    assert bundle.source_backend.kind == "cpu"
    assert bundle.source_backend.name == "cpu-reference"
    assert bundle.source_backend.device == "host"
    assert bundle.source_backend.device_index == 0
    assert bundle.source_backend.native

    document = _document(path)
    document["source_backend"]["kind"] = "opencl"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CheckpointError, match="unknown backend kind"):
        load_checkpoint_bundle(path)


def test_version_one_checkpoint_migrates_to_an_empty_signal_state(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "legacy-v1.cm2.json"
    save_checkpoint(simulation, path)
    document = _document(path)
    document["version"] = 1
    del document["channel_metadata"]
    del document["integrity"]["channel_metadata"]
    del document["controller"]
    del document["integrity"]["controller"]
    del document["simulation"]["signal_grid"]
    del document["simulation"]["coupled_rate_plan"]
    _remove_fixed_fields(document)
    _remove_constraint_boxes(document)
    _remove_grid_obstacles(document)
    _rewrite_with_state_digest(path, document)

    restored = load_checkpoint(path)
    assert not restored.has_signal_grid
    assert restored.signal_count == 0
    assert restored.time == simulation.time


def test_version_two_checkpoint_migrates_without_a_coupled_plan(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "legacy-v2.cm2.json"
    save_checkpoint(simulation, path)
    document = _document(path)
    document["version"] = 2
    del document["channel_metadata"]
    del document["integrity"]["channel_metadata"]
    del document["controller"]
    del document["integrity"]["controller"]
    del document["simulation"]["coupled_rate_plan"]
    del document["simulation"]["signal_grid"]["spec"]["integration"]
    del document["simulation"]["signal_grid"]["spec"]["solver"]
    _remove_affine_reaction(document)
    _remove_fixed_fields(document)
    _remove_constraint_boxes(document)
    _remove_grid_obstacles(document)
    _rewrite_with_state_digest(path, document)

    restored = load_checkpoint(path)
    assert restored.has_signal_grid
    assert not restored.has_coupled_rate_plan
    assert restored.signal_levels == simulation.signal_levels


def test_version_three_checkpoint_migrates_without_controller_state(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "legacy-v3.cm2.json"
    save_checkpoint(simulation, path)
    document = _document(path)
    document["version"] = 3
    del document["channel_metadata"]
    del document["integrity"]["channel_metadata"]
    del document["controller"]
    del document["integrity"]["controller"]
    del document["simulation"]["signal_grid"]["spec"]["integration"]
    del document["simulation"]["signal_grid"]["spec"]["solver"]
    _remove_affine_reaction(document)
    _remove_fixed_fields(document)
    _remove_constraint_boxes(document)
    _remove_grid_obstacles(document)
    _rewrite_with_state_digest(path, document)

    bundle = load_checkpoint_bundle(path)
    assert isinstance(bundle, CheckpointBundle)
    assert bundle.controller is None
    assert all(not cell.fixed for cell in bundle.simulation.cells())
    _assert_cells_exact(bundle.simulation, simulation, compare_fixed=False)


def test_version_four_signal_grid_migrates_to_forward_euler(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "legacy-v4.cm2.json"
    save_checkpoint(simulation, path)
    document = _document(path)
    document["version"] = 4
    del document["channel_metadata"]
    del document["integrity"]["channel_metadata"]
    del document["simulation"]["signal_grid"]["spec"]["integration"]
    del document["simulation"]["signal_grid"]["spec"]["solver"]
    _remove_affine_reaction(document)
    _remove_fixed_fields(document)
    _remove_constraint_boxes(document)
    _remove_grid_obstacles(document)
    _rewrite_with_state_digest(path, document)

    restored = load_checkpoint(path)
    restored.step(0.25)
    simulation.step(0.25)
    assert restored.signal_levels == simulation.signal_levels


def test_version_five_cells_migrate_to_movable(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "legacy-v5.cm2.json"
    save_checkpoint(simulation, path)
    document = _document(path)
    document["version"] = 5
    del document["channel_metadata"]
    del document["integrity"]["channel_metadata"]
    _remove_affine_reaction(document)
    _remove_fixed_fields(document)
    _remove_constraint_boxes(document)
    _remove_grid_obstacles(document)
    _rewrite_with_state_digest(path, document)

    restored = load_checkpoint(path)
    assert all(not cell.fixed for cell in restored.cells())


def test_version_six_signal_grid_migrates_without_affine_reactions(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "legacy-v6.cm2.json"
    save_checkpoint(simulation, path)
    document = _document(path)
    document["version"] = 6
    del document["channel_metadata"]
    del document["integrity"]["channel_metadata"]
    _remove_affine_reaction(document)
    _remove_constraint_boxes(document)
    _remove_grid_obstacles(document)
    _rewrite_with_state_digest(path, document)

    restored = load_checkpoint(path)
    checkpoint = restored._checkpoint()
    assert checkpoint.signal_grid is not None
    assert checkpoint.signal_grid.spec.reaction is None


def test_version_seven_checkpoint_migrates_without_boxes(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "legacy-v7.cm2.json"
    save_checkpoint(simulation, path)
    document = _document(path)
    document["version"] = 7
    del document["channel_metadata"]
    del document["integrity"]["channel_metadata"]
    _remove_constraint_boxes(document)
    _remove_grid_obstacles(document)
    _rewrite_with_state_digest(path, document)

    restored = load_checkpoint(path)
    checkpoint = restored._checkpoint()
    assert checkpoint.constraints.boxes == []
    assert checkpoint.constraints.cylinders == []
    assert len(checkpoint.constraints.planes) == 1
    assert len(checkpoint.constraints.spheres) == 1
    box = BoxConstraintInit()
    assert restored.add_box_constraint(box) == 4


def test_grid_obstacles_round_trip_exactly(tmp_path: Path) -> None:
    simulation = Simulation()
    shape = GridShape()
    shape.x = 3
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.diffusion = [0.5]
    grid.advection = [Vec3()]
    grid.obstacles = [0, 1, 0]
    simulation.configure_signal_grid(grid, [2.0, 0.0, 0.5])
    path = tmp_path / "masked.cm2.json"
    save_checkpoint(simulation, path)

    document = _document(path)
    assert document["simulation"]["signal_grid"]["spec"]["obstacles"] == [0, 1, 0]

    restored = load_checkpoint(path)
    checkpoint = restored._checkpoint()
    assert checkpoint.signal_grid is not None
    assert list(checkpoint.signal_grid.spec.obstacles) == [0, 1, 0]
    restored.step(0.25)
    assert restored.signal_levels == [2.0, 0.0, 0.5]


def test_affine_grid_reaction_round_trips_exactly(tmp_path: Path) -> None:
    simulation = Simulation()
    shape = GridShape()
    shape.x = 2
    shape.y = 1
    shape.z = 1
    reaction = SignalGridAffineReaction()
    reaction.source_rates = [2.0, 0.0]
    reaction.loss_rates = [0.5, 1.0]
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    grid.reaction = reaction
    simulation.configure_signal_grid(grid, [1.0, 2.0])
    path = tmp_path / "affine-reaction.cm2.json"

    save_checkpoint(simulation, path)

    document = _document(path)
    assert document["simulation"]["signal_grid"]["spec"]["reaction"] == {
        "source_rates": [2.0, 0.0],
        "loss_rates": [0.5, 1.0],
    }
    restored = load_checkpoint(path)
    checkpoint = restored._checkpoint()
    assert checkpoint.signal_grid is not None
    restored_reaction = checkpoint.signal_grid.spec.reaction
    assert restored_reaction is not None
    assert restored_reaction.source_rates == [2.0, 0.0]
    assert restored_reaction.loss_rates == [0.5, 1.0]
    restored.step(0.5)
    assert restored.signal_levels == [1.75, 1.0]

    invalid = _document(path)
    invalid["simulation"]["signal_grid"]["spec"]["reaction"]["loss_rates"][0] = -0.5
    _rewrite_with_state_digest(path, invalid)
    with pytest.raises(CheckpointError, match="affine loss rates"):
        load_checkpoint(path)


def test_crank_nicolson_signal_configuration_round_trips(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    checkpoint = simulation._checkpoint()
    assert checkpoint.signal_grid is not None
    checkpoint.signal_grid.spec.integration = SignalIntegrationKind.CRANK_NICOLSON
    checkpoint.signal_grid.spec.solver.max_iterations = 321
    checkpoint.signal_grid.spec.solver.absolute_tolerance = 2.0e-7
    checkpoint.signal_grid.spec.solver.relative_tolerance = 3.0e-6
    simulation = Simulation(BackendKind.CPU, checkpoint)
    path = tmp_path / "crank-nicolson.cm2.json"
    save_checkpoint(simulation, path)

    document = _document(path)
    assert document["simulation"]["signal_grid"]["spec"]["integration"] == "crank_nicolson"
    restored = load_checkpoint(path)
    restored_checkpoint = restored._checkpoint()
    assert restored_checkpoint.signal_grid is not None
    spec = restored_checkpoint.signal_grid.spec
    assert spec.integration is SignalIntegrationKind.CRANK_NICOLSON
    assert spec.solver.max_iterations == 321
    assert math.isclose(spec.solver.absolute_tolerance, 2.0e-7, rel_tol=1.0e-6)
    assert math.isclose(spec.solver.relative_tolerance, 3.0e-6, rel_tol=1.0e-6)


def test_controller_state_is_authenticated_and_cannot_be_silently_discarded(
    tmp_path: Path,
) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "controlled.cm2.json"
    controller: dict[str, JSONValue] = {
        "kind": "test-controller",
        "cells": {"2": {"threshold": 3.5}},
    }
    save_checkpoint(
        simulation,
        path,
        provenance={"model": "controlled-test"},
        controller=controller,
    )

    bundle = load_checkpoint_bundle(path)
    assert bundle.controller == controller
    assert bundle.provenance == {"model": "controlled-test"}
    _assert_cells_exact(bundle.simulation, simulation)
    with pytest.raises(CheckpointError, match="controller state"):
        load_checkpoint(path)

    document = _document(path)
    document["controller"]["cells"]["2"]["threshold"] = 4.0
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(CheckpointError, match="controller digest"):
        load_checkpoint_bundle(path)


def test_coupled_plan_round_trip_is_exact(tmp_path: Path) -> None:
    simulation = Simulation(species_count=1)
    shape = GridShape()
    shape.x = 1
    shape.y = 1
    shape.z = 1
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.diffusion = [0.0]
    grid.advection = [Vec3()]
    simulation.configure_signal_grid(grid, [2.0])
    cell = CellInit()
    cell.growth_rate = 0.0
    cell.species = [1.0]
    simulation.add_cell(cell)
    simulation.set_coupled_rate_plan(
        CoupledRatePlan(
            1,
            1,
            [
                _instruction(RateOp.SIGNAL, first=0),
                _instruction(RateOp.CONSTANT, value=0.0),
            ],
            [0],
            [1],
        )
    )
    path = tmp_path / "coupled.cm2.json"
    save_checkpoint(simulation, path)

    document = _document(path)
    coupled = document["simulation"]["coupled_rate_plan"]
    assert coupled["instructions"][0]["operation"] == "signal"
    restored = load_checkpoint(path)
    assert restored.has_coupled_rate_plan
    restored.step(0.5)
    simulation.step(0.5)
    _assert_cells_exact(restored, simulation)

    invalid = _document(path)
    invalid["simulation"]["coupled_rate_plan"]["signal_outputs"] = []
    _rewrite_with_state_digest(path, invalid)
    with pytest.raises(CheckpointError, match="coupled signal output count"):
        load_checkpoint(path)


def test_checkpoint_rejects_corruption_and_invalid_state(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "colony.cm2.json"
    save_checkpoint(simulation, path)

    corrupted = _document(path)
    corrupted["simulation"]["world"]["cells"][0]["length"] = 99.0
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(CheckpointError, match="digest"):
        load_checkpoint(path)

    save_checkpoint(simulation, path)
    invalid = _document(path)
    invalid["simulation"]["world"]["cells"][0]["slot"] = 7
    _rewrite_with_state_digest(path, invalid)
    with pytest.raises(CheckpointError, match="slots"):
        load_checkpoint(path)

    save_checkpoint(simulation, path)
    unsupported = _document(path)
    unsupported["version"] = CHECKPOINT_VERSION + 1
    path.write_text(json.dumps(unsupported), encoding="utf-8")
    with pytest.raises(CheckpointError, match="unsupported checkpoint version"):
        load_checkpoint(path)

    save_checkpoint(simulation, path)
    unknown = _document(path)
    unknown["simulation"]["world"]["mystery"] = 1
    _rewrite_with_state_digest(path, unknown)
    with pytest.raises(CheckpointError, match="unknown keys"):
        load_checkpoint(path)


def test_checkpoint_rejects_executable_or_non_json_values(tmp_path: Path) -> None:
    simulation, _, _ = _make_simulation()
    path = tmp_path / "colony.cm2.json"
    with pytest.raises(CheckpointError, match="provenance"):
        save_checkpoint(simulation, path, provenance={"callback": object()})  # type: ignore[dict-item]
    assert not path.exists()

    path.write_text('{"format":"microsimulator-checkpoint","value":NaN}', encoding="utf-8")
    with pytest.raises(CheckpointError, match="non-finite"):
        load_checkpoint(path)

    path.write_text('{"format":"first","format":"second"}', encoding="utf-8")
    with pytest.raises(CheckpointError, match="duplicate key"):
        load_checkpoint(path)
