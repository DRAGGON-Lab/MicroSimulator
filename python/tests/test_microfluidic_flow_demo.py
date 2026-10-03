from pathlib import Path

from microsimulator import (
    BackendKind,
    ModelContext,
    NativeController,
    build_model,
    capture_scene,
    load_checkpoint_bundle,
    run_simulation,
)

MODEL = Path(__file__).resolve().parents[2] / "examples/microfluidic_flow.py"


def test_dye_advects_around_solid_pillars_and_restarts(tmp_path: Path) -> None:
    context = ModelContext(BackendKind.CPU, 0, seed=42, parameters={"cells": False})
    model, provenance = build_model(MODEL, context)
    assert isinstance(model, NativeController)
    grid = model.simulation._checkpoint().signal_grid
    assert grid is not None
    field = grid.spec.velocity_field
    assert field is not None
    assert max(field.x_faces) > 1
    assert min(field.x_faces) < -1
    assert max(field.y_faces) > 40
    assert model.simulation.cell_count == 0

    checkpoint = tmp_path / "flow.json"
    run_simulation(model, steps=80, dt=0.025, output=checkpoint, provenance=provenance)
    scene = capture_scene(model.simulation, channel_metadata=model.channel_metadata)
    assert len(scene.constraints.cylinders) == 5
    assert scene.signal_grid is not None
    assert scene.channel_metadata.signals == (
        "Flow tracer (relative concentration)",
        "Nutrient (relative concentration)",
    )
    levels = scene.signal_grid.levels[: grid.spec.site_count]
    ny, nz = grid.spec.shape.y, grid.spec.shape.z
    total = sum(levels)
    dye_center_y = (
        sum(
            value * (grid.spec.origin.y + ((i // nz) % ny) * grid.spec.spacing.y)
            for i, value in enumerate(levels)
        )
        / total
    )
    assert dye_center_y > -80  # The pulse has moved beyond the inlet strip.
    assert all(
        value == 0 for value, solid in zip(levels, grid.spec.obstacles, strict=True) if solid
    )

    resumed, _ = build_model(MODEL, context, checkpoint=load_checkpoint_bundle(checkpoint))
    assert isinstance(resumed, NativeController)

    for _ in range(50):
        model.step(0.025)
        resumed.step(0.025)

    assert resumed.simulation.time == model.simulation.time
    assert resumed.simulation.signal_levels == model.simulation.signal_levels
    assert model.controller_state() == resumed.controller_state()


def test_cells_grow_divide_drift_consume_nutrients_and_restart(tmp_path: Path) -> None:
    context = ModelContext(BackendKind.CPU, 0, seed=42)
    model, provenance = build_model(MODEL, context)
    assert isinstance(model, NativeController)
    initial = {cell.id: cell for cell in model.simulation.cells()}
    assert len(initial) == 32
    assert sum(cell.fixed for cell in initial.values()) == 20

    model.step(0.025)
    current = {cell.id: cell for cell in model.simulation.cells()}
    assert any(cell.length > initial[cell.id].length for cell in current.values())
    assert any(
        cell.position.y > initial[cell.id].position.y + 0.1
        for cell in current.values()
        if not cell.fixed
    )
    assert all(
        cell.position.y == initial[cell.id].position.y for cell in current.values() if cell.fixed
    )

    # An otherwise identical cell-free run separates uptake from outlet losses.
    control, _ = build_model(
        MODEL, ModelContext(BackendKind.CPU, 0, seed=42, parameters={"cells": False})
    )
    assert isinstance(control, NativeController)
    control.step(0.025)
    site_count = len(model.simulation.signal_levels) // 2
    assert sum(model.simulation.signal_levels[site_count:]) < sum(
        control.simulation.signal_levels[site_count:]
    )

    checkpoint = tmp_path / "cells.json"
    run_simulation(model, steps=40, dt=0.025, output=checkpoint, provenance=provenance)
    assert model.simulation.cell_count > len(initial)
    assert sum(cell.fixed for cell in model.simulation.cells()) == 20
    assert any(cell.parent_id is not None for cell in capture_scene(model.simulation).cells)

    resumed, _ = build_model(MODEL, context, checkpoint=load_checkpoint_bundle(checkpoint))
    assert isinstance(resumed, NativeController)

    for _ in range(20):
        model.step(0.025)
        resumed.step(0.025)

    assert capture_scene(model.simulation) == capture_scene(resumed.simulation)
    assert model.controller_state() == resumed.controller_state()


def test_starved_cells_do_not_grow() -> None:
    model, _ = build_model(MODEL, ModelContext(BackendKind.CPU, 0, seed=42))
    assert isinstance(model, NativeController)
    levels = model.simulation.signal_levels
    site_count = len(levels) // 2
    levels[site_count:] = [0.0] * site_count
    model.simulation.set_signal_levels(levels)
    initial_lengths = {cell.id: cell.length for cell in model.simulation.cells()}
    model.step(0.025)

    assert {cell.id: cell.length for cell in model.simulation.cells()} == initial_lengths
    assert all(cell.growth_rate == 0 for cell in model.simulation.cells())
