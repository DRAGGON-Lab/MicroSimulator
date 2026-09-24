from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest
from microsimulator import (
    BackendKind,
    ModelContext,
    SimulationController,
    Vec3,
    build_model,
    load_checkpoint_bundle,
    run_simulation,
)
from microsimulator.checkpoint import JSONValue

_ROOT = Path(__file__).resolve().parents[2]
_TUTORIALS = _ROOT / "examples" / "tutorials"

_MODELS: tuple[tuple[str, dict[str, JSONValue], float], ...] = (
    *(("biophysics.py", {"scenario": scenario}, 0.001) for scenario in (
        "basics",
        "two_types",
        "short_cells",
        "competition",
        "box",
    )),
    *(("gene_expression.py", {"scenario": scenario}, 0.001) for scenario in (
        "constitutive",
        "legacy_constitutive",
        "dilution",
        "derepression",
        "oscillator",
    )),
    *(("signaling.py", {"scenario": scenario}, 0.001) for scenario in (
        "single_gene",
        "communication",
        "mutualism",
    )),
    *(("simbol_circuits.py", {"circuit": circuit}, 0.001) for circuit in (
        "bba_0001",
        "bba_0002",
        "bba_0003",
        "bba_0004",
        "bba_0005",
        "bba_i5200",
    )),
    ("plasmid_segregation.py", {"copies_per_cell": 10}, 0.001),
    ("conjugation.py", {"transfer_probability": 0.1}, 0.001),
    ("danino_clock.py", {}, 0.001),
    ("biopixel_trap.py", {}, 0.001),
    ("pillar_channel.py", {}, 0.001),
)


@pytest.mark.parametrize(("filename", "parameters", "dt"), _MODELS)
def test_all_tutorial_scenarios_build_step_and_checkpoint(
    filename: str,
    parameters: dict[str, JSONValue],
    dt: float,
    tmp_path: Path,
) -> None:
    model, provenance = build_model(
        _TUTORIALS / filename,
        ModelContext(BackendKind.CPU, 0, seed=17, parameters=parameters),
    )
    assert isinstance(model, SimulationController)
    output = tmp_path / f"{filename}-{next(iter(parameters.values()), 'default')}.cm2.json"
    run_simulation(
        model,
        steps=1,
        dt=dt,
        output=output,
        provenance=provenance,
    )
    checkpoint = load_checkpoint_bundle(output)
    checkpoint.simulation.validate()
    assert checkpoint.controller is not None


def test_plasmid_tutorial_keeps_exact_copy_counts_and_published_fractions() -> None:
    model, _ = build_model(
        _TUTORIALS / "plasmid_segregation.py",
        ModelContext(BackendKind.CPU, 0, seed=23, parameters={"copies_per_cell": 6}),
    )
    assert isinstance(model, SimulationController)
    model.step(0.01)
    state = cast(dict[str, object], model.controller_state())
    model_state = cast(dict[str, object], state["state"])
    plasmids = cast(dict[str, dict[str, int]], model_state["plasmids"])
    for cell in model.simulation.cells():
        counts = plasmids[str(cell.id)]
        assert counts["a"] + counts["b"] == 6
        assert math.isclose(cell.species[0] + cell.species[1], 1.0, abs_tol=1.0e-7)


def test_conjugation_tutorial_uses_current_contact_graph() -> None:
    model, _ = build_model(
        _TUTORIALS / "conjugation.py",
        ModelContext(
            BackendKind.CPU,
            0,
            seed=29,
            parameters={"transfer_probability": 1.0},
        ),
    )
    assert isinstance(model, SimulationController)
    acceptor, donor = model.simulation.cells()
    model.simulation.set_cell_geometry(
        donor.id,
        Vec3(acceptor.position.x, acceptor.position.y + 0.7, acceptor.position.z),
        donor.direction,
        donor.length,
    )
    model.step(0.0)
    assert model.simulation.cell(acceptor.id).cell_type == 2


def test_danino_tutorial_uses_device_flow_obstacles_and_washout() -> None:
    model, _ = build_model(
        _TUTORIALS / "danino_clock.py",
        ModelContext(BackendKind.CPU, 0, seed=31),
    )
    assert isinstance(model, SimulationController)
    assert model.simulation.species_count == 3
    assert model.simulation.signal_count == 2
    checkpoint = model.simulation._checkpoint()
    assert checkpoint.signal_grid is not None
    spec = checkpoint.signal_grid.spec
    assert spec.reaction is None
    assert spec.velocity_field is not None
    assert any(value != 0.0 for value in spec.velocity_field.y_faces)
    # The solved field is dominated by the axial channel flow; transverse
    # components exist only as weak circulation at the trap mouth.
    assert max(abs(value) for value in spec.velocity_field.x_faces) < max(
        abs(value) for value in spec.velocity_field.y_faces
    )
    solid = sum(spec.obstacles)
    assert 0 < solid < len(spec.obstacles)
    assert spec.y_lower.values == [0.0, 10.0]
    assert len(checkpoint.constraints.boxes) == 4


def test_pillar_channel_anchors_sheds_and_washes_out() -> None:
    model, _ = build_model(
        _TUTORIALS / "pillar_channel.py",
        ModelContext(BackendKind.CPU, 0, seed=7),
    )
    assert isinstance(model, SimulationController)
    # 250 steps crosses the Brinkman re-solve cadence at step 100 and sheds
    # daughters from every anchored lineage into the stream.
    for _ in range(250):
        model.step(0.02)

    adhesion_sites = ((-20.0, -46.0), (20.0, -46.0), (0.0, 14.0))
    cells = model.simulation.cells()
    anchored = [cell for cell in cells if cell.fixed]
    released = [cell for cell in cells if not cell.fixed]
    assert len(anchored) == 3
    assert len(released) > 3
    for cell in anchored:
        nearest = min(
            math.hypot(cell.position.x - x, cell.position.y - y) for x, y in adhesion_sites
        )
        assert nearest < 4.0
    # Released cells drift downstream of the anchors; the flow is doing work.
    assert any(cell.position.y > 30.0 for cell in released)
    for cell in cells:
        assert cell.position.z == 0.0
        assert abs(cell.position.x) < 40.0
        assert abs(cell.position.y) < 120.0


def test_plasmid_tutorial_resume_is_exact(tmp_path: Path) -> None:
    model_path = _TUTORIALS / "plasmid_segregation.py"
    parameters: dict[str, JSONValue] = {"copies_per_cell": 6}

    uninterrupted, _ = build_model(
        model_path,
        ModelContext(BackendKind.CPU, 0, seed=37, parameters=parameters),
    )
    assert isinstance(uninterrupted, SimulationController)
    for _ in range(6):
        uninterrupted.step(0.1)

    split, provenance = build_model(
        model_path,
        ModelContext(BackendKind.CPU, 0, seed=37, parameters=parameters),
    )
    assert isinstance(split, SimulationController)
    midpoint = tmp_path / "plasmid-midpoint.cm2.json"
    run_simulation(
        split,
        steps=3,
        dt=0.1,
        output=midpoint,
        provenance=provenance,
    )
    resumed, _ = build_model(
        model_path,
        ModelContext(BackendKind.CPU, 0, seed=37, parameters=parameters),
        checkpoint=load_checkpoint_bundle(midpoint),
    )
    assert isinstance(resumed, SimulationController)
    for _ in range(3):
        resumed.step(0.1)

    assert resumed.controller_state() == uninterrupted.controller_state()
    assert resumed.simulation.time == uninterrupted.simulation.time
    assert [
        (
            cell.id,
            resumed.simulation.lineage_parent(cell.id),
            cell.position.x,
            cell.position.y,
            cell.position.z,
            cell.direction.x,
            cell.direction.y,
            cell.direction.z,
            cell.length,
            cell.cell_type,
            cell.species,
        )
        for cell in resumed.simulation.cells()
    ] == [
        (
            cell.id,
            uninterrupted.simulation.lineage_parent(cell.id),
            cell.position.x,
            cell.position.y,
            cell.position.z,
            cell.direction.x,
            cell.direction.y,
            cell.direction.z,
            cell.length,
            cell.cell_type,
            cell.species,
        )
        for cell in uninterrupted.simulation.cells()
    ]


_FOUNDER_MODELS: tuple[tuple[str, dict[str, JSONValue], float], ...] = (
    *_MODELS,
    ("../culture_dish.py", {}, 0.001),
    ("../microfluidic_trap.py", {}, 0.001),
)


def _founder_targets(model: SimulationController) -> dict[str, float]:
    controller = cast(dict[str, JSONValue], model.controller_state())
    state = cast(dict[str, JSONValue], controller["state"])
    if "division_targets" in state:
        return cast(dict[str, float], state["division_targets"])
    policy = cast(dict[str, JSONValue], state["length_division"])
    return cast(dict[str, float], policy["targets"])


@pytest.mark.parametrize("seed", (0, 7, 17, 71))
@pytest.mark.parametrize(("filename", "parameters", "dt"), _FOUNDER_MODELS)
def test_tutorial_founders_do_not_divide_without_growth(
    filename: str, parameters: dict[str, JSONValue], dt: float, seed: int,
) -> None:
    model, _ = build_model(
        _TUTORIALS / filename,
        ModelContext(BackendKind.CPU, 0, seed=seed, parameters=parameters),
    )
    assert isinstance(model, SimulationController)
    targets = _founder_targets(model)
    ids = [cell.id for cell in model.simulation.cells()]
    assert set(targets) == {str(cell_id) for cell_id in ids}
    assert all(cell.length <= targets[str(cell.id)] for cell in model.simulation.cells())
    model.step(0.0)
    assert [cell.id for cell in model.simulation.cells()] == ids
    # Existing strict comparison still divides each founder after its length grows.
    for cell in model.simulation.cells():
        model.simulation.set_cell_geometry(
            cell.id, cell.position, cell.direction, targets[str(cell.id)] + 0.01,
        )
    model.step(0.0)
    assert not set(ids) & {cell.id for cell in model.simulation.cells()}


@pytest.mark.parametrize(("filename", "parameters", "dt"), _FOUNDER_MODELS)
def test_tutorial_founder_initialization_is_deterministic_and_resume_does_not_cap(
    filename: str, parameters: dict[str, JSONValue], dt: float, tmp_path: Path,
) -> None:
    context = ModelContext(BackendKind.CPU, 0, seed=71, parameters=parameters)
    first, provenance = build_model(_TUTORIALS / filename, context)
    second, _ = build_model(
        _TUTORIALS / filename,
        ModelContext(BackendKind.CPU, 0, seed=71, parameters=parameters),
    )
    assert isinstance(first, SimulationController)
    assert isinstance(second, SimulationController)
    assert first.controller_state() == second.controller_state()
    assert [cell.length for cell in first.simulation.cells()] == [
        cell.length for cell in second.simulation.cells()
    ]
    for cell in first.simulation.cells():
        first.simulation.set_cell_geometry(cell.id, cell.position, cell.direction, 8.0)
    path = tmp_path / "oversized.cm2.json"
    run_simulation(first, steps=0, dt=dt, output=path, provenance=provenance)
    restored, _ = build_model(
        _TUTORIALS / filename,
        ModelContext(BackendKind.CPU, 0, seed=71, parameters=parameters),
        checkpoint=load_checkpoint_bundle(path),
    )
    assert isinstance(restored, SimulationController)
    assert restored.controller_state() == first.controller_state()
    assert all(cell.length == 8.0 for cell in restored.simulation.cells())


def test_conjugation_rare_short_gaussian_target_is_not_resampled() -> None:
    import random

    model, _ = build_model(
        _TUTORIALS / "conjugation.py",
        ModelContext(BackendKind.CPU, 0, seed=3103),
    )
    assert isinstance(model, SimulationController)
    expected = random.Random(3103)
    targets = _founder_targets(model)
    cells = model.simulation.cells()
    # Native CellInit stores the requested 1.9 in single precision.
    requested = 1.899999976158142
    for cell in cells:
        assert targets[str(cell.id)] == requested + expected.gauss(1.9, 0.45)
        assert cell.length <= targets[str(cell.id)]
    assert cells[0].length < requested
    ids = [cell.id for cell in cells]
    model.step(0.0)
    assert [cell.id for cell in model.simulation.cells()] == ids
