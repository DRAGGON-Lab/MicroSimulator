from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, cast

import microsimulator as cm
import pytest
from microsimulator.analysis import export_dataset, open_dataset
from microsimulator.controller import ControllerStep, NativeController, StepPlan
from microsimulator.scene import capture_scene, dumps_scene, parse_scene


def configured(*, division: bool = False) -> cm.Simulation:
    sim = cm.Simulation(species_count=1)
    cell = cm.CellInit()
    cell.position = cm.Vec3(8.5 if division else 5.5, 5.5, 5.5)
    cell.radius, cell.length = 2, 8 if division else 1
    cell.growth_rate = 0
    cell.fixed = True
    cell.species = [3]
    sim.add_cell(cell)
    config = cm.CultureConfiguration()
    config.grid.shape.x = 18 if division else 12
    config.grid.shape.y = config.grid.shape.z = 12
    config.maximum_retries = 1
    config.stepping.minimum_gap_m = 0.1e-6
    solute = cm.Solute()
    solute.name, solute.diffusion = "tracer", 0.1
    config.solutes = [solute]
    sim._configure_culture(config, [2], [80])

    return sim


def test_explicit_biochemical_volume_and_checkpoint(tmp_path: Path) -> None:
    sim = configured()
    instruction = cm.RateInstruction()
    instruction.operation = cm.RateOp.CELL_VOLUME
    sim.set_species_rate_plan(cm.SpeciesRatePlan(1, [instruction], [0]))
    sim.step(0.125)
    state = sim.culture_checkpoint
    assert state is not None
    assert state.cells[0].biochemical_volume == 80
    assert state.cells[0].species_amounts == [1040]
    assert sim.cell(1).species == [13]
    path = tmp_path / "media.json"
    cm.save_checkpoint(sim, path)
    restored = cm.load_checkpoint(path)
    assert restored.has_culture
    saved = restored.culture_checkpoint
    assert saved is not None
    assert saved.cells[0].species_amounts == state.cells[0].species_amounts
    assert saved.extracellular_amounts == state.extracellular_amounts
    assert (
        saved.last_report.transport.mass_balance_error
        == state.last_report.transport.mass_balance_error
    )
    sim.step(0.125)
    restored.step(0.125)
    assert restored.cell(1).species == sim.cell(1).species


def test_volume_and_amount_conserving_division() -> None:
    sim = configured(division=True)
    before = sim.culture_checkpoint
    assert before is not None
    first, second = sim.divide(1, 0.4)
    after = sim.culture_checkpoint
    assert after is not None
    assert {c.body.id for c in after.cells} == {first, second}
    assert math.isclose(
        sum(c.body.geometric_volume for c in after.cells),
        before.cells[0].body.geometric_volume,
        rel_tol=1e-13,
    )
    assert sum(c.biochemical_volume for c in after.cells) == 80
    assert sum(c.species_amounts[0] for c in after.cells) == 240
    assert math.isclose(
        math.fsum(after.extracellular_amounts),
        math.fsum(before.extracellular_amounts),
        rel_tol=2e-5,
    )
    assert sim.time == 0
    assert sim.lineage_parent(first) == sim.lineage_parent(second) == 1

    for daughter in after.cells:
        assert sim.cell(daughter.body.id).species == [3]


def test_controller_restores_regulation_rng_and_native_state() -> None:
    sim = configured()
    rng = random.Random(42)
    random_state = rng.getstate()

    def regulate(context: ControllerStep) -> StepPlan:
        context.rng.random()
        context.state["attempt"] = 99
        context.simulation.set_species(1, [10])
        raise RuntimeError("regulation failed")

    controller = NativeController(
        sim,
        model_id="media-test",
        model_version=1,
        rng=rng,
        regulate=regulate,
        state={"attempt": 0},
    )

    with pytest.raises(RuntimeError, match="regulation failed"):
        controller.step(0.1)

    assert controller.state == {"attempt": 0}
    assert rng.getstate() == random_state
    assert controller.completed_steps == 0
    assert sim.time == 0
    assert sim.cell(1).species == [3]
    assert sim.culture_checkpoint is not None
    assert sim.culture_checkpoint.cells[0].species_amounts == [240]


def test_culture_mode_rejects_unmodelled_geometry_mutations() -> None:
    sim = configured()

    with pytest.raises(RuntimeError, match="poses change"):
        sim.set_cell_geometry(1, cm.Vec3(5, 5, 5), cm.Vec3(1, 0, 0), 2)

    with pytest.raises(RuntimeError, match="removal"):
        sim.remove_cell(1)

    with pytest.raises(RuntimeError, match="seed cells"):
        sim.add_cell(cm.CellInit())

    assert sim.cell_count == 1


def test_culture_scene_and_analysis_preserve_fragment_amounts(tmp_path: Path) -> None:
    sim = configured()
    frame = capture_scene(sim)
    assert frame.culture is not None
    assert frame.culture.length_unit_m == 1e-6
    assert frame.culture.cells[0].biochemical_volume == 80
    assert frame.culture.cells[0].species_amounts == (240,)
    assert parse_scene(dumps_scene(frame)) == frame
    assert frame.signal_grid is not None

    for fragment in frame.culture.fragments:
        assert math.isclose(fragment.amounts[0] / fragment.volume, 2)

    source = tmp_path / "source.json"
    cm.save_checkpoint(sim, source)
    output = tmp_path / "analysis"
    export_dataset([source], output)
    dataset = open_dataset(output)
    assert dataset.has_table("fluid_fragments.parquet")
    assert dataset.has_table("culture_cells.parquet")
    cells = dataset.scan_table("culture_cells.parquet").collect()
    assert cells["biochemical_volume"].to_list() == [80]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()


def _save_version_ten(sim: cm.Simulation, path: Path) -> None:
    cm.save_checkpoint(sim, path)
    document = cast(dict[str, Any], json.loads(path.read_text()))
    document["version"] = 10
    state = document["simulation"].pop("culture")
    document["simulation"]["media_flow"] = state
    config = state["configuration"]
    config["medium"] = config.pop("fluid")

    for key in (
        "growth",
        "events",
        "authoring_json",
        "coupling_tolerance",
        "maximum_coupling_iterations",
    ):
        config.pop(key)

    for solute in config["solutes"]:
        solute.pop("amount_unit")

    for boundary in config["reservoirs"]:
        boundary.pop("kind")
        boundary.pop("allow_backflow")

    for cell in state["cells"]:
        for key in ("uptake_totals", "realized_specific_rate", "biomass_produced"):
            cell.pop(key)

    state.pop("time")
    state.pop("event_index")
    document["integrity"]["simulation"] = _digest(document["simulation"])
    path.write_text(json.dumps(document))


def test_version_ten_checkpoint_preserves_reserve_semantics(tmp_path: Path) -> None:
    sim = configured()
    path = tmp_path / "legacy.json"
    _save_version_ten(sim, path)
    restored = cm.load_checkpoint(path)
    current = restored.culture_checkpoint
    assert current is not None
    assert current.configuration.growth == []
    assert current.configuration.solutes[0].amount_unit == "model"
    assert current.cells[0].species_amounts == [240]
    assert current.cells[0].biochemical_volume == 80
    sim.step(0.125)
    restored.step(0.125)
    assert restored.culture_checkpoint is not None
    assert sim.culture_checkpoint is not None
    assert (
        restored.culture_checkpoint.extracellular_amounts
        == sim.culture_checkpoint.extracellular_amounts
    )


def test_version_four_scene_reads_into_culture() -> None:
    fixture = Path(__file__).parents[2] / "viewer/tests/fixtures/media-v4.scene.json"
    frame = parse_scene(fixture.read_text())
    assert frame.culture is not None
    assert frame.culture.cells[0].species_amounts == (240,)
    assert frame.culture.solute_amount_units == ("model",)
    assert parse_scene(dumps_scene(frame)) == frame


def test_version_four_analysis_preserves_identity_and_uses_canonical_names(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    cm.save_checkpoint(configured(), source)
    output = tmp_path / "dataset"
    export_dataset([source], output)
    path = output / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(path.read_text()))
    manifest["version"] = 4

    for current, legacy in {
        "culture_frames.parquet": "media_frames.parquet",
        "culture_cells.parquet": "media_cells.parquet",
        "fluid_fragments.parquet": "media_fragments.parquet",
        "chemical_transfers.parquet": "media_reservoirs.parquet",
    }.items():
        (output / current).rename(output / legacy)
        manifest["tables"][legacy] = manifest["tables"].pop(current)

    manifest["dataset_id"] = _digest(
        {
            key: manifest[key]
            for key in ("format", "version", "sources", "options", "tables", "signals")
        }
    )
    path.write_text(json.dumps(manifest))
    dataset = open_dataset(output)
    assert dataset.verified
    assert dataset.manifest["dataset_id"] == manifest["dataset_id"]
    assert dataset.has_table("culture_cells.parquet")
    assert dataset.scan_table("culture_cells.parquet").collect()[
        "biochemical_volume"
    ].to_list() == [80]


@pytest.mark.parametrize(
    "malformation", ["missing_fluid", "new_growth", "new_unit", "invalid_array"]
)
def test_version_ten_rejects_malformed_original_schema(tmp_path: Path, malformation: str) -> None:
    path = tmp_path / "legacy.json"
    _save_version_ten(configured(), path)
    document = cast(dict[str, Any], json.loads(path.read_text()))
    config = document["simulation"]["media_flow"]["configuration"]

    if malformation == "missing_fluid":
        config.pop("medium")
    elif malformation == "new_growth":
        config["growth"] = []
    elif malformation == "new_unit":
        config["solutes"][0]["amount_unit"] = "mol"
    else:
        config["solutes"] = {}

    document["integrity"]["simulation"] = _digest(document["simulation"])
    path.write_text(json.dumps(document))

    with pytest.raises(cm.CheckpointError, match=r"schema|array"):
        cm.load_checkpoint(path)
