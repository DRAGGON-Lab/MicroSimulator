from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from microsimulator import (
    LegacyPickleError,
    import_legacy_pickle,
    load_checkpoint_bundle,
    save_checkpoint,
)
from microsimulator.checkpoint import JSONValue
from microsimulator.cli import main


def _snapshot_bytes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tuple_format: bool = False,
    dict_style: bool = False,
) -> bytes:
    package = ModuleType("CellModeller")
    module = ModuleType("CellModeller.CellState")
    bases = (dict,) if dict_style else ()
    namespace: dict[str, object] = {"__module__": "CellModeller.CellState"}
    if dict_style:
        dict_type = cast(Any, dict)
        namespace.update(
            {
                "__getattr__": dict_type.get,
                "__setattr__": dict_type.__setitem__,
            }
        )
    cell_class = cast(type[Any], type("CellState", bases, namespace))
    module.CellState = cell_class  # type: ignore[attr-defined]
    package.CellState = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "CellModeller", package)
    monkeypatch.setitem(sys.modules, "CellModeller.CellState", module)

    first = cell_class()
    first.id = 3
    first.idx = 0
    first.pos = [1.0, 2.0, 3.0]
    first.dir = [2.0, 0.0, 0.0]
    first.length = 4.0
    first.radius = 0.5
    first.growthRate = 0.25
    first.cellType = 7
    first.species = np.asarray([1.5, 2.5], dtype=np.float32)
    first.targetVol = 6.0
    first.color = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    second = cell_class()
    second.id = 7
    second.idx = 1
    second.pos = [-1.0, -2.0, -3.0]
    second.dir = [0.0, 3.0, 0.0]
    second.length = 5.0
    second.radius = 0.6
    second.growthRate = 0.5
    second.cellType = 9
    second.species = np.asarray([3.5, 4.5], dtype=np.float32)
    second.divideFlag = False

    cells = {7: second, 3: first}
    lineage = {3: 1, 7: 1}
    if tuple_format:
        return pickle.dumps((cells, lineage), protocol=2)
    return pickle.dumps(
        {
            "cellStates": cells,
            "lineage": lineage,
            "stepNum": 4,
            "moduleName": "legacy_growth",
            "moduleStr": "def setup(sim):\n    pass\n",
        },
        protocol=4,
    )


def test_trusted_pickle_migrates_geometry_species_identity_and_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "legacy.pickle"
    source.write_bytes(_snapshot_bytes(monkeypatch))

    imported = import_legacy_pickle(
        source,
        dt=0.25,
        trusted=True,
        native_state_only=True,
    )
    assert imported.simulation.time == 1.0
    assert [cell.id for cell in imported.simulation.cells()] == [3, 7]
    assert [cell.slot for cell in imported.simulation.cells()] == [0, 1]
    assert imported.simulation.cell(3).direction.x == 1.0
    assert imported.simulation.cell(7).direction.y == 1.0
    assert imported.simulation.cell(3).species == [1.5, 2.5]
    assert imported.simulation.lineage_parent(3) == 1
    assert imported.simulation.lineage_parent(7) == 1
    assert imported.dropped_cell_fields == ("color", "divideFlag", "targetVol")

    provenance = cast(dict[str, JSONValue], imported.provenance["legacy_pickle"])
    assert provenance["step_number"] == 4
    assert provenance["time_basis"] == "step-number-times-dt"
    assert provenance["migration_mode"] == "geometry-species"

    converted = tmp_path / "converted.cm2.json"
    save_checkpoint(imported.simulation, converted, provenance=imported.provenance)
    bundle = load_checkpoint_bundle(converted)
    assert bundle.simulation.cell(3).species == [1.5, 2.5]
    assert bundle.simulation.lineage_parent(7) == 1


def test_tuple_snapshot_requires_explicit_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "old.pickle"
    source.write_bytes(_snapshot_bytes(monkeypatch, tuple_format=True, dict_style=True))

    with pytest.raises(LegacyPickleError, match="explicit physical time"):
        import_legacy_pickle(
            source,
            dt=0.1,
            trusted=True,
            native_state_only=True,
        )
    imported = import_legacy_pickle(
        source,
        time=12.5,
        trusted=True,
        native_state_only=True,
    )
    assert imported.simulation.time == 12.5


def test_restricted_unpickler_rejects_executable_global(tmp_path: Path) -> None:
    sentinel = tmp_path / "executed"

    class Malicious:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return eval, (f"open({str(sentinel)!r}, 'w').close()",)

    source = tmp_path / "malicious.pickle"
    source.write_bytes(pickle.dumps(Malicious(), protocol=4))
    with pytest.raises(LegacyPickleError, match="forbidden global"):
        import_legacy_pickle(
            source,
            time=0.0,
            trusted=True,
            native_state_only=True,
        )
    assert not sentinel.exists()


def test_cli_requires_explicit_trust_and_geometry_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "legacy.pickle"
    source.write_bytes(_snapshot_bytes(monkeypatch))
    output = tmp_path / "converted.cm2.json"
    arguments = [
        "import-legacy-pickle",
        str(source),
        "--output",
        str(output),
        "--dt",
        "0.25",
    ]
    assert main(arguments) == 2
    assert "trusted=True" in capsys.readouterr().err
    assert not output.exists()

    assert main([*arguments, "--trust-legacy-pickle", "--native-state-only"]) == 0
    assert output.exists()
    assert "dropped_fields=3" in capsys.readouterr().out
