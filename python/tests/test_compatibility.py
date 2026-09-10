from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from microsimulator.compatibility import (
    LEGACY_EXAMPLE_MATRIX_FORMAT,
    LegacyExampleMatrixError,
    load_legacy_example_matrix,
)

_ROOT = Path(__file__).resolve().parents[2]
_MATRIX = _ROOT / "compatibility" / "legacy-examples-v1.json"


def test_complete_legacy_example_matrix_has_closed_classifications() -> None:
    matrix = load_legacy_example_matrix(_MATRIX)

    assert matrix.legacy_commit == "4896f543c6250f053eea2312e628cc3a96bf7408"
    assert len(matrix.examples) == 25
    counts = {
        status: sum(example.status == status for example in matrix.examples)
        for status in (
            "runnable",
            "migrated",
            "deliberately_retired",
            "migration_only",
        )
    }
    assert counts == {
        "runnable": 15,
        "migrated": 9,
        "deliberately_retired": 0,
        "migration_only": 1,
    }
    assert {example.path for example in matrix.examples} == {
        "ACS2012/EdgeDetectorChamber.py",
        "Conjugation.py",
        "TimRudgeThesis/Meristem.py",
        "Tutorial_1/Tutorial_1a.py",
        "Tutorial_1/Tutorial_1b.py",
        "Tutorial_1/Tutorial_1c.py",
        "Tutorial_2/Tutorial_2a.py",
        "Tutorial_2/Tutorial_2b.py",
        "Tutorial_3/Tutorial_3.py",
        "colorWalk_planes_3d.py",
        "ex1_simpleGrowth.py",
        "ex1_simpleGrowth2D.py",
        "ex1a_simpleGrowth2D.py",
        "ex1a_simpleGrowth2Types.py",
        "ex1b_simpleGrowth2D.py",
        "ex1b_simpleGrowthRoundCell.py",
        "ex2_constGene.py",
        "ex2a_dilution.py",
        "ex2b_diluteRepression.py",
        "ex3_simpleSignal.py",
        "ex4_simpleCellCellSignaling.py",
        "ex5_colonySector.py",
        "ex5_colonySector_3d.py",
        "load.py",
        "sphere_constraints.py",
    }


def test_legacy_example_matrix_requires_all_25_rows(tmp_path: Path) -> None:
    document = cast(dict[str, Any], json.loads(_MATRIX.read_text(encoding="utf-8")))
    assert document["format"] == LEGACY_EXAMPLE_MATRIX_FORMAT
    cast(list[object], document["examples"]).pop()
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(LegacyExampleMatrixError, match="exactly 25"):
        load_legacy_example_matrix(path)


def test_legacy_example_matrix_rejects_implementation_drift(tmp_path: Path) -> None:
    document = cast(dict[str, Any], json.loads(_MATRIX.read_text(encoding="utf-8")))
    examples = cast(list[dict[str, Any]], document["examples"])
    examples[0]["implementation_sha256"] = "not-a-digest"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(LegacyExampleMatrixError, match="lowercase SHA-256"):
        load_legacy_example_matrix(path)
