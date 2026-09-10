from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, cast

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DISPOSITIONS = _ROOT / "compatibility" / "legacy-renderers-v1.json"
_CLASSES = {
    "GLSphereRenderer",
    "GLGridRenderer",
    "GLPlantSignalRenderer",
    "GLPlantRenderer",
    "GLBacteriumRenderer",
    "GLBacteriumRendererWithPeriodicImages",
    "GLWillsMeshRenderer",
    "GLStaticMeshRenderer",
    "GLCelBacteriumRenderer",
    "GL2DBacteriumRenderer",
}


def _document() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_DISPOSITIONS.read_text(encoding="utf-8")))


def test_every_legacy_renderer_has_a_closed_disposition() -> None:
    document = _document()
    assert set(document) == {
        "format",
        "version",
        "legacy_repository",
        "legacy_commit",
        "source",
        "source_sha256",
        "families",
    }
    assert document["format"] == "microsimulator-legacy-renderer-dispositions"
    assert document["version"] == 1
    assert document["legacy_commit"] == "4896f543c6250f053eea2312e628cc3a96bf7408"
    assert document["source"] == "CellModeller/GUI/Renderers.py"
    assert document["source_sha256"] == (
        "50859ca88ba6409c57ac9b81fd943f9daa689793cfc4f863dc16f1aae6e51a36"
    )
    families = cast(list[dict[str, Any]], document["families"])
    assert [family["id"] for family in families] == [
        "rod_cells",
        "signal_grid",
        "sphere_cells",
        "plant_cells",
        "periodic_cell_images",
        "dynamic_collision_mesh",
        "static_triangle_mesh",
    ]
    classes = [name for family in families for name in cast(list[str], family["classes"])]
    assert len(classes) == len(set(classes))
    assert set(classes) == _CLASSES
    for family in families:
        assert set(family) == {
            "id",
            "classes",
            "bundled_example_call_sites",
            "disposition",
            "replacement",
            "reason",
        }
        assert family["disposition"] in {"replaced", "deliberately_retired"}
        assert isinstance(family["reason"], str) and family["reason"]
        if family["disposition"] == "replaced":
            assert isinstance(family["replacement"], str) and family["replacement"]
        else:
            assert family["replacement"] is None


def test_pinned_legacy_renderer_source_and_call_sites() -> None:
    legacy_root_value = os.environ.get("CM_LEGACY_ROOT")
    if legacy_root_value is None:
        pytest.skip("CM_LEGACY_ROOT is required to authenticate legacy renderer sources")
    legacy_root = Path(legacy_root_value)
    document = _document()
    source = legacy_root / cast(str, document["source"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == document["source_sha256"]
    defined_classes = set(
        re.findall(
            r"^class\s+([A-Za-z0-9_]+)\s*:", source.read_text(encoding="utf-8"), re.MULTILINE
        )
    )
    assert defined_classes == _CLASSES

    examples = tuple((legacy_root / "Examples").rglob("*.py"))
    families = cast(list[dict[str, Any]], document["families"])
    for family in families:
        classes = cast(list[str], family["classes"])
        call_sites = sum(
            any(
                f"Renderers.{class_name}(" in path.read_text(encoding="utf-8")
                for class_name in classes
            )
            for path in examples
        )
        assert call_sites == family["bundled_example_call_sites"]
