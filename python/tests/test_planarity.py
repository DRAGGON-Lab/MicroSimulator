"""Shared native-backend regressions for the documented three-dimensional contract."""

from __future__ import annotations

import math
import runpy
from itertools import pairwise
from pathlib import Path

import pytest
from microsimulator import (
    BackendKind,
    ModelContext,
    NativeController,
    Simulation,
    backend_available,
    build_model,
)
from microsimulator.scene import capture_scene, dumps_scene

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = runpy.run_path(str(ROOT / "scripts" / "diagnose_planarity.py"))


@pytest.mark.parametrize("backend", list(BackendKind))
def test_three_dimensional_diagnostic_fixtures(backend: BackendKind) -> None:
    if not backend_available(backend):
        pytest.skip(f"{backend} runtime unavailable; no fallback")
    results = DIAGNOSTIC["fixtures"](backend, 17, 0.02)
    for name in ("separated_planar", "planar_division"):
        assert results[name]["first_out_of_plane"] is None
        assert all(
            stage["max_center_displacement_from_plane"] < 1e-6 for stage in results[name]["stages"]
        )
        assert all(stage["max_direction_z"] < 1e-6 for stage in results[name]["stages"])
    for name in ("crossing", "coincident_parallel"):
        result = results[name]
        assert any(abs(normal[2]) > 0.99 for normal in result["contact_normals"])
        assert result["first_out_of_plane"]["stage"] == "contact_and_constraint_relaxation"
        negative, positive = sorted(cell["center"][2] for cell in result["final_geometry"])
        assert math.isclose(negative, -0.4, abs_tol=1e-6)
        assert math.isclose(positive, 0.4, abs_tol=1e-6)
    inherited = results["inherited_tilt"]
    assert inherited["first_out_of_plane"]["stage"] == "initialization"
    assert any(
        stage["max_center_displacement_from_plane"] > 0.2
        for stage in inherited["stages"]
        if stage["stage"] == "division"
    )
    jitter = [stage for stage in inherited["stages"] if stage["stage"] == "geometry_edit"]
    assert len(jitter) == 2
    assert all(stage["requested_direction_delta_z"] == 0 for stage in jitter)
    assert any(stage["max_stage_direction_change_z"] > 1e-6 for stage in jitter)
    confined = results["finite_height_constraints"]
    default = confined["default_wall_penetration_by_pass"]
    tight = confined["tight_wall_penetration_by_pass"]
    assert math.isclose(default[0], 0.3, abs_tol=1e-6)
    assert 1e-6 < default[-1] < 0.005
    assert tight[-1] < 1e-6
    assert all(after <= before + 1e-7 for before, after in pairwise(tight))
    final = confined["final_geometry"][0]
    axial_z_extent = abs(final["direction"][2]) * final["length"] / 2 + final["radius"]
    assert final["center"][2] + axial_z_extent <= 1 + 1e-6
    assert final["center"][2] > 0.49  # walls permit nonzero Z; they do not impose z=0
    flow = results["vertical_flow"]
    assert flow["first_out_of_plane"]["stage"] == "flow_drift"
    assert math.isclose(flow["final_geometry"][0]["center"][2], 1.004, abs_tol=1e-6)


def test_diagnostics_preserve_rng_and_restore_native_methods_on_failure() -> None:
    model_path = ROOT / "examples/tutorials/biophysics.py"
    traced, _ = build_model(
        model_path, ModelContext(BackendKind.CPU, 0, seed=17, parameters={"scenario": "two_types"})
    )
    normal, _ = build_model(
        model_path, ModelContext(BackendKind.CPU, 0, seed=17, parameters={"scenario": "two_types"})
    )
    assert isinstance(traced, NativeController)
    assert isinstance(normal, NativeController)
    trace = DIAGNOSTIC["Trace"](traced.simulation)
    originals = {
        name: getattr(Simulation, name)
        for name in (
            "divide",
            "divide_equal",
            "remove_cell",
            "step",
            "apply_flow_drift",
            "relax_cell_mechanics",
            "set_cell_geometry",
        )
    }
    with trace.instrument():
        for _ in range(5):
            traced.step(0.02)
    for _ in range(5):
        normal.step(0.02)
    assert dumps_scene(capture_scene(traced.simulation)) == dumps_scene(
        capture_scene(normal.simulation)
    )
    assert traced.controller_state() == normal.controller_state()
    assert traced.simulation.cell_count > len(trace.initial["cells"])
    assert any(stage["stage"] == "geometry_edit" for stage in trace.events)
    assert all(getattr(Simulation, name) is method for name, method in originals.items())
    with pytest.raises(RuntimeError, match="probe"), trace.instrument():
        raise RuntimeError("probe")
    assert all(getattr(Simulation, name) is method for name, method in originals.items())
    assert trace.initial["cells"]
    assert "constraints" in trace.report()
