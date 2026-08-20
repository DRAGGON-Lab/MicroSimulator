"""Multi-step runs of every model that integrates signals implicitly.

A Crank-Nicolson step can fail long after a model starts: the solver's
convergence threshold is compared against a residual whose floor rises with
the magnitude of the field, so a model that converges from a near-empty grid
can stop converging once its signals have grown. One step proves nothing about
that. These runs advance each implicit model far enough for its field to
develop, and require every step to converge and commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cellmodeller2 import (
    BackendKind,
    ModelContext,
    SimulationController,
    build_model,
)
from cellmodeller2.checkpoint import JSONValue

_ROOT = Path(__file__).resolve().parents[2]

# One case per model that selects Crank-Nicolson, with the time step its
# documentation recommends.
_IMPLICIT_MODELS: tuple[tuple[str, dict[str, JSONValue], float], ...] = (
    ("examples/tutorials/signaling.py", {"scenario": "communication"}, 0.02),
    ("examples/tutorials/simbol_circuits.py", {"circuit": "bba_0003"}, 0.02),
    ("examples/legacy/ex4_simpleCellCellSignaling.py", {}, 0.02),
    ("examples/legacy/Tutorial_3/Tutorial_3.py", {}, 0.02),
    ("examples/legacy/ACS2012/EdgeDetectorChamber.py", {}, 0.02),
    ("examples/microfluidic_trap.py", {}, 0.02),
    ("examples/tutorials/danino_clock.py", {}, 0.005),
    ("examples/tutorials/pillar_channel.py", {}, 0.01),
    ("examples/tutorials/biopixel_trap.py", {}, 0.02),
)

_STEPS = 200


@pytest.mark.parametrize(("filename", "parameters", "dt"), _IMPLICIT_MODELS)
def test_implicit_models_converge_over_a_long_run(
    filename: str, parameters: dict[str, JSONValue], dt: float
) -> None:
    model, _ = build_model(
        _ROOT / filename,
        ModelContext(BackendKind.CPU, 0, seed=17, parameters=parameters),
    )
    assert isinstance(model, SimulationController)
    assert model.simulation.has_signal_grid

    for step in range(_STEPS):
        model.step(dt)
        report = model.simulation.last_signal_solve_report
        assert report is not None, f"step {step} reported no signal solve"
        assert report.converged, f"step {step} committed an unconverged field"

    # The engine rejects a non-finite or negative field, so reaching here means
    # every step committed a valid one.
    assert model.simulation.cell_count > 0
