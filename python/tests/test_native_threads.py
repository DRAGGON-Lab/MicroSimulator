from __future__ import annotations

import threading
import time

import pytest
from microsimulator import (
    GridShape,
    SignalGridSpec,
    SignalIntegrationKind,
    Simulation,
    Vec3,
)


def _slow_implicit_simulation(size: int = 40) -> Simulation:
    shape = GridShape()
    shape.x = shape.y = shape.z = size
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.diffusion = [1.0]
    spec.advection = [Vec3()]
    spec.integration = SignalIntegrationKind.BACKWARD_EULER
    simulation = Simulation()
    simulation.configure_signal_grid(spec, [float(index % 13) for index in range(size**3)])
    return simulation


def test_long_native_calls_release_the_gil_and_reject_concurrent_use() -> None:
    simulation = _slow_implicit_simulation()
    steps = 4
    failures: list[BaseException] = []

    def advance() -> None:
        try:
            for _ in range(steps):
                simulation.step(2.0)
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=advance)
    worker.start()
    busy_observations = 0
    deadline = time.monotonic() + 120.0
    while worker.is_alive() and time.monotonic() < deadline:
        try:
            _ = simulation.time
        except RuntimeError as error:
            assert "in use by another thread" in str(error)
            busy_observations += 1
    worker.join(timeout=120.0)

    assert not worker.is_alive()
    assert failures == []
    assert busy_observations > 0
    assert simulation.time == 2.0 * steps
    simulation.validate()


def test_a_failed_native_call_leaves_the_object_usable() -> None:
    simulation = _slow_implicit_simulation(size=4)
    with pytest.raises(ValueError, match="time step"):
        simulation.step(float("nan"))
    simulation.step(0.5)
    assert simulation.time == 0.5
