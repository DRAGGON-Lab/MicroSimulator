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
    cpu_thread_count,
    set_cpu_thread_count,
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


def test_results_do_not_depend_on_the_native_thread_count() -> None:
    default = cpu_thread_count()
    assert default >= 1
    try:
        levels: list[list[float]] = []
        for threads in (1, 2, 5):
            set_cpu_thread_count(threads)
            assert cpu_thread_count() == threads
            simulation = _slow_implicit_simulation(size=28)
            simulation.step(0.5)
            levels.append(simulation.signal_levels)
        assert levels[1] == levels[0]
        assert levels[2] == levels[0]
    finally:
        set_cpu_thread_count(0)
    assert cpu_thread_count() == default


def test_an_invalid_native_thread_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="too large"):
        set_cpu_thread_count(1 << 20)
    with pytest.raises(TypeError):
        set_cpu_thread_count(-1)
