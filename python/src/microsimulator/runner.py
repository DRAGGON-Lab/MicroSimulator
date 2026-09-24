"""Deterministic batch construction and execution primitives."""

from __future__ import annotations

import hashlib
import math
import random
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Literal, cast

from ._artifact_paths import periodic_checkpoint_path
from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    Simulation,
    backend_available,
)
from .channels import ChannelMetadata, ChannelMetadataError
from .checkpoint import CheckpointBundle, JSONValue, save_checkpoint
from .controller import SimulationController

_UINT64_MAX = (1 << 64) - 1


class BatchError(RuntimeError):
    """Raised when a batch run cannot be configured or completed safely."""


def _empty_parameters() -> dict[str, JSONValue]:
    return {}


@dataclass(slots=True)
class ModelContext:
    """Inputs supplied to a batch model's ``build(context)`` function."""

    backend: BackendKind
    device_index: int
    seed: int
    parameters: Mapping[str, JSONValue] = field(default_factory=_empty_parameters)
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise BatchError("device index must be non-negative")
        if self.seed < 0 or self.seed > _UINT64_MAX:
            raise BatchError("seed must be an unsigned 64-bit integer")
        if not backend_available(self.backend, self.device_index):
            raise BatchError(
                f"backend {self.backend.name.lower()} device {self.device_index} is unavailable"
            )
        self.parameters = MappingProxyType(dict(self.parameters))
        self.rng = random.Random(self.seed)

    def simulation(self, *, reserved_capacity: int = 0, species_count: int = 0) -> Simulation:
        """Construct a simulation on the backend selected by the runner."""

        return Simulation(
            self.backend,
            reserved_capacity=reserved_capacity,
            species_count=species_count,
            device_index=self.device_index,
        )


@dataclass(frozen=True, slots=True)
class RunProgress:
    completed_steps: int
    requested_steps: int
    time: float
    cell_count: int


@dataclass(frozen=True, slots=True)
class RunSummary:
    completed_steps: int
    stop_reason: RunStopReason
    cell_count_threshold: int | None
    time: float
    cell_count: int
    output: Path
    periodic_checkpoints: tuple[Path, ...]


type ProgressCallback = Callable[[RunProgress], None]
type RunnableModel = Simulation | SimulationController
type RunStopReason = Literal["step_limit", "cell_count"]


def native_simulation(model: object) -> Simulation:
    """Return the native state owner behind any supported runnable model."""

    if isinstance(model, Simulation):
        return model
    if not isinstance(model, SimulationController):
        raise BatchError("runnable model does not implement the controller protocol")
    simulation = cast(object, model.simulation)
    if not isinstance(simulation, Simulation):
        raise BatchError("controller simulation is not a native Simulation")
    return simulation


def model_channel_metadata(model: RunnableModel) -> ChannelMetadata:
    """Read optional model labels without extending the required controller protocol."""

    value = cast(object, getattr(model, "channel_metadata", ChannelMetadata()))
    if not isinstance(value, ChannelMetadata):
        raise BatchError("model channel_metadata must be ChannelMetadata")
    native = native_simulation(model)
    try:
        return value.resolved(native.species_count, native.signal_count)
    except ChannelMetadataError as error:
        raise BatchError(str(error)) from error


def controller_state(model: RunnableModel) -> JSONValue:
    """Capture optional data-only controller state for a runnable model."""

    if isinstance(model, Simulation):
        return None
    state = model.controller_state()
    if state is None:
        raise BatchError("controller_state() must return non-null JSON data")
    return state


def _checkpoint_model_context(
    source_path: Path,
    digest: str,
    context: ModelContext,
    checkpoint: CheckpointBundle,
) -> None:
    model = checkpoint.provenance.get("model")
    if not isinstance(model, dict):
        raise BatchError("checkpoint is missing model provenance")
    saved_digest = model.get("sha256")
    saved_seed = model.get("seed")
    saved_parameters = model.get("parameters")
    if (
        not isinstance(saved_digest, str)
        or not isinstance(saved_seed, int)
        or isinstance(saved_seed, bool)
        or not isinstance(saved_parameters, dict)
    ):
        raise BatchError("checkpoint model provenance is invalid")
    if digest != saved_digest:
        raise BatchError(f"model digest does not match checkpoint: {source_path}")
    if saved_seed != context.seed or saved_parameters != dict(context.parameters):
        raise BatchError("resume context differs from checkpoint model provenance")


def _periodic_path(output: Path, step: int) -> Path:
    return periodic_checkpoint_path(output, step)


def _run_provenance(
    base: Mapping[str, JSONValue],
    *,
    status: str,
    completed_steps: int,
    steps: int,
    dt: float,
    stop_reason: RunStopReason | None,
    stop_cell_count: int | None,
) -> dict[str, JSONValue]:
    result = dict(base)
    result["run"] = {
        "status": status,
        "completed_steps": completed_steps,
        "requested_steps": steps,
        "dt": dt,
        "stop_reason": stop_reason,
        "stopping": {
            "maximum_steps": steps,
            "cell_count": stop_cell_count,
        },
    }
    return result


def run_simulation(
    simulation: RunnableModel,
    *,
    steps: int,
    dt: float,
    output: str | Path,
    checkpoint_every: int = 0,
    stop_cell_count: int | None = None,
    overwrite: bool = False,
    provenance: Mapping[str, JSONValue] | None = None,
    progress: ProgressCallback | None = None,
) -> RunSummary:
    """Advance a simulation and write periodic and final atomic checkpoints."""

    if steps < 0:
        raise BatchError("steps must be non-negative")
    if steps > _UINT64_MAX:
        raise BatchError("steps must be an unsigned 64-bit integer")
    if not math.isfinite(dt) or dt < 0.0:
        raise BatchError("time step must be finite and non-negative")
    if checkpoint_every < 0:
        raise BatchError("checkpoint interval must be non-negative")
    if checkpoint_every > _UINT64_MAX:
        raise BatchError("checkpoint interval must be an unsigned 64-bit integer")
    if stop_cell_count is not None and (
        isinstance(stop_cell_count, bool) or stop_cell_count <= 0 or stop_cell_count > _UINT64_MAX
    ):
        raise BatchError("cell-count threshold must be a positive uint64 value")
    native = native_simulation(simulation)
    native.validate()
    model_channel_metadata(simulation)

    destination = Path(output)
    periodic_steps = (
        tuple(range(checkpoint_every, steps + 1, checkpoint_every)) if checkpoint_every > 0 else ()
    )
    periodic_paths = tuple(_periodic_path(destination, step) for step in periodic_steps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        collisions = [path for path in (*periodic_paths, destination) if path.exists()]
        if collisions:
            raise BatchError(f"output already exists: {collisions[0]}")

    base_provenance = dict(provenance) if provenance is not None else {}
    periodic_by_step = dict(zip(periodic_steps, periodic_paths, strict=True))
    written_periodic: list[Path] = []
    completed_steps = 0
    stop_reason: RunStopReason = "step_limit"
    if stop_cell_count is not None and native.cell_count >= stop_cell_count:
        stop_reason = "cell_count"
    if stop_reason != "cell_count":
        for step_number in range(1, steps + 1):
            simulation.step(dt)
            completed_steps = step_number
            reached_cell_count = (
                stop_cell_count is not None and native.cell_count >= stop_cell_count
            )
            if progress is not None:
                progress(
                    RunProgress(
                        completed_steps=completed_steps,
                        requested_steps=steps,
                        time=native.time,
                        cell_count=native.cell_count,
                    )
                )
            periodic = periodic_by_step.get(completed_steps)
            if periodic is not None:
                finished = reached_cell_count or completed_steps == steps
                save_checkpoint(
                    native,
                    periodic,
                    provenance=_run_provenance(
                        base_provenance,
                        status="complete" if finished else "running",
                        completed_steps=completed_steps,
                        steps=steps,
                        dt=dt,
                        stop_reason=(
                            "cell_count"
                            if reached_cell_count
                            else "step_limit"
                            if completed_steps == steps
                            else None
                        ),
                        stop_cell_count=stop_cell_count,
                    ),
                    controller=controller_state(simulation),
                    channel_metadata=model_channel_metadata(simulation),
                )
                written_periodic.append(periodic)
            if reached_cell_count:
                stop_reason = "cell_count"
                break

    save_checkpoint(
        native,
        destination,
        provenance=_run_provenance(
            base_provenance,
            status="complete",
            completed_steps=completed_steps,
            steps=steps,
            dt=dt,
            stop_reason=stop_reason,
            stop_cell_count=stop_cell_count,
        ),
        controller=controller_state(simulation),
        channel_metadata=model_channel_metadata(simulation),
    )
    return RunSummary(
        completed_steps=completed_steps,
        stop_reason=stop_reason,
        cell_count_threshold=stop_cell_count,
        time=native.time,
        cell_count=native.cell_count,
        output=destination,
        periodic_checkpoints=tuple(written_periodic),
    )


def build_model(
    path: str | Path,
    context: ModelContext,
    *,
    expected_sha256: str | None = None,
    checkpoint: CheckpointBundle | None = None,
) -> tuple[RunnableModel, dict[str, JSONValue]]:
    """Build or resume a runnable model from explicitly selected Python source."""

    source_path = Path(path).resolve()
    try:
        source = source_path.read_bytes()
    except OSError as error:
        raise BatchError(f"could not read model {source_path}") from error
    digest = hashlib.sha256(source).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise BatchError(f"model digest does not match manifest: {source_path}")
    if checkpoint is not None:
        _checkpoint_model_context(source_path, digest, context, checkpoint)
    module_name = f"_microsimulator_model_{digest[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""

    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    sys.path.insert(0, str(source_path.parent))
    try:
        code = compile(source, str(source_path), "exec")
        exec(code, module.__dict__)
        if checkpoint is None:
            build_value = module.__dict__.get("build")
            if not callable(build_value):
                raise BatchError(f"model {source_path} must define build(context)")
            build = cast(Callable[[ModelContext], object], build_value)
            model_value = build(context)
            entrypoint = "build(context)"
        else:
            resume_value = module.__dict__.get("resume")
            if not callable(resume_value):
                raise BatchError(f"model {source_path} must define resume(context, checkpoint)")
            resume = cast(Callable[[ModelContext, CheckpointBundle], object], resume_value)
            model_value = resume(context, checkpoint)
            entrypoint = "resume(context, checkpoint)"
    except BatchError:
        raise
    except Exception as error:
        raise BatchError(f"model {source_path} failed: {error}") from error
    finally:
        sys.path.pop(0)
        if previous_module is None:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = previous_module

    if not isinstance(model_value, Simulation | SimulationController):
        raise BatchError(
            f"model {source_path} {entrypoint} did not return a Simulation or SimulationController"
        )
    simulation = native_simulation(model_value)
    if checkpoint is not None and simulation is not checkpoint.simulation:
        raise BatchError(
            f"model {source_path} resume(context, checkpoint) did not use checkpoint.simulation"
        )
    info = simulation.backend_info
    if info.kind != context.backend or info.device_index != context.device_index:
        raise BatchError("model returned a simulation on a different backend or device")
    simulation.validate()
    labels = model_channel_metadata(model_value)
    if checkpoint is not None and labels != checkpoint.channel_metadata.resolved(
        simulation.species_count, simulation.signal_count
    ):
        raise BatchError("resumed model channel metadata differs from checkpoint")
    provenance: dict[str, JSONValue] = {
        "model": {
            "path": str(source_path),
            "sha256": digest,
            "seed": context.seed,
            "parameters": dict(context.parameters),
        }
    }
    return model_value, provenance
