"""Strict data-only experiment manifests for reproducible batch jobs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

from ._artifact_paths import periodic_checkpoint_parts
from ._core import BackendKind  # pyright: ignore[reportMissingModuleSource]
from .checkpoint import JSONValue
from .runner import (
    BatchError,
    ModelContext,
    ProgressCallback,
    RunSummary,
    build_model,
    run_simulation,
)

RUN_MANIFEST_FORMAT = "microsimulator-run-manifest"
RUN_MANIFEST_VERSION = 1
MAX_RUN_MANIFEST_BYTES = 1 << 24

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_BACKENDS = {
    "cpu": BackendKind.CPU,
    "metal": BackendKind.METAL,
    "cuda": BackendKind.CUDA,
}


class RunManifestError(BatchError):
    """Raised when a run manifest is malformed or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class RunJob:
    """One fully explicit model invocation from a run manifest."""

    id: str
    model: Path
    model_sha256: str
    backend: BackendKind
    device_index: int
    seed: int
    parameters: Mapping[str, JSONValue]
    maximum_steps: int
    dt: float
    stop_cell_count: int | None
    checkpoint_every: int
    output: Path


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Validated jobs plus identity of their source manifest."""

    source: Path
    sha256: str
    jobs: tuple[RunJob, ...]

    def job(self, job_id: str) -> RunJob:
        """Select one job by its stable manifest ID."""

        for job in self.jobs:
            if job.id == job_id:
                return job
        raise RunManifestError(f"run manifest has no job {job_id!r}")


def _fail(path: str, message: str) -> NoReturn:
    raise RunManifestError(f"{path}: {message}")


def _reject_constant(value: str) -> NoReturn:
    raise RunManifestError(f"run manifest contains non-finite JSON number {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RunManifestError(f"run manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        _fail(path, "expected an object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        _fail(path, "expected string object keys")
    return cast(dict[str, object], mapping)


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        _fail(path, "expected an array")
    return cast(list[object], value)


def _keys(value: dict[str, object], path: str, required: set[str]) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required
    if missing:
        _fail(path, f"missing keys {sorted(missing)}")
    if unknown:
        _fail(path, f"unknown keys {sorted(unknown)}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        _fail(path, "expected a string")
    return value


def _integer(value: object, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, "expected an integer")
    if value < minimum or value > maximum:
        _fail(path, f"integer is outside [{minimum}, {maximum}]")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(path, "expected a number")
    result = float(value)
    if not math.isfinite(result):
        _fail(path, "number must be finite")
    return result


def _digest(value: object, path: str) -> str:
    result = _string(value, path)
    if len(result) != 64 or result != result.lower():
        _fail(path, "expected a lowercase SHA-256 digest")
    try:
        bytes.fromhex(result)
    except ValueError:
        _fail(path, "expected a lowercase SHA-256 digest")
    return result


def _path(value: object, path: str, directory: Path) -> Path:
    encoded = _string(value, path)
    if not encoded or "\0" in encoded:
        _fail(path, "expected a nonempty filesystem path")
    result = Path(encoded)
    return (result if result.is_absolute() else directory / result).resolve()


def _json_value(value: object, path: str) -> JSONValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "number must be finite")
        return value
    if isinstance(value, list):
        return [
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(cast(list[object], value))
        ]
    if isinstance(value, dict):
        return {
            key: _json_value(item, f"{path}.{key}")
            for key, item in _object(cast(object, value), path).items()
        }
    _fail(path, "expected JSON data")


def _parameters(value: object, path: str) -> Mapping[str, JSONValue]:
    return MappingProxyType(
        {key: _json_value(item, f"{path}.{key}") for key, item in _object(value, path).items()}
    )


def _job(value: object, path: str, directory: Path) -> RunJob:
    data = _object(value, path)
    _keys(
        data,
        path,
        {
            "id",
            "model",
            "backend",
            "device_index",
            "seed",
            "parameters",
            "stopping",
            "checkpoint_every",
            "output",
        },
    )
    job_id = _string(data["id"], f"{path}.id")
    if _RUN_ID_PATTERN.fullmatch(job_id) is None:
        _fail(f"{path}.id", "expected 1-128 ASCII letters, digits, '.', '_', or '-'")

    model = _object(data["model"], f"{path}.model")
    _keys(model, f"{path}.model", {"path", "sha256"})
    backend_name = _string(data["backend"], f"{path}.backend")
    backend = _BACKENDS.get(backend_name)
    if backend is None:
        _fail(f"{path}.backend", f"unknown backend {backend_name!r}")
    stopping = _object(data["stopping"], f"{path}.stopping")
    _keys(stopping, f"{path}.stopping", {"maximum_steps", "dt", "cell_count"})
    stop_value = stopping["cell_count"]
    stop_cell_count = (
        None
        if stop_value is None
        else _integer(stop_value, f"{path}.stopping.cell_count", 1, _UINT64_MAX)
    )
    dt = _number(stopping["dt"], f"{path}.stopping.dt")
    if dt < 0.0:
        _fail(f"{path}.stopping.dt", "number must be non-negative")
    return RunJob(
        id=job_id,
        model=_path(model["path"], f"{path}.model.path", directory),
        model_sha256=_digest(model["sha256"], f"{path}.model.sha256"),
        backend=backend,
        device_index=_integer(data["device_index"], f"{path}.device_index", 0, _UINT32_MAX),
        seed=_integer(data["seed"], f"{path}.seed", 0, _UINT64_MAX),
        parameters=_parameters(data["parameters"], f"{path}.parameters"),
        maximum_steps=_integer(
            stopping["maximum_steps"], f"{path}.stopping.maximum_steps", 0, _UINT64_MAX
        ),
        dt=dt,
        stop_cell_count=stop_cell_count,
        checkpoint_every=_integer(
            data["checkpoint_every"], f"{path}.checkpoint_every", 0, _UINT64_MAX
        ),
        output=_path(data["output"], f"{path}.output", directory),
    )


def _periodic_contains(job: RunJob, candidate: Path) -> bool:
    if job.checkpoint_every == 0 or job.checkpoint_every > job.maximum_steps:
        return False
    parent, stem, suffix = periodic_checkpoint_parts(job.output)
    if candidate.parent != parent:
        return False
    prefix = f"{stem}.step-"
    name = candidate.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return False
    encoded_step = name[len(prefix) : -len(suffix)]
    if len(encoded_step) < 8 or not encoded_step.isascii() or not encoded_step.isdigit():
        return False
    step = int(encoded_step)
    return 0 < step <= job.maximum_steps and step % job.checkpoint_every == 0


def _validate_output_disjointness(jobs: tuple[RunJob, ...]) -> None:
    for first, second in combinations(jobs, 2):
        if first.output == second.output:
            raise RunManifestError(
                f"jobs {first.id!r} and {second.id!r} use the same output {first.output}"
            )
        if _periodic_contains(first, second.output) or _periodic_contains(second, first.output):
            raise RunManifestError(
                f"jobs {first.id!r} and {second.id!r} have colliding final/periodic outputs"
            )
        first_parent, first_stem, first_suffix = periodic_checkpoint_parts(first.output)
        second_parent, second_stem, second_suffix = periodic_checkpoint_parts(second.output)
        if (
            first.checkpoint_every > 0
            and second.checkpoint_every > 0
            and first.checkpoint_every <= first.maximum_steps
            and second.checkpoint_every <= second.maximum_steps
            and first_parent == second_parent
            and first_stem == second_stem
            and first_suffix == second_suffix
            and math.lcm(first.checkpoint_every, second.checkpoint_every)
            <= min(first.maximum_steps, second.maximum_steps)
        ):
            raise RunManifestError(
                f"jobs {first.id!r} and {second.id!r} have colliding periodic outputs"
            )


def load_run_manifest(path: str | os.PathLike[str]) -> RunManifest:
    """Parse a strict manifest without importing or executing any model."""

    source = Path(path).resolve()
    try:
        with source.open("rb") as stream:
            encoded = stream.read(MAX_RUN_MANIFEST_BYTES + 1)
    except OSError as error:
        raise RunManifestError(f"could not read run manifest {source}") from error
    if not encoded:
        raise RunManifestError("run manifest is empty")
    if len(encoded) > MAX_RUN_MANIFEST_BYTES:
        raise RunManifestError(f"run manifest exceeds the {MAX_RUN_MANIFEST_BYTES}-byte limit")
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        decoded = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except RunManifestError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as error:
        raise RunManifestError(f"run manifest is not valid UTF-8 JSON: {error}") from error

    root = _object(cast(object, decoded), "$")
    _keys(root, "$", {"format", "version", "jobs"})
    if _string(root["format"], "$.format") not in (
        RUN_MANIFEST_FORMAT, "cellmodeller2-run-manifest"
    ):
        _fail("$.format", "not a MicroSimulator run manifest")
    version = _integer(root["version"], "$.version", 0, _UINT32_MAX)
    if version != RUN_MANIFEST_VERSION:
        _fail("$.version", f"unsupported run manifest version {version}")
    values = _array(root["jobs"], "$.jobs")
    if not values:
        _fail("$.jobs", "at least one job is required")
    jobs = tuple(
        _job(value, f"$.jobs[{index}]", source.parent) for index, value in enumerate(values)
    )
    ids = [job.id for job in jobs]
    if len(ids) != len(set(ids)):
        raise RunManifestError("run manifest job IDs must be unique")
    _validate_output_disjointness(jobs)
    return RunManifest(source=source, sha256=digest, jobs=jobs)


def execute_run_job(
    manifest: RunManifest,
    job_id: str,
    *,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> RunSummary:
    """Execute exactly one named manifest job through the ordinary batch runner."""

    job = manifest.job(job_id)
    context = ModelContext(
        backend=job.backend,
        device_index=job.device_index,
        seed=job.seed,
        parameters=job.parameters,
    )
    simulation, model_provenance = build_model(
        job.model,
        context,
        expected_sha256=job.model_sha256,
    )
    provenance = dict(model_provenance)
    provenance["experiment"] = {
        "manifest": {
            "path": str(manifest.source),
            "sha256": manifest.sha256,
        },
        "job_id": job.id,
    }
    return run_simulation(
        simulation,
        steps=job.maximum_steps,
        dt=job.dt,
        output=job.output,
        checkpoint_every=job.checkpoint_every,
        stop_cell_count=job.stop_cell_count,
        overwrite=overwrite,
        provenance=provenance,
        progress=progress,
    )
