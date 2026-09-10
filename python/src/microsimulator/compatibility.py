"""Executable compatibility matrix for the pinned CellModeller example corpus."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    backend_device_count,
)
from .checkpoint import JSONValue
from .legacy_loader import build_legacy_model
from .runner import ModelContext, build_model, native_simulation

LEGACY_EXAMPLE_MATRIX_FORMAT = "microsimulator-legacy-example-matrix"
LEGACY_EXAMPLE_MATRIX_VERSION = 1
MAX_LEGACY_EXAMPLE_MATRIX_BYTES = 256 * 1024

type LegacyExampleStatus = Literal["runnable", "migrated", "deliberately_retired", "migration_only"]

_STATUSES: frozenset[str] = frozenset(
    {"runnable", "migrated", "deliberately_retired", "migration_only"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class LegacyExampleMatrixError(RuntimeError):
    """Raised when the legacy example matrix or its execution is invalid."""


@dataclass(frozen=True, slots=True)
class LegacyExample:
    path: str
    sha256: str
    status: LegacyExampleStatus
    implementation: str | None
    implementation_sha256: str | None
    steps: int
    dt: float
    note: str


@dataclass(frozen=True, slots=True)
class LegacyExampleMatrix:
    legacy_repository: str
    legacy_commit: str
    examples: tuple[LegacyExample, ...]


@dataclass(frozen=True, slots=True)
class BackendTarget:
    backend: BackendKind
    device_index: int


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LegacyExampleMatrixError(f"{path} must be an object")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise LegacyExampleMatrixError(f"{path} must use string keys")
    return cast(dict[str, object], mapping)


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise LegacyExampleMatrixError(f"{path} must be a non-empty string")
    return value


def _relative_python_path(value: object, path: str) -> str:
    result = _string(value, path)
    parsed = PurePosixPath(result)
    if parsed.is_absolute() or parsed.suffix != ".py" or ".." in parsed.parts:
        raise LegacyExampleMatrixError(f"{path} must be a relative Python path")
    return result


def _digest(value: object, path: str) -> str:
    result = _string(value, path)
    if _SHA256.fullmatch(result) is None:
        raise LegacyExampleMatrixError(f"{path} must be a lowercase SHA-256 digest")
    return result


def _number(value: object, path: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise LegacyExampleMatrixError(f"{path} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise LegacyExampleMatrixError(f"{path} must be finite and non-negative")
    return result


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LegacyExampleMatrixError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_legacy_example_matrix(path: str | Path) -> LegacyExampleMatrix:
    """Load and strictly validate the complete 25-example compatibility matrix."""

    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as error:
        raise LegacyExampleMatrixError(f"could not read compatibility matrix {source}") from error
    if len(data) > MAX_LEGACY_EXAMPLE_MATRIX_BYTES:
        raise LegacyExampleMatrixError("compatibility matrix is too large")
    try:
        document = cast(object, json.loads(data, object_pairs_hook=_pairs))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise LegacyExampleMatrixError("compatibility matrix is not valid JSON") from error
    root = _object(document, "matrix")
    if set(root) != {
        "format",
        "version",
        "legacy_repository",
        "legacy_commit",
        "examples",
    }:
        raise LegacyExampleMatrixError("compatibility matrix has unexpected fields")
    if root["format"] not in (LEGACY_EXAMPLE_MATRIX_FORMAT, "cellmodeller2-legacy-example-matrix"):
        raise LegacyExampleMatrixError("compatibility matrix format is unsupported")
    if root["version"] != LEGACY_EXAMPLE_MATRIX_VERSION:
        raise LegacyExampleMatrixError("compatibility matrix version is unsupported")
    repository = _string(root["legacy_repository"], "matrix.legacy_repository")
    commit = _string(root["legacy_commit"], "matrix.legacy_commit")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise LegacyExampleMatrixError("matrix.legacy_commit must be a full commit ID")
    examples_value = root["examples"]
    if not isinstance(examples_value, list):
        raise LegacyExampleMatrixError("matrix.examples must contain exactly 25 rows")
    example_values = cast(list[object], examples_value)
    if len(example_values) != 25:
        raise LegacyExampleMatrixError("matrix.examples must contain exactly 25 rows")

    examples: list[LegacyExample] = []
    for index, value in enumerate(example_values):
        row_path = f"matrix.examples[{index}]"
        row = _object(value, row_path)
        if set(row) != {
            "path",
            "sha256",
            "status",
            "implementation",
            "implementation_sha256",
            "steps",
            "dt",
            "note",
        }:
            raise LegacyExampleMatrixError(f"{row_path} has unexpected fields")
        legacy_path = _relative_python_path(row["path"], f"{row_path}.path")
        status_value = _string(row["status"], f"{row_path}.status")
        if status_value not in _STATUSES:
            raise LegacyExampleMatrixError(f"{row_path}.status is unsupported")
        status = cast(LegacyExampleStatus, status_value)
        implementation_value = row["implementation"]
        implementation = (
            None
            if implementation_value is None
            else _relative_python_path(implementation_value, f"{row_path}.implementation")
        )
        implementation_digest_value = row["implementation_sha256"]
        implementation_digest = (
            None
            if implementation_digest_value is None
            else _digest(
                implementation_digest_value,
                f"{row_path}.implementation_sha256",
            )
        )
        steps_value = row["steps"]
        if not isinstance(steps_value, int) or isinstance(steps_value, bool) or steps_value < 0:
            raise LegacyExampleMatrixError(f"{row_path}.steps must be a non-negative integer")
        dt = _number(row["dt"], f"{row_path}.dt")
        note = _string(row["note"], f"{row_path}.note")
        if status == "migrated":
            if implementation is None or implementation_digest is None or steps_value == 0:
                raise LegacyExampleMatrixError(
                    f"{row_path} migrated rows require an implementation and execution"
                )
        elif implementation is not None or implementation_digest is not None:
            raise LegacyExampleMatrixError(
                f"{row_path} only migrated rows may name an implementation"
            )
        if status == "runnable" and steps_value == 0:
            raise LegacyExampleMatrixError(f"{row_path} runnable rows must execute")
        if status in {"deliberately_retired", "migration_only"} and (steps_value != 0 or dt != 0.0):
            raise LegacyExampleMatrixError(f"{row_path} non-runnable rows cannot execute")
        examples.append(
            LegacyExample(
                path=legacy_path,
                sha256=_digest(row["sha256"], f"{row_path}.sha256"),
                status=status,
                implementation=implementation,
                implementation_sha256=implementation_digest,
                steps=steps_value,
                dt=dt,
                note=note,
            )
        )

    paths = [example.path for example in examples]
    if len(set(paths)) != len(paths):
        raise LegacyExampleMatrixError("matrix.examples contains duplicate paths")
    if paths != sorted(paths):
        raise LegacyExampleMatrixError("matrix.examples must be sorted by path")
    return LegacyExampleMatrix(repository, commit, tuple(examples))


def enumerate_backend_targets(backends: Sequence[BackendKind]) -> tuple[BackendTarget, ...]:
    """Resolve every device for each explicitly requested native backend."""

    if not backends:
        raise LegacyExampleMatrixError("at least one backend must be requested")
    if len(set(backends)) != len(backends):
        raise LegacyExampleMatrixError("requested backends must be unique")
    targets: list[BackendTarget] = []
    for backend in backends:
        count = backend_device_count(backend)
        if count == 0:
            raise LegacyExampleMatrixError(
                f"requested backend {backend.name.lower()} is unavailable"
            )
        targets.extend(BackendTarget(backend, index) for index in range(count))
    return tuple(targets)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise LegacyExampleMatrixError(f"could not read {path}") from error


def run_legacy_example_matrix(
    matrix: LegacyExampleMatrix,
    *,
    legacy_root: str | Path,
    project_root: str | Path,
    targets: Sequence[BackendTarget],
    seed: int = 1729,
) -> dict[str, JSONValue]:
    """Verify pinned sources and exercise every runnable row on every target."""

    if not targets:
        raise LegacyExampleMatrixError("at least one backend target is required")
    legacy_directory = Path(legacy_root).resolve()
    project_directory = Path(project_root).resolve()
    rows: list[JSONValue] = []
    passed = True
    for example in matrix.examples:
        legacy_source = legacy_directory / "Examples" / example.path
        source_error: str | None = None
        try:
            actual_digest = _sha256(legacy_source)
            if actual_digest != example.sha256:
                raise LegacyExampleMatrixError(f"legacy source digest mismatch for {example.path}")
            if example.implementation is not None:
                implementation_source = project_directory / example.implementation
                implementation_digest = _sha256(implementation_source)
                if implementation_digest != example.implementation_sha256:
                    raise LegacyExampleMatrixError(
                        f"implementation digest mismatch for {example.path}"
                    )
        except LegacyExampleMatrixError as error:
            source_error = str(error)
            passed = False

        runs: list[JSONValue] = []
        if source_error is None and example.status in {"runnable", "migrated"}:
            for target in targets:
                backend_name = target.backend.name.lower()
                try:
                    context = ModelContext(target.backend, target.device_index, seed=seed)
                    if example.status == "runnable":
                        model, _ = build_legacy_model(legacy_source, context)
                    else:
                        if example.implementation is None:
                            raise AssertionError("migrated example has no implementation")
                        model, _ = build_model(
                            project_directory / example.implementation,
                            context,
                            expected_sha256=example.implementation_sha256,
                        )
                    for _ in range(example.steps):
                        model.step(example.dt)
                    simulation = native_simulation(model)
                    simulation.validate()
                    info = simulation.backend_info
                    runs.append(
                        {
                            "backend": backend_name,
                            "device_index": target.device_index,
                            "device": info.device,
                            "result": "pass",
                            "steps": example.steps,
                            "time": simulation.time,
                            "cell_count": simulation.cell_count,
                        }
                    )
                except Exception as error:  # keep the complete matrix in the report
                    passed = False
                    runs.append(
                        {
                            "backend": backend_name,
                            "device_index": target.device_index,
                            "result": "fail",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        row: dict[str, JSONValue] = {
            "path": example.path,
            "sha256": example.sha256,
            "status": example.status,
            "implementation": example.implementation,
            "note": example.note,
            "source_result": "fail" if source_error is not None else "pass",
            "runs": runs,
        }
        if source_error is not None:
            row["source_error"] = source_error
        rows.append(row)

    return {
        "format": "microsimulator-legacy-example-matrix-report",
        "version": 1,
        "result": "pass" if passed else "fail",
        "legacy_repository": matrix.legacy_repository,
        "legacy_commit": matrix.legacy_commit,
        "seed": seed,
        "rows": rows,
    }
