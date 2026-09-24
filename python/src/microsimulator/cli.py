"""Command-line entry point for MicroSimulator batch work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn, cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    BackendKind,
    Simulation,
    backend_available,
    backend_device_count,
)
from .checkpoint import (
    CheckpointError,
    JSONValue,
    load_checkpoint,
    load_checkpoint_bundle,
    save_checkpoint,
)
from .legacy_loader import build_legacy_model, resume_legacy_model
from .legacy_pickle import LegacyPickleError, import_legacy_pickle
from .runner import (
    BatchError,
    ModelContext,
    RunnableModel,
    RunProgress,
    build_model,
    run_simulation,
)

_BACKENDS = {
    "cpu": BackendKind.CPU,
    "metal": BackendKind.METAL,
    "cuda": BackendKind.CUDA,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microsimulator", description="MicroSimulator batch runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    devices = commands.add_parser("devices", help="list available native compute devices")
    devices.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    legacy_pickle = commands.add_parser(
        "import-legacy-pickle", help="migrate a trusted CellModeller 1 snapshot"
    )
    legacy_pickle.add_argument("input", type=Path, help="legacy .pickle snapshot")
    legacy_pickle.add_argument("--output", type=Path, required=True)
    time_source = legacy_pickle.add_mutually_exclusive_group(required=True)
    time_source.add_argument("--time", type=float, help="physical simulation time")
    time_source.add_argument("--dt", type=float, help="legacy step duration")
    legacy_pickle.add_argument("--trust-legacy-pickle", action="store_true")
    legacy_pickle.add_argument("--native-state-only", action="store_true")
    legacy_pickle.add_argument("--overwrite", action="store_true")

    run = commands.add_parser("run", help="run a model or resume a checkpoint")
    _add_source_arguments(run)
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--dt", type=float, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint-every", type=int, default=0)
    run.add_argument(
        "--stop-cell-count",
        type=int,
        help="stop after the first step reaching this many cells",
    )
    run.add_argument("--progress-every", type=int, default=100)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--quiet", action="store_true")

    view = commands.add_parser("view", help="control a model in the local scene viewer")
    _add_source_arguments(view)
    view.add_argument("--dt", type=float, required=True)
    view.add_argument("--host", choices=("127.0.0.1", "::1", "localhost"), default="127.0.0.1")
    view.add_argument("--port", type=int, default=8765)
    view.add_argument("--frame-steps", type=int, default=1)
    view.add_argument("--fps", type=float, default=30.0)
    view.add_argument("--checkpoint-output", type=Path)
    view.add_argument("--viewer-dist", type=Path)
    view.add_argument("--open", action="store_true", help="open the live URL in a browser")

    analysis = commands.add_parser(
        "export-analysis", help="export ordered checkpoints to Parquet and Zarr"
    )
    analysis.add_argument("checkpoints", nargs="+", type=Path)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--backend", choices=tuple(_BACKENDS), default="cpu")
    analysis.add_argument("--device-index", type=int, default=0)
    analysis.add_argument("--contacts", action="store_true", help="derive cell contacts")
    analysis.add_argument(
        "--external-contacts", action="store_true", help="derive constraint contacts"
    )
    analysis.add_argument(
        "--path-provenance",
        action="store_true",
        help="record absolute checkpoint paths in the manifest",
    )
    analysis.add_argument("--overwrite", action="store_true")

    replay = commands.add_parser(
        "export-replay", help="export explicitly ordered checkpoints for offline replay"
    )
    replay.add_argument("checkpoints", nargs="+", type=Path)
    replay.add_argument("--output", type=Path, required=True, help="new replay bundle directory")

    manifest = commands.add_parser(
        "run-manifest", help="execute one named job from a data-only run manifest"
    )
    manifest.add_argument("manifest", type=Path)
    manifest.add_argument("--job", required=True, help="stable job ID to execute")
    manifest.add_argument("--progress-every", type=int, default=100)
    manifest.add_argument("--overwrite", action="store_true")
    manifest.add_argument("--quiet", action="store_true")
    return parser


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--model",
        type=Path,
        help=(
            "Python file defining build(context) and, for controller resume, "
            "resume(context, checkpoint)"
        ),
    )
    source.add_argument("--legacy-model", type=Path, help="CellModeller 1 growth/mechanics model")
    parser.add_argument("--resume", type=Path, help="MicroSimulator checkpoint to resume")
    parser.add_argument("--backend", choices=tuple(_BACKENDS), default="cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="model-construction seed")
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="model parameter; repeat for multiple values",
    )


def _json_value(value: object, path: str) -> JSONValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BatchError(f"{path} must be finite JSON")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{path}[]") for item in cast(list[object], value)]
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in mapping):
            raise BatchError(f"{path} must use string object keys")
        return {cast(str, key): _json_value(item, f"{path}.{key}") for key, item in mapping.items()}
    raise BatchError(f"{path} is not JSON data")


def _parameters(values: Sequence[str]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for value in values:
        name, separator, encoded = value.partition("=")
        if not separator or not name:
            raise BatchError(f"invalid parameter {value!r}; expected NAME=JSON")
        if name in result:
            raise BatchError(f"duplicate parameter {name!r}")

        def reject_constant(constant: str, parameter_name: str = name) -> NoReturn:
            raise BatchError(f"parameter {parameter_name!r} contains {constant}")

        try:
            decoded = json.loads(encoded, parse_constant=reject_constant)
        except json.JSONDecodeError as error:
            raise BatchError(f"parameter {name!r} is not valid JSON: {error.msg}") from error
        result[name] = _json_value(cast(object, decoded), f"parameter {name!r}")
    return result


def _device_records() -> list[dict[str, JSONValue]]:
    records: list[dict[str, JSONValue]] = []
    for name, backend in _BACKENDS.items():
        count = backend_device_count(backend)
        if count == 0:
            records.append({"backend": name, "available": False, "devices": []})
            continue
        devices: list[JSONValue] = []
        for device_index in range(count):
            info = Simulation(backend, device_index=device_index).backend_info
            devices.append({"index": info.device_index, "name": info.device})
        records.append({"backend": name, "available": True, "devices": devices})
    return records


def _devices(json_output: bool) -> int:
    records = _device_records()
    if json_output:
        print(json.dumps(records, indent=2, sort_keys=True))
        return 0
    for record in records:
        backend = cast(str, record["backend"])
        devices = cast(list[JSONValue], record["devices"])
        if not devices:
            print(f"{backend}: unavailable")
            continue
        for device in devices:
            device_record = cast(dict[str, JSONValue], device)
            print(f"{backend}:{device_record['index']} {device_record['name']}")
    return 0


def _resume_provenance(path: Path) -> dict[str, JSONValue]:
    source = path.resolve()
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as error:
        raise BatchError(f"could not read checkpoint {source}") from error
    return {"resume": {"path": str(source), "sha256": digest}}


def _progress_printer(interval: int, quiet: bool):
    if quiet:
        return None
    if interval <= 0:
        raise BatchError("progress interval must be positive unless --quiet is used")

    def report(progress: RunProgress) -> None:
        if progress.completed_steps % interval == 0 or (
            progress.completed_steps == progress.requested_steps
        ):
            print(
                f"step {progress.completed_steps}/{progress.requested_steps} "
                f"time={progress.time:.9g} cells={progress.cell_count}",
                file=sys.stderr,
            )

    return report


def _model_factory(
    arguments: argparse.Namespace,
) -> Callable[[], tuple[RunnableModel, dict[str, JSONValue]]]:
    backend = _BACKENDS[cast(str, arguments.backend)]
    device_index = cast(int, arguments.device_index)
    if not backend_available(backend, device_index):
        count = backend_device_count(backend)
        raise BatchError(
            f"backend {arguments.backend} device {device_index} is unavailable "
            f"({count} device(s) found)"
        )
    parameters = _parameters(cast(list[str], arguments.parameter))
    model_path = cast(Path | None, arguments.model)
    legacy_model_path = cast(Path | None, arguments.legacy_model)
    resume_path = cast(Path | None, arguments.resume)
    if model_path is not None:
        if resume_path is None:
            seed = cast(int, arguments.seed)

            def build_native_model() -> tuple[RunnableModel, dict[str, JSONValue]]:
                context = ModelContext(backend, device_index, seed, parameters)
                return build_model(model_path, context)

            return build_native_model
        if parameters:
            raise BatchError(
                "native resume uses the checkpoint parameters; do not pass --parameter"
            )

        def resume_native_model() -> tuple[RunnableModel, dict[str, JSONValue]]:
            bundle = load_checkpoint_bundle(
                resume_path,
                backend=backend,
                device_index=device_index,
            )
            model_value = bundle.provenance.get("model")
            if not isinstance(model_value, dict):
                raise BatchError("native checkpoint is missing model provenance")
            seed_value = model_value.get("seed")
            saved_parameters = model_value.get("parameters")
            if (
                not isinstance(seed_value, int)
                or isinstance(seed_value, bool)
                or not isinstance(saved_parameters, dict)
            ):
                raise BatchError("native checkpoint model provenance is invalid")
            context = ModelContext(
                backend=backend,
                device_index=device_index,
                seed=seed_value,
                parameters=cast(dict[str, JSONValue], saved_parameters),
            )
            model, model_provenance = build_model(
                model_path,
                context,
                checkpoint=bundle,
            )
            provenance = dict(model_provenance)
            provenance.update(_resume_provenance(resume_path))
            return model, provenance

        return resume_native_model
    elif legacy_model_path is not None:
        if resume_path is None:
            seed = cast(int, arguments.seed)

            def build_legacy() -> tuple[RunnableModel, dict[str, JSONValue]]:
                context = ModelContext(backend, device_index, seed, parameters)
                return build_legacy_model(legacy_model_path, context)

            return build_legacy
        if parameters:
            raise BatchError(
                "legacy resume uses the checkpoint parameters; do not pass --parameter"
            )

        def resume_legacy() -> tuple[RunnableModel, dict[str, JSONValue]]:
            bundle = load_checkpoint_bundle(
                resume_path,
                backend=backend,
                device_index=device_index,
            )
            model_value = bundle.provenance.get("model")
            if not isinstance(model_value, dict):
                raise BatchError("legacy checkpoint is missing model provenance")
            seed_value = model_value.get("seed")
            saved_parameters = model_value.get("parameters")
            if (
                not isinstance(seed_value, int)
                or isinstance(seed_value, bool)
                or not isinstance(saved_parameters, dict)
            ):
                raise BatchError("legacy checkpoint model provenance is invalid")
            context = ModelContext(
                backend=backend,
                device_index=device_index,
                seed=seed_value,
                parameters=cast(dict[str, JSONValue], saved_parameters),
            )
            simulation, model_provenance = resume_legacy_model(
                legacy_model_path,
                context,
                bundle,
            )
            provenance = dict(model_provenance)
            provenance.update(_resume_provenance(resume_path))
            return simulation, provenance

        return resume_legacy
    else:
        if resume_path is None:
            raise BatchError("a model or checkpoint is required")
        if parameters:
            raise BatchError("--parameter is only valid with --model")

        def resume_native() -> tuple[RunnableModel, dict[str, JSONValue]]:
            simulation = load_checkpoint(
                resume_path,
                backend=backend,
                device_index=device_index,
            )
            return simulation, _resume_provenance(resume_path)

        return resume_native


def _run(arguments: argparse.Namespace) -> int:
    simulation, provenance = _model_factory(arguments)()

    summary = run_simulation(
        simulation,
        steps=cast(int, arguments.steps),
        dt=cast(float, arguments.dt),
        output=cast(Path, arguments.output),
        checkpoint_every=cast(int, arguments.checkpoint_every),
        stop_cell_count=cast(int | None, arguments.stop_cell_count),
        overwrite=cast(bool, arguments.overwrite),
        provenance=provenance,
        progress=_progress_printer(
            cast(int, arguments.progress_every), cast(bool, arguments.quiet)
        ),
    )
    print(
        f"wrote {summary.output} steps={summary.completed_steps} "
        f"time={summary.time:.9g} cells={summary.cell_count} stop={summary.stop_reason}"
    )
    return 0


def _viewer_distribution(value: Path | None) -> Path:
    if value is not None:
        return value.resolve()
    source_distribution = Path(__file__).resolve().parents[3] / "viewer" / "dist"
    if source_distribution.is_dir():
        return source_distribution
    raise BatchError("viewer build not found; run `pnpm --dir viewer build` or pass --viewer-dist")


def _view(arguments: argparse.Namespace) -> int:
    try:
        from .viewer_server import LiveSession, serve_live
    except ModuleNotFoundError as error:
        if error.name == "aiohttp":
            raise BatchError("live viewer requires `microsimulator[viewer]`") from error
        raise

    viewer_dist = _viewer_distribution(cast(Path | None, arguments.viewer_dist))
    session = LiveSession(
        _model_factory(arguments),
        dt=cast(float, arguments.dt),
        checkpoint_output=cast(Path | None, arguments.checkpoint_output),
    )
    serve_live(
        session,
        viewer_dist,
        host=cast(str, arguments.host),
        port=cast(int, arguments.port),
        frame_steps=cast(int, arguments.frame_steps),
        fps=cast(float, arguments.fps),
        open_browser=cast(bool, arguments.open),
    )
    return 0


def _import_legacy_pickle(arguments: argparse.Namespace) -> int:
    output = cast(Path, arguments.output)
    if output.exists() and not cast(bool, arguments.overwrite):
        raise BatchError(f"output already exists: {output}")
    imported = import_legacy_pickle(
        cast(Path, arguments.input),
        time=cast(float | None, arguments.time),
        dt=cast(float | None, arguments.dt),
        trusted=cast(bool, arguments.trust_legacy_pickle),
        native_state_only=cast(bool, arguments.native_state_only),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(imported.simulation, output, provenance=imported.provenance)
    print(
        f"wrote {output} cells={imported.simulation.cell_count} "
        f"dropped_fields={len(imported.dropped_cell_fields)}"
    )
    return 0


def _export_analysis(arguments: argparse.Namespace) -> int:
    try:
        from .analysis import export_dataset
    except ModuleNotFoundError as error:
        if error.name in {"pyarrow", "zarr"}:
            raise BatchError("analysis export requires `microsimulator[analysis]`") from error
        raise

    backend_name = cast(str, arguments.backend)
    backend = _BACKENDS[backend_name]
    device_index = cast(int, arguments.device_index)
    if not backend_available(backend, device_index):
        count = backend_device_count(backend)
        raise BatchError(
            f"backend {backend_name} device {device_index} is unavailable ({count} device(s) found)"
        )
    summary = export_dataset(
        cast(list[Path], arguments.checkpoints),
        cast(Path, arguments.output),
        backend=backend,
        device_index=device_index,
        include_contacts=cast(bool, arguments.contacts),
        include_external_contacts=cast(bool, arguments.external_contacts),
        path_provenance=cast(bool, arguments.path_provenance),
        replace=cast(bool, arguments.overwrite),
    )
    print(
        f"wrote {summary.output} frames={summary.frame_count} "
        f"cells={summary.cell_rows} signal_epochs={summary.signal_epochs}"
    )
    return 0


def _run_manifest(arguments: argparse.Namespace) -> int:
    from .run_manifest import execute_run_job, load_run_manifest

    manifest = load_run_manifest(cast(Path, arguments.manifest))
    summary = execute_run_job(
        manifest,
        cast(str, arguments.job),
        overwrite=cast(bool, arguments.overwrite),
        progress=_progress_printer(
            cast(int, arguments.progress_every), cast(bool, arguments.quiet)
        ),
    )
    print(
        f"wrote {summary.output} steps={summary.completed_steps} "
        f"time={summary.time:.9g} cells={summary.cell_count} stop={summary.stop_reason}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``microsimulator`` command and return its process status."""

    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "devices":
            return _devices(cast(bool, arguments.json))
        if arguments.command == "import-legacy-pickle":
            return _import_legacy_pickle(arguments)
        if arguments.command == "view":
            return _view(arguments)
        if arguments.command == "export-analysis":
            return _export_analysis(arguments)
        if arguments.command == "export-replay":
            from .replay import export_replay

            summary = export_replay(
                cast(list[Path], arguments.checkpoints), cast(Path, arguments.output)
            )
            print(f"wrote {summary.output} frames={summary.frame_count}")
            return 0
        if arguments.command == "run-manifest":
            return _run_manifest(arguments)
        return _run(arguments)
    except (
        BatchError,
        CheckpointError,
        LegacyPickleError,
        IndexError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"microsimulator: {error}", file=sys.stderr)
        return 2
