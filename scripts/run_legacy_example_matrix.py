#!/usr/bin/env python3
"""Execute the pinned legacy example matrix on selected native backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microsimulator import BackendKind
from microsimulator.compatibility import (
    LegacyExampleMatrixError,
    enumerate_backend_targets,
    load_legacy_example_matrix,
    run_legacy_example_matrix,
)

_BACKENDS = {
    "cpu": BackendKind.CPU,
    "metal": BackendKind.METAL,
    "cuda": BackendKind.CUDA,
}


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=project_root / "compatibility" / "legacy-examples-v1.json",
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=tuple(_BACKENDS),
        required=True,
        help="backend to exercise; repeat to request multiple backends",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        matrix = load_legacy_example_matrix(arguments.matrix)
        targets = enumerate_backend_targets(tuple(_BACKENDS[name] for name in arguments.backend))
        report = run_legacy_example_matrix(
            matrix,
            legacy_root=arguments.legacy_root,
            project_root=arguments.project_root,
            targets=targets,
            seed=arguments.seed,
        )
    except LegacyExampleMatrixError as error:
        _parser().error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(arguments.output)
    print(f"legacy example matrix {report['result']}: {arguments.output}")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
