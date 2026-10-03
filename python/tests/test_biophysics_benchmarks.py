from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from microsimulator import BackendKind


def _benchmark_module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "run_biophysics_benchmarks.py"
    spec = importlib.util.spec_from_file_location("run_biophysics_benchmarks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quick_biophysics_checks_pass_on_cpu() -> None:
    module = _benchmark_module()
    checks = [
        module.growth_recurrence,
        module.biomass_and_dilution,
        module.division_conservation,
        module.contacts_and_mechanics,
        module.flow_translation_and_rotation,
        module.transport,
        module.membrane_exchange,
        module.monod_and_yield,
    ]
    for check in checks:
        passed, metrics = check(BackendKind.CPU)
        assert passed, f"{check.__name__}: {json.dumps(metrics, sort_keys=True)}"
