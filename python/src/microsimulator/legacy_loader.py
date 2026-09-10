"""Opt-in loader for maintained CellModeller 1 growth/mechanics models."""

from __future__ import annotations

import hashlib
import random as stdlib_random
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    CellInit,
    MechanicsParameters,
    PlaneConstraintInit,
    Simulation,
    SphereConstraintInit,
    SphereRegion,
    Vec3,
)
from .checkpoint import CheckpointBundle, JSONValue
from .legacy import (
    DivideCallback,
    InitCallback,
    LegacyCell,
    LegacyCompatibilityError,
    LegacyModelAdapter,
    UpdateCallback,
)
from .runner import BatchError, ModelContext

_MISSING = object()


def _vec3(value: object, name: str) -> Vec3:
    try:
        coordinates = list(cast(Any, value))
    except TypeError as error:
        raise LegacyCompatibilityError(f"legacy {name} must contain three coordinates") from error
    if len(coordinates) != 3:
        raise LegacyCompatibilityError(f"legacy {name} must contain three coordinates")
    try:
        return Vec3(float(coordinates[0]), float(coordinates[1]), float(coordinates[2]))
    except (TypeError, ValueError, OverflowError) as error:
        raise LegacyCompatibilityError(f"legacy {name} coordinates must be numbers") from error


class _LegacyModuleRegulator:
    def __init__(self, simulator: _LegacySetupFacade, *_: object, **__: object) -> None:
        self.simulator = simulator


class _LegacyRenderer:
    def __init__(self, *_: object, **__: object) -> None:
        pass


class _UnsupportedLegacyComponent:
    def __init__(self, *_: object, **__: object) -> None:
        raise LegacyCompatibilityError(
            "legacy OpenCL integrators and signaling objects are not supported; "
            "use SpeciesRatePlan or CoupledRatePlan"
        )


class _LegacyCLBacterium:
    def __init__(
        self,
        simulator: _LegacySetupFacade,
        max_substeps: int = 8,
        max_cells: int = 10_000,
        max_contacts: int = 24,
        max_planes: int = 1,
        max_spheres: int = 1,
        max_sqs: int = 192**2,
        grid_spacing: float = 5.0,
        muA: float = 1.0,  # noqa: N803 - legacy keyword
        gamma: float = 10.0,
        dt: float | None = None,
        cgs_tol: float = 5.0e-3,
        jitter_z: bool = True,
        alternate_divisions: bool = False,
        printing: bool = True,
        compNeighbours: bool = False,  # noqa: N803 - legacy keyword
    ) -> None:
        del max_cells, max_contacts, max_planes, max_spheres
        del max_sqs, grid_spacing, printing
        if dt is not None:
            raise LegacyCompatibilityError("a CLBacterium-specific time step is not supported")
        self._setup = simulator
        self.jitter_z = bool(jitter_z)
        self.alternate_divisions = bool(alternate_divisions)
        max_substeps_value = cast(object, max_substeps)
        if (
            not isinstance(max_substeps_value, int)
            or isinstance(max_substeps_value, bool)
            or max_substeps_value < 0
            or max_substeps_value > (1 << 32) - 1
        ):
            raise LegacyCompatibilityError("legacy max_substeps must be an unsigned 32-bit integer")
        self.max_substeps = max_substeps_value
        self.compute_neighbors = bool(compNeighbours)
        self.mechanics_parameters = MechanicsParameters()
        self.mechanics_parameters.mu_a = float(muA)
        self.mechanics_parameters.gamma = float(gamma)
        self.mechanics_parameters.residual_rms_tolerance = float(cgs_tol)

    def addPlane(  # noqa: N802 - legacy API
        self, point: object, normal: object, coefficient: float
    ) -> int:
        plane = PlaneConstraintInit()
        plane.point = _vec3(point, "plane point")
        plane.inward_normal = _vec3(normal, "plane normal")
        plane.coefficient = float(coefficient)
        if self._setup.restoring:
            return self._setup.record_restored_constraint()
        return self._setup.simulation.add_plane_constraint(plane)

    def addSphere(  # noqa: N802 - legacy API
        self,
        center: object,
        radius: float,
        coefficient: float,
        normal_sign: float,
    ) -> int:
        if normal_sign not in (-1, 1):
            raise LegacyCompatibilityError("legacy sphere normal sign must be -1 or 1")
        sphere = SphereConstraintInit()
        sphere.center = _vec3(center, "sphere center")
        sphere.radius = float(radius)
        sphere.coefficient = float(coefficient)
        sphere.allowed_region = (
            SphereRegion.INSIDE if normal_sign == -1 else SphereRegion.OUTSIDE
        )
        if self._setup.restoring:
            return self._setup.record_restored_constraint()
        return self._setup.simulation.add_sphere_constraint(sphere)


class _LegacySetupFacade:
    def __init__(
        self,
        context: ModelContext,
        module_name: str,
        source: bytes,
        *,
        simulation: Simulation | None = None,
        controller: JSONValue = None,
    ) -> None:
        self.simulation = simulation if simulation is not None else context.simulation()
        self.moduleName = module_name
        self.moduleStr = source.decode("utf-8")
        self.module: ModuleType | None = None
        self.is_gui = False
        self.pickleSteps = 10
        self.saveOutput = False
        self._context = context
        self._controller = controller
        self._adapter: LegacyModelAdapter | None = None
        self._restored_cell_cursor = 0
        self._restored_constraint_cursor = 0

    @property
    def restoring(self) -> bool:
        return self._controller is not None

    def record_restored_constraint(self) -> int:
        self._restored_constraint_cursor += 1
        return self._restored_constraint_cursor

    @property
    def adapter(self) -> LegacyModelAdapter:
        if self._adapter is None:
            raise LegacyCompatibilityError("legacy setup did not call sim.init")
        return self._adapter

    @property
    def cellStates(self) -> dict[int, LegacyCell]:  # noqa: N802 - legacy API
        return self.adapter._cells  # pyright: ignore[reportPrivateUsage]

    def init(
        self,
        biophysics: object,
        regulator: object,
        signaling: object | None,
        integrator: object | None,
    ) -> None:
        del regulator
        if self._adapter is not None:
            raise LegacyCompatibilityError("legacy setup called sim.init more than once")
        if not isinstance(biophysics, _LegacyCLBacterium):
            raise LegacyCompatibilityError("only legacy CLBacterium physics is supported")
        if signaling is not None or integrator is not None:
            raise LegacyCompatibilityError(
                "legacy signaling and integration objects must be replaced by typed rate plans"
            )
        if self.module is None:
            raise AssertionError("legacy module is not attached")
        initialize = cast(InitCallback, _required_callback(self.module, "init"))
        update = cast(UpdateCallback, _required_callback(self.module, "update"))
        divide_value = _optional_callback(self.module, "divide")
        divide = cast(DivideCallback, divide_value) if divide_value is not None else None
        if self.restoring:
            self._adapter = LegacyModelAdapter.from_controller_state(
                self.simulation,
                self._controller,
                init=initialize,
                update=update,
                divide=divide,
            )
        else:
            self._adapter = LegacyModelAdapter(
                self.simulation,
                init=initialize,
                update=update,
                divide=divide,
                mechanics=True,
                compute_neighbors=biophysics.compute_neighbors,
                division_jitter_z=(
                    None if biophysics.alternate_divisions else biophysics.jitter_z
                ),
                alternate_divisions=biophysics.alternate_divisions,
                max_substeps=biophysics.max_substeps,
                rng=self._context.rng,
                mechanics_parameters=biophysics.mechanics_parameters,
            )

    def addCell(self, **values: object) -> int:  # noqa: N802 - legacy API
        aliases = {
            "cellType",
            "cellAdh",
            "pos",
            "dir",
            "len",
            "length",
            "rad",
            "radius",
            "color",
        }
        unknown = values.keys() - aliases
        if unknown:
            raise LegacyCompatibilityError(f"unsupported legacy addCell fields: {sorted(unknown)}")
        if "len" in values and "length" in values:
            raise LegacyCompatibilityError("legacy addCell supplied both len and length")
        if "rad" in values and "radius" in values:
            raise LegacyCompatibilityError("legacy addCell supplied both rad and radius")
        if self.restoring:
            setup_ids = self.adapter._setup_cell_ids  # pyright: ignore[reportPrivateUsage]
            if self._restored_cell_cursor >= len(setup_ids):
                raise LegacyCompatibilityError(
                    "legacy setup adds more cells than the saved controller"
                )
            cell_id = setup_ids[self._restored_cell_cursor]
            self._restored_cell_cursor += 1
            return cell_id
        cell = CellInit()
        cell.position = _vec3(values.get("pos", (0.0, 0.0, 0.0)), "cell position")
        cell.direction = _vec3(values.get("dir", (1.0, 0.0, 0.0)), "cell direction")
        cell.length = float(cast(Any, values.get("length", values.get("len", 3.5))))
        cell.radius = float(cast(Any, values.get("radius", values.get("rad", 0.5))))
        cell.cell_type = int(cast(Any, values.get("cellType", 0)))
        cell_id = self.adapter.add_cell(cell)
        self.adapter._setup_cell_ids.append(cell_id)  # pyright: ignore[reportPrivateUsage]
        legacy_cell = self.adapter.cells[cell_id]
        legacy_cell.cellAdh = int(cast(Any, values.get("cellAdh", 0)))
        if "color" in values:
            legacy_cell.color = values["color"]
        return cell_id

    def addRenderer(self, renderer: object) -> None:  # noqa: N802 - legacy API
        del renderer

    def loadFromPickle(self, data: object) -> None:  # noqa: N802 - legacy API
        del data
        raise LegacyCompatibilityError(
            "legacy pickle loading requires the separate one-way migration tool"
        )

    def validate_setup_complete(self) -> None:
        if self.restoring and self._restored_cell_cursor != len(
            self.adapter._setup_cell_ids  # pyright: ignore[reportPrivateUsage]
        ):
            raise LegacyCompatibilityError(
                "legacy setup adds fewer cells than the saved controller"
            )


def _required_callback(module: ModuleType, name: str) -> Callable[..., Any]:
    value = module.__dict__.get(name)
    if not callable(value):
        raise LegacyCompatibilityError(f"legacy model must define {name}(...)")
    return value


def _optional_callback(module: ModuleType, name: str) -> Callable[..., Any] | None:
    value = module.__dict__.get(name)
    if value is None:
        return None
    if not callable(value):
        raise LegacyCompatibilityError(f"legacy model {name} must be callable")
    return value


def _module(name: str) -> ModuleType:
    result = ModuleType(name)
    result.__package__ = name.rpartition(".")[0]
    return result


def _legacy_modules() -> dict[str, ModuleType]:
    root = _module("CellModeller")
    regulation = _module("CellModeller.Regulation")
    regulator = _module("CellModeller.Regulation.ModuleRegulator")
    regulator.ModuleRegulator = _LegacyModuleRegulator  # type: ignore[attr-defined]
    biophysics = _module("CellModeller.Biophysics")
    bacterial = _module("CellModeller.Biophysics.BacterialModels")
    cl_bacterium = _module("CellModeller.Biophysics.BacterialModels.CLBacterium")
    cl_bacterium.CLBacterium = _LegacyCLBacterium  # type: ignore[attr-defined]
    gui = _module("CellModeller.GUI")
    renderers = _module("CellModeller.GUI.Renderers")
    renderers.GLBacteriumRenderer = _LegacyRenderer  # type: ignore[attr-defined]
    renderers.GLGridRenderer = _LegacyRenderer  # type: ignore[attr-defined]
    gui.Renderers = renderers  # type: ignore[attr-defined]
    integration = _module("CellModeller.Integration")
    signaling = _module("CellModeller.Signalling")
    modules = {
        module.__name__: module
        for module in (
            root,
            regulation,
            regulator,
            biophysics,
            bacterial,
            cl_bacterium,
            gui,
            renderers,
            integration,
            signaling,
        )
    }
    for component in ("CLEulerIntegrator", "CLEulerSigIntegrator", "CLCrankNicIntegrator"):
        module = _module(f"CellModeller.Integration.{component}")
        setattr(module, component, _UnsupportedLegacyComponent)
        modules[module.__name__] = module
    grid = _module("CellModeller.Signalling.GridDiffusion")
    grid.GridDiffusion = _UnsupportedLegacyComponent  # type: ignore[attr-defined]
    modules[grid.__name__] = grid
    return modules


@contextmanager
def _installed_legacy_modules() -> Generator[None]:
    modules = _legacy_modules()
    previous = {name: sys.modules.get(name, _MISSING) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                del sys.modules[name]
            else:
                sys.modules[name] = cast(ModuleType, value)


def build_legacy_model(
    path: str | Path, context: ModelContext
) -> tuple[LegacyModelAdapter, dict[str, JSONValue]]:
    """Load an unchanged growth/mechanics model through the explicit compatibility facade."""

    return _load_legacy_model(Path(path).resolve(), context)


def resume_legacy_model(
    path: str | Path,
    context: ModelContext,
    bundle: CheckpointBundle,
) -> tuple[LegacyModelAdapter, dict[str, JSONValue]]:
    """Restore an authenticated legacy controller using the exact recorded model source."""

    if bundle.controller is None:
        raise LegacyCompatibilityError("checkpoint does not contain legacy controller state")
    source_path = Path(path).resolve()
    model_value = bundle.provenance.get("model")
    if not isinstance(model_value, dict):
        raise LegacyCompatibilityError("checkpoint is missing legacy model provenance")
    digest_value = model_value.get("sha256")
    seed_value = model_value.get("seed")
    parameters_value = model_value.get("parameters")
    compatibility_value = model_value.get("compatibility")
    if (
        not isinstance(digest_value, str)
        or not isinstance(seed_value, int)
        or isinstance(seed_value, bool)
        or not isinstance(parameters_value, dict)
        or compatibility_value != "legacy-python-callbacks-v1"
    ):
        raise LegacyCompatibilityError("checkpoint legacy model provenance is invalid")
    if seed_value != context.seed or parameters_value != dict(context.parameters):
        raise LegacyCompatibilityError("legacy resume context differs from checkpoint provenance")
    return _load_legacy_model(
        source_path,
        context,
        simulation=bundle.simulation,
        controller=bundle.controller,
        expected_digest=digest_value,
    )


def _load_legacy_model(
    source_path: Path,
    context: ModelContext,
    *,
    simulation: Simulation | None = None,
    controller: JSONValue = None,
    expected_digest: str | None = None,
) -> tuple[LegacyModelAdapter, dict[str, JSONValue]]:
    try:
        source = source_path.read_bytes()
    except OSError as error:
        raise BatchError(f"could not read legacy model {source_path}") from error
    digest = hashlib.sha256(source).hexdigest()
    if expected_digest is not None and digest != expected_digest:
        raise LegacyCompatibilityError("legacy model source digest does not match checkpoint")
    module_name = f"_microsimulator_legacy_{digest[:16]}"
    module = ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__package__ = ""
    setup_facade = _LegacySetupFacade(
        context,
        module_name,
        source,
        simulation=simulation,
        controller=controller,
    )
    setup_facade.module = module

    previous_module = sys.modules.get(module_name, _MISSING)
    sys.modules[module_name] = module
    sys.path.insert(0, str(source_path.parent))
    try:
        with _installed_legacy_modules():
            code = compile(source, str(source_path), "exec")
            exec(code, module.__dict__)
            if module.__dict__.get("random") is stdlib_random:
                module.__dict__["random"] = context.rng
            setup = module.__dict__.get("setup")
            if not callable(setup):
                raise LegacyCompatibilityError("legacy model must define setup(sim)")
            cast(Callable[[_LegacySetupFacade], object], setup)(setup_facade)
            adapter = setup_facade.adapter
            setup_facade.validate_setup_complete()
            if setup_facade.restoring:
                restored_rng = adapter._rng  # pyright: ignore[reportPrivateUsage]
                if restored_rng is not None:
                    module.__dict__["random"] = restored_rng
                    context.rng = restored_rng
    except (BatchError, LegacyCompatibilityError):
        raise
    except Exception as error:
        raise BatchError(f"legacy model {source_path} failed: {error}") from error
    finally:
        sys.path.pop(0)
        if previous_module is _MISSING:
            del sys.modules[module_name]
        else:
            sys.modules[module_name] = cast(ModuleType, previous_module)

    provenance: dict[str, JSONValue] = {
        "model": {
            "path": str(source_path),
            "sha256": digest,
            "seed": context.seed,
            "parameters": dict(context.parameters),
            "compatibility": "legacy-python-callbacks-v1",
        }
    }
    return adapter, provenance
