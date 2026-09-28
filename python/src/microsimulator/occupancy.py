"""Native Metal/CUDA primitives for ADR 0025's coarse geometric porosity model.

The method signatures and value types match :mod:`occupancy_reference`. Arrays
and device arithmetic are float32; ledgers are accumulated in host float64.
This standalone API does not enable occupancy in Simulation or checkpoints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._core import BackendKind, OccupancyCapsule, OccupancyFace, OccupancyReservoir
from ._core import OccupancySolver as NativeOccupancySolver
from .occupancy_reference import Balance, Capsule, Face, ReservoirFace

__all__ = ["Balance", "Capsule", "Face", "OccupancySolver", "ReservoirFace", "SolverReport"]

Array = NDArray[np.float32]


def _array(values: ArrayLike, name: str) -> Array:
    original = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = original.astype(np.float32)
    if not np.isfinite(result).all() or np.any((original != 0) & (result == 0)):
        raise ValueError(f"{name} must be finite and representable in float32")
    return result


def _vector(values: ArrayLike, name: str, count: int | None = None) -> list[float]:
    result = _array(values, name)
    if result.ndim != 1 or (count is not None and len(result) != count):
        raise ValueError(f"{name} must be a vector with matching size")
    return cast(list[float], result.tolist())


def _index(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(cast(object, value), int)
        or not 0 <= value < 2**32 - 1
    ):
        raise ValueError(f"{name} must be a nonnegative integer below 2**32 - 1")
    return value


@dataclass(frozen=True)
class SolverReport:
    iterations: int
    relative_residual: float


class OccupancySolver:
    """Explicit GPU device and reproducible cutoff; no automatic CPU fallback.

    Pass ``"metal"`` or ``"cuda"`` (or the corresponding BackendKind). Use
    ``occupancy_reference`` for the independent float64 CPU implementation.
    Each method returns a candidate without mutating input arrays. The caller
    owns geometry barriers, support connectivity, intrinsic face velocities,
    and the commit/rollback of a complete simulation transaction.
    """

    def __init__(
        self,
        backend: str | BackendKind,
        *,
        device_index: int = 0,
        epsilon_cutoff: float = 1e-8,
    ) -> None:
        if isinstance(backend, str):
            if backend not in ("metal", "cuda"):
                raise ValueError("native occupancy requires 'metal' or 'cuda'")
            backend = BackendKind.METAL if backend == "metal" else BackendKind.CUDA
        self._native = NativeOccupancySolver(
            backend, _index(device_index, "device index"), epsilon_cutoff
        )
        self._backend = backend
        self._device_index = device_index
        self._epsilon_cutoff = float(np.float32(epsilon_cutoff))
        self._last_report: SolverReport | None = None

    @property
    def backend(self) -> BackendKind:
        return self._backend

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def epsilon_cutoff(self) -> float:
        return self._epsilon_cutoff

    @property
    def last_report(self) -> SolverReport | None:
        """Report from the last successfully returned backward-Euler candidate."""
        return self._last_report

    def geometric_porosity(
        self,
        centers: ArrayLike,
        spacing: tuple[float, float, float],
        cells: Sequence[Capsule],
        *,
        subdivisions: int = 8,
        walls: Sequence[bool] | None = None,
    ) -> Array:
        """Midpoint capsule-union quadrature; m is restricted to 1..256."""
        points = _array(centers, "centers")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("centers must be finite N by 3 coordinates")
        mask: list[int] = []
        if walls is not None:
            solid = np.asarray(walls)
            if solid.shape != (len(points),) or solid.dtype != np.bool_:
                raise ValueError("walls must contain one Boolean per voxel")
            mask = solid.astype(np.uint32).tolist()
        return np.asarray(
            self._native.geometric_porosity(
                cast(list[tuple[float, float, float]], points.tolist()),
                spacing,
                [
                    OccupancyCapsule(cell.center, cell.direction, cell.length, cell.radius)
                    for cell in cells
                ],
                _index(subdivisions, "subdivisions"),
                mask,
            ),
            dtype=np.float32,
        )

    def accessible_volumes(self, porosity: ArrayLike, voxel_volume: float) -> Array:
        return np.asarray(
            self._native.accessible_volumes(_vector(porosity, "porosity"), voxel_volume),
            dtype=np.float32,
        )

    def concentration(self, amount: ArrayLike, volume: ArrayLike) -> Array:
        return np.asarray(
            self._native.concentration(_vector(amount, "amount"), _vector(volume, "volume")),
            dtype=np.float32,
        )

    def porosity_face(
        self,
        first: int,
        second: int,
        epsilon_first: float,
        epsilon_second: float,
        *,
        diffusion: float,
        area: float,
        distance: float,
        intrinsic_velocity: float = 0.0,
    ) -> Face:
        face = self._native.porosity_face(
            _index(first, "first site"),
            _index(second, "second site"),
            epsilon_first,
            epsilon_second,
            diffusion,
            area,
            distance,
            intrinsic_velocity,
        )
        return Face(face.first, face.second, face.conductance, face.volume_flux)

    def remap_amounts(
        self,
        amount: ArrayLike,
        old_volume: ArrayLike,
        new_volume: ArrayLike,
        neighbors: Sequence[tuple[int, int]],
    ) -> Array:
        return np.asarray(
            self._native.remap_amounts(
                _vector(amount, "amount"),
                _vector(old_volume, "old volume"),
                _vector(new_volume, "new volume"),
                [(_index(i, "neighbor"), _index(j, "neighbor")) for i, j in neighbors],
            ),
            dtype=np.float32,
        )

    def exchange_weights(self, kernel: ArrayLike, volume: ArrayLike) -> Array:
        """Normalize phi*W on caller-supplied, already connected physical support."""
        return np.asarray(
            self._native.exchange_weights(_vector(kernel, "kernel"), _vector(volume, "volume")),
            dtype=np.float32,
        )

    def backward_euler(
        self,
        amount: ArrayLike,
        volume: ArrayLike,
        faces: Sequence[Face],
        dt: float,
        *,
        source: ArrayLike | None = None,
        loss: ArrayLike | None = None,
        reservoirs: Sequence[ReservoirFace] = (),
        max_iterations: int = 20000,
        relative_tolerance: float = 1e-7,
    ) -> tuple[Array, Balance]:
        """Sparse float32 Jacobi solve with explicit amounts and a signed ledger.

        Source is amount/time and loss is 1/time. Supply accessible-fluid affine
        production as W*b in source. Face coefficients already include aperture.
        Periodicity is an explicit shared internal face; no-flux faces are absent.
        Reject negative candidates, nonconvergence, or a ledger residual above
        5e-6 times the normalized amount scale. No amount renormalization occurs.
        """
        amounts = _vector(amount, "amount")
        result = self._native.backward_euler(
            amounts,
            _vector(volume, "volume", len(amounts)),
            [
                OccupancyFace(
                    _index(f.first, "first site"),
                    _index(f.second, "second site"),
                    f.conductance,
                    f.volume_flux,
                )
                for f in faces
            ],
            dt,
            [] if source is None else _vector(source, "source", len(amounts)),
            [] if loss is None else _vector(loss, "loss", len(amounts)),
            [
                OccupancyReservoir(
                    _index(f.site, "reservoir site"), f.concentration, f.conductance, f.volume_flux
                )
                for f in reservoirs
            ],
            _index(max_iterations, "max iterations"),
            relative_tolerance,
        )
        balance = result.balance
        self._last_report = SolverReport(result.iterations, result.relative_residual)
        return np.asarray(result.amount, dtype=np.float32), Balance(
            balance.before, balance.after, balance.source, balance.reaction, balance.boundary
        )
