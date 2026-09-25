"""Float64 design reference for ADR 0025; not a native transport implementation.

Amounts are authoritative. Coarse geometric porosity is distinct from the
biochemical biomass density used by the flow-resistance closure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
EPSILON_CUTOFF = 1.0e-8


@dataclass(frozen=True)
class Capsule:
    center: tuple[float, float, float]
    direction: tuple[float, float, float]
    length: float
    radius: float

    def __post_init__(self) -> None:
        values = (*self.center, *self.direction, self.length, self.radius)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("capsule geometry must be finite")
        if len(self.center) != 3 or len(self.direction) != 3:
            raise ValueError("capsule vectors must have three coordinates")
        if self.length < 0 or self.radius <= 0 or math.hypot(*self.direction) == 0:
            raise ValueError("capsule requires nonnegative length, positive radius and direction")


def geometric_porosity(
    centers: Array,
    spacing: tuple[float, float, float],
    cells: Sequence[Capsule],
    *,
    subdivisions: int = 8,
    walls: Sequence[bool] | None = None,
) -> Array:
    """Midpoint quadrature of the union of capsules, clipped to fluid voxels.

    centers use native lattice-center convention. Walls use the existing binary
    voxel mask, not a second interpretation of mechanical constraint surfaces.
    Overlaps count once. The result is a coarse storage fraction, not a resolved
    aperture or a sub-voxel connectivity claim.
    """
    points = np.asarray(centers, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("centers must be finite N by 3 coordinates")
    if len(spacing) != 3 or any(not math.isfinite(h) or h <= 0 for h in spacing):
        raise ValueError("spacing must contain three positive finite lengths")
    if (
        isinstance(subdivisions, bool)
        or not isinstance(cast(object, subdivisions), int)
        or subdivisions < 1
    ):
        raise ValueError("subdivisions must be a positive integer")
    solid = np.zeros(len(points), dtype=np.bool_) if walls is None else np.asarray(walls)
    if solid.shape != (len(points),) or solid.dtype != np.bool_:
        raise ValueError("walls must contain one Boolean per voxel")
    samples = (np.arange(subdivisions, dtype=np.float64) + 0.5) / subdivisions - 0.5
    offsets = np.stack(np.meshgrid(samples, samples, samples, indexing="ij"), axis=-1)
    offsets = offsets.reshape(-1, 3) * np.asarray(spacing)
    result = np.zeros(len(points))
    for index, center in enumerate(points):
        if solid[index]:
            continue
        coordinates = center + offsets
        occupied = np.zeros(len(coordinates), dtype=np.bool_)
        for cell in cells:
            direction = np.asarray(cell.direction, dtype=np.float64)
            direction = direction / np.linalg.norm(direction)
            relative = coordinates - np.asarray(cell.center)
            axial = np.clip(relative @ direction, -cell.length / 2, cell.length / 2)
            distance = relative - axial[:, None] * direction
            squared = cast(Array, np.sum(distance * distance, axis=1))
            within = squared <= cell.radius * cell.radius
            occupied = np.logical_or(occupied, within)
        result[index] = 1.0 - np.mean(occupied)
    result[result < EPSILON_CUTOFF] = 0
    return result


def _vector(values: Sequence[float] | Array, name: str, count: int | None = None) -> Array:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    if count is not None and result.shape != (count,):
        raise ValueError(f"{name} size mismatch")
    return result


def accessible_volumes(porosity: Sequence[float] | Array, voxel_volume: float) -> Array:
    epsilon = _vector(porosity, "porosity")
    if np.any(epsilon < 0) or np.any(epsilon > 1):
        raise ValueError("porosity must lie in [0, 1]")
    if not math.isfinite(voxel_volume) or voxel_volume <= 0:
        raise ValueError("voxel volume must be finite and positive")
    return np.where(epsilon < EPSILON_CUTOFF, 0.0, epsilon) * voxel_volume


def concentration(amount: Sequence[float] | Array, volume: Sequence[float] | Array) -> Array:
    n = _vector(amount, "amount")
    w = _vector(volume, "accessible volume", len(n))
    if np.any(n < 0) or np.any(w < 0) or np.any((w == 0) & (n != 0)):
        raise ValueError("nonnegative amounts require accessible storage")
    return np.divide(n, w, out=np.zeros_like(n), where=w > 0)


def remap_amounts(
    amount: Sequence[float] | Array,
    old_volume: Sequence[float] | Array,
    new_volume: Sequence[float] | Array,
    neighbors: Sequence[tuple[int, int]],
) -> Array:
    """Keep surviving voxel amounts; expel closing storage conservatively.

    Recipients are all newly accessible voxels in the old-or-new accessible
    face-connected component, weighted by new accessible volume. A closing
    component with nonzero amount fails atomically. Inputs are never mutated.
    """
    n = _vector(amount, "amount")
    old = _vector(old_volume, "old volume", len(n))
    new = _vector(new_volume, "new volume", len(n))
    concentration(n, old)
    if np.any(new < 0):
        raise ValueError("new volume must be nonnegative")
    result = n.copy()
    adjacency: list[list[int]] = [[] for _ in n]
    for first, second in neighbors:
        if first == second or not 0 <= first < len(n) or not 0 <= second < len(n):
            raise ValueError("invalid neighbor edge")
        adjacency[first].append(second)
        adjacency[second].append(first)
    active = (old > 0) | (new > 0)
    visited: set[int] = set()
    for start in range(len(n)):
        if not active[start] or start in visited:
            continue
        pending = [start]
        component: list[int] = []
        while pending:
            index = pending.pop()
            if index in visited or not active[index]:
                continue
            visited.add(index)
            component.append(index)
            pending.extend(adjacency[index])
        component.sort()
        donors = [index for index in component if new[index] == 0]
        recipients = [index for index in component if new[index] > 0]
        expelled = math.fsum(float(n[index]) for index in donors)
        if expelled > 0 and not recipients:
            raise ValueError(
                f"closing component at voxel {start} has solute but no accessible recipient"
            )
        result[donors] = 0
        if expelled:
            capacity = math.fsum(float(new[index]) for index in recipients)
            for index in recipients:
                result[index] += expelled * new[index] / capacity
    concentration(result, new)
    return result


@dataclass(frozen=True)
class Face:
    """Internal oriented face: diffusive conductance L^3/T, fluid flux L^3/T."""

    first: int
    second: int
    conductance: float
    volume_flux: float = 0.0


def porosity_face(
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
    """Harmonic porosity closure; aperture is applied exactly once to flux."""
    values = (epsilon_first, epsilon_second, diffusion, area, distance, intrinsic_velocity)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("face data must be finite")
    if not 0 <= epsilon_first <= 1 or not 0 <= epsilon_second <= 1:
        raise ValueError("face porosities must lie in [0, 1]")
    if diffusion < 0 or area <= 0 or distance <= 0:
        raise ValueError("invalid face geometry or diffusion")
    aperture = (
        0.0
        if min(epsilon_first, epsilon_second) < EPSILON_CUTOFF
        else 2 * epsilon_first * epsilon_second / (epsilon_first + epsilon_second)
    )
    return Face(
        first, second, diffusion * aperture * area / distance, aperture * area * intrinsic_velocity
    )


@dataclass(frozen=True)
class ReservoirFace:
    """Exterior reservoir: positive flux leaves domain; concentration is amount/L^3."""

    site: int
    concentration: float
    conductance: float = 0.0
    volume_flux: float = 0.0


@dataclass(frozen=True)
class Balance:
    before: float
    after: float
    source: float
    reaction: float
    boundary: float

    @property
    def residual(self) -> float:
        return self.after - self.before - self.source - self.reaction - self.boundary


def backward_euler(
    amount: Sequence[float] | Array,
    volume: Sequence[float] | Array,
    faces: Sequence[Face],
    dt: float,
    *,
    source: Sequence[float] | Array | None = None,
    loss: Sequence[float] | Array | None = None,
    reservoirs: Sequence[ReservoirFace] = (),
) -> tuple[Array, Balance]:
    """Dense float64 finite-volume reference with an explicit amount ledger.

    source is amount/time, loss is 1/time. Interior faces are equal/opposite;
    first-order advection and reservoir/loss terms are implicit. Not scalable.
    """
    n = _vector(amount, "amount")
    w = _vector(volume, "volume", len(n))
    concentration(n, w)
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    s = np.zeros_like(n) if source is None else _vector(source, "source", len(n))
    k = np.zeros_like(n) if loss is None else _vector(loss, "loss", len(n))
    if np.any(k < 0) or np.any((w == 0) & (s != 0)):
        raise ValueError("loss must be nonnegative; sources require accessible storage")
    operator = np.diag(k * w)
    rhs = n + dt * s
    for face in faces:
        i, j, g, q = face.first, face.second, face.conductance, face.volume_flux
        if i == j or not 0 <= i < len(n) or not 0 <= j < len(n):
            raise ValueError("invalid transport face indices")
        if not math.isfinite(g) or g < 0 or not math.isfinite(q):
            raise ValueError("invalid transport coefficients")
        if (w[i] == 0 or w[j] == 0) and (g != 0 or q != 0):
            raise ValueError("closed storage cannot have an open face")
        operator[i, i] += g + max(q, 0)
        operator[j, j] += g + max(-q, 0)
        operator[i, j] -= g + max(-q, 0)
        operator[j, i] -= g + max(q, 0)
    for face in reservoirs:
        i, c, g, q = face.site, face.concentration, face.conductance, face.volume_flux
        if not 0 <= i < len(n) or w[i] == 0:
            raise ValueError("reservoir must connect accessible storage")
        if not all(math.isfinite(value) for value in (c, g, q)) or c < 0 or g < 0:
            raise ValueError("invalid reservoir coefficients")
        operator[i, i] += g + max(q, 0)
        rhs[i] += dt * (g + max(-q, 0)) * c
    matrix = np.diag(w) + dt * operator
    for i in range(len(n)):
        if w[i] == 0:
            matrix[i, i] = 1.0
    c = np.linalg.solve(matrix, rhs)
    updated = c * w
    if not np.isfinite(updated).all() or np.any(updated < 0):
        raise ValueError("step produced invalid amount; no clipping is permitted")
    boundary = dt * math.fsum(
        f.conductance * (f.concentration - c[f.site])
        - f.volume_flux * (c[f.site] if f.volume_flux >= 0 else f.concentration)
        for f in reservoirs
    )
    return updated, Balance(
        float(n.sum()),
        float(updated.sum()),
        float(dt * s.sum()),
        float(-dt * np.dot(k, updated)),
        float(boundary),
    )


def exchange_weights(
    kernel: Sequence[float] | Array,
    volume: Sequence[float] | Array,
) -> Array:
    """Accessible-volume weighted partition of unity in a declared connected support."""
    base = _vector(kernel, "kernel")
    accessible = _vector(volume, "volume", len(base))
    if np.any(base < 0) or np.any(accessible < 0):
        raise ValueError("exchange kernel and volume must be nonnegative")
    weights = base * accessible
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("cell has no valid accessible exchange support")
    return weights / weights.sum()
