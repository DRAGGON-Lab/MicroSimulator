"""Exact and tabulated reference solutions for validating flow solves.

These are the analytic answers the flow solvers are measured against, kept in
one place so a benchmark run and a test suite check the same physics. They
also give a model author a way to validate a device grid: build the reference
geometry with `duct_grid`, solve it, and compare.

References:
- Plane Poiseuille and rectangular duct series: F. M. White, "Viscous Fluid
  Flow" (3rd ed., ch. 3); the duct peak-to-mean ratio is tabulated in R. K.
  Shah and A. L. London, "Laminar Flow Forced Convection in Ducts" (1978).
- Two-layer Brinkman channel: the exact solution of the Brinkman equation
  (H. C. Brinkman, Appl. Sci. Res. A1, 1949) matched in value and slope across
  the fluid-porous interface.
- Hele-Shaw closure: depth-averaged Stokes flow in a thin gap obeys Darcy's
  law with mobility proportional to the squared gap height (H. S. Hele-Shaw,
  1898).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ._core import GridBoundaryKind, GridShape, SignalGridSpec, Vec3

_Profile = NDArray[np.float64]

SQUARE_DUCT_PEAK_TO_MEAN = 2.0962
"""Peak-to-mean axial velocity of fully developed flow in a square duct."""


def duct_grid(
    nx: int, ny: int, nz: int, spacing: tuple[float, float, float]
) -> SignalGridSpec:
    """A duct grid flowing along y between fixed inlet and outlet boundaries."""

    shape = GridShape()
    shape.x, shape.y, shape.z = nx, ny, nz
    spec = SignalGridSpec()
    spec.signal_count = 1
    spec.shape = shape
    spec.spacing = Vec3(*spacing)
    spec.diffusion = [1.0]
    spec.advection = [Vec3()]
    for name in ("y_lower", "y_upper"):
        boundary = getattr(spec, name)
        boundary.kind = GridBoundaryKind.FIXED
        boundary.values = [0.0]
        setattr(spec, name, boundary)
    return spec


def site_index(spec: SignalGridSpec, x: int, y: int, z: int) -> int:
    """The flat site index of a lattice position in the grid's site order."""

    return (x * spec.shape.y + y) * spec.shape.z + z


def plane_poiseuille(positions: _Profile) -> _Profile:
    """The unit-mean parabolic profile across a channel of unit width."""

    return 6.0 * positions * (1.0 - positions)


def two_layer_brinkman(drag: float, positions: _Profile) -> _Profile:
    """The exact profile of a channel half open and half porous.

    A unit pressure gradient drives flow across a unit-width channel with
    no-slip walls at both edges. The lower half is open Stokes flow and the
    upper half carries Brinkman drag ``drag``, an inverse permeability. The
    two branches match in value and in slope at the interface, and the
    solution is unscaled: its amplitude is part of the reference.
    """

    if drag <= 0.0:
        raise ValueError("drag must be positive")
    root = math.sqrt(drag)
    matrix = np.array(
        [
            [0.0, math.cosh(root), math.sinh(root)],
            [0.5, -math.cosh(root * 0.5), -math.sinh(root * 0.5)],
            [1.0, -root * math.sinh(root * 0.5), -root * math.cosh(root * 0.5)],
        ]
    )
    rhs = np.array([-1.0 / drag, 1.0 / drag + 0.125, 0.5])
    linear, cosh_c, sinh_c = (float(value) for value in np.linalg.solve(matrix, rhs))
    profile: _Profile = np.where(
        positions < 0.5,
        -positions * positions / 2.0 + linear * positions,
        1.0 / drag + cosh_c * np.cosh(root * positions) + sinh_c * np.sinh(root * positions),
    )
    return profile


def centerline_value(profile: _Profile) -> float:
    """Interpolate a cell-centered symmetric profile to its center.

    Cell centers straddle the axis of a duct with an even cell count, so the
    largest sampled value understates the peak. The symmetric four-point
    combination ``(9 * inner - outer) / 8`` per axis recovers the center value
    exactly for a parabola and to fourth order for a smooth even profile.
    """

    result = profile
    for _ in range(profile.ndim):
        count = result.shape[0]
        if count < 4 or count % 2 != 0:
            raise ValueError("centerline interpolation needs at least four cells per axis")
        middle = count // 2
        inner = 0.5 * (result[middle - 1] + result[middle])
        outer = 0.5 * (result[middle - 2] + result[middle + 1])
        result = (9.0 * inner - outer) / 8.0
    return float(result)
