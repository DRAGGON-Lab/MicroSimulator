"""The conserved biochemical volume and its distinct geometric counterpart."""

import math


def biomass_volume(length: float, radius: float) -> float:
    """Return pi*r**2*(length + 2*r), conserved by native cell division.

    At constant biomass density, multiplying this measure by density gives
    biomass mass. It is the same volume used by native concentration dilution.
    """
    if not math.isfinite(length) or length < 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("length must be finite and nonnegative; radius finite and positive")
    return math.pi * radius * radius * (length + 2 * radius)


def capsule_volume(length: float, radius: float) -> float:
    """Return the geometric capsule volume, which is not the biomass measure."""
    return biomass_volume(length, radius) - (2 / 3) * math.pi * radius**3
