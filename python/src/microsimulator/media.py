"""Medium compositions and explicit concentration units; no organism kinetics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .schedules import PiecewiseConstant as PiecewiseConstant


@dataclass(frozen=True, slots=True)
class Concentration:
    value: float
    unit: Literal["mM", "mol/m3", "mol/L", "g/L"]

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("concentration must be finite and nonnegative")

        if self.unit not in ("mM", "mol/m3", "mol/L", "g/L"):
            raise ValueError("unsupported concentration unit")

    @classmethod
    def mM(cls, value: float) -> Concentration:  # noqa: N802
        return cls(value, "mM")

    def canonical(
        self, amount_unit: str, length_unit_m: float, molar_mass_g_per_mol: float | None = None
    ) -> float:
        basis = "g" if self.unit == "g/L" else "mol"
        per_m3 = self.value * (1000 if self.unit in ("g/L", "mol/L") else 1)

        if basis != amount_unit:
            if molar_mass_g_per_mol is None:
                raise ValueError("mass/molar conversion requires molecular weight")

            per_m3 *= molar_mass_g_per_mol if basis == "mol" else 1 / molar_mass_g_per_mol

        return per_m3 * length_unit_m**3


@dataclass(frozen=True, slots=True)
class Medium:
    name: str
    concentrations: Mapping[str, Concentration]
    provenance: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("medium needs a name")

        if any(
            not key or not isinstance(c, Concentration)  # pyright: ignore[reportUnnecessaryIsInstance]
            for key, c in self.concentrations.items()
        ):
            raise ValueError("medium entries require solute identifiers and Concentration values")

        object.__setattr__(self, "concentrations", MappingProxyType(dict(self.concentrations)))

    @classmethod
    def millimolar(
        cls, name: str, concentrations: Mapping[str, float], *, provenance: str | None = None
    ) -> Medium:
        return cls(
            name,
            {key: Concentration.mM(value) for key, value in concentrations.items()},
            provenance,
        )


@dataclass(frozen=True, slots=True)
class Perfusion:
    """Convenience pairing; imports keep physics independent of recipes."""

    port: str
    drive: object
    medium: Medium | PiecewiseConstant[Medium]

    def boundaries(self) -> tuple[object, object]:
        from .transport import AdvectiveFeed

        return self.drive, AdvectiveFeed(self.medium)
