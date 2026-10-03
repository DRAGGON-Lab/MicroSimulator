"""Solute identities, diffusion and chemical boundary authoring."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .media import Medium
from .schedules import PiecewiseConstant


@dataclass(frozen=True, slots=True)
class Solute:
    id: str
    diffusion_um2_per_s: float
    amount_unit: Literal["mol", "g"] = "mol"
    label: str | None = None
    molar_mass_g_per_mol: float | None = None

    def __post_init__(self) -> None:
        if (
            not self.id
            or not math.isfinite(self.diffusion_um2_per_s)
            or self.diffusion_um2_per_s < 0
        ):
            raise ValueError("invalid solute identifier or diffusion coefficient")

        if self.amount_unit not in ("mol", "g"):
            raise ValueError("solute amount_unit must be mol or g")

        if self.molar_mass_g_per_mol is not None and (
            not math.isfinite(self.molar_mass_g_per_mol) or self.molar_mass_g_per_mol <= 0
        ):
            raise ValueError("molecular weight must be finite and positive")


@dataclass(frozen=True, slots=True)
class AdvectiveFeed:
    medium: Medium | PiecewiseConstant[Medium]


@dataclass(frozen=True, slots=True)
class ReservoirContact:
    medium: Medium | PiecewiseConstant[Medium]


@dataclass(frozen=True, slots=True)
class ConvectiveOutflow:
    backflow: Literal["error"] | Medium | PiecewiseConstant[Medium] = "error"


@dataclass(frozen=True, slots=True)
class SoluteTransport:
    solutes: tuple[Solute, ...]
    boundaries: Mapping[str, AdvectiveFeed | ReservoirContact | ConvectiveOutflow]
    initial_medium: Medium

    def __post_init__(self) -> None:
        object.__setattr__(self, "solutes", tuple(self.solutes))
        object.__setattr__(self, "boundaries", MappingProxyType(dict(self.boundaries)))
        ids = [s.id for s in self.solutes]

        if len(set(ids)) != len(ids):
            raise ValueError("duplicate solute identifier")
