"""Cell-specific nutritional kinetics and explicit biomass conversion."""

from __future__ import annotations

# ruff: noqa: N803, N815 -- mM denotes millimolar, not millimeters.
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .media import Concentration


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class BiomassYield:
    grams: float
    amount_unit: Literal["mol", "g"]

    def __post_init__(self) -> None:
        _positive(self.grams, "biomass yield")

        if self.amount_unit not in ("mol", "g"):
            raise ValueError("yield basis must be mol or g")

    @classmethod
    def grams_per_mol(cls, value: float) -> BiomassYield:
        return cls(value, "mol")


@dataclass(frozen=True, slots=True)
class MonodGrowth:
    substrate: str
    mu_max_per_hour: float
    half_saturation: Concentration
    biomass_yield: BiomassYield

    def __post_init__(self) -> None:
        if (
            not self.substrate
            or not math.isfinite(self.mu_max_per_hour)
            or self.mu_max_per_hour < 0
        ):
            raise ValueError("invalid substrate or specific growth rate")

        _positive(self.half_saturation.value, "half saturation")

    @classmethod
    def molar(
        cls,
        *,
        substrate: str,
        mu_max_per_hour: float,
        half_saturation_mM: float,
        yield_g_per_mol: float,
    ) -> MonodGrowth:
        return cls(
            substrate,
            mu_max_per_hour,
            Concentration.mM(half_saturation_mM),
            BiomassYield.grams_per_mol(yield_g_per_mol),
        )


@dataclass(frozen=True, slots=True)
class NutrientRequirement:
    half_saturation_mM: float
    yield_g_per_mol: float

    def __post_init__(self) -> None:
        _positive(self.half_saturation_mM, "half saturation")
        _positive(self.yield_g_per_mol, "biomass yield")


@dataclass(frozen=True, slots=True)
class EssentialNutrientGrowth:
    mu_max_per_hour: float
    requirements: Mapping[str, NutrientRequirement]
    limitation: Literal["liebig_minimum"] = "liebig_minimum"

    def __post_init__(self) -> None:
        if self.limitation != "liebig_minimum" or not self.requirements:
            raise ValueError("essential growth requires nutrients and liebig_minimum")

        if not math.isfinite(self.mu_max_per_hour) or self.mu_max_per_hour < 0:
            raise ValueError("invalid specific growth rate")

        object.__setattr__(self, "requirements", MappingProxyType(dict(self.requirements)))

    @classmethod
    def molar(
        cls,
        *,
        mu_max_per_hour: float,
        requirements: Mapping[str, NutrientRequirement],
        limitation: Literal["liebig_minimum"] = "liebig_minimum",
    ) -> EssentialNutrientGrowth:
        return cls(mu_max_per_hour, requirements, limitation)


@dataclass(frozen=True, slots=True)
class ReserveBudgetGrowth:
    """Existing prescribed elongation, limited by named intracellular pools.

    Keys are intracellular species indices; values are amount per biochemical
    volume. Surface exchange remains explicit in the native configuration.
    """

    requirements: Mapping[int, float]

    def __post_init__(self) -> None:
        for species, amount in self.requirements.items():
            if species < 0:
                raise ValueError("invalid intracellular species index")

            _positive(amount, "reserve requirement")

        object.__setattr__(self, "requirements", MappingProxyType(dict(self.requirements)))


@dataclass(frozen=True, slots=True)
class BiomassConversion:
    density_g_per_um3: float
    biochemical_volume_per_geometric_volume: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.density_g_per_um3, "biomass density")
        _positive(self.biochemical_volume_per_geometric_volume, "biochemical volume ratio")


@dataclass(frozen=True, slots=True)
class CellGrowth:
    model: MonodGrowth | EssentialNutrientGrowth | ReserveBudgetGrowth
    biomass: BiomassConversion
