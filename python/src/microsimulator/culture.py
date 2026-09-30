"""Compile fluid, chemical and biological definitions into one native transaction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from . import _core as core  # pyright: ignore[reportPrivateUsage, reportMissingModuleSource]
from .growth import CellGrowth, EssentialNutrientGrowth, MonodGrowth, ReserveBudgetGrowth
from .media import Concentration, Medium
from .schedules import at, times
from .stokes import StokesFlow
from .transport import AdvectiveFeed, ReservoirContact, Solute, SoluteTransport


def _concentrations(medium: Medium, solutes: tuple[Solute, ...], length: float) -> list[float]:
    unknown = set(medium.concentrations) - {s.id for s in solutes}

    if unknown:
        raise ValueError(f"medium {medium.name!r} refers to unknown solutes: {sorted(unknown)}")

    return [
        medium.concentrations[s.id].canonical(s.amount_unit, length, s.molar_mass_g_per_mol)
        if s.id in medium.concentrations
        else 0.0
        for s in solutes
    ]


def _chemical_boundaries(
    fluid: StokesFlow,
    transport: SoluteTransport,
    length_unit_m: float,
    recipes: dict[str, object],
    seconds: float,
) -> list[core.ChemicalBoundary]:
    result: list[core.ChemicalBoundary] = []

    for port in fluid.domain.ports:
        authored = transport.boundaries[port]
        value = core.ChemicalBoundary()
        value.port = port
        medium: Medium | None

        if isinstance(authored, (AdvectiveFeed, ReservoirContact)):
            value.kind = (
                core.ChemicalBoundaryKind.ADVECTIVE
                if isinstance(authored, AdvectiveFeed)
                else core.ChemicalBoundaryKind.RESERVOIR_CONTACT
            )
            medium = at(authored.medium, seconds)
        else:
            value.kind = core.ChemicalBoundaryKind.OUTFLOW
            value.allow_backflow = authored.backflow != "error"
            medium = None if authored.backflow == "error" else at(authored.backflow, seconds)

        value.concentrations = (
            [0.0] * len(transport.solutes)
            if medium is None
            else _concentrations(medium, transport.solutes, length_unit_m)
        )

        if medium is not None:
            definition = {
                "name": medium.name,
                "provenance": medium.provenance,
                "concentrations": {
                    s: {"value": c.value, "unit": c.unit} for s, c in medium.concentrations.items()
                },
            }

            if medium.name in recipes and recipes[medium.name] != definition:
                raise ValueError("distinct medium recipes must have distinct names")

            recipes[medium.name] = definition

        result.append(value)

    return result


def _configure_growth(
    config: core.CultureConfiguration,
    simulation: core.Simulation,
    transport: SoluteTransport,
    cell_growth: Mapping[int, CellGrowth],
) -> None:
    solute_index = {s.id: i for i, s in enumerate(transport.solutes)}
    existing = {c.id: c for c in simulation.cells()}

    if set(cell_growth) - set(existing):
        raise ValueError("growth binding refers to absent cells")

    models: list[core.CellGrowthModel] = []
    reserve_models = [b for b in cell_growth.values() if isinstance(b.model, ReserveBudgetGrowth)]

    if reserve_models and len(reserve_models) != len(cell_growth):
        raise ValueError("reserve and kinetic growth cannot share a culture configuration")

    for cell_id, binding in cell_growth.items():
        if isinstance(binding.model, ReserveBudgetGrowth):
            if binding != reserve_models[0]:
                raise ValueError("the reserve model currently requires shared parameters")

            config.biomass_per_geometric_volume = (
                binding.biomass.biochemical_volume_per_geometric_volume
            )
            requirements: list[core.ReserveRequirement] = []

            for species, amount in binding.model.requirements.items():
                r = core.ReserveRequirement()
                r.species, r.amount_per_biomass = species, amount
                requirements.append(r)

            config.biomass_requirements = requirements
            continue

        if existing[cell_id].growth_rate != 0:
            raise ValueError("set prescribed growth_rate to zero when using a kinetic model")

        model = _kinetic_growth_model(cell_id, binding, config, transport, solute_index)
        models.append(model)

    config.growth = models


def _kinetic_growth_model(
    cell_id: int,
    binding: CellGrowth,
    config: core.CultureConfiguration,
    transport: SoluteTransport,
    solute_index: dict[str, int],
) -> core.CellGrowthModel:
    growth = binding.model

    if isinstance(growth, ReserveBudgetGrowth):
        raise TypeError("kinetic growth requires a kinetic model")

    model = core.CellGrowthModel()
    model.cell_id = cell_id
    model.biomass_density = (
        binding.biomass.density_g_per_um3 * (config.grid.length_unit_m / 1e-6) ** 3
    )
    model.volume_ratio = binding.biomass.biochemical_volume_per_geometric_volume
    model.mu_max = growth.mu_max_per_hour / 3600 * config.grid.time_unit_s

    if isinstance(growth, MonodGrowth):
        model.kind = core.GrowthKind.MONOD
        requirements_data = [
            (
                growth.substrate,
                growth.half_saturation,
                growth.biomass_yield.grams,
                growth.biomass_yield.amount_unit,
            )
        ]
    elif isinstance(growth, EssentialNutrientGrowth):  # pyright: ignore[reportUnnecessaryIsInstance]
        model.kind = core.GrowthKind.ESSENTIAL
        requirements_data = [
            (s, Concentration.mM(r.half_saturation_mM), r.yield_g_per_mol, "mol")
            for s, r in growth.requirements.items()
        ]
    else:
        raise TypeError("unsupported growth model")

    native_requirements: list[core.GrowthRequirement] = []

    for name, concentration, yield_value, yield_basis in requirements_data:
        if name not in solute_index:
            raise ValueError(f"growth requires unknown solute {name!r}")

        solute = transport.solutes[solute_index[name]]
        requirement = core.GrowthRequirement()
        requirement.solute = solute_index[name]
        requirement.half_saturation = concentration.canonical(
            solute.amount_unit, config.grid.length_unit_m, solute.molar_mass_g_per_mol
        )

        if yield_basis != solute.amount_unit:
            if solute.molar_mass_g_per_mol is None:
                raise ValueError("yield unit conversion requires molecular weight")

            yield_value *= (
                1 / solute.molar_mass_g_per_mol
                if yield_basis == "mol"
                else solute.molar_mass_g_per_mol
            )

        requirement.biomass_yield = yield_value
        native_requirements.append(requirement)

    model.requirements = native_requirements

    return model


def configure_culture(
    simulation: core.Simulation,
    fluid: StokesFlow,
    transport: SoluteTransport,
    cell_growth: Mapping[int, CellGrowth],
) -> None:
    if set(transport.boundaries) != set(fluid.domain.ports):
        raise ValueError("each fluid port needs exactly one chemical boundary")

    config = core.CultureConfiguration()
    config.grid, config.fluid = fluid.domain.native_grid(), fluid.properties.native()
    config.solver, config.geometry, config.stepping = fluid.solver, fluid.geometry, fluid.stepping
    native_solutes: list[core.Solute] = []

    for solute in transport.solutes:
        value = core.Solute()
        value.name, value.amount_unit = solute.id, solute.amount_unit
        value.diffusion = (
            solute.diffusion_um2_per_s
            * 1e-12
            / config.grid.length_unit_m**2
            * config.grid.time_unit_s
        )
        native_solutes.append(value)

    config.solutes = native_solutes
    event_times: set[float] = set()

    for drive in fluid.boundaries.values():
        event_times.update(times(drive))

    for boundary in transport.boundaries.values():
        if isinstance(boundary, (AdvectiveFeed, ReservoirContact)):
            event_times.update(times(boundary.medium))
        elif boundary.backflow != "error":
            event_times.update(times(boundary.backflow))

    recipes: dict[str, object] = {}

    config.ports, config.reservoirs = (
        fluid.native_ports(),
        _chemical_boundaries(fluid, transport, config.grid.length_unit_m, recipes, 0.0),
    )
    events: list[core.CultureEvent] = []

    for seconds in sorted(event_times):
        event = core.CultureEvent()
        event.time = seconds / config.grid.time_unit_s
        event.ports, event.reservoirs = (
            fluid.native_ports(seconds),
            _chemical_boundaries(fluid, transport, config.grid.length_unit_m, recipes, seconds),
        )
        events.append(event)

    config.events = events
    _configure_growth(config, simulation, transport, cell_growth)
    config.authoring_json = json.dumps(
        {
            "version": 1,
            "recipes": recipes,
            "initial_medium": {
                "name": transport.initial_medium.name,
                "provenance": transport.initial_medium.provenance,
                "concentrations": {
                    s: {"value": c.value, "unit": c.unit}
                    for s, c in transport.initial_medium.concentrations.items()
                },
            },
            "solutes": [
                {
                    "id": s.id,
                    "label": s.label,
                    "amount_unit": s.amount_unit,
                    "molar_mass_g_per_mol": s.molar_mass_g_per_mol,
                }
                for s in transport.solutes
            ],
        },
        sort_keys=True,
        allow_nan=False,
    )
    initial = _concentrations(
        transport.initial_medium, transport.solutes, config.grid.length_unit_m
    )
    simulation._configure_culture(config, initial)


@dataclass(frozen=True, slots=True)
class CultureCell:
    dry_biomass_g: float | None
    biomass_produced_g: float
    realized_specific_growth_rate_per_hour: float
    surface_concentrations_mM: Mapping[str, float]  # noqa: N815
    uptake: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CultureState:
    time_seconds: float
    cells: Mapping[int, CultureCell]
    extracellular_amounts: Mapping[str, float]
    boundary_transfer: Mapping[str, Mapping[str, float]]
    budget_residual: Mapping[str, float]
    native: core.CultureCheckpoint


def capture_culture(simulation: core.Simulation) -> CultureState | None:
    state = simulation.culture_checkpoint

    if state is None:
        return None

    config = state.configuration
    models = {m.cell_id: m for m in config.growth}
    cells: dict[int, CultureCell] = {}

    for cell in state.cells:
        model = models.get(cell.body.id)
        concentrations = simulation.cell_surface_concentrations(cell.body.id)
        cells[cell.body.id] = CultureCell(
            None if model is None else cell.biochemical_volume * model.biomass_density,
            cell.biomass_produced,
            cell.realized_specific_rate * 3600 / config.grid.time_unit_s,
            MappingProxyType(
                {
                    s.name: concentrations[i] / config.grid.length_unit_m**3
                    for i, s in enumerate(config.solutes)
                    if s.amount_unit == "mol"
                }
            ),
            MappingProxyType({s.name: cell.uptake_totals[i] for i, s in enumerate(config.solutes)}),
        )

    ns = len(config.solutes)

    return CultureState(
        state.time * config.grid.time_unit_s,
        MappingProxyType(cells),
        MappingProxyType(
            {s.name: sum(state.extracellular_amounts[i::ns]) for i, s in enumerate(config.solutes)}
        ),
        MappingProxyType(
            {
                t.port: MappingProxyType(
                    {s.name: t.amounts[i] for i, s in enumerate(config.solutes)}
                )
                for t in state.reservoir_totals
            }
        ),
        MappingProxyType(
            {
                s.name: state.last_report.transport.mass_balance_error[i]
                if i < len(state.last_report.transport.mass_balance_error)
                else 0.0
                for i, s in enumerate(config.solutes)
            }
        ),
        state,
    )
