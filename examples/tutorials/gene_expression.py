"""Native ports of the CellModeller intracellular-dynamics tutorials.

Select ``constitutive``, ``legacy_constitutive``, ``dilution``,
``derepression``, or ``oscillator`` with the JSON-valued ``scenario`` model
parameter.
"""

from __future__ import annotations

from collections.abc import Mapping

from microsimulator import (
    CellInit,
    CellUpdate,
    ControllerStep,
    MechanicsConfig,
    ModelContext,
    NativeController,
    RatePlanBuilder,
    SpeciesRatePlan,
    StepPlan,
    UniformLengthDivision,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "tutorials.gene-expression"
MODEL_VERSION = 1
_SCENARIOS = frozenset(
    {"constitutive", "legacy_constitutive", "dilution", "derepression", "oscillator"}
)


def _scenario(parameters: Mapping[str, JSONValue]) -> str:
    value = parameters.get("scenario", "constitutive")
    if not isinstance(value, str) or value not in _SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(_SCENARIOS)}")
    return value


def _division(scenario: str) -> UniformLengthDivision:
    if scenario in {"constitutive", "oscillator"}:
        return UniformLengthDivision(3.0, 3.5, jitter_z=False)
    return UniformLengthDivision(2.5, 3.0, jitter_z=False)


def _growth_rate(scenario: str) -> float:
    return 0.6 if scenario == "oscillator" else 1.0 if scenario == "constitutive" else 2.0


def _initial_species(scenario: str) -> list[float]:
    if scenario == "dilution":
        return [10.0]
    if scenario == "derepression":
        return [10.0, 0.0]
    if scenario == "oscillator":
        return [0.0, 0.0]
    return [0.0]


def _rate_plan(scenario: str) -> SpeciesRatePlan:
    rates = RatePlanBuilder()
    if scenario == "constitutive":
        return rates.species_plan(1, (rates.constant(2.0),))
    if scenario == "legacy_constitutive":
        return rates.species_plan(1, (rates.constant(1.0),))
    if scenario == "dilution":
        return rates.species_plan(1, (rates.constant(0.0),))
    if scenario == "derepression":
        x0 = rates.species(0)
        return rates.species_plan(
            2,
            (rates.constant(0.0), 4.0 / (4.0 + x0 * x0)),
        )

    activator = rates.species(0)
    inhibitor = rates.species(1)
    activator_squared = activator * activator
    activator_rate = (
        2.0 * (1.0 + activator_squared)
        / (1.0 + activator_squared + inhibitor * inhibitor)
        - activator
    )
    inhibitor_rate = 2.0 * (1.0 + activator_squared) / (1.0 + activator_squared) - inhibitor
    return rates.species_plan(2, (activator_rate, inhibitor_rate))


def _callbacks(scenario: str):
    division = _division(scenario)

    def regulate(step: ControllerStep) -> StepPlan:
        return StepPlan(
            updates=tuple(
                CellUpdate(cell.id, growth_rate=_growth_rate(scenario)) for cell in step.cells
            ),
            divisions=division.requests(step),
        )

    return division, regulate


def build(context: ModelContext) -> NativeController:
    scenario = _scenario(context.parameters)
    initial_species = _initial_species(scenario)
    simulation = context.simulation(
        reserved_capacity=100_000,
        species_count=len(initial_species),
    )
    simulation.set_species_rate_plan(_rate_plan(scenario))

    founder = CellInit()
    founder.length = 3.5
    founder.radius = 0.5
    founder.growth_rate = _growth_rate(scenario)
    founder.species = initial_species
    founder_id = simulation.add_cell(founder)

    division, regulate = _callbacks(scenario)
    state: dict[str, JSONValue] = {"scenario": scenario}
    division.initialize(state, context.rng, (founder_id,))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=regulate,
        on_division=division.on_division,
        mechanics=MechanicsConfig(),
        state=state,
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    scenario = _scenario(context.parameters)
    division, regulate = _callbacks(scenario)
    controller = NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=regulate,
        on_division=division.on_division,
    )
    if controller.state.get("scenario") != scenario:
        raise ValueError("checkpoint scenario does not match model parameters")
    return controller
