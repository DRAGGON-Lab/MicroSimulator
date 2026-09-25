"""Typed MicroSimulator ports of SimBOL's six BioBrick circuit examples.

Choose ``bba_0001`` through ``bba_0005`` or ``bba_i5200`` with the
``circuit`` model parameter.  Circuit 3 also accepts ``precursor_concentration``;
circuits 2, 4, and 5 accept ``inducer_concentration``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from microsimulator import (
    CellInit,
    CellUpdate,
    ControllerStep,
    CoupledRatePlan,
    GridShape,
    MechanicsConfig,
    ModelContext,
    NativeController,
    RateExpression,
    RatePlanBuilder,
    SignalGridSpec,
    SignalIntegrationKind,
    SpeciesRatePlan,
    StepPlan,
    UniformLengthDivision,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "tutorials.simbol-circuits"
MODEL_VERSION = 1
DIVISION = UniformLengthDivision(3.5, 3.505, jitter_z=False)
_CIRCUITS = frozenset(
    {"bba_0001", "bba_0002", "bba_0003", "bba_0004", "bba_0005", "bba_i5200"}
)


def _circuit(parameters: Mapping[str, JSONValue]) -> str:
    value = parameters.get("circuit", "bba_0001")
    if not isinstance(value, str) or value not in _CIRCUITS:
        raise ValueError(f"circuit must be one of {sorted(_CIRCUITS)}")
    return value


def _number(parameters: Mapping[str, JSONValue], name: str, default: float) -> float:
    value = parameters.get(name, default)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _repression(
    rates: RatePlanBuilder,
    repressor: RateExpression,
) -> RateExpression:
    return 16.0 / (16.0 + repressor**4.0)


def _activity_fraction(rates: RatePlanBuilder, inducer: float) -> RateExpression:
    return rates.constant(16.0 / (16.0 + inducer**4.0))


def _species_model(
    circuit: str,
    parameters: Mapping[str, JSONValue],
) -> tuple[list[float], SpeciesRatePlan]:
    rates = RatePlanBuilder()
    if circuit == "bba_0001":
        gfp = rates.species(0)
        return [1.0], rates.species_plan(1, (1.0 - 0.05 * gfp,))

    if circuit == "bba_0002":
        rfp = rates.species(0)
        tetr = rates.species(1)
        active_tetr = tetr * _activity_fraction(
            rates,
            _number(parameters, "inducer_concentration", 0.0),
        )
        return [2.0, 1.0], rates.species_plan(
            2,
            (_repression(rates, active_tetr) - 0.05 * rfp, 1.0 - 0.05 * tetr),
        )

    if circuit == "bba_0004":
        laci = rates.species(0)
        gfp = rates.species(1)
        active_laci = laci * _activity_fraction(
            rates,
            _number(parameters, "inducer_concentration", 1.0),
        )
        return [0.0, 0.0], rates.species_plan(
            2,
            (2.0 - 0.1 * laci, 2.0 * _repression(rates, active_laci) - 0.1 * gfp),
        )

    if circuit == "bba_0005":
        tetr = rates.species(0)
        gfp = rates.species(1)
        ci = rates.species(2)
        laci = rates.species(3)
        active_tetr = tetr * _activity_fraction(
            rates,
            _number(parameters, "inducer_concentration", 1.0),
        )
        k909012 = _repression(rates, ci) * _repression(rates, laci)
        return [1.0, 0.0, 0.0, 0.0], rates.species_plan(
            4,
            (
                2.0 - 0.1 * tetr,
                2.0 * k909012 - 0.1 * gfp,
                2.0 * _repression(rates, active_tetr) - 0.1 * ci,
                2.0 * k909012 - 0.1 * laci,
            ),
        )

    if circuit == "bba_i5200":
        ci = rates.species(0)
        gfp = rates.species(1)
        laci = rates.species(2)
        tetr = rates.species(3)
        return [1.0, 0.0, 0.0, 0.0], rates.species_plan(
            4,
            (
                2.0 * _repression(rates, tetr) - 0.1 * ci,
                2.0 * _repression(rates, laci) - 0.1 * gfp,
                2.0 * _repression(rates, ci) - 0.1 * laci,
                2.0 * _repression(rates, laci) - 0.1 * tetr,
            ),
        )
    raise AssertionError("signaling circuit must use the coupled model")


def _signal_grid() -> SignalGridSpec:
    shape = GridShape()
    shape.x, shape.y, shape.z = 80, 80, 3
    grid = SignalGridSpec()
    grid.signal_count = 1
    grid.shape = shape
    grid.origin = Vec3(-40.0, -40.0, -1.0)
    grid.spacing = Vec3(1.0, 1.0, 1.0)
    grid.diffusion = [0.01]
    grid.advection = [Vec3()]
    grid.integration = SignalIntegrationKind.CRANK_NICOLSON
    grid.solver.absolute_tolerance = 1.0e-12
    return grid


def _signaling_model(
    parameters: Mapping[str, JSONValue],
) -> tuple[list[float], CoupledRatePlan]:
    precursor = _number(parameters, "precursor_concentration", 5.0)
    rates = RatePlanBuilder()
    luxr = rates.species(0)
    gfp = rates.species(1)
    luxi = rates.species(2)
    complex_pool = rates.species(3)
    extracellular = rates.signal(0)
    signal_fourth = extracellular**4.0
    activation = signal_fourth / (16.0 + signal_fourth)
    exchange_amount = 0.1 * (complex_pool - extracellular) * rates.cell_surface_area()
    synthesis = 0.1 * (luxr + luxi) * precursor
    plan = rates.coupled_plan(
        4,
        1,
        (
            2.0 - 0.1 * luxr,
            2.0 * activation - 0.1 * gfp,
            2.0 - 0.1 * luxi,
            synthesis - 0.1 * complex_pool - exchange_amount / rates.cell_volume(),
        ),
        (exchange_amount,),
    )
    return [0.0, 0.0, 0.0, 0.0], plan


def _regulate(step: ControllerStep) -> StepPlan:
    return StepPlan(
        updates=tuple(CellUpdate(cell.id, growth_rate=1.0) for cell in step.cells),
        divisions=DIVISION.requests(step),
    )


def build(context: ModelContext) -> NativeController:
    circuit = _circuit(context.parameters)
    if circuit == "bba_0003":
        initial_species, coupled_plan = _signaling_model(context.parameters)
        simulation = context.simulation(reserved_capacity=10_000, species_count=4)
        simulation.configure_signal_grid(_signal_grid())
        simulation.set_coupled_rate_plan(coupled_plan)
    else:
        initial_species, species_plan = _species_model(circuit, context.parameters)
        simulation = context.simulation(
            reserved_capacity=10_000,
            species_count=len(initial_species),
        )
        simulation.set_species_rate_plan(species_plan)

    founder = CellInit()
    founder.length = 3.0
    founder.radius = 0.5
    founder.growth_rate = 1.0
    founder.species = initial_species
    state: dict[str, JSONValue] = {"circuit": circuit}
    DIVISION.initialize_founders(simulation, state, context.rng, (founder,))
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=_regulate,
        on_division=DIVISION.on_division,
        mechanics=MechanicsConfig(gamma=100.0),
        state=state,
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    circuit = _circuit(context.parameters)
    controller = NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=_regulate,
        on_division=DIVISION.on_division,
    )
    if controller.state.get("circuit") != circuit:
        raise ValueError("checkpoint circuit does not match model parameters")
    return controller
