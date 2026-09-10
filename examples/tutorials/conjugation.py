"""Native contact-dependent plasmid conjugation tutorial."""

from __future__ import annotations

import math
from collections.abc import Mapping

from microsimulator import (
    CellInit,
    CellUpdate,
    ControllerStep,
    DivisionEvent,
    DivisionRequest,
    MechanicsConfig,
    ModelContext,
    NativeController,
    StepPlan,
    Vec3,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "tutorials.conjugation"
MODEL_VERSION = 1


def _probability(parameters: Mapping[str, JSONValue]) -> float:
    value = parameters.get("transfer_probability", 0.1)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0.0
        or value > 1.0
    ):
        raise ValueError("transfer_probability must be finite and in [0, 1]")
    return float(value)


def _targets(step: ControllerStep) -> dict[str, JSONValue]:
    value = step.state.get("division_targets")
    if not isinstance(value, dict):
        raise ValueError("conjugation division state is invalid")
    return value


def _callbacks(transfer_probability: float):
    def regulate(step: ControllerStep) -> StepPlan:
        targets = _targets(step)
        if set(targets) != {str(cell.id) for cell in step.cells}:
            raise ValueError("conjugation division state does not match active cells")
        graph = step.simulation.find_cell_contacts()
        cell_types = {cell.id: cell.cell_type for cell in step.cells}
        updates: list[CellUpdate] = []
        divisions: list[DivisionRequest] = []
        for cell in step.cells:
            next_type = cell.cell_type
            if cell.cell_type == 0:
                infectious_neighbors = (
                    neighbor
                    for neighbor in graph.neighbor_ids(cell.slot)
                    if cell_types[neighbor] != 0
                )
                if any(step.rng.random() < transfer_probability for _ in infectious_neighbors):
                    next_type = 2
            updates.append(CellUpdate(cell.id, growth_rate=1.0, cell_type=next_type))
            target = targets[str(cell.id)]
            if not isinstance(target, int | float) or isinstance(target, bool):
                raise ValueError("conjugation division target is invalid")
            if cell.length > float(target):
                divisions.append(DivisionRequest(cell.id))
        return StepPlan(updates=tuple(updates), divisions=tuple(divisions))

    def divided(step: ControllerStep, event: DivisionEvent) -> None:
        targets = _targets(step)
        del targets[str(event.parent.id)]
        for daughter in (event.first, event.second):
            sampled = daughter.length + step.rng.gauss(1.9, 0.45)
            targets[str(daughter.id)] = max(daughter.length + 0.1, sampled)

    return regulate, divided


def build(context: ModelContext) -> NativeController:
    transfer_probability = _probability(context.parameters)
    simulation = context.simulation(reserved_capacity=50_000)
    targets: dict[str, JSONValue] = {}
    for cell_type, x in ((0, -5.0), (1, 5.0)):
        founder = CellInit()
        founder.position = Vec3(x, 0.0, 0.0)
        founder.direction = Vec3(1.0, 0.0, 0.0)
        founder.length = 1.9
        founder.radius = 0.4
        founder.growth_rate = 1.0
        founder.cell_type = cell_type
        founder_id = simulation.add_cell(founder)
        targets[str(founder_id)] = founder.length + context.rng.gauss(1.9, 0.45)

    regulate, divided = _callbacks(transfer_probability)
    return NativeController(
        simulation,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        rng=context.rng,
        regulate=regulate,
        on_division=divided,
        mechanics=MechanicsConfig(gamma=10.0),
        state={
            "transfer_probability": transfer_probability,
            "division_targets": targets,
        },
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> NativeController:
    transfer_probability = _probability(context.parameters)
    regulate, divided = _callbacks(transfer_probability)
    controller = NativeController.from_checkpoint(
        checkpoint,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        regulate=regulate,
        on_division=divided,
    )
    if controller.state.get("transfer_probability") != transfer_probability:
        raise ValueError("checkpoint transfer probability does not match model parameters")
    return controller
