"""Discrete incompatible-plasmid segregation with exact checkpointed copy counts."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import cast

from microsimulator import (
    CellInit,
    ModelContext,
    Simulation,
    capped_founder_length,
    capture_random_state,
    restore_random_state,
)
from microsimulator.checkpoint import CheckpointBundle, JSONValue

MODEL_ID = "tutorials.plasmid-segregation"
MODEL_VERSION = 1


def _copies(parameters: Mapping[str, JSONValue]) -> int:
    value = parameters.get("copies_per_cell", 10)
    if not isinstance(value, int) or isinstance(value, bool) or value < 2 or value > 1_000:
        raise ValueError("copies_per_cell must be an integer in [2, 1000]")
    return value


def _records(state: dict[str, JSONValue], name: str) -> dict[str, JSONValue]:
    value = state.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"plasmid controller {name} state is invalid")
    return value


def _count(record: JSONValue, name: str) -> int:
    if not isinstance(record, dict) or set(record) != {"a", "b"}:
        raise ValueError("plasmid count state is invalid")
    value = record[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("plasmid count state is invalid")
    return value


class PlasmidController:
    """Orchestrate discrete copy-number state around the native cell engine."""

    def __init__(
        self,
        simulation: Simulation,
        rng: random.Random,
        state: dict[str, JSONValue],
        *,
        copies_per_cell: int,
        completed_steps: int = 0,
    ) -> None:
        self.simulation = simulation
        self._rng = rng
        self._state = state
        self._copies_per_cell = copies_per_cell
        self._completed_steps = completed_steps
        self._validate()

    def _validate(self) -> None:
        cells = {str(cell.id) for cell in self.simulation.cells()}
        plasmids = _records(self._state, "plasmids")
        targets = _records(self._state, "division_targets")
        if set(plasmids) != cells or set(targets) != cells:
            raise ValueError("plasmid controller state does not match active cells")
        for cell_id in cells:
            a_count = _count(plasmids[cell_id], "a")
            b_count = _count(plasmids[cell_id], "b")
            if a_count + b_count != self._copies_per_cell:
                raise ValueError("plasmid copy-number total is invalid")
            target = targets[cell_id]
            if (
                not isinstance(target, int | float)
                or isinstance(target, bool)
                or not math.isfinite(target)
                or target <= 0.0
            ):
                raise ValueError("plasmid division target is invalid")

    def _partition(self, a_count: int, b_count: int) -> tuple[tuple[int, int], tuple[int, int]]:
        plasmids = ([0] * (2 * a_count)) + ([1] * (2 * b_count))
        self._rng.shuffle(plasmids)
        split = self._copies_per_cell
        first = plasmids[:split]
        second = plasmids[split:]
        return (
            (first.count(0), first.count(1)),
            (second.count(0), second.count(1)),
        )

    def _present_state(self) -> None:
        plasmids = _records(self._state, "plasmids")
        total = float(self._copies_per_cell)
        for cell in self.simulation.cells():
            record = plasmids[str(cell.id)]
            a_count = _count(record, "a")
            b_count = _count(record, "b")
            cell_type = 1 if b_count == 0 else 2 if a_count == 0 else 0
            self.simulation.set_cell_attributes(cell.id, cell.growth_rate, cell_type)
            self.simulation.set_species(cell.id, [a_count / total, b_count / total])

    def step(self, dt: float) -> None:
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError("time step must be finite and non-negative")
        self._validate()
        plasmids = _records(self._state, "plasmids")
        targets = _records(self._state, "division_targets")
        dividing = [
            cell
            for cell in self.simulation.cells()
            if cell.length > float(cast(int | float, targets[str(cell.id)]))
        ]
        for parent in dividing:
            parent_key = str(parent.id)
            record = plasmids[parent_key]
            a_count = _count(record, "a")
            b_count = _count(record, "b")
            first_counts, second_counts = self._partition(a_count, b_count)
            first_id, second_id = self.simulation.divide_equal(parent.id)
            del plasmids[parent_key]
            del targets[parent_key]
            for daughter_id, (daughter_a, daughter_b) in (
                (first_id, first_counts),
                (second_id, second_counts),
            ):
                plasmids[str(daughter_id)] = {"a": daughter_a, "b": daughter_b}
                targets[str(daughter_id)] = self._rng.uniform(3.5, 4.0)
            if first_counts[0] + second_counts[0] != 2 * a_count:
                raise AssertionError("plasmid A copies were not conserved")
            if first_counts[1] + second_counts[1] != 2 * b_count:
                raise AssertionError("plasmid B copies were not conserved")

        self.simulation.step(dt)
        if self.simulation.cell_count:
            self.simulation.relax_cell_mechanics()
        self._present_state()
        self._completed_steps += 1
        self._validate()

    def controller_state(self) -> dict[str, JSONValue]:
        self._validate()
        return {
            "kind": MODEL_ID,
            "version": MODEL_VERSION,
            "completed_steps": self._completed_steps,
            "copies_per_cell": self._copies_per_cell,
            "random": capture_random_state(self._rng),
            "state": self._state,
        }


def build(context: ModelContext) -> PlasmidController:
    copies_per_cell = _copies(context.parameters)
    simulation = context.simulation(reserved_capacity=100_000, species_count=2)
    founder = CellInit()
    founder.length = 3.5
    founder.radius = 0.5
    founder.growth_rate = 1.0
    first_count = copies_per_cell // 2
    second_count = copies_per_cell - first_count
    founder.species = [first_count / copies_per_cell, second_count / copies_per_cell]
    target = context.rng.uniform(3.5, 4.0)
    founder.length = capped_founder_length(founder.length, target)
    founder_id = simulation.add_cell(founder)
    state: dict[str, JSONValue] = {
        "plasmids": {str(founder_id): {"a": first_count, "b": second_count}},
        "division_targets": {str(founder_id): target},
    }
    return PlasmidController(
        simulation,
        context.rng,
        state,
        copies_per_cell=copies_per_cell,
    )


def resume(context: ModelContext, checkpoint: CheckpointBundle) -> PlasmidController:
    copies_per_cell = _copies(context.parameters)
    value = checkpoint.controller
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "version",
        "completed_steps",
        "copies_per_cell",
        "random",
        "state",
    }:
        raise ValueError("plasmid checkpoint controller state is invalid")
    if (
        value["kind"] != MODEL_ID
        or value["version"] != MODEL_VERSION
        or value["copies_per_cell"] != copies_per_cell
    ):
        raise ValueError("plasmid checkpoint model identity does not match")
    completed_steps = value["completed_steps"]
    state = value["state"]
    if (
        not isinstance(completed_steps, int)
        or isinstance(completed_steps, bool)
        or completed_steps < 0
    ):
        raise ValueError("plasmid checkpoint step count is invalid")
    if not isinstance(state, dict):
        raise ValueError("plasmid checkpoint model state is invalid")
    return PlasmidController(
        checkpoint.simulation,
        restore_random_state(value["random"]),
        cast(dict[str, JSONValue], state),
        copies_per_cell=copies_per_cell,
        completed_steps=completed_steps,
    )
