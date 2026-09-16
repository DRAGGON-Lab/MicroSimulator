"""Reusable deterministic division policies for native controllers."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import cast

from ._core import Vec3  # pyright: ignore[reportMissingModuleSource]
from .checkpoint import JSONValue
from .controller import ControllerStateError, ControllerStep, DivisionEvent, DivisionRequest

_STATE_KEY = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,127}\Z")


def _valid_cell_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True, slots=True)
class UniformLengthDivision:
    """Divide above per-cell thresholds sampled from one uniform distribution."""

    minimum: float
    maximum: float
    jitter_z: bool | None = None
    state_key: str = "length_division"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum)
            or not math.isfinite(self.maximum)
            or self.minimum < 0.0
            or self.maximum < self.minimum
        ):
            raise ValueError("division target range must be finite, ordered, and non-negative")
        if self.jitter_z is not None and not isinstance(cast(object, self.jitter_z), bool):
            raise ValueError("division jitter_z must be Boolean or None")
        if _STATE_KEY.fullmatch(self.state_key) is None:
            raise ValueError("division state key is invalid")

    def _sample(self, rng: random.Random) -> float:
        return rng.uniform(self.minimum, self.maximum)

    def initialize(
        self,
        state: dict[str, JSONValue],
        rng: random.Random,
        cell_ids: tuple[int, ...],
    ) -> None:
        """Create targets for founder cells before constructing a controller."""

        if self.state_key in state:
            raise ControllerStateError(f"controller state already contains {self.state_key!r}")
        if len(cell_ids) != len(set(cell_ids)) or any(
            not _valid_cell_id(cell_id) for cell_id in cell_ids
        ):
            raise ControllerStateError("founder cell IDs are invalid")
        state[self.state_key] = {
            "targets": {str(cell_id): self._sample(rng) for cell_id in cell_ids}
        }

    def _targets(self, step: ControllerStep) -> dict[str, JSONValue]:
        policy = step.state.get(self.state_key)
        if not isinstance(policy, dict) or set(policy) != {"targets"}:
            raise ControllerStateError(f"controller state {self.state_key!r} is invalid")
        targets = policy["targets"]
        if not isinstance(targets, dict):
            raise ControllerStateError(f"controller state {self.state_key!r} targets are invalid")
        for target in targets.values():
            if (
                not isinstance(target, int | float)
                or isinstance(target, bool)
                or not math.isfinite(target)
                or target < 0.0
            ):
                raise ControllerStateError("division target is invalid")
        return targets

    def requests(self, step: ControllerStep) -> tuple[DivisionRequest, ...]:
        """Return stable-ID-ordered division requests for cells above target length."""

        targets = self._targets(step)
        if set(targets) != {str(cell.id) for cell in step.cells}:
            raise ControllerStateError("division targets do not match active cell identities")
        return tuple(
            DivisionRequest(cell.id)
            for cell in step.cells
            if cell.length > float(cast(int | float, targets[str(cell.id)]))
        )

    def forget(self, step: ControllerStep, cell_ids: tuple[int, ...]) -> None:
        """Drop division targets for cells the returned plan removes."""

        targets = self._targets(step)
        for cell_id in cell_ids:
            targets.pop(str(cell_id), None)

    def on_division(self, step: ControllerStep, event: DivisionEvent) -> None:
        """Transfer policy state, apply optional jitter, and sample daughter targets."""

        targets = self._targets(step)
        parent_key = str(event.parent.id)
        daughter_keys = {str(event.first.id), str(event.second.id)}
        active = {str(cell.id) for cell in step.cells}
        # Plan removals apply after divisions, so cells already forgotten for
        # this step's removals are still active here; targets may be a subset
        # of the pre-division identities but never contain anything else.
        expected = (active - daughter_keys) | {parent_key}
        if parent_key not in targets or not set(targets) <= expected:
            raise ControllerStateError("division targets do not match pre-division identities")
        del targets[parent_key]
        if self.jitter_z is not None:
            for daughter in (event.first, event.second):
                jitter = [step.rng.uniform(-1.0e-3, 1.0e-3) for _ in range(3)]
                if not self.jitter_z:
                    jitter[2] = 0.0
                direction = Vec3(
                    daughter.direction.x + jitter[0],
                    daughter.direction.y + jitter[1],
                    daughter.direction.z + jitter[2],
                )
                step.simulation.set_cell_geometry(
                    daughter.id,
                    daughter.position,
                    direction,
                    daughter.length,
                )
        targets[str(event.first.id)] = self._sample(step.rng)
        targets[str(event.second.id)] = self._sample(step.rng)
