"""Readable construction of validated native species and coupled rate plans."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    CoupledRatePlan,
    RateInstruction,
    RateOp,
    SpeciesRatePlan,
)

_UINT32_MAX = (1 << 32) - 1
_FLOAT32_MAX = 3.4028234663852886e38


class RatePlanError(ValueError):
    """Raised when a symbolic rate expression is malformed."""


@dataclass(frozen=True, slots=True)
class RateExpression:
    """One immutable reference into a :class:`RatePlanBuilder`."""

    _builder: RatePlanBuilder
    index: int

    def __add__(self, other: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.ADD, self, other)

    def __radd__(self, other: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.ADD, other, self)

    def __sub__(self, other: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.SUBTRACT, self, other)

    def __rsub__(self, other: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.SUBTRACT, other, self)

    def __mul__(self, other: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.MULTIPLY, self, other)

    def __rmul__(self, other: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.MULTIPLY, other, self)

    def __truediv__(self, other: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.DIVIDE, self, other)

    def __rtruediv__(self, other: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.DIVIDE, other, self)

    def __pow__(self, exponent: float | int | RateExpression) -> RateExpression:
        return self._builder._binary(RateOp.POWER, self, exponent)

    def __neg__(self) -> RateExpression:
        return self._builder._unary(RateOp.NEGATE, self)

    def __bool__(self) -> bool:
        raise TypeError("rate expressions cannot be evaluated as Python Booleans")


class RatePlanBuilder:
    """Build an acyclic typed expression graph with ordinary arithmetic syntax."""

    def __init__(self) -> None:
        self._instructions: list[RateInstruction] = []

    @property
    def instruction_count(self) -> int:
        return len(self._instructions)

    def _emit(
        self,
        operation: RateOp,
        *,
        first: int = 0,
        second: int = 0,
        third: int = 0,
        value: float = 0.0,
    ) -> RateExpression:
        if len(self._instructions) >= _UINT32_MAX:
            raise RatePlanError("rate plan exceeds the uint32 instruction space")
        instruction = RateInstruction()
        instruction.operation = operation
        instruction.first = first
        instruction.second = second
        instruction.third = third
        instruction.value = value
        self._instructions.append(instruction)
        return RateExpression(self, len(self._instructions) - 1)

    def _coerce(self, value: float | int | RateExpression) -> RateExpression:
        if isinstance(value, RateExpression):
            if value._builder is not self:
                raise RatePlanError("rate expressions from different builders cannot be mixed")
            return value
        return self.constant(value)

    def _unary(self, operation: RateOp, value: RateExpression) -> RateExpression:
        operand = self._coerce(value)
        return self._emit(operation, first=operand.index)

    def _binary(
        self,
        operation: RateOp,
        first: float | int | RateExpression,
        second: float | int | RateExpression,
    ) -> RateExpression:
        left = self._coerce(first)
        right = self._coerce(second)
        return self._emit(operation, first=left.index, second=right.index)

    def constant(self, value: float | int) -> RateExpression:
        candidate = cast(object, value)
        if (
            not isinstance(candidate, int | float)
            or isinstance(candidate, bool)
            or not math.isfinite(candidate)
            or abs(candidate) > _FLOAT32_MAX
        ):
            raise RatePlanError("rate constant must be a finite float32 value")
        return self._emit(RateOp.CONSTANT, value=float(candidate))

    def _source(self, operation: RateOp, index: int = 0) -> RateExpression:
        candidate = cast(object, index)
        if (
            not isinstance(candidate, int)
            or isinstance(candidate, bool)
            or candidate < 0
            or candidate > _UINT32_MAX
        ):
            raise RatePlanError("rate source index must be an unsigned 32-bit integer")
        return self._emit(operation, first=candidate)

    def species(self, index: int) -> RateExpression:
        return self._source(RateOp.SPECIES, index)

    def signal(self, index: int) -> RateExpression:
        return self._source(RateOp.SIGNAL, index)

    def position_x(self) -> RateExpression:
        return self._source(RateOp.POSITION_X)

    def position_y(self) -> RateExpression:
        return self._source(RateOp.POSITION_Y)

    def position_z(self) -> RateExpression:
        return self._source(RateOp.POSITION_Z)

    def cell_length(self) -> RateExpression:
        return self._source(RateOp.CELL_LENGTH)

    def cell_radius(self) -> RateExpression:
        return self._source(RateOp.CELL_RADIUS)

    def growth_rate(self) -> RateExpression:
        return self._source(RateOp.GROWTH_RATE)

    def cell_type(self) -> RateExpression:
        return self._source(RateOp.CELL_TYPE)

    def cell_volume(self) -> RateExpression:
        return self._source(RateOp.CELL_VOLUME)

    def cell_surface_area(self) -> RateExpression:
        return self._source(RateOp.CELL_SURFACE_AREA)

    def minimum(
        self,
        first: float | int | RateExpression,
        second: float | int | RateExpression,
    ) -> RateExpression:
        return self._binary(RateOp.MINIMUM, first, second)

    def maximum(
        self,
        first: float | int | RateExpression,
        second: float | int | RateExpression,
    ) -> RateExpression:
        return self._binary(RateOp.MAXIMUM, first, second)

    def exponential(self, value: RateExpression) -> RateExpression:
        return self._unary(RateOp.EXPONENTIAL, value)

    def logarithm(self, value: RateExpression) -> RateExpression:
        return self._unary(RateOp.LOGARITHM, value)

    def less(
        self,
        first: float | int | RateExpression,
        second: float | int | RateExpression,
    ) -> RateExpression:
        return self._binary(RateOp.LESS, first, second)

    def less_equal(
        self,
        first: float | int | RateExpression,
        second: float | int | RateExpression,
    ) -> RateExpression:
        return self._binary(RateOp.LESS_EQUAL, first, second)

    def greater(
        self,
        first: float | int | RateExpression,
        second: float | int | RateExpression,
    ) -> RateExpression:
        return self._binary(RateOp.GREATER, first, second)

    def greater_equal(
        self,
        first: float | int | RateExpression,
        second: float | int | RateExpression,
    ) -> RateExpression:
        return self._binary(RateOp.GREATER_EQUAL, first, second)

    def equal(
        self,
        first: float | int | RateExpression,
        second: float | int | RateExpression,
    ) -> RateExpression:
        return self._binary(RateOp.EQUAL, first, second)

    def select(
        self,
        condition: RateExpression,
        when_true: float | int | RateExpression,
        when_false: float | int | RateExpression,
    ) -> RateExpression:
        predicate = self._coerce(condition)
        selected = self._coerce(when_true)
        fallback = self._coerce(when_false)
        return self._emit(
            RateOp.SELECT,
            first=predicate.index,
            second=selected.index,
            third=fallback.index,
        )

    def _outputs(self, values: tuple[RateExpression, ...]) -> list[int]:
        return [self._coerce(value).index for value in values]

    def species_plan(
        self,
        species_count: int,
        outputs: tuple[RateExpression, ...],
    ) -> SpeciesRatePlan:
        return SpeciesRatePlan(
            species_count,
            list(self._instructions),
            self._outputs(outputs),
        )

    def coupled_plan(
        self,
        species_count: int,
        signal_count: int,
        species_outputs: tuple[RateExpression, ...],
        signal_outputs: tuple[RateExpression, ...],
    ) -> CoupledRatePlan:
        return CoupledRatePlan(
            species_count,
            signal_count,
            list(self._instructions),
            self._outputs(species_outputs),
            self._outputs(signal_outputs),
        )
