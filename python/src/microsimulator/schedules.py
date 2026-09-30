"""Immutable, right-continuous simulation schedules, authored in seconds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True, slots=True)
class PiecewiseConstant[T]:
    initial: T
    changes: tuple[tuple[float, T], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "changes", tuple((float(t), v) for t, v in self.changes))
        previous = 0.0

        for time, _ in self.changes:
            if not math.isfinite(time) or time <= previous:
                raise ValueError("schedule times must be finite, positive and strictly increasing")

            previous = time

    def at(self, seconds: float) -> T:
        result = self.initial

        for time, value in self.changes:
            if time > seconds:
                break

            result = value

        return result


def at[T](value: T | PiecewiseConstant[T], seconds: float) -> T:
    return (
        cast(PiecewiseConstant[T], value).at(seconds)
        if isinstance(value, PiecewiseConstant)
        else cast(T, value)
    )


def times[T](value: T | PiecewiseConstant[T]) -> tuple[float, ...]:
    return (
        tuple(t for t, _ in cast(PiecewiseConstant[T], value).changes)
        if isinstance(value, PiecewiseConstant)
        else ()
    )
