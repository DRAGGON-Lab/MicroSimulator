"""Optional presentation labels; numerical channel indices remain authoritative."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .checkpoint import JSONValue


class ChannelMetadataError(ValueError):
    """Raised when labels cannot describe the declared numerical channels."""


@dataclass(frozen=True, slots=True)
class ChannelMetadata:
    """Ordered labels, with ``None`` for an unnamed group or individual channel.

    Explicit tuples must match channel counts exactly. Empty/whitespace-only
    strings are treated as missing labels for display, but preserved on disk.
    """

    species: tuple[str | None, ...] | None = None
    signals: tuple[str | None, ...] | None = None

    def __post_init__(self) -> None:
        for kind in ("species", "signals"):
            labels = cast(object, getattr(self, kind))
            if labels is None:
                continue
            if not isinstance(labels, tuple):
                raise ChannelMetadataError(f"channel_metadata.{kind}: expected a tuple or None")
            for index, label in enumerate(cast(tuple[object, ...], labels)):
                if label is not None and not isinstance(label, str):
                    raise ChannelMetadataError(
                        f"channel_metadata.{kind}[{index}]: expected a string or None"
                    )
                if isinstance(label, str):
                    try:
                        label.encode("utf-8")
                    except UnicodeEncodeError as error:
                        raise ChannelMetadataError(
                            f"channel_metadata.{kind}[{index}]: invalid Unicode scalar value"
                        ) from error

    def resolved(self, species_count: int, signal_count: int) -> ChannelMetadata:
        """Validate explicit counts and expand unspecified groups to null labels."""

        groups: list[tuple[str | None, ...]] = []
        for kind, labels, count in (
            ("species", self.species, species_count),
            ("signals", self.signals, signal_count),
        ):
            if labels is not None and len(labels) != count:
                raise ChannelMetadataError(
                    f"channel_metadata.{kind}: expected {count} labels, got {len(labels)}"
                )
            groups.append((None,) * count if labels is None else labels)
        return ChannelMetadata(species=groups[0], signals=groups[1])

    def to_json(self, species_count: int, signal_count: int) -> dict[str, JSONValue]:
        labels = self.resolved(species_count, signal_count)
        return {"species": list(labels.species or ()), "signals": list(labels.signals or ())}

    @classmethod
    def from_json(cls, value: object, species_count: int, signal_count: int) -> ChannelMetadata:
        """Decode the closed data-only representation shared by scenes/checkpoints."""

        if not isinstance(value, dict) or set(cast(dict[object, object], value)) != {
            "species",
            "signals",
        }:
            raise ChannelMetadataError("channel_metadata: expected exactly species and signals")
        data = cast(dict[str, object], value)
        groups: list[tuple[str | None, ...]] = []
        for kind in ("species", "signals"):
            values = data[kind]
            if not isinstance(values, list):
                raise ChannelMetadataError(f"channel_metadata.{kind}: expected an array")
            groups.append(cast(tuple[str | None, ...], tuple(cast(list[object], values))))
        return cls(species=groups[0], signals=groups[1]).resolved(species_count, signal_count)


UNNAMED_CHANNELS = ChannelMetadata()
