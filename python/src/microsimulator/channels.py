"""Optional presentation labels; numerical channel indices remain authoritative.

``ChannelMetadata`` holds ordered labels for species and signal channels, with
``None`` for an unnamed group or an individual unnamed channel. Explicit tuples
must match the channel counts exactly. Empty and whitespace-only strings are
treated as missing labels for display but are preserved on disk. The type is
implemented by the native core.
"""

from __future__ import annotations

from ._core import (  # pyright: ignore[reportMissingModuleSource]
    ChannelMetadata,
    ChannelMetadataError,
)

UNNAMED_CHANNELS = ChannelMetadata()

__all__ = ["UNNAMED_CHANNELS", "ChannelMetadata", "ChannelMetadataError"]
