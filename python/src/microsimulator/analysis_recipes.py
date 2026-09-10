"""Lazy Polars recipes over versioned MicroSimulator analysis datasets."""

# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import zarr

from .analysis import AnalysisDataset, AnalysisError, open_dataset

type DatasetSource = AnalysisDataset | str | os.PathLike[str]
type LengthField = Literal["cylinder_length", "capsule_length"]
type SignalAxis = Literal["x", "y", "z"]

_UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class SignalSlice:
    """One materialized signal plane with named remaining dimensions."""

    frame_index: int
    time: float
    channel: int
    fixed_axis: SignalAxis
    fixed_index: int
    dimensions: tuple[SignalAxis, SignalAxis]
    values: np.ndarray[Any, np.dtype[np.float32]]


def _dataset(source: DatasetSource) -> AnalysisDataset:
    return source if isinstance(source, AnalysisDataset) else open_dataset(source)


def _edges(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) < 2:
        raise AnalysisError(f"{name} requires at least two edges")
    if not all(math.isfinite(value) for value in result):
        raise AnalysisError(f"{name} edges must be finite")
    if any(right <= left for left, right in pairwise(result)):
        raise AnalysisError(f"{name} edges must be strictly increasing")
    if len(result) - 1 > _UINT32_MAX:
        raise AnalysisError(f"{name} has too many bins")
    return result


def _bins(values: Sequence[float], prefix: str) -> tuple[pl.LazyFrame, int]:
    edges = _edges(values, prefix)
    count = len(edges) - 1
    frame = pl.DataFrame(
        {
            f"{prefix}_bin": pl.Series(range(count), dtype=pl.UInt32),
            f"{prefix}_left": pl.Series(edges[:-1], dtype=pl.Float64),
            f"{prefix}_right": pl.Series(edges[1:], dtype=pl.Float64),
            f"{prefix}_center": pl.Series(
                [(left + right) / 2.0 for left, right in pairwise(edges)],
                dtype=pl.Float64,
            ),
        }
    )
    return frame.lazy(), count - 1


def _inside(value: str, prefix: str, last_bin: int) -> pl.Expr:
    return (pl.col(value) >= pl.col(f"{prefix}_left")) & (
        (pl.col(value) < pl.col(f"{prefix}_right"))
        | ((pl.col(f"{prefix}_bin") == last_bin) & (pl.col(value) <= pl.col(f"{prefix}_right")))
    )


def _frame_grid(dataset: AnalysisDataset, bins: pl.LazyFrame) -> pl.LazyFrame:
    return (
        dataset.scan_table("frames.parquet").select("frame_index", "time").join(bins, how="cross")
    )


def cells_with_radial_position(source: DatasetSource) -> pl.LazyFrame:
    """Add radial XY position without discarding any typed cell column."""

    cells = _dataset(source).scan_table("cells.parquet")
    return cells.with_columns(
        (
            pl.col("position_x").cast(pl.Float64).pow(2)
            + pl.col("position_y").cast(pl.Float64).pow(2)
        )
        .sqrt()
        .alias("radial_xy")
    )


def radial_counts(source: DatasetSource, edges: Sequence[float]) -> pl.LazyFrame:
    """Count cells in explicit radial XY bins for each frame.

    Bins are left-closed and right-open, except the final bin includes its
    right edge. Empty bins are retained with a zero count.
    """

    dataset = _dataset(source)
    bins, last_bin = _bins(edges, "radial")
    assigned = (
        cells_with_radial_position(dataset)
        .select("frame_index", "radial_xy")
        .join(bins, how="cross")
        .filter(_inside("radial_xy", "radial", last_bin))
    )
    counts = assigned.group_by("frame_index", "radial_bin").agg(
        pl.len().cast(pl.UInt64).alias("cell_count")
    )
    return (
        _frame_grid(dataset, bins)
        .join(counts, on=["frame_index", "radial_bin"], how="left")
        .with_columns(pl.col("cell_count").fill_null(0).cast(pl.UInt64))
        .sort("frame_index", "radial_bin")
    )


def radial_species_mean(
    source: DatasetSource,
    channel: int,
    edges: Sequence[float],
) -> pl.LazyFrame:
    """Compute a per-frame species mean in explicit radial XY bins.

    Empty bins retain a zero cell count and a null mean. An unavailable channel
    therefore produces null means rather than silently substituting zero.
    """

    if isinstance(channel, bool) or channel < 0 or channel > _UINT32_MAX:
        raise AnalysisError("species channel must be a uint32 value")
    dataset = _dataset(source)
    bins, last_bin = _bins(edges, "radial")
    positions = cells_with_radial_position(dataset).select(
        "frame_index", pl.col("id").alias("cell_id"), "radial_xy"
    )
    values = (
        dataset.scan_table("species.parquet")
        .filter(pl.col("channel") == channel)
        .join(positions, on=["frame_index", "cell_id"], how="inner")
        .join(bins, how="cross")
        .filter(_inside("radial_xy", "radial", last_bin))
    )
    means = values.group_by("frame_index", "radial_bin").agg(
        pl.len().cast(pl.UInt64).alias("cell_count"),
        pl.col("level").cast(pl.Float64).mean().alias("species_mean"),
    )
    return (
        _frame_grid(dataset, bins)
        .join(means, on=["frame_index", "radial_bin"], how="left")
        .with_columns(pl.col("cell_count").fill_null(0).cast(pl.UInt64))
        .sort("frame_index", "radial_bin")
    )


def length_histogram(
    source: DatasetSource,
    edges: Sequence[float],
    *,
    length: LengthField = "capsule_length",
) -> pl.LazyFrame:
    """Count cells by cylinder or full-capsule length for each frame."""

    if length not in {"cylinder_length", "capsule_length"}:
        raise AnalysisError(f"unknown length field {length!r}")
    dataset = _dataset(source)
    bins, last_bin = _bins(edges, "length")
    assigned = (
        dataset.scan_table("cells.parquet")
        .select("frame_index", pl.col(length).cast(pl.Float64).alias("length_value"))
        .join(bins, how="cross")
        .filter(_inside("length_value", "length", last_bin))
    )
    counts = assigned.group_by("frame_index", "length_bin").agg(
        pl.len().cast(pl.UInt64).alias("cell_count")
    )
    return (
        _frame_grid(dataset, bins)
        .join(counts, on=["frame_index", "length_bin"], how="left")
        .with_columns(
            pl.lit(length).alias("length_field"),
            pl.col("cell_count").fill_null(0).cast(pl.UInt64),
        )
        .sort("frame_index", "length_bin")
    )


def line_density_xy(
    source: DatasetSource,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
) -> pl.LazyFrame:
    """Sum full capsule length in explicit XY bins for each frame.

    The result is the legacy notebook's length-weighted histogram proxy. It is
    not area, volume, biomass, packing fraction, or normalized spatial density.
    """

    dataset = _dataset(source)
    x_bins, last_x = _bins(x_edges, "x")
    y_bins, last_y = _bins(y_edges, "y")
    bin_grid = x_bins.join(y_bins, how="cross")
    assigned = (
        dataset.scan_table("cells.parquet")
        .select(
            "frame_index",
            pl.col("position_x").cast(pl.Float64),
            pl.col("position_y").cast(pl.Float64),
            pl.col("capsule_length").cast(pl.Float64),
        )
        .join(bin_grid, how="cross")
        .filter(_inside("position_x", "x", last_x) & _inside("position_y", "y", last_y))
    )
    density = assigned.group_by("frame_index", "x_bin", "y_bin").agg(
        pl.len().cast(pl.UInt64).alias("cell_count"),
        pl.col("capsule_length").sum().alias("line_density_proxy"),
    )
    return (
        _frame_grid(dataset, bin_grid)
        .join(density, on=["frame_index", "x_bin", "y_bin"], how="left")
        .with_columns(
            pl.col("cell_count").fill_null(0).cast(pl.UInt64),
            pl.col("line_density_proxy").fill_null(0.0).cast(pl.Float64),
        )
        .sort("frame_index", "x_bin", "y_bin")
    )


def unique_neighbor_edges(source: DatasetSource) -> pl.LazyFrame:
    """Collapse parallel geometric contact rows into stable undirected edges."""

    dataset = _dataset(source)
    contacts = dataset.scan_table("contacts.parquet")
    normalized = contacts.select(
        "frame_index",
        pl.min_horizontal("first_id", "second_id").alias("first_id"),
        pl.max_horizontal("first_id", "second_id").alias("second_id"),
        "signed_separation",
        "overlap",
        "weight",
    )
    return (
        normalized.group_by("frame_index", "first_id", "second_id")
        .agg(
            pl.len().cast(pl.UInt32).alias("contact_row_count"),
            pl.col("signed_separation").min().alias("minimum_signed_separation"),
            pl.col("overlap").max().alias("maximum_overlap"),
            pl.col("weight").sum().alias("total_weight"),
        )
        .sort("frame_index", "first_id", "second_id")
    )


def sister_neighbor_counts(source: DatasetSource) -> pl.LazyFrame:
    """Count unique active neighbors sharing a non-null lineage parent."""

    dataset = _dataset(source)
    cells = dataset.scan_table("cells.parquet").select(
        "frame_index", pl.col("id").alias("cell_id"), "parent_id"
    )
    first_lineage = cells.rename({"cell_id": "first_id", "parent_id": "first_parent_id"})
    second_lineage = cells.rename({"cell_id": "second_id", "parent_id": "second_parent_id"})
    sisters = (
        unique_neighbor_edges(dataset)
        .join(first_lineage, on=["frame_index", "first_id"], how="inner")
        .join(second_lineage, on=["frame_index", "second_id"], how="inner")
        .filter(
            pl.col("first_parent_id").is_not_null()
            & (pl.col("first_parent_id") == pl.col("second_parent_id"))
        )
    )
    directed = pl.concat(
        [
            sisters.select("frame_index", pl.col("first_id").alias("cell_id")),
            sisters.select("frame_index", pl.col("second_id").alias("cell_id")),
        ]
    )
    counts = directed.group_by("frame_index", "cell_id").agg(
        pl.len().cast(pl.UInt32).alias("sister_neighbor_count")
    )
    frames = dataset.scan_table("frames.parquet").select("frame_index", "time")
    return (
        cells.join(frames, on="frame_index", how="left")
        .join(counts, on=["frame_index", "cell_id"], how="left")
        .with_columns(pl.col("sister_neighbor_count").fill_null(0).cast(pl.UInt32))
        .select(
            "frame_index",
            "time",
            "cell_id",
            "parent_id",
            "sister_neighbor_count",
        )
        .sort("frame_index", "cell_id")
    )


def _signal_epoch(
    source: DatasetSource, epoch_index: int
) -> tuple[AnalysisDataset, dict[str, object], Any]:
    if isinstance(epoch_index, bool) or epoch_index < 0:
        raise AnalysisError("signal epoch index must be non-negative")
    dataset = _dataset(source)
    signal_value = dataset.manifest["signals"]
    if not isinstance(signal_value, dict):
        raise AnalysisError("analysis dataset does not contain signals")
    signal_record = cast(dict[str, object], signal_value)
    epochs = signal_record.get("epochs")
    if not isinstance(epochs, list) or epoch_index >= len(epochs):
        raise AnalysisError(f"signal epoch {epoch_index} is unavailable")
    epoch_value = epochs[epoch_index]
    if not isinstance(epoch_value, dict):
        raise AnalysisError(f"signal epoch {epoch_index} metadata is invalid")
    epoch = cast(dict[str, object], epoch_value)
    expected_name = f"epoch-{epoch_index:04d}"
    if epoch.get("name") != expected_name:
        raise AnalysisError(f"signal epoch {epoch_index} name is invalid")
    root = zarr.open_group(dataset.root / "signals.zarr", mode="r")
    return dataset, epoch, root[expected_name]


def _signal_shape(epoch: dict[str, object]) -> tuple[int, int, int, int]:
    signal_count = epoch.get("signal_count")
    shape = epoch.get("shape")
    if (
        isinstance(signal_count, bool)
        or not isinstance(signal_count, int)
        or signal_count <= 0
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        )
    ):
        raise AnalysisError("signal epoch shape metadata is invalid")
    dimensions = cast(list[int], shape)
    return signal_count, dimensions[0], dimensions[1], dimensions[2]


def signal_slice(
    source: DatasetSource,
    *,
    epoch: int,
    local_frame: int,
    channel: int,
    axis: SignalAxis,
    index: int,
) -> SignalSlice:
    """Read one named 2D plane from a signal epoch.

    ``local_frame`` indexes frames inside the epoch; the returned record also
    reports the dataset-wide frame index and physical time.
    """

    if axis not in {"x", "y", "z"}:
        raise AnalysisError(f"unknown signal slice axis {axis!r}")
    _, metadata, group = _signal_epoch(source, epoch)
    signal_count, x_size, y_size, z_size = _signal_shape(metadata)
    frame_indices = metadata.get("frame_indices")
    if not isinstance(frame_indices, list):
        raise AnalysisError("signal epoch frame metadata is invalid")
    if isinstance(local_frame, bool) or local_frame < 0 or local_frame >= len(frame_indices):
        raise AnalysisError("signal local frame index is out of range")
    if isinstance(channel, bool) or channel < 0 or channel >= signal_count:
        raise AnalysisError("signal channel is out of range")
    axis_sizes = {"x": x_size, "y": y_size, "z": z_size}
    if isinstance(index, bool) or index < 0 or index >= axis_sizes[axis]:
        raise AnalysisError(f"signal {axis} index is out of range")
    levels = group["levels"]
    if axis == "x":
        values = levels[local_frame, channel, index, :, :]
        dimensions: tuple[SignalAxis, SignalAxis] = ("y", "z")
    elif axis == "y":
        values = levels[local_frame, channel, :, index, :]
        dimensions = ("x", "z")
    else:
        values = levels[local_frame, channel, :, :, index]
        dimensions = ("x", "y")
    return SignalSlice(
        frame_index=cast(int, frame_indices[local_frame]),
        time=float(group["time"][local_frame]),
        channel=channel,
        fixed_axis=axis,
        fixed_index=index,
        dimensions=dimensions,
        values=np.asarray(values, dtype=np.float32),
    )


def signal_time_course(
    source: DatasetSource,
    *,
    epoch: int,
    channel: int,
    x: int,
    y: int,
    z: int,
) -> pl.DataFrame:
    """Read one voxel's values across the frames of a signal epoch."""

    _, metadata, group = _signal_epoch(source, epoch)
    signal_count, x_size, y_size, z_size = _signal_shape(metadata)
    coordinates = {"x": x, "y": y, "z": z}
    limits = {"x": x_size, "y": y_size, "z": z_size}
    if isinstance(channel, bool) or channel < 0 or channel >= signal_count:
        raise AnalysisError("signal channel is out of range")
    for name, value in coordinates.items():
        if isinstance(value, bool) or value < 0 or value >= limits[name]:
            raise AnalysisError(f"signal {name} index is out of range")
    frame_indices = np.asarray(group["frame_index"][:], dtype=np.uint32)
    times = np.asarray(group["time"][:], dtype=np.float64)
    frame_count = frame_indices.shape[0]
    return pl.DataFrame(
        {
            "frame_index": pl.Series(frame_indices, dtype=pl.UInt32),
            "time": pl.Series(times, dtype=pl.Float64),
            "channel": pl.Series([channel] * frame_count, dtype=pl.UInt32),
            "x": pl.Series([x] * frame_count, dtype=pl.UInt32),
            "y": pl.Series([y] * frame_count, dtype=pl.UInt32),
            "z": pl.Series([z] * frame_count, dtype=pl.UInt32),
            "level": pl.Series(group["levels"][:, channel, x, y, z], dtype=pl.Float32),
        }
    )
