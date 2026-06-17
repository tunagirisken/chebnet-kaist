"""Sliding-segment extraction from CAN frame streams."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from chebnet_kaist.data.ingestion.kaist_loader import load_kaist

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

DEFAULT_SEGMENT_SIZE = 0.1  # seconds (100 ms)

CLASS_ORDER = ["normal", "dos", "fuzzy", "rpm", "gear"]


def iter_segments(
    df: pd.DataFrame,
    segment_size: float = DEFAULT_SEGMENT_SIZE,
    show_progress: bool = False,
) -> Iterator[tuple[pd.DataFrame, int]]:
    """Yield non-overlapping sliding segments from a CAN frame DataFrame.

    Segment label is the mode of attack frames inside the segment, or 0 (normal)
    when no attack frame is present. Uses ``searchsorted`` for O(log n) slicing.

    Args:
        df: CAN frames with ``timestamp``, ``flag``, and ``label`` columns.
        segment_size: Segment duration in seconds.
        show_progress: Whether to display a tqdm progress bar.

    Yields:
        Tuples of ``(segment_df, label)`` where each row in ``segment_df`` is a frame.
    """
    timestamps = df["timestamp"].values
    t_start = timestamps[0]
    t_end = timestamps[-1]
    total = int((t_end - t_start) / segment_size)

    iterator = range(total)
    if show_progress and _HAS_TQDM:
        iterator = tqdm(iterator, desc="Segments", unit="seg", leave=False)

    for index in iterator:
        t_left = t_start + index * segment_size
        t_right = t_left + segment_size

        idx_left = np.searchsorted(timestamps, t_left, side="left")
        idx_right = np.searchsorted(timestamps, t_right, side="left")

        if idx_left >= idx_right:
            continue

        segment_df = df.iloc[idx_left:idx_right]

        if segment_df["flag"].any():
            label = int(segment_df.loc[segment_df["flag"] == 1, "label"].mode().iloc[0])
        else:
            label = 0

        yield segment_df, label


def iter_all_segments(
    data_dir: str | Path,
    segment_size: float = DEFAULT_SEGMENT_SIZE,
) -> Iterator[tuple[pd.DataFrame, int]]:
    """Yield segments for all five KAIST classes, processed file-by-file.

    Each recording session is segmented independently because timestamps are
    not contiguous across files.

    Args:
        data_dir: Root directory of the KAIST dataset.
        segment_size: Segment duration in seconds.

    Yields:
        Tuples of ``(segment_df, label)``.
    """
    for attack_type in CLASS_ORDER:
        df = load_kaist(data_dir, attack_type)
        n_segments = int((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / segment_size)
        print(f"  {attack_type:8s}: {len(df):>9,} CAN frames -> ~{n_segments:>6,} segments")
        yield from iter_segments(df, segment_size, show_progress=True)
