"""Convert CAN frame segments into PyTorch Geometric graph objects."""

from collections.abc import Iterable

import pandas as pd
import torch
from torch_geometric.data import Data

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _id_priority(can_id: int) -> float:
    """Map a CAN ID to a normalized priority bucket in [0, 1].

    Low IDs carry high-priority safety traffic (DoS attacks often use 0x000).

    Args:
        can_id: Raw CAN identifier.

    Returns:
        Priority score: 0.0 (high), 0.5 (normal), or 1.0 (low).
    """
    if can_id <= 0x100:
        return 0.0
    if can_id <= 0x600:
        return 0.5
    return 1.0


def segment_to_graph(segment_df: pd.DataFrame, label: int) -> Data:
    """Build a PyG ``Data`` graph from one sliding segment of CAN frames.

    Graph definition:
        - Nodes: unique CAN IDs (ECUs) in the segment.
        - Edges: consecutive frame transitions between distinct IDs.
        - Node features (3-d): log frequency, mean DLC / 8, ID priority.
        - Edge attr: transition count (edge weight).

    Args:
        segment_df: CAN frames with ``can_id``, ``dlc``, and ``timestamp``.
        label: Graph-level class label (0-4).

    Returns:
        PyTorch Geometric ``Data`` object.
    """
    unique_ids = sorted(segment_df["can_id"].unique())
    id_to_index = {can_id: index for index, can_id in enumerate(unique_ids)}

    duration = segment_df["timestamp"].iloc[-1] - segment_df["timestamp"].iloc[0] + 1e-9
    frequency = segment_df.groupby("can_id").size() / duration
    mean_dlc = segment_df.groupby("can_id")["dlc"].mean()

    node_features: list[list[float]] = []
    for can_id in unique_ids:
        freq = float(frequency.get(can_id, 0.0))
        dlc = float(mean_dlc.get(can_id, 0.0)) / 8.0
        priority = _id_priority(can_id)
        node_features.append([freq, dlc, priority])

    x = torch.tensor(node_features, dtype=torch.float)
    x[:, 0] = torch.log1p(x[:, 0])

    edge_counts: dict[tuple[int, int], int] = {}
    id_sequence = segment_df["can_id"].tolist()
    for index in range(len(id_sequence) - 1):
        source = id_to_index[id_sequence[index]]
        target = id_to_index[id_sequence[index + 1]]
        if source != target:
            key = (source, target)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    if edge_counts:
        sources, targets, weights = [], [], []
        for (src, dst), count in edge_counts.items():
            sources.append(src)
            targets.append(dst)
            weights.append(float(count))
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    else:
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr = torch.ones((1, 1), dtype=torch.float)

    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def build_graph_dataset(
    segments: Iterable[tuple[pd.DataFrame, int]],
    show_progress: bool = True,
) -> list[Data]:
    """Convert a segment iterator into a list of PyG graphs.

    Args:
        segments: Iterable of ``(segment_df, label)`` pairs.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        List of graph objects ready for DataLoader batching.
    """
    if show_progress and _HAS_TQDM:
        segments = tqdm(segments, desc="Building graphs", unit="graph")

    return [segment_to_graph(df, label) for df, label in segments]
