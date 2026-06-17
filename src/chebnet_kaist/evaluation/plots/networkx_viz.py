"""NetworkX visualization for CAN segment graphs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
from chebnet_kaist.constants import CLASS_NAMES
from chebnet_kaist.data.graphs.builder import segment_to_graph
from chebnet_kaist.evaluation.plots.style import FIG_DPI, save_figure, set_publication_style
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx


def _scalar_weight(weight: object) -> float:
    """Coerce PyG edge attributes to a scalar float for NetworkX layouts."""
    if isinstance(weight, torch.Tensor):
        return float(weight.reshape(-1)[0].item())
    if isinstance(weight, (list, tuple)):
        return float(weight[0])
    return float(weight)


def can_id_labels_from_segment(segment_df: pd.DataFrame) -> list[str]:
    """Build hex CAN ID labels aligned with ``segment_to_graph`` node ordering.

    Args:
        segment_df: CAN frames with a ``can_id`` column.

    Returns:
        One label string per unique CAN ID (sorted ascending).
    """
    unique_ids = sorted(segment_df["can_id"].unique())
    return [f"0x{int(can_id):03X}" for can_id in unique_ids]


def data_to_networkx(
    graph: Data,
    node_labels: list[str] | None = None,
) -> nx.DiGraph:
    """Convert a PyG ``Data`` object to a NetworkX directed graph.

    Args:
        graph: PyTorch Geometric graph with ``edge_index`` and optional ``edge_attr``.
        node_labels: Optional human-readable node names (length must equal ``num_nodes``).

    Returns:
        Directed NetworkX graph with ``weight`` on edges and ``features`` on nodes.

    Raises:
        ValueError: If ``node_labels`` length does not match the node count.
    """
    nx_graph = to_networkx(
        graph,
        edge_attrs=["edge_attr"],
        node_attrs=["x"],
        to_undirected=False,
    )
    if not isinstance(nx_graph, nx.DiGraph):
        nx_graph = nx.DiGraph(nx_graph)

    for node in nx_graph.nodes:
        feat = nx_graph.nodes[node].get("x")
        if feat is not None:
            if isinstance(feat, torch.Tensor):
                feat = feat.detach().cpu().tolist()
            nx_graph.nodes[node]["features"] = feat

    for _u, _v, data in nx_graph.edges(data=True):
        weight = data.pop("edge_attr", 1.0)
        data["weight"] = _scalar_weight(weight)

    if node_labels is not None:
        if len(node_labels) != graph.num_nodes:
            raise ValueError(f"Expected {graph.num_nodes} node labels, got {len(node_labels)}")
        mapping = {index: label for index, label in enumerate(node_labels)}
        nx_graph = nx.relabel_nodes(nx_graph, mapping)

    return nx_graph


def draw_networkx_graph(
    nx_graph: nx.DiGraph,
    *,
    title: str,
    output_stem: Path,
    class_color: str | None = None,
) -> tuple[Path, Path]:
    """Render a NetworkX graph and save PNG + SVG.

    Args:
        nx_graph: Graph to draw (node ``features[0]`` scales node size when present).
        title: Figure title.
        output_stem: Output path without extension.
        class_color: Optional node fill color (hex).

    Returns:
        Tuple of ``(png_path, svg_path)``.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    if nx_graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty graph", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        return save_figure(fig, output_stem)

    layout_seed = 42
    if nx_graph.number_of_nodes() == 1:
        sole = next(iter(nx_graph.nodes))
        pos = {sole: (0.0, 0.0)}
    else:
        pos = nx.spring_layout(nx_graph, seed=layout_seed, weight="weight")

    node_sizes: list[float] = []
    for node in nx_graph.nodes:
        feat = nx_graph.nodes[node].get("features")
        if feat:
            freq = float(feat[0])
            node_sizes.append(400 + 200 * max(freq, 0.1))
        else:
            node_sizes.append(600)

    edge_widths = [
        0.8 + 0.4 * float(data.get("weight", 1.0)) for _u, _v, data in nx_graph.edges(data=True)
    ]
    fill = class_color or "#2563eb"

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        node_size=node_sizes,
        node_color=fill,
        alpha=0.85,
        ax=ax,
    )
    nx.draw_networkx_labels(nx_graph, pos, font_size=9, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(
        nx_graph,
        pos,
        width=edge_widths,
        edge_color="#64748b",
        alpha=0.7,
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    svg_path = output_stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    plt.close(fig)
    return png_path, svg_path


def visualize_segment_graph(
    segment_df: pd.DataFrame,
    label: int,
    output_stem: Path,
) -> tuple[Path, Path]:
    """Build a CAN segment graph and export a NetworkX figure.

    Args:
        segment_df: CAN frame segment.
        label: Graph class index (0-4).
        output_stem: Output path without extension.

    Returns:
        Tuple of ``(png_path, svg_path)``.
    """
    graph = segment_to_graph(segment_df, label)
    labels = can_id_labels_from_segment(segment_df)
    nx_graph = data_to_networkx(graph, node_labels=labels)
    class_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
    title = (
        f"CAN Segment Graph (NetworkX)\n"
        f"Class: {class_name} | Nodes: {graph.num_nodes} | Edges: {graph.num_edges}"
    )
    colors = ["#2563eb", "#dc2626", "#d97706", "#7c3aed", "#059669"]
    class_color = colors[label] if 0 <= label < len(colors) else None
    return draw_networkx_graph(
        nx_graph,
        title=title,
        output_stem=output_stem,
        class_color=class_color,
    )


def demo_segment_df() -> pd.DataFrame:
    """Return a small synthetic CAN segment for demos and tests."""
    return pd.DataFrame(
        {
            "timestamp": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
            "can_id": [0x000, 0x18F, 0x000, 0x316, 0x18F, 0x000],
            "dlc": [8, 4, 8, 6, 4, 8],
            "flag": [1, 0, 1, 0, 0, 1],
            "label": [1, 1, 1, 1, 1, 1],
        }
    )
