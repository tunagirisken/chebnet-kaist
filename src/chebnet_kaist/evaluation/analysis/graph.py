"""Graph-level structural analysis (Part C).

Spectral time-series helpers are KAIST-only: they call ``load_kaist`` and
iterate attack types ``dos``, ``fuzzy``, ``rpm``, ``gear``. Do not invoke them
when ``config.paths.data_dir`` points at another dataset (e.g. ROAD).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from chebnet_kaist.constants import CLASS_NAMES, CLASS_NAMES_SHORT
from chebnet_kaist.data.graphs.builder import segment_to_graph
from chebnet_kaist.data.ingestion.kaist_loader import load_kaist
from chebnet_kaist.data.preprocessing.segments import iter_segments
from chebnet_kaist.evaluation.plots.style import (
    CLASS_COLORS,
    model_title,
    save_figure,
    set_publication_style,
)
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from torch_geometric.data import Data


def _adjacency_matrix(graph: Data) -> np.ndarray:
    """Build a dense symmetric adjacency matrix from edge_index."""
    n = graph.num_nodes
    adj = np.zeros((n, n), dtype=np.float64)
    src = graph.edge_index[0].cpu().numpy()
    dst = graph.edge_index[1].cpu().numpy()
    w = (
        graph.edge_attr.squeeze(-1).cpu().numpy()
        if graph.edge_attr is not None
        else np.ones(len(src))
    )
    for s, d, weight in zip(src, dst, w):
        adj[s, d] += weight
        adj[d, s] += weight
    return adj


def compute_graph_stats(graph: Data) -> dict[str, float]:
    """Compute structural statistics for one graph."""
    n = graph.num_nodes
    e = graph.num_edges
    if n <= 1:
        return {
            "node_count": float(n),
            "edge_count": float(e),
            "avg_degree": 0.0,
            "density": 0.0,
            "fiedler": 0.0,
            "spectral_radius": 0.0,
            "graph_energy": 0.0,
        }

    adj = _adjacency_matrix(graph)
    deg = adj.sum(axis=1)
    avg_degree = float(deg.mean())
    max_edges = n * (n - 1)
    density = float(e / max_edges) if max_edges > 0 else 0.0

    lap = csgraph.laplacian(adj, normed=True)
    try:
        eigvals = eigsh(lap, k=min(2, n - 1), which="SM", return_eigenvectors=False)
        fiedler = float(eigvals[1]) if len(eigvals) > 1 else 0.0
    except Exception:
        fiedler = 0.0

    sym_eigvals = np.linalg.eigvalsh(adj)
    spectral_radius = float(np.max(np.abs(sym_eigvals)))
    graph_energy = float(np.sum(sym_eigvals**2))

    return {
        "node_count": float(n),
        "edge_count": float(e),
        "avg_degree": avg_degree,
        "density": density,
        "fiedler": fiedler,
        "spectral_radius": spectral_radius,
        "graph_energy": graph_energy,
    }


def plot_feature_distributions(
    graphs: list[Data],
    output_stem: Path,
    model_name: str,
    class_names: list[str] | None = None,
) -> tuple[Path, Path]:
    """Violin plots of graph features per class (C1)."""
    set_publication_style()
    features = [
        "node_count",
        "edge_count",
        "avg_degree",
        "density",
        "fiedler",
        "spectral_radius",
        "graph_energy",
    ]
    labels = [
        "Node Count",
        "Edge Count",
        "Avg Degree",
        "Density",
        "Fiedler (lambda_1)",
        "Spectral Radius",
        "Graph Energy E(G)",
    ]

    present = sorted({int(g.y.item()) for g in graphs})
    names = class_names or CLASS_NAMES
    name_for = {i: names[i] if i < len(names) else str(i) for i in present}

    by_class: dict[int, dict[str, list[float]]] = {c: {f: [] for f in features} for c in present}
    for graph in graphs:
        cls = int(graph.y.item())
        stats = compute_graph_stats(graph)
        for f in features:
            by_class[cls][f].append(stats[f])

    fig, axes = plt.subplots(len(features), 1, figsize=(12, 3.2 * len(features)))
    for ax, feat, label in zip(axes, features, labels):
        data = [by_class[c][feat] for c in present if by_class[c][feat]]
        positions = [i for i, c in enumerate(present) if by_class[c][feat]]
        if not data:
            ax.set_visible(False)
            continue
        parts = ax.violinplot(
            data,
            positions=positions,
            showmeans=True,
            showmedians=True,
        )
        shown = [c for c in present if by_class[c][feat]]
        for i, body in enumerate(parts["bodies"]):
            cls = shown[i]
            color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else "#64748b"
            body.set_facecolor(color)
            body.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([name_for[c] for c in shown], rotation=20, ha="right")
        ax.set_ylabel(label)
    axes[0].set_title(
        model_title(model_name, "Graph Feature Distributions by Class"),
        pad=14,
    )
    fig.text(
        0.5,
        0.005,
        "Key takeaway: spectral parameters differ across attack types at the graph level.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def plot_spectral_timeseries(
    data_dir: Path,
    attack_type: str,
    output_stem: Path,
    model_name: str,
    segment_size: float = 0.1,
    n_before: int = 40,
    n_after: int = 40,
) -> tuple[Path, Path] | None:
    """Plot spectral parameters around attack onset (C2)."""
    set_publication_style()
    df = load_kaist(data_dir, attack_type)
    segments = list(iter_segments(df, segment_size, show_progress=False))

    onset_idx = None
    for i, (_, label) in enumerate(segments):
        if label != 0:
            onset_idx = i
            break
    if onset_idx is None:
        return None

    start = max(0, onset_idx - n_before)
    end = min(len(segments), onset_idx + n_after)
    onset_slice = segments[start:end]

    fiedler_vals, radius_vals, energy_vals = [], [], []
    for segment_df, _ in onset_slice:
        graph = segment_to_graph(segment_df, 0)
        stats = compute_graph_stats(graph)
        fiedler_vals.append(stats["fiedler"])
        radius_vals.append(stats["spectral_radius"])
        energy_vals.append(stats["graph_energy"])

    x = np.arange(len(onset_slice))
    attack_x = onset_idx - start

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    series = [
        (fiedler_vals, "Fiedler Value (lambda_1)", "#2563eb"),
        (radius_vals, "Spectral Radius (lambda_max)", "#dc2626"),
        (energy_vals, "Graph Energy E(G)", "#059669"),
    ]
    class_label = (
        CLASS_NAMES[CLASS_NAMES_SHORT.index(attack_type)] if attack_type != "normal" else "Normal"
    )

    for ax, (vals, ylabel, color) in zip(axes, series):
        ax.plot(x, vals, color=color, lw=2.0)
        ax.axvline(attack_x, color="#111827", linestyle="--", lw=1.5, label="Attack onset")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Segment Index (100 ms each)")
    axes[0].set_title(
        model_title(model_name, f"Spectral Parameters at {class_label} Attack Onset"),
        pad=14,
    )
    fig.text(
        0.5,
        0.01,
        "Key takeaway: spectral shifts at attack onset are visible without model inference.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def generate_spectral_timeseries_all(
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    segment_size: float = 0.1,
) -> list[Path]:
    """Generate spectral time series for each attack type where onset is visible."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for attack in ["dos", "fuzzy", "rpm", "gear"]:
        result = plot_spectral_timeseries(
            data_dir,
            attack,
            output_dir / f"spectral_timeseries_{attack}",
            model_name,
            segment_size,
        )
        if result:
            saved.append(result[0])
    return saved
