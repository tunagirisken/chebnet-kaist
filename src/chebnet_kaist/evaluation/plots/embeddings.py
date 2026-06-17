"""ChebNet embedding analysis figures (Part B2)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from chebnet_kaist.constants import CLASS_NAMES
from chebnet_kaist.evaluation.plots.style import (
    CLASS_COLORS,
    model_title,
    save_figure,
    set_publication_style,
)
from chebnet_kaist.models.base import NUM_CLASSES
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

try:
    import umap

    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


@torch.no_grad()
def _collect_embeddings(
    model: nn.Module,
    graphs: list[Data],
    device: torch.device,
    batch_size: int = 64,
    max_graphs: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect node embeddings, graph labels, node degrees, and norms."""
    model.eval()
    subset = graphs[:max_graphs] if len(graphs) > max_graphs else graphs
    all_emb, all_labels, all_degrees, all_norms = [], [], [], []

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    graph_idx = 0
    for batch in loader:
        batch = batch.to(device)
        _, (node_emb, _) = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            return_viz=True,
        )
        emb_np = node_emb.cpu().numpy()
        batch_vec = batch.batch.cpu().numpy()
        deg = degree(batch.edge_index[0], num_nodes=batch.num_nodes).cpu().numpy()

        for i in range(batch.num_graphs):
            mask = batch_vec == i
            g = subset[graph_idx]
            all_emb.append(emb_np[mask])
            all_labels.extend([int(g.y.item())] * int(mask.sum()))
            all_degrees.extend(deg[mask].tolist())
            norms = np.linalg.norm(emb_np[mask], axis=1)
            all_norms.extend(norms.tolist())
            graph_idx += 1

    return (
        np.vstack(all_emb),
        np.array(all_labels),
        np.array(all_degrees),
        np.array(all_norms),
    )


def plot_tsne_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_stem: Path,
    model_name: str,
    seed: int = 42,
) -> tuple[Path, Path]:
    """t-SNE (two seeds) and UMAP side by side."""
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, tsne_seed in zip(axes[:2], [42, 123]):
        perplexity = min(30, max(5, len(embeddings) // 50))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=tsne_seed,
            max_iter=1000,
        ).fit_transform(embeddings)
        for cls in range(NUM_CLASSES):
            mask = labels == cls
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls],
                s=12,
                alpha=0.6,
            )
        ax.set_title(f"t-SNE (seed={tsne_seed})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    ax = axes[2]
    if _HAS_UMAP:
        coords = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15).fit_transform(
            embeddings
        )
        for cls in range(NUM_CLASSES):
            mask = labels == cls
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=CLASS_COLORS[cls],
                label=CLASS_NAMES[cls],
                s=12,
                alpha=0.6,
            )
        ax.set_title("UMAP")
    else:
        ax.text(
            0.5, 0.5, "umap-learn not installed", ha="center", va="center", transform=ax.transAxes
        )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    axes[0].legend(fontsize=8, loc="best", markerscale=2)
    fig.suptitle(
        model_title(model_name, "Node Embedding Projections (final ChebNet layer)"),
        fontsize=14,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Key takeaway: well-separated clusters indicate class-discriminative embeddings.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def plot_class_distance_heatmap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_stem: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """5x5 cosine distance heatmap between class centroids."""
    set_publication_style()
    centroids = []
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        if mask.any():
            centroids.append(embeddings[mask].mean(axis=0))
        else:
            centroids.append(np.zeros(embeddings.shape[1]))

    centroids = np.stack(centroids)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    normed = centroids / norms
    dist = 1.0 - normed @ normed.T

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist, cmap="YlOrRd", vmin=0, vmax=dist.max() + 1e-6)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, label="Cosine Distance")
    ax.set_title(model_title(model_name, "Inter-Class Embedding Distance"), pad=14)
    fig.text(
        0.5,
        0.01,
        "Key takeaway: high off-diagonal values indicate well-separated classes.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def plot_degree_vs_norm(
    degrees: np.ndarray,
    norms: np.ndarray,
    labels: np.ndarray,
    output_stem: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Scatter of node degree vs embedding L2 norm."""
    set_publication_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        ax.scatter(
            degrees[mask],
            norms[mask],
            c=CLASS_COLORS[cls],
            label=CLASS_NAMES[cls],
            s=10,
            alpha=0.4,
        )
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Embedding L2 Norm")
    ax.set_title(model_title(model_name, "Node Degree vs Embedding Norm"), pad=14)
    ax.legend(markerscale=3, fontsize=9)
    fig.text(
        0.5,
        0.01,
        "Key takeaway: high-degree attack nodes may show distinct embedding norms.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def generate_all_embedding_figures(
    model: nn.Module,
    graphs: list[Data],
    device: torch.device,
    output_dir: Path,
    model_name: str,
    batch_size: int = 64,
) -> list[Path]:
    """Generate all ChebNet embedding figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    emb, labels, degrees, norms = _collect_embeddings(model, graphs, device, batch_size)

    p = plot_tsne_umap(emb, labels, output_dir / "tsne_umap_comparison", model_name)
    saved.append(p[0])
    p = plot_class_distance_heatmap(emb, labels, output_dir / "class_distance_heatmap", model_name)
    saved.append(p[0])
    p = plot_degree_vs_norm(
        degrees, norms, labels, output_dir / "degree_vs_embedding_norm", model_name
    )
    saved.append(p[0])
    return saved
