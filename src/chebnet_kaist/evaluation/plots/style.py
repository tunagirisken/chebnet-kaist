"""Shared matplotlib style and export helpers."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Single palette reused wherever class identity is encoded by color.
CLASS_COLORS = ["#2563eb", "#dc2626", "#d97706", "#7c3aed", "#059669"]

METRIC_COLORS = {
    "precision": "#2563eb",
    "recall": "#059669",
    "f1-score": "#d97706",
}

DATASET_NAME = "KAIST Car-Hacking"
FIG_DPI = 300
MIN_FIGSIZE = (10, 7)


def set_publication_style() -> None:
    """Apply consistent academic figure styling."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: Path, tight: bool = True) -> tuple[Path, Path]:
    """Save a figure as PNG (300 DPI) and SVG.

    Args:
        fig: Matplotlib figure to export.
        stem: Output path without extension.
        tight: Whether to call ``tight_layout`` before saving.

    Returns:
        Tuple of ``(png_path, svg_path)``.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()

    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight", format="svg")
    plt.close(fig)
    return png_path, svg_path


def model_title(model_name: str, subtitle: str) -> str:
    """Build a standard figure title with model and dataset context."""
    from chebnet_kaist.constants import MODEL_LABELS

    label = MODEL_LABELS.get(model_name, model_name.upper())
    return f"{label} | {DATASET_NAME}\n{subtitle}"
