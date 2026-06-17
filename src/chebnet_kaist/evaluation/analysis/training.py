"""Training diagnostic figures (Part D)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from chebnet_kaist.constants import CLASS_NAMES
from chebnet_kaist.evaluation.plots.style import (
    CLASS_COLORS,
    model_title,
    save_figure,
    set_publication_style,
)
from chebnet_kaist.models.base import NUM_CLASSES


def load_training_history(checkpoint_dir: Path, history_stem: str) -> dict | None:
    """Load training history JSON if it exists."""
    path = checkpoint_dir / f"training_history_{history_stem}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_training_curves(
    history: dict,
    output_stem: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Dual-axis training/validation loss and F1 curves (D1)."""
    set_publication_style()
    epochs = [e["epoch"] for e in history["epochs"]]
    train_loss = [e["train_loss"] for e in history["epochs"]]
    val_loss = [e["val_loss"] for e in history["epochs"]]
    val_f1 = [e["val_f1"] for e in history["epochs"]]
    best_epoch = history.get("best_epoch", 0)

    # Rolling average on validation F1
    window = 5
    val_f1_smooth = np.convolve(val_f1, np.ones(window) / window, mode="same")

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(epochs, train_loss, color="#2563eb", lw=1.8, label="Train Loss")
    ax1.plot(epochs, val_loss, color="#dc2626", lw=1.8, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, color="#059669", lw=1.0, alpha=0.4, label="Val F1 (raw)")
    ax2.plot(epochs, val_f1_smooth, color="#059669", lw=2.2, label="Val F1 (MA-5)")
    ax2.set_ylabel("F1 Score")
    ax2.set_ylim(0, 1.05)

    if best_epoch:
        ax1.axvline(
            best_epoch,
            color="#111827",
            linestyle="--",
            lw=1.5,
            label=f"Best epoch ({best_epoch})",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.set_title(model_title(model_name, "Training Curves"), pad=14)
    fig.text(
        0.5,
        0.01,
        "Key takeaway: dashed line marks the checkpoint saved by best validation F1.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def plot_class_imbalance(
    history: dict,
    output_stem: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Horizontal bar chart of class counts and sampler weights (D2)."""
    set_publication_style()
    counts = history.get("train_class_counts", {})
    weights = history.get("sampler_weights", {})
    total = sum(counts.values()) or 1

    fig, ax1 = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(NUM_CLASSES)
    vals = [counts.get(str(i), counts.get(i, 0)) for i in range(NUM_CLASSES)]
    bars = ax1.barh(y_pos, vals, color=CLASS_COLORS, alpha=0.85, height=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(CLASS_NAMES)
    ax1.set_xlabel("Training Sample Count")
    ax1.invert_yaxis()

    for bar, val in zip(bars, vals):
        pct = val / total * 100
        ax1.text(
            bar.get_width() + max(vals) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({pct:.1f}%)",
            va="center",
            fontsize=9,
        )

    ax2 = ax1.twiny()
    w_vals = [weights.get(str(i), weights.get(i, 0)) for i in range(NUM_CLASSES)]
    ax2.plot(w_vals, y_pos, "ko-", markersize=6, label="Sampler Weight")
    ax2.set_xlabel("WeightedRandomSampler Weight")
    ax2.legend(loc="lower right", fontsize=9)

    ax1.set_title(model_title(model_name, "Training Class Imbalance"), pad=14)
    fig.text(
        0.5,
        0.01,
        "Key takeaway: minority classes receive higher sampling weight during training.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def generate_training_figures(
    checkpoint_dir: Path,
    output_dir: Path,
    history_stem: str,
    model_name: str | None = None,
) -> list[Path]:
    """Generate training diagnostic figures if history exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_training_history(checkpoint_dir, history_stem)
    plot_label = model_name or history_stem
    if history is None:
        return []

    saved: list[Path] = []
    p = plot_training_curves(history, output_dir / "training_curves", plot_label)
    saved.append(p[0])
    p = plot_class_imbalance(history, output_dir / "class_imbalance", plot_label)
    saved.append(p[0])

    history_out = output_dir / "history.json"
    with history_out.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return saved
