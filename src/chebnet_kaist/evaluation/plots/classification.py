"""Classification performance figures (Part A)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from chebnet_kaist.constants import CLASS_NAMES
from chebnet_kaist.evaluation.plots.style import (
    CLASS_COLORS,
    METRIC_COLORS,
    MIN_FIGSIZE,
    model_title,
    save_figure,
    set_publication_style,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def _plot_classes(
    class_names: list[str] | None = None,
    num_classes: int | None = None,
) -> tuple[list[str], int]:
    """Resolve class labels and count for plotting helpers."""
    names = class_names or CLASS_NAMES
    count = num_classes or len(names)
    return names, count


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_stem: Path,
    model_name: str,
    class_names: list[str] | None = None,
    num_classes: int | None = None,
) -> tuple[Path, Path]:
    """Save a high-resolution confusion matrix with counts and percentages."""
    set_publication_style()
    names, n_cls = _plot_classes(class_names, num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = plt.cm.Blues
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=1)

    for row in range(n_cls):
        for col in range(n_cls):
            is_diag = row == col
            bg = cm_pct[row, col]
            text_color = "white" if bg > 0.55 else "black"
            edgecolor = "#059669" if is_diag else "#dc2626" if cm[row, col] > 0 else "none"
            linewidth = 2.0 if (not is_diag and cm[row, col] > 0) else 0.0
            ax.add_patch(
                plt.Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )
            )
            ax.text(
                col,
                row,
                f"{cm[row, col]:,}\n({cm_pct[row, col]:.1%})",
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(
        model_title(model_name, f"Confusion Matrix  |  Macro F1 = {macro_f1:.4f}"),
        pad=16,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row %")
    fig.text(
        0.5,
        0.02,
        "Key takeaway: off-diagonal cells highlight misclassifications; green borders = correct.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def plot_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_stem: Path,
    model_name: str,
    class_names: list[str] | None = None,
    num_classes: int | None = None,
) -> tuple[Path, Path]:
    """Grouped bar chart sorted by F1 descending."""
    set_publication_style()
    names, n_cls = _plot_classes(class_names, num_classes)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(n_cls)),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    f1_scores = [(name, report[name]["f1-score"]) for name in names]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_names = [n for n, _ in f1_scores]

    x = np.arange(len(sorted_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))

    for offset, metric in zip([-width, 0, width], ["precision", "recall", "f1-score"]):
        values = [report[name][metric] for name in sorted_names]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric.replace("-", " ").title(),
            color=METRIC_COLORS[metric],
            alpha=0.9,
            edgecolor="white",
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(0.90, color="#dc2626", linestyle="--", linewidth=1.2, label="Target (0.90)")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(model_title(model_name, "Per-Class Metrics (sorted by F1)"), pad=14)
    ax.legend(loc="lower right")
    fig.text(
        0.5,
        0.01,
        "Key takeaway: classes left of the chart are hardest to detect.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_stem: Path,
    model_name: str,
    class_names: list[str] | None = None,
    num_classes: int | None = None,
) -> tuple[Path, Path]:
    """One-vs-rest ROC curves with highlighted macro-average curve."""
    set_publication_style()
    names, n_cls = _plot_classes(class_names, num_classes)
    colors = CLASS_COLORS[:n_cls]
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=MIN_FIGSIZE)
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5, label="Random baseline")

    # Macro-average ROC
    all_fpr = np.unique(
        np.concatenate([roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(n_cls)])
    )
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_cls):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_cls
    macro_auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    ax.plot(
        all_fpr,
        mean_tpr,
        color="#111827",
        lw=3.0,
        label=f"Macro average  AUC={macro_auc:.4f}",
        zorder=5,
    )

    for i, (name, color) in enumerate(zip(names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_val = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name}  AUC={auc_val:.3f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(model_title(model_name, f"ROC Curves  |  Macro AUC = {macro_auc:.4f}"), pad=14)
    ax.legend(fontsize=9, loc="lower right")
    fig.text(
        0.5,
        0.01,
        "Key takeaway: curves above the diagonal indicate discriminative power per class.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_stem: Path,
    model_name: str,
    class_names: list[str] | None = None,
    num_classes: int | None = None,
) -> tuple[Path, Path]:
    """One-vs-rest PR curves with macro-average highlighted."""
    set_publication_style()
    names, n_cls = _plot_classes(class_names, num_classes)
    colors = CLASS_COLORS[:n_cls]
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))

    fig, ax = plt.subplots(figsize=MIN_FIGSIZE)

    # Macro-average PR
    precisions, recalls = [], []
    for i in range(n_cls):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        precisions.append(p)
        recalls.append(r)
    all_recall = np.unique(np.concatenate(recalls))
    mean_precision = np.zeros_like(all_recall)
    for p, r in zip(precisions, recalls):
        mean_precision += np.interp(all_recall, r[::-1], p[::-1])
    mean_precision /= n_cls
    macro_ap = average_precision_score(y_bin, y_prob, average="macro")
    ax.plot(
        all_recall,
        mean_precision,
        color="#111827",
        lw=3.0,
        label=f"Macro average  AP={macro_ap:.4f}",
        zorder=5,
    )

    for i, (name, color) in enumerate(zip(names, colors)):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(r, p, color=color, lw=1.8, label=f"{name}  AP={ap:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        model_title(model_name, f"Precision-Recall Curves  |  Macro AP = {macro_ap:.4f}"), pad=14
    )
    ax.legend(fontsize=9, loc="lower left")
    fig.text(
        0.5,
        0.01,
        "Key takeaway: PR curves matter most for imbalanced classes such as Normal (~13%).",
        ha="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )
    return save_figure(fig, output_stem, tight=False)
