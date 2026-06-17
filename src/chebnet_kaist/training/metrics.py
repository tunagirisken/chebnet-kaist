"""Classification metrics for graph-level intrusion detection."""

from typing import Any

import numpy as np
from chebnet_kaist.models.base import NUM_CLASSES
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[list[float]] | np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    """Compute macro-averaged classification metrics.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        y_prob: Predicted class probabilities ``(N, num_classes)``.
        num_classes: Total number of classes.

    Returns:
        Dict with keys ``f1``, ``precision``, ``recall``, and ``auc``.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)

    f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    precision = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)

    try:
        if num_classes == 2:
            y_bin = np.column_stack(
                [
                    (y_true_arr == 0).astype(int),
                    (y_true_arr == 1).astype(int),
                ]
            )
        else:
            y_bin = label_binarize(y_true_arr, classes=list(range(num_classes)))
        auc = roc_auc_score(y_bin, y_prob_arr, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {"f1": f1, "precision": precision, "recall": recall, "auc": auc}


def metrics_to_log_line(epoch: int, train_loss: float, val_metrics: dict[str, Any]) -> str:
    """Format one epoch's metrics as a human-readable log line.

    Args:
        epoch: Current epoch number (1-based).
        train_loss: Mean training loss for the epoch.
        val_metrics: Validation metrics dict including ``loss`` key.

    Returns:
        Formatted log string.
    """
    return (
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"F1={val_metrics['f1']:.4f} | "
        f"Prec={val_metrics['precision']:.4f} | "
        f"Rec={val_metrics['recall']:.4f} | "
        f"AUC={val_metrics['auc']:.4f}"
    )
