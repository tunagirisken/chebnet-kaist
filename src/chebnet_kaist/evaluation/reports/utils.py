"""Shared helpers for evaluation report pipelines."""

import logging
import traceback
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from chebnet_kaist.config.schema import ExperimentConfig
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_run(label: str, fn: Callable[[], T], *, log_prefix: str = "Visualization") -> T | None:
    """Run a report step without aborting the pipeline on failure."""
    try:
        return fn()
    except Exception:
        logger.warning("%s failed: %s\n%s", log_prefix, label, traceback.format_exc())
        return None


def split_validation_graphs(
    graphs: list[Data],
    config: ExperimentConfig,
) -> list[Data]:
    """Return the stratified validation subset used by full reports."""
    labels = [int(g.y.item()) for g in graphs]
    _, val_idx = train_test_split(
        range(len(graphs)),
        test_size=config.training.val_split,
        random_state=config.seed,
        stratify=labels,
    )
    return [graphs[i] for i in val_idx]


def compute_binary_attack_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute normal-vs-attack binary metrics from multi-class outputs."""
    if y_prob.shape[1] == 2:
        y_bin_true = y_true.astype(int)
        y_bin_pred = y_pred.astype(int)
        attack_prob = y_prob[:, 1]
    else:
        y_bin_true = (y_true > 0).astype(int)
        y_bin_pred = (y_pred > 0).astype(int)
        attack_prob = 1.0 - y_prob[:, 0]
    return {
        "f1": float(f1_score(y_bin_true, y_bin_pred, zero_division=0)),
        "precision": float(precision_score(y_bin_true, y_bin_pred, zero_division=0)),
        "recall": float(recall_score(y_bin_true, y_bin_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_bin_true, attack_prob)),
    }


def training_history_stem(config: ExperimentConfig) -> str:
    """Resolve the training history JSON stem written by ``Trainer``."""
    return config.experiment_tag or config.model.name
