"""Checkpoint loading and batched model evaluation."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from chebnet_kaist.config.schema import ExperimentConfig
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def resolve_checkpoint_path(model_name: str, checkpoint: Path | None) -> Path:
    """Resolve the checkpoint file for a given model.

    Falls back to legacy checkpoint names when newer names are absent.

    Args:
        model_name: Registered model name.
        checkpoint: Explicit checkpoint path, or None for default.

    Returns:
        Resolved checkpoint path (may not exist).
    """
    if checkpoint is not None:
        return checkpoint

    candidates = [
        Path(f"checkpoints/best_model_{model_name}.pth"),
        Path(f"checkpoints/en_iyi_model_{model_name}.pth"),
        Path("checkpoints/en_iyi_model.pth"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


class Evaluator:
    """Run inference on a graph dataset."""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        """Initialize evaluator with a loaded model.

        Args:
            model: Trained graph classifier.
            config: Experiment configuration.
            device: Compute device.
        """
        self.model = model
        self.config = config
        self.device = device

    @torch.no_grad()
    def predict(
        self,
        graphs: list[Data],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run batched inference on a graph list.

        Args:
            graphs: Graphs to evaluate.

        Returns:
            Tuple of ``(y_true, y_pred, y_prob)`` numpy arrays.
        """
        loader = DataLoader(
            graphs,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
        )

        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        y_prob: list[list[float]] = []

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            y_true.extend(batch.y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

        return np.array(y_true), np.array(y_pred), np.array(y_prob)
