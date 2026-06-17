"""Model-agnostic training loop for graph classifiers."""

import json
from pathlib import Path

import torch
import torch.nn as nn
from chebnet_kaist.config.schema import ExperimentConfig
from chebnet_kaist.training.metrics import compute_metrics, metrics_to_log_line
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class Trainer:
    """Train a graph classifier with stratified split and class balancing.

    The trainer is decoupled from any specific GNN architecture; it only
    requires a model that implements the shared forward signature.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        """Initialize optimizer, scheduler, and loss for training.

        Args:
            model: Graph classifier module.
            config: Full experiment configuration.
            device: Target compute device.
        """
        self.model = model
        self.config = config
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=config.training.lr_scheduler_patience,
            factor=config.training.lr_scheduler_factor,
        )

    def _split_graphs(self, graphs: list[Data]) -> tuple[list[Data], list[Data]]:
        """Stratified train/validation split.

        Args:
            graphs: Full graph dataset.

        Returns:
            Tuple of ``(train_graphs, val_graphs)``.
        """
        labels = [int(graph.y.item()) for graph in graphs]
        train_idx, val_idx = train_test_split(
            range(len(graphs)),
            test_size=self.config.training.val_split,
            random_state=self.config.seed,
            stratify=labels,
        )
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        return train_graphs, val_graphs

    def _build_loaders(
        self,
        train_graphs: list[Data],
        val_graphs: list[Data],
    ) -> tuple[DataLoader, DataLoader]:
        """Create train and validation DataLoaders with weighted sampling.

        Args:
            train_graphs: Training graph list.
            val_graphs: Validation graph list.

        Returns:
            Tuple of ``(train_loader, val_loader)``.
        """
        num_classes = self.config.num_classes
        class_counts = torch.zeros(num_classes)
        for graph in train_graphs:
            class_counts[graph.y.item()] += 1

        weights = 1.0 / (class_counts + 1e-6)
        sample_weights = torch.tensor([weights[g.y.item()] for g in train_graphs])
        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True,
        )

        batch_size = self.config.training.batch_size
        train_loader = DataLoader(train_graphs, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch.

        Args:
            loader: Training DataLoader.

        Returns:
            Mean training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc="Train", leave=False, unit="batch"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(logits, batch.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on the validation set.

        Args:
            loader: Validation DataLoader.

        Returns:
            Metrics dict including ``loss``, ``f1``, ``precision``, ``recall``, ``auc``.
        """
        self.model.eval()
        total_loss = 0.0
        y_true: list[int] = []
        y_pred: list[int] = []
        y_prob: list[list[float]] = []

        for batch in tqdm(loader, desc="Validate", leave=False, unit="batch"):
            batch = batch.to(self.device)
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            y_true.extend(batch.y.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

        metrics = compute_metrics(y_true, y_pred, y_prob, self.config.num_classes)
        metrics["loss"] = total_loss / len(loader.dataset)
        return metrics

    def fit(self, graphs: list[Data], checkpoint_path: Path) -> float:
        """Train the model and save the best checkpoint by validation F1.

        Args:
            graphs: Full graph dataset.
            checkpoint_path: Path to write the best model state dict.

        Returns:
            Best validation F1 score achieved during training.
        """
        train_graphs, val_graphs = self._split_graphs(graphs)
        print(f"Split: train={len(train_graphs):,}  val={len(val_graphs):,}")

        train_loader, val_loader = self._build_loaders(train_graphs, val_graphs)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        num_classes = self.config.num_classes
        class_counts = torch.zeros(num_classes)
        for graph in train_graphs:
            class_counts[graph.y.item()] += 1
        sampler_weights = (1.0 / (class_counts + 1e-6)).tolist()

        history: dict = {
            "model": self.config.model.name,
            "experiment_tag": self.config.experiment_tag,
            "dataset": self.config.dataset,
            "label_mode": self.config.label_mode,
            "epochs": [],
            "best_epoch": 0,
            "best_f1": 0.0,
            "train_class_counts": {int(i): int(c) for i, c in enumerate(class_counts)},
            "sampler_weights": {int(i): float(w) for i, w in enumerate(sampler_weights)},
        }

        best_f1 = 0.0
        best_epoch = 0
        epoch_bar = tqdm(
            range(1, self.config.training.epochs + 1),
            desc="Epochs",
            unit="epoch",
        )

        for epoch in epoch_bar:
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            self.scheduler.step(val_metrics["f1"])

            line = metrics_to_log_line(epoch, train_loss, val_metrics)
            tqdm.write(line)
            epoch_bar.set_postfix(
                F1=f"{val_metrics['f1']:.4f}",
                AUC=f"{val_metrics['auc']:.4f}",
            )

            history["epochs"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_auc": val_metrics["auc"],
                }
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                torch.save(self.model.state_dict(), checkpoint_path)
                tqdm.write(f"  -> Best model saved: {checkpoint_path} (F1={best_f1:.4f})")

        history["best_epoch"] = best_epoch
        history["best_f1"] = best_f1
        history_stem = self.config.experiment_tag or self.config.model.name
        history_path = checkpoint_path.parent / f"training_history_{history_stem}.json"
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        print(f"Training history saved: {history_path}")

        return best_f1
