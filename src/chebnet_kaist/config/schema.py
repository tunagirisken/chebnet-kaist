"""Typed configuration structures."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathConfig:
    """Filesystem paths used across train and evaluate flows."""

    data_dir: Path
    cache_file: Path
    checkpoint_dir: Path
    results_dir: Path
    checkpoint_name: str | None = None


@dataclass
class ModelConfig:
    """Model hyperparameters shared by all GNN architectures."""

    name: str
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.3
    cheb_k: int = 3
    num_classes: int = 5
    in_channels: int = 3


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.30
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    ignore_cache: bool = False
    init_checkpoint: Path | None = None


@dataclass
class EvaluationConfig:
    """Evaluation-specific settings."""

    batch_size: int = 256
    checkpoint: Path | None = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    seed: int = 42
    experiment_tag: str | None = None
    dataset: str = "kaist"
    label_mode: str = "multiclass"
    num_classes: int = 5
    node_features: int = 3
    segment_size_sec: float = 0.1
    paths: PathConfig = field(
        default_factory=lambda: PathConfig(
            data_dir=Path("data/kaist"),
            cache_file=Path("data/kaist/graphs_cache.pt"),
            checkpoint_dir=Path("checkpoints"),
            results_dir=Path("results"),
        )
    )
    model: ModelConfig = field(default_factory=lambda: ModelConfig(name="chebnet"))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
