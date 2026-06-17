"""YAML configuration loading and merging."""

from pathlib import Path
from typing import Any

import yaml
from chebnet_kaist.config.schema import (
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
)

_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load and merge experiment configuration from YAML files.

    Supports a simple ``defaults`` list that references other YAML files
    in the ``configs/`` directory (e.g. ``defaults: [default]``).

    Args:
        config_path: Path to the primary YAML config file.

    Returns:
        Parsed :class:`ExperimentConfig` instance.
    """
    path = Path(config_path)
    if not path.is_absolute():
        candidate = _CONFIGS_DIR / path.name
        path = candidate if candidate.exists() else path

    raw = _load_yaml(path)

    defaults = raw.pop("defaults", [])
    merged: dict[str, Any] = {}
    for default_name in defaults:
        default_path = _CONFIGS_DIR / f"{default_name}.yaml"
        if default_path.exists():
            default_raw = _load_yaml(default_path)
            default_raw.pop("defaults", None)
            merged = _deep_merge(merged, default_raw)

    merged = _deep_merge(merged, raw)

    paths_raw = merged.get("paths", {})
    model_raw = merged.get("model", {})
    training_raw = merged.get("training", {})
    evaluation_raw = merged.get("evaluation", {})

    return ExperimentConfig(
        seed=merged.get("seed", 42),
        experiment_tag=merged.get("experiment_tag"),
        dataset=merged.get("dataset", "kaist"),
        label_mode=merged.get("label_mode", "multiclass"),
        num_classes=merged.get("num_classes", 5),
        node_features=merged.get("node_features", 3),
        segment_size_sec=merged.get("segment_size_sec", 0.1),
        paths=PathConfig(
            data_dir=Path(paths_raw.get("data_dir", "data/kaist")),
            cache_file=Path(paths_raw.get("cache_file", "data/kaist/graphs_cache.pt")),
            checkpoint_dir=Path(paths_raw.get("checkpoint_dir", "checkpoints")),
            results_dir=Path(paths_raw.get("results_dir", "results")),
            checkpoint_name=paths_raw.get("checkpoint_name"),
        ),
        model=ModelConfig(
            name=model_raw.get("name", "chebnet"),
            hidden_dim=model_raw.get("hidden_dim", 64),
            num_heads=model_raw.get("num_heads", 4),
            dropout=model_raw.get("dropout", 0.3),
            cheb_k=model_raw.get("cheb_k", 3),
            num_classes=model_raw.get("num_classes", merged.get("num_classes", 5)),
            in_channels=merged.get("node_features", 3),
        ),
        training=TrainingConfig(
            epochs=training_raw.get("epochs", 50),
            batch_size=training_raw.get("batch_size", 64),
            learning_rate=training_raw.get("learning_rate", 1e-3),
            weight_decay=training_raw.get("weight_decay", 1e-4),
            val_split=training_raw.get("val_split", 0.30),
            lr_scheduler_patience=training_raw.get("lr_scheduler_patience", 5),
            lr_scheduler_factor=training_raw.get("lr_scheduler_factor", 0.5),
            ignore_cache=training_raw.get("ignore_cache", False),
            init_checkpoint=(
                Path(training_raw["init_checkpoint"])
                if training_raw.get("init_checkpoint")
                else None
            ),
        ),
        evaluation=EvaluationConfig(
            batch_size=evaluation_raw.get("batch_size", 256),
            checkpoint=(
                Path(evaluation_raw["checkpoint"]) if evaluation_raw.get("checkpoint") else None
            ),
        ),
    )
