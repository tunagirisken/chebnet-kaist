"""Lightweight helpers for evaluation report output paths and metadata."""

import json
from datetime import datetime, timezone
from pathlib import Path

from chebnet_kaist.config.schema import ExperimentConfig


def model_output_dir(
    results_root: Path,
    model_name: str,
    experiment_tag: str | None = None,
) -> Path:
    """Return isolated output directory for one model or experiment."""
    tag = experiment_tag or model_name
    return results_root / tag


def write_run_config(
    output_dir: Path,
    config: ExperimentConfig,
    checkpoint: Path,
    full_report: bool,
) -> Path:
    """Save run metadata alongside evaluation outputs."""
    payload = {
        "model": config.model.name,
        "checkpoint": str(checkpoint),
        "dataset": str(config.paths.data_dir),
        "cache_file": str(config.paths.cache_file),
        "date": datetime.now(timezone.utc).isoformat(),
        "full_report": full_report,
        "seed": config.seed,
        "hyperparameters": {
            "hidden_dim": config.model.hidden_dim,
            "num_heads": config.model.num_heads,
            "dropout": config.model.dropout,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epochs,
            "val_split": config.training.val_split,
        },
    }
    path = output_dir / "run_config.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
