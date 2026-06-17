"""CLI entry point for model training."""

import argparse
from collections import Counter
from pathlib import Path

import torch
from chebnet_kaist.config import load_config
from chebnet_kaist.data import load_graphs_for_experiment
from chebnet_kaist.data.providers.registry import get_dataset
from chebnet_kaist.evaluation.checkpoint_loader import load_pretrained_encoder
from chebnet_kaist.models import create_model
from chebnet_kaist.training import Trainer
from chebnet_kaist.utils import set_seed


def run_training(config_path: str | Path, ignore_cache: bool = False) -> float:
    """Execute the full training pipeline from a YAML config.

    Args:
        config_path: Path to a training YAML config file.
        ignore_cache: When True, rebuild the graph cache from raw data.

    Returns:
        Best validation F1 score.
    """
    config = load_config(config_path)
    if ignore_cache:
        config.training.ignore_cache = True

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    graphs = load_graphs_for_experiment(config)

    provider = get_dataset(config.dataset)
    class_names = provider.class_names(config.label_mode)
    distribution = Counter(int(g.y.item()) for g in graphs)
    print("Class distribution:")
    for label, count in sorted(distribution.items()):
        name = class_names[label] if label < len(class_names) else str(label)
        print(f"  {name:8s} ({label}): {count:>7,}  ({count / len(graphs) * 100:.1f}%)")

    model = create_model(
        config.model.name,
        in_channels=config.model.in_channels,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        cheb_k=config.model.cheb_k,
        num_classes=config.model.num_classes,
    ).to(device)

    if config.training.init_checkpoint is not None:
        skip_head = config.model.num_classes != 5
        print(f"Loading init checkpoint: {config.training.init_checkpoint}")
        load_pretrained_encoder(
            model,
            config.training.init_checkpoint,
            device,
            skip_head=skip_head,
        )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model.name} | parameters: {param_count:,}")

    checkpoint_name = config.paths.checkpoint_name or f"best_model_{config.model.name}.pth"
    checkpoint_path = config.paths.checkpoint_dir / checkpoint_name
    trainer = Trainer(model, config, device)
    best_f1 = trainer.fit(graphs, checkpoint_path)

    print(f"\nTraining complete. Best validation F1: {best_f1:.4f}")
    return best_f1


def main() -> None:
    """Parse CLI arguments and start training."""
    parser = argparse.ArgumentParser(description="Train a GNN CAN IDS model.")
    parser.add_argument(
        "--config",
        default="configs/train_chebnet.yaml",
        help="Path to YAML config file (default: configs/train_chebnet.yaml)",
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Rebuild graph cache even if it exists",
    )
    args = parser.parse_args()
    run_training(args.config, ignore_cache=args.ignore_cache)


if __name__ == "__main__":
    main()
