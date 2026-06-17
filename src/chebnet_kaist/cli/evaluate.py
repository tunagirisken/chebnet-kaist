"""CLI entry point for model evaluation (delegates to report runner)."""

import argparse
from pathlib import Path

import torch
from chebnet_kaist.config import load_config
from chebnet_kaist.data import load_graphs_for_experiment
from chebnet_kaist.evaluation.checkpoint_loader import load_model_weights
from chebnet_kaist.evaluation.evaluator import resolve_checkpoint_path
from chebnet_kaist.evaluation.reports.runner import run_evaluation
from chebnet_kaist.models import create_model
from chebnet_kaist.utils import set_seed

DEFAULT_CONFIG = "configs/train_chebnet.yaml"


def main() -> None:
    """Parse CLI arguments and start evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ChebNet CAN IDS model.")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to YAML config (default: configs/train_chebnet.yaml)",
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Generate all figures and HTML report",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path override",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config.model.name
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = load_graphs_for_experiment(config)
    ckpt = resolve_checkpoint_path(
        model_name,
        Path(args.checkpoint) if args.checkpoint else config.evaluation.checkpoint,
    )

    model = create_model(
        model_name,
        in_channels=config.model.in_channels,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        cheb_k=config.model.cheb_k,
        num_classes=config.model.num_classes,
    ).to(device)
    load_model_weights(model, ckpt, device)

    metrics = run_evaluation(
        model,
        config,
        device,
        graphs,
        ckpt,
        full_report=args.full_report,
    )
    print(f"Macro F1: {metrics['f1']:.4f} | Macro AUC: {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()
