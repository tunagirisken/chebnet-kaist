"""Orchestrates fast and full evaluation report generation."""

import shutil
from pathlib import Path

import torch
import torch.nn as nn
from chebnet_kaist.config.schema import ExperimentConfig
from chebnet_kaist.constants import EMBEDDING_VIZ_MODELS, MODEL_LABELS
from chebnet_kaist.evaluation import metrics_tables
from chebnet_kaist.evaluation.analysis import graph as graph_analysis
from chebnet_kaist.evaluation.analysis import training as training_viz
from chebnet_kaist.evaluation.evaluator import Evaluator
from chebnet_kaist.evaluation.plots import classification
from chebnet_kaist.evaluation.plots.style import set_publication_style
from chebnet_kaist.evaluation.reports.builder import build_html_report, generate_key_findings
from chebnet_kaist.evaluation.reports.io import model_output_dir, write_run_config
from chebnet_kaist.evaluation.reports.utils import (
    safe_run,
    split_validation_graphs,
    training_history_stem,
)
from chebnet_kaist.training.metrics import compute_metrics
from chebnet_kaist.utils.seed import set_seed
from torch_geometric.data import Data


def migrate_legacy_results(results_root: Path) -> None:
    """Move legacy flat result files into results/raw/ untouched."""
    raw_dir = results_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    legacy_patterns = ["*.png", "*.txt", "*.html", "rapor.html", "metrics.txt"]
    for pattern in legacy_patterns:
        for path in results_root.glob(pattern):
            if path.is_file() and path.parent == results_root:
                dest = raw_dir / path.name
                if not dest.exists():
                    shutil.move(str(path), str(dest))


def run_evaluation(
    model: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
    graphs: list[Data],
    checkpoint: Path,
    full_report: bool = False,
) -> dict[str, float]:
    """Run fast or full evaluation for a single model.

    Args:
        model: Loaded graph classifier.
        config: Experiment configuration.
        device: Compute device.
        graphs: Full cached graph dataset.
        checkpoint: Path to model checkpoint used.
        full_report: When True, generate all figures and HTML report.

    Returns:
        Macro metrics dict.
    """
    set_seed(config.seed)
    set_publication_style()

    model_name = config.model.name
    results_root = Path(config.paths.results_dir)
    migrate_legacy_results(results_root)
    out = model_output_dir(results_root, model_name, config.experiment_tag)

    for sub in ("metrics", "figures", "graphs", "training", "embeddings"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    write_run_config(out, config, checkpoint, full_report)

    val_graphs = split_validation_graphs(graphs, config)
    print(f"  Validation set: {len(val_graphs):,} graphs")

    evaluator = Evaluator(model, config, device)
    y_true, y_pred, y_prob = evaluator.predict(val_graphs)
    metrics = compute_metrics(y_true, y_pred, y_prob, config.num_classes)
    rows = metrics_tables.build_summary_rows(y_true, y_pred, y_prob)

    metrics_tables.write_csv(rows, out / "metrics" / "summary.csv")
    metrics_tables.write_html(rows, out / "metrics" / "summary.html", model_name)
    metrics_tables.write_json(
        rows,
        out / "metrics" / "metrics.json",
        extra={"macro": metrics, "model": model_name},
    )
    print(f"  Metrics saved to {out / 'metrics'}/")

    if not full_report:
        print("  Fast mode: skipping figure generation.")
        return metrics

    fig_dir = out / "figures"
    graph_dir = out / "graphs"
    train_dir = out / "training"
    history_stem = training_history_stem(config)

    # Part A — classification figures
    safe_run(
        "confusion_matrix",
        lambda: classification.plot_confusion_matrix(
            y_true,
            y_pred,
            fig_dir / "confusion_matrix",
            model_name,
        ),
    )
    safe_run(
        "per_class_metrics",
        lambda: classification.plot_per_class_metrics(
            y_true,
            y_pred,
            fig_dir / "per_class_metrics",
            model_name,
        ),
    )
    safe_run(
        "roc_curves",
        lambda: classification.plot_roc_curves(
            y_true,
            y_prob,
            fig_dir / "roc_curves",
            model_name,
        ),
    )
    safe_run(
        "pr_curves",
        lambda: classification.plot_precision_recall_curves(
            y_true,
            y_prob,
            fig_dir / "precision_recall_curves",
            model_name,
        ),
    )

    # Part B — ChebNet embedding figures
    if model_name in EMBEDDING_VIZ_MODELS:
        from chebnet_kaist.evaluation.plots import embeddings

        safe_run(
            "embeddings",
            lambda: embeddings.generate_all_embedding_figures(
                model,
                val_graphs,
                device,
                out / "embeddings",
                model_name,
                config.evaluation.batch_size,
            ),
        )

    # Part C — graph structure (spectral plots are KAIST-only; see graph_analysis)
    safe_run(
        "feature_distributions",
        lambda: graph_analysis.plot_feature_distributions(
            val_graphs,
            graph_dir / "feature_distributions",
            model_name,
        ),
    )
    safe_run(
        "spectral_timeseries",
        lambda: graph_analysis.generate_spectral_timeseries_all(
            config.paths.data_dir,
            graph_dir,
            model_name,
            config.segment_size_sec,
        ),
    )

    # Part D — training diagnostics (history file uses experiment_tag when set)
    safe_run(
        "training",
        lambda: training_viz.generate_training_figures(
            config.paths.checkpoint_dir,
            train_dir,
            history_stem,
            model_name,
        ),
    )

    # Part E — HTML report
    findings = generate_key_findings(rows, model_name)
    config_summary = {
        "Model": MODEL_LABELS.get(model_name, model_name),
        "Checkpoint": str(checkpoint),
        "Hidden dim": config.model.hidden_dim,
        "Dropout": config.model.dropout,
        "Batch size": config.training.batch_size,
        "Learning rate": config.training.learning_rate,
        "Seed": config.seed,
        "Val split": config.training.val_split,
    }

    figure_sections: list[tuple[str, list[tuple[Path, str]]]] = [
        (
            "Classification Performance",
            [
                (
                    fig_dir / "confusion_matrix.png",
                    "Confusion matrix with raw counts and row percentages.",
                ),
                (
                    fig_dir / "per_class_metrics.png",
                    "Per-class precision, recall, and F1 sorted by difficulty.",
                ),
                (
                    fig_dir / "roc_curves.png",
                    "One-vs-rest ROC curves; thick black line = macro average.",
                ),
                (
                    fig_dir / "precision_recall_curves.png",
                    "PR curves highlight performance on imbalanced classes.",
                ),
            ],
        ),
    ]

    if model_name in EMBEDDING_VIZ_MODELS:
        emb = out / "embeddings"
        label = MODEL_LABELS.get(model_name, model_name)
        figure_sections.append(
            (
                f"Embedding Analysis ({label})",
                [
                    (
                        emb / "tsne_umap_comparison.png",
                        "t-SNE (two seeds) and UMAP projections of node embeddings.",
                    ),
                    (
                        emb / "class_distance_heatmap.png",
                        "Cosine distance between class embedding centroids.",
                    ),
                    (
                        emb / "degree_vs_embedding_norm.png",
                        "Node degree vs embedding norm, colored by class.",
                    ),
                ],
            )
        )

    figure_sections.append(
        (
            "Graph Structure Analysis",
            [
                (
                    graph_dir / "feature_distributions.png",
                    "Violin plots of spectral and structural graph features per class.",
                ),
                (
                    graph_dir / "spectral_timeseries_dos.png",
                    "Spectral parameter shift at DoS attack onset (model-independent).",
                ),
            ],
        )
    )
    figure_sections.append(
        (
            "Training Diagnostics",
            [
                (
                    train_dir / "training_curves.png",
                    "Training/validation loss and F1; dashed line = best checkpoint epoch.",
                ),
                (
                    train_dir / "class_imbalance.png",
                    "Training class distribution and WeightedRandomSampler weights.",
                ),
            ],
        )
    )

    safe_run(
        "html_report",
        lambda: build_html_report(
            model_name,
            out / "report.html",
            out / "metrics" / "summary.html",
            figure_sections,
            findings,
            config_summary,
        ),
    )

    print(f"  Full report: {out / 'report.html'}")
    return metrics
