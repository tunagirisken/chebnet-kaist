"""HTML report generation and orchestration."""

from chebnet_kaist.evaluation.reports.builder import build_html_report, generate_key_findings
from chebnet_kaist.evaluation.reports.io import model_output_dir, write_run_config
from chebnet_kaist.evaluation.reports.runner import migrate_legacy_results, run_evaluation
from chebnet_kaist.evaluation.reports.utils import (
    compute_binary_attack_metrics,
    safe_run,
    split_validation_graphs,
    training_history_stem,
)

__all__ = [
    "build_html_report",
    "compute_binary_attack_metrics",
    "generate_key_findings",
    "migrate_legacy_results",
    "model_output_dir",
    "run_evaluation",
    "safe_run",
    "split_validation_graphs",
    "training_history_stem",
    "write_run_config",
]
