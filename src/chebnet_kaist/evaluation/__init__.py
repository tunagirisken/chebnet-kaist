"""Model evaluation, reporting, and visualization."""

from chebnet_kaist.evaluation.evaluator import Evaluator, resolve_checkpoint_path

__all__ = ["Evaluator", "resolve_checkpoint_path", "run_evaluation"]


def run_evaluation(*args, **kwargs):
    """Run fast or full evaluation (lazy import avoids heavy viz deps at import time).

    Returns:
        Whatever ``reports.runner.run_evaluation`` returns.
    """
    from chebnet_kaist.evaluation.reports.runner import run_evaluation as _run_evaluation

    return _run_evaluation(*args, **kwargs)
