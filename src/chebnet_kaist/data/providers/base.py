"""Dataset provider protocol for graph-level CAN IDS pipelines."""

from typing import Protocol

from chebnet_kaist.config.schema import ExperimentConfig
from torch_geometric.data import Data


class GraphDatasetProvider(Protocol):
    """Abstract interface for raw CAN logs -> PyG graph datasets.

    Each concrete provider owns dataset-specific parsing, segmentation, labeling,
    and cache paths. Training and evaluation code must depend only on this
    protocol — never on a specific dataset module.
    """

    @property
    def name(self) -> str:
        """Registry key (e.g. ``kaist``, ``road``)."""
        ...

    def load_graphs(self, config: ExperimentConfig) -> list[Data]:
        """Load or build graphs for an experiment configuration.

        Args:
            config: Full experiment config including ``paths``, ``dataset``,
                ``label_mode``, and training cache flags.

        Returns:
            List of PyG ``Data`` graphs ready for the model-agnostic trainer.
        """
        ...

    def class_names(self, label_mode: str = "multiclass") -> list[str]:
        """Human-readable class names for metrics and plots.

        Args:
            label_mode: ``multiclass`` or ``binary``.

        Returns:
            Ordered class name list matching label indices.
        """
        ...
