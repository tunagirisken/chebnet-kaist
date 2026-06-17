"""Shared model interface and constants."""

from typing import Any, Protocol

from torch import Tensor

NUM_CLASSES = 5  # Normal, DoS, Fuzzy, RPM, Gear


class GraphClassifier(Protocol):
    """Protocol for graph-level CAN intrusion classifiers.

    All models must accept the same forward signature so the training loop
    and evaluator remain architecture-agnostic.
    """

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor | None = None,
        return_viz: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Any, Any]]:
        """Run a forward pass and optionally return visualization artifacts.

        Args:
            x: Node feature matrix ``(N, F)``.
            edge_index: COO edge index ``(2, E)``.
            batch: Batch assignment vector ``(N,)``.
            edge_attr: Optional edge features (unused by current models).
            return_viz: When True, return auxiliary data for plotting.

        Returns:
            Logits tensor ``(B, num_classes)``, or logits plus viz tuple.
        """
        ...
