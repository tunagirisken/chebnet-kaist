"""ChebNet graph classifier for CAN intrusion detection."""

import torch.nn as nn
import torch.nn.functional as F
from chebnet_kaist.models.base import NUM_CLASSES
from torch import Tensor
from torch_geometric.nn import ChebConv, global_mean_pool


class ChebNetIDS(nn.Module):
    """Chebyshev spectral GNN for graph-level intrusion classification.

    Uses three ChebConv layers (K-order polynomial filters) with ELU activations
    and global mean pooling. When ``return_viz=True``, pre-pooling node
    embeddings are returned for embedding visualization.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
        cheb_k: int = 3,
    ) -> None:
        """Initialize ChebNet layers and classifier head.

        Args:
            in_channels: Input node feature dimension.
            hidden_dim: Hidden dimension for ChebConv layers.
            num_heads: Unused; kept for a shared constructor signature.
            dropout: Dropout rate for the classifier head.
            num_classes: Number of output classes.
            cheb_k: Chebyshev polynomial order for ChebConv layers.
        """
        super().__init__()
        del num_heads

        self.layer1 = ChebConv(in_channels, hidden_dim, K=cheb_k)
        self.layer2 = ChebConv(hidden_dim, hidden_dim, K=cheb_k)
        self.layer3 = ChebConv(hidden_dim, hidden_dim * 2, K=cheb_k)

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_attr: Tensor | None = None,
        return_viz: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor | None, None]]:
        """Forward pass with optional node embedding export.

        Args:
            x: Node features.
            edge_index: Graph connectivity.
            batch: Batch vector for pooling.
            edge_attr: Optional edge weights used by ChebConv if provided.
            return_viz: Return layer-3 node embeddings before pooling.

        Returns:
            Class logits, optionally paired with ``(node_embeddings, None)``.
        """
        x = F.elu(self.layer1(x, edge_index, edge_attr))
        x = F.elu(self.layer2(x, edge_index, edge_attr))
        x = F.elu(self.layer3(x, edge_index, edge_attr))

        node_embeddings = x if return_viz else None

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc_hidden(x))
        x = self.dropout(x)
        logits = self.fc_out(x)

        if return_viz:
            return logits, (node_embeddings, None)
        return logits
