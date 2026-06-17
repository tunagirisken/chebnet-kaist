"""Load or build graphs according to experiment configuration."""

from chebnet_kaist.config.schema import ExperimentConfig
from chebnet_kaist.data.providers.registry import get_dataset
from torch_geometric.data import Data


def load_graphs_for_experiment(config: ExperimentConfig) -> list[Data]:
    """Load graphs via the dataset registry (model-agnostic entry point).

    The ``dataset`` field in the YAML config selects the provider. Models never
    import KAIST or ROAD modules directly — only this function (or the registry).

    Args:
        config: Full experiment configuration.

    Returns:
        Graph list ready for training or evaluation.
    """
    provider = get_dataset(config.dataset)
    print(f"Dataset provider: {provider.name}")
    return provider.load_graphs(config)
