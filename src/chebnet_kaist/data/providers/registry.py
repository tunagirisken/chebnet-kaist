"""Registry for graph dataset providers."""

from chebnet_kaist.data.providers.base import GraphDatasetProvider
from chebnet_kaist.data.providers.kaist import KAIST_PROVIDER

_DATASET_REGISTRY: dict[str, GraphDatasetProvider] = {
    KAIST_PROVIDER.name: KAIST_PROVIDER,
}


def register_dataset(provider: GraphDatasetProvider) -> None:
    """Register a dataset provider.

    Args:
        provider: Provider instance implementing :class:`GraphDatasetProvider`.
    """
    _DATASET_REGISTRY[provider.name] = provider


def get_dataset(name: str) -> GraphDatasetProvider:
    """Return a registered dataset provider by name.

    Args:
        name: Dataset key (``kaist``).

    Returns:
        Provider instance.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _DATASET_REGISTRY:
        registered = ", ".join(sorted(_DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset: {name!r}. Available: {registered}")
    return _DATASET_REGISTRY[name]


def list_datasets() -> list[str]:
    """Return sorted registered dataset names."""
    return sorted(_DATASET_REGISTRY)
