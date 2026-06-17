"""Model registry and factory function."""

import torch.nn as nn
from chebnet_kaist.models.chebnet import ChebNetIDS

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "chebnet": ChebNetIDS,
}


def create_model(name: str, **kwargs: object) -> nn.Module:
    """Instantiate ChebNet by name.

    Args:
        name: Model identifier (``chebnet``).
        **kwargs: Constructor keyword arguments forwarded to the model class.

    Returns:
        Initialized model instance.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if name not in _MODEL_REGISTRY:
        registered = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model: {name!r}. Available models: {registered}")

    return _MODEL_REGISTRY[name](**kwargs)
