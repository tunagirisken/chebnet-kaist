"""Checkpoint loading with legacy key name remapping."""

from pathlib import Path

import torch
import torch.nn as nn

# Legacy Turkish layer names from pre-refactor models.
_LEGACY_KEY_MAP = {
    "katman1.": "layer1.",
    "katman2.": "layer2.",
    "katman3.": "layer3.",
    "tam_bagli.": "fc_hidden.",
    "cikis.": "fc_out.",
}


def _remap_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remap legacy checkpoint keys to current module attribute names."""
    if not any(k.startswith("katman") for k in state_dict):
        return state_dict
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in _LEGACY_KEY_MAP.items():
            if key.startswith(old):
                new_key = new + key[len(old) :]
                break
        remapped[new_key] = value
    return remapped


def load_pretrained_encoder(
    model: nn.Module,
    checkpoint: Path,
    device: torch.device,
    *,
    skip_head: bool = False,
) -> None:
    """Load weights from a checkpoint, optionally skipping the classifier head.

    Useful when transferring encoder weights between compatible runs while
    replacing or resizing the classifier head.

    Args:
        model: Target model instance.
        checkpoint: Source ``.pth`` file.
        device: Target device.
        skip_head: When True, omit ``fc_out`` keys with shape mismatches.
    """
    state_dict = _remap_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    if skip_head:
        model_state = model.state_dict()
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
    model.load_state_dict(state_dict, strict=not skip_head)


def load_model_weights(model: nn.Module, checkpoint: Path, device: torch.device) -> None:
    """Load model weights, supporting legacy checkpoint key names.

    Args:
        model: Model instance to load weights into.
        checkpoint: Path to ``.pth`` state dict file.
        device: Target device for tensor mapping.
    """
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = _remap_state_dict(state_dict)
    model.load_state_dict(state_dict)
