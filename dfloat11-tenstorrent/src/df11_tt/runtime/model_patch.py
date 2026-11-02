"""Model patching utilities: swap dense weights with compressed runtime path."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ..utils.logging import get_logger
from ..compress.compressor import compress_weight
from .compressed_linear import CompressedLinearRuntime

log = get_logger("model_patch")


def get_patchable_layers(state_dict: dict, targets: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """Identify layers suitable for compression from a state dict.

    Args:
        state_dict: model state dict
        targets: list of layer name patterns to match. If None, returns all bf16 2D weight tensors.

    Returns:
        Dict mapping layer name -> tensor for patchable layers
    """
    patchable = {}
    for name, param in state_dict.items():
        if param.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            continue
        if param.ndim < 2:
            continue

        if targets:
            if any(t in name for t in targets):
                patchable[name] = param
        else:
            patchable[name] = param

    log.info(f"Found {len(patchable)} patchable layers"
             + (f" matching targets {targets}" if targets else ""))
    return patchable


def extract_and_compress_layers(
    state_dict: dict,
    layer_names: List[str],
    artifact_dir: str,
    verbose: bool = True,
) -> Dict[str, str]:
    """Extract and compress specified layers from a state dict.

    Args:
        state_dict: model state dict
        layer_names: names of layers to compress
        artifact_dir: output directory for artifacts
        verbose: print details

    Returns:
        Dict mapping layer_name -> artifact_path
    """
    artifact_map = {}
    for name in layer_names:
        if name not in state_dict:
            log.warning(f"Layer '{name}' not found in state dict, skipping")
            continue

        weight = state_dict[name]
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)

        artifact = compress_weight(weight, name, artifact_dir, verbose=verbose)
        artifact_path = str(Path(artifact_dir) / name.replace(".", "_"))
        artifact_map[name] = artifact_path

    manifest_path = Path(artifact_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(artifact_map, f, indent=2)

    if verbose:
        log.info(f"Saved manifest with {len(artifact_map)} layers to {manifest_path}")

    return artifact_map


def patch_model_layer(
    artifact_path: str,
    device=None,
    cache_decoded: bool = False,
    verify: bool = True,
    verbose: bool = True,
) -> CompressedLinearRuntime:
    """Create a compressed linear runtime for a given artifact.

    Args:
        artifact_path: path to the compressed artifact directory
        device: TT device
        cache_decoded: whether to cache decoded weights across calls
        verify: verify checksums on decode
        verbose: print details

    Returns:
        CompressedLinearRuntime instance
    """
    return CompressedLinearRuntime(
        artifact_path=artifact_path,
        device=device,
        cache_decoded=cache_decoded,
        verify=verify,
        verbose=verbose,
    )
