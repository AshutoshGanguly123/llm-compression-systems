"""TT device adapter: upload decoded CPU tensors to TT device."""

import torch
from ..utils.logging import get_logger
from ..utils.timing import Timer

log = get_logger("tt_adapter")

_TTNN_AVAILABLE = False
try:
    import ttnn
    _TTNN_AVAILABLE = True
except ImportError:
    pass


def to_tt_weight(
    weight: torch.Tensor,
    device=None,
    dtype=None,
    layout=None,
    memory_config=None,
    verbose: bool = True,
):
    """Upload a CPU BF16 tensor to the TT device.

    Args:
        weight: CPU torch.bfloat16 tensor
        device: TT device object (from ttnn)
        dtype: TT dtype (default: ttnn.bfloat16)
        layout: TT layout (default: ttnn.TILE_LAYOUT)
        memory_config: optional memory config

    Returns:
        TT tensor on device, or the original tensor if ttnn is not available
    """
    if not _TTNN_AVAILABLE or device is None:
        if verbose:
            log.warning("ttnn not available or no device; returning CPU tensor")
        return weight

    if dtype is None:
        dtype = ttnn.bfloat16
    if layout is None:
        layout = ttnn.TILE_LAYOUT

    with Timer() as t:
        tt_tensor = ttnn.from_torch(
            weight,
            device=device,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )

    if verbose:
        log.info(f"Uploaded tensor to TT device: shape={tuple(weight.shape)}, "
                 f"dtype={dtype}, layout={layout} ({t.elapsed_ms:.1f}ms)")

    return tt_tensor


def cleanup_tt_tensor(tt_tensor, verbose: bool = False):
    """Deallocate a TT tensor from device memory."""
    if not _TTNN_AVAILABLE:
        return
    try:
        ttnn.deallocate(tt_tensor)
        if verbose:
            log.info("Deallocated TT tensor")
    except Exception as e:
        if verbose:
            log.warning(f"Failed to deallocate TT tensor: {e}")
