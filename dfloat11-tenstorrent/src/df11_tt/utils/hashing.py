"""Checksumming for tensors to verify bit-exact reconstruction."""

import hashlib
import torch
import numpy as np


def tensor_checksum(t: torch.Tensor) -> str:
    """Compute SHA-256 of the raw bytes of a tensor (detached, contiguous, CPU)."""
    data = t.detach().cpu().contiguous()
    if data.dtype == torch.bfloat16:
        raw = data.view(torch.int16).numpy().view(np.uint8).tobytes()
    else:
        raw = data.numpy().view(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


def verify_bit_exact(original: torch.Tensor, reconstructed: torch.Tensor) -> bool:
    """Check two BF16 tensors are bitwise identical via raw 16-bit representation."""
    if original.shape != reconstructed.shape:
        return False
    if original.dtype != reconstructed.dtype:
        return False
    o_bytes = original.detach().cpu().contiguous().view(torch.int16)
    r_bytes = reconstructed.detach().cpu().contiguous().view(torch.int16)
    return torch.equal(o_bytes, r_bytes)
