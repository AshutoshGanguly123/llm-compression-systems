"""Bit-level operations for BF16 weight decomposition and reconstruction.

BF16 layout (16 bits): [sign (1)] [exponent (8)] [mantissa (7)]
"""

import torch
import numpy as np


def split_bf16(tensor: torch.Tensor) -> tuple:
    """Split a BF16 tensor into sign, exponent, and mantissa arrays.

    Args:
        tensor: A torch.bfloat16 tensor (any shape).

    Returns:
        (signs, exponents, mantissas) as uint8 numpy arrays, all with shape = (N,)
        where N is the number of elements.
    """
    assert tensor.dtype == torch.bfloat16, f"Expected bfloat16, got {tensor.dtype}"
    flat = tensor.detach().cpu().contiguous().view(-1)
    raw = flat.view(torch.int16).numpy().astype(np.uint16)

    signs = ((raw >> 15) & 0x1).astype(np.uint8)
    exponents = ((raw >> 7) & 0xFF).astype(np.uint8)
    mantissas = (raw & 0x7F).astype(np.uint8)

    return signs, exponents, mantissas


def merge_bf16(signs: np.ndarray, exponents: np.ndarray, mantissas: np.ndarray,
               shape: tuple) -> torch.Tensor:
    """Reconstruct a BF16 tensor from sign, exponent, and mantissa arrays.

    Args:
        signs: uint8 array of shape (N,), values in {0, 1}
        exponents: uint8 array of shape (N,), values in [0, 255]
        mantissas: uint8 array of shape (N,), values in [0, 127]
        shape: original tensor shape

    Returns:
        torch.bfloat16 tensor with the given shape
    """
    raw = (signs.astype(np.uint16) << 15) | (exponents.astype(np.uint16) << 7) | mantissas.astype(np.uint16)
    raw_i16 = raw.astype(np.int16)
    t = torch.from_numpy(raw_i16.copy()).view(torch.bfloat16)
    return t.reshape(shape)


def pack_sign_mantissa(signs: np.ndarray, mantissas: np.ndarray) -> np.ndarray:
    """Pack sign (1 bit) and mantissa (7 bits) into a single uint8 per element."""
    return ((signs.astype(np.uint8) << 7) | mantissas.astype(np.uint8))


def unpack_sign_mantissa(packed: np.ndarray) -> tuple:
    """Unpack sign and mantissa from packed uint8 array."""
    signs = ((packed >> 7) & 0x1).astype(np.uint8)
    mantissas = (packed & 0x7F).astype(np.uint8)
    return signs, mantissas
