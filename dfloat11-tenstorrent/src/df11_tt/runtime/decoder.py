"""Host-side CPU decoder: reconstruct BF16 tensor from compressed artifact."""

import torch
import numpy as np

from ..compress.artifact import load_artifact
from ..compress.bitops import merge_bf16, unpack_sign_mantissa
from ..utils.hashing import tensor_checksum, verify_bit_exact
from ..utils.logging import get_logger
from ..utils.timing import Timer

log = get_logger("decoder")


def decode_artifact(
    artifact_path: str,
    verify: bool = False,
    verbose: bool = True,
) -> torch.Tensor:
    """Decode a compressed artifact back to a BF16 tensor on CPU.

    Args:
        artifact_path: path to the artifact directory
        verify: if True, verify checksum against stored original
        verbose: print timing info

    Returns:
        torch.bfloat16 tensor with the original shape
    """
    with Timer() as t_load:
        artifact, codec, exponent_bitstream, sign_mantissa_packed = load_artifact(artifact_path)

    if verbose:
        log.info(f"Loaded artifact '{artifact.layer_name}': shape={artifact.shape}, "
                 f"compressed={artifact.compressed_bytes:,} bytes ({t_load.elapsed_ms:.1f}ms)")

    with Timer() as t_decode_exp:
        exponents = codec.decode(
            exponent_bitstream,
            artifact.num_bits_exponent,
            artifact.num_elements,
        )

    if verbose:
        log.info(f"  Decoded exponents: {artifact.num_elements:,} symbols ({t_decode_exp.elapsed_ms:.1f}ms)")

    with Timer() as t_unpack:
        signs, mantissas = unpack_sign_mantissa(sign_mantissa_packed)

    if verbose:
        log.info(f"  Unpacked sign+mantissa ({t_unpack.elapsed_ms:.1f}ms)")

    with Timer() as t_merge:
        tensor = merge_bf16(signs, exponents, mantissas, artifact.shape)

    if verbose:
        log.info(f"  Merged BF16 tensor ({t_merge.elapsed_ms:.1f}ms)")

    if verify:
        chk = tensor_checksum(tensor)
        if chk != artifact.checksum:
            raise ValueError(
                f"Checksum mismatch for '{artifact.layer_name}': "
                f"expected {artifact.checksum}, got {chk}"
            )
        if verbose:
            log.info(f"  Checksum verified OK")

    total_ms = t_load.elapsed_ms + t_decode_exp.elapsed_ms + t_unpack.elapsed_ms + t_merge.elapsed_ms
    if verbose:
        log.info(f"  Total decode time: {total_ms:.1f}ms")

    return tensor
