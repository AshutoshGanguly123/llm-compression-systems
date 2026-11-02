"""Offline BF16 weight compression using DFloat11-style exponent Huffman coding.

Compresses a BF16 tensor by:
1. Splitting into sign, exponent, mantissa components
2. Huffman-encoding the exponent stream (lowest entropy due to clustering)
3. Packing sign+mantissa as raw payload (8 bits each: 1 sign + 7 mantissa)
4. Saving compressed artifact with metadata for exact reconstruction
"""

import torch
import numpy as np

from ..utils.hashing import tensor_checksum
from ..utils.logging import get_logger
from ..utils.timing import Timer
from .bitops import split_bf16, pack_sign_mantissa
from .huffman import HuffmanCodec
from .artifact import CompressionArtifact, save_artifact

log = get_logger("compress")


def compress_weight(
    weight: torch.Tensor,
    layer_name: str,
    out_dir: str,
    codec: str = "huffman_exponent",
    verbose: bool = True,
) -> CompressionArtifact:
    """Compress a BF16 weight tensor and save the artifact.

    Args:
        weight: torch.bfloat16 tensor to compress
        layer_name: identifier for the layer
        out_dir: directory to store artifacts
        codec: compression strategy (only "huffman_exponent" supported)
        verbose: print detailed info

    Returns:
        CompressionArtifact with compression metadata
    """
    assert weight.dtype == torch.bfloat16, f"Expected bfloat16, got {weight.dtype}"
    assert codec == "huffman_exponent", f"Unknown codec: {codec}"

    original_bytes = weight.nelement() * 2  # 2 bytes per bf16

    if verbose:
        log.info(f"Compressing layer '{layer_name}': shape={tuple(weight.shape)}, "
                 f"elements={weight.nelement():,}, original={original_bytes:,} bytes")

    chk = tensor_checksum(weight)

    with Timer() as t_split:
        signs, exponents, mantissas = split_bf16(weight)
    if verbose:
        log.info(f"  Split BF16 into components: {t_split.elapsed_ms:.1f}ms")

    with Timer() as t_codec:
        huffman = HuffmanCodec().build_from_data(exponents)
    if verbose:
        stats = huffman.stats(exponents)
        log.info(f"  Built Huffman codec: {stats['unique_symbols']} unique exponents, "
                 f"exponent compression ratio: {stats['compression_ratio_exponent']:.2f}x")

    with Timer() as t_encode:
        exponent_bitstream, num_bits = huffman.encode(exponents)
    if verbose:
        log.info(f"  Encoded exponents: {len(exponents):,} -> {len(exponent_bitstream):,} bytes "
                 f"({t_encode.elapsed_ms:.1f}ms)")

    with Timer() as t_pack:
        sign_mantissa_packed = pack_sign_mantissa(signs, mantissas)
    if verbose:
        log.info(f"  Packed sign+mantissa: {len(sign_mantissa_packed):,} bytes ({t_pack.elapsed_ms:.1f}ms)")

    compressed_bytes = len(exponent_bitstream) + len(sign_mantissa_packed)
    ratio = original_bytes / max(compressed_bytes, 1)

    artifact = CompressionArtifact(
        layer_name=layer_name,
        shape=tuple(weight.shape),
        dtype="bfloat16",
        original_bytes=original_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=round(ratio, 4),
        checksum=chk,
        num_elements=weight.nelement(),
        num_bits_exponent=num_bits,
    )

    with Timer() as t_save:
        artifact_path = save_artifact(artifact, huffman, exponent_bitstream, sign_mantissa_packed, out_dir)

    if verbose:
        savings_pct = (1 - compressed_bytes / original_bytes) * 100
        log.info(f"  Saved artifact to '{artifact_path}' ({t_save.elapsed_ms:.1f}ms)")
        log.info(f"  Compression: {original_bytes:,} -> {compressed_bytes:,} bytes "
                 f"(ratio={ratio:.3f}x, saved {savings_pct:.1f}%)")

    return artifact
