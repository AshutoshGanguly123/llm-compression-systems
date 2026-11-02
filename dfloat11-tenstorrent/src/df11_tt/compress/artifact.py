"""Compressed artifact management: save/load DFloat11-style compressed weight artifacts."""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .huffman import HuffmanCodec


@dataclass
class CompressionArtifact:
    layer_name: str
    shape: tuple
    dtype: str  # "bfloat16"
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    checksum: str
    num_elements: int
    num_bits_exponent: int
    codec_version: str = "huffman_exponent_v1"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(d["shape"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CompressionArtifact":
        d["shape"] = tuple(d["shape"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def save_artifact(
    artifact: CompressionArtifact,
    codec: HuffmanCodec,
    exponent_bitstream: bytes,
    sign_mantissa_packed: np.ndarray,
    out_dir: str,
) -> str:
    """Save compressed artifact to disk.

    Creates:
      <out_dir>/<layer_name>/metadata.json
      <out_dir>/<layer_name>/exponent_bitstream.bin
      <out_dir>/<layer_name>/sign_mantissa.npy
      <out_dir>/<layer_name>/codec.json
    """
    artifact_dir = Path(out_dir) / artifact.layer_name.replace(".", "_")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(artifact.to_dict(), f, indent=2)

    with open(artifact_dir / "exponent_bitstream.bin", "wb") as f:
        f.write(exponent_bitstream)

    np.save(str(artifact_dir / "sign_mantissa.npy"), sign_mantissa_packed)

    with open(artifact_dir / "codec.json", "w") as f:
        json.dump(codec.to_dict(), f, indent=2)

    return str(artifact_dir)


def load_artifact(artifact_path: str) -> tuple:
    """Load a compressed artifact from disk.

    Returns:
        (artifact, codec, exponent_bitstream, sign_mantissa_packed)
    """
    artifact_dir = Path(artifact_path)

    with open(artifact_dir / "metadata.json") as f:
        artifact = CompressionArtifact.from_dict(json.load(f))

    with open(artifact_dir / "codec.json") as f:
        codec = HuffmanCodec.from_dict(json.load(f))

    with open(artifact_dir / "exponent_bitstream.bin", "rb") as f:
        exponent_bitstream = f.read()

    sign_mantissa_packed = np.load(str(artifact_dir / "sign_mantissa.npy"))

    return artifact, codec, exponent_bitstream, sign_mantissa_packed
