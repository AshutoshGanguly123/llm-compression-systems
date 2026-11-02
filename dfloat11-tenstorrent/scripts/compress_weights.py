#!/usr/bin/env python3
"""Compress extracted BF16 weights using DFloat11-style Huffman exponent coding.

Usage:
    python scripts/compress_weights.py \
        --extracted-dir outputs/extracted \
        --artifact-dir outputs/artifacts \
        --layers lm_head output.weight
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.compress import compress_weight
from df11_tt.utils import get_logger

log = get_logger("compress_weights")


def compress_from_extracted(extracted_dir: str, artifact_dir: str, layer_patterns: list = None):
    """Compress all extracted weight files, or those matching layer_patterns."""
    manifest_path = Path(extracted_dir) / "extraction_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    pt_files = list(Path(extracted_dir).glob("*.pt"))
    if not pt_files:
        log.error(f"No .pt files found in {extracted_dir}")
        return

    # Build reverse map from filename stem to actual layer name using manifest
    stem_to_name = {}
    for layer_name in manifest:
        stem_to_name[layer_name.replace(".", "_")] = layer_name

    artifacts = {}
    for pt_file in pt_files:
        stem = pt_file.stem
        layer_name = stem_to_name.get(stem, stem)

        if layer_patterns:
            if not any(p in layer_name for p in layer_patterns):
                continue

        log.info(f"Loading {pt_file} (layer: {layer_name})...")
        weight = torch.load(str(pt_file), map_location="cpu", weights_only=True)
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)

        artifact = compress_weight(weight, layer_name, artifact_dir)
        artifacts[layer_name] = {
            "artifact_path": str(Path(artifact_dir) / layer_name.replace(".", "_")),
            "original_bytes": artifact.original_bytes,
            "compressed_bytes": artifact.compressed_bytes,
            "compression_ratio": artifact.compression_ratio,
        }

    summary_path = Path(artifact_dir) / "compression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(artifacts, f, indent=2)

    total_orig = sum(a["original_bytes"] for a in artifacts.values())
    total_comp = sum(a["compressed_bytes"] for a in artifacts.values())
    total_ratio = total_orig / max(total_comp, 1)
    savings_pct = (1 - total_comp / max(total_orig, 1)) * 100

    log.info(f"\nCompression Summary:")
    log.info(f"  Layers compressed: {len(artifacts)}")
    log.info(f"  Total original:    {total_orig:,} bytes")
    log.info(f"  Total compressed:  {total_comp:,} bytes")
    log.info(f"  Overall ratio:     {total_ratio:.3f}x (saved {savings_pct:.1f}%)")


def compress_from_state_dict(state_dict: dict, artifact_dir: str, layer_patterns: list):
    """Compress layers directly from a state dict (for pipeline use)."""
    artifacts = {}
    for name, param in state_dict.items():
        if not any(p in name for p in layer_patterns):
            continue
        if param.dtype != torch.bfloat16:
            param = param.to(torch.bfloat16)

        artifact = compress_weight(param, name, artifact_dir)
        artifacts[name] = str(Path(artifact_dir) / name.replace(".", "_"))

    return artifacts


def main():
    parser = argparse.ArgumentParser(description="Compress extracted BF16 weights")
    parser.add_argument("--extracted-dir", default="outputs/extracted",
                        help="Directory containing extracted .pt files")
    parser.add_argument("--artifact-dir", default="outputs/artifacts",
                        help="Output directory for compressed artifacts")
    parser.add_argument("--layers", nargs="*", default=None,
                        help="Layer name patterns to compress (default: all)")
    args = parser.parse_args()

    compress_from_extracted(args.extracted_dir, args.artifact_dir, args.layers)


if __name__ == "__main__":
    main()
