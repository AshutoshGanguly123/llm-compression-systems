"""Create a small test artifact for device decode verification."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from df11_tt.compress.compressor import compress_weight

def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'test_artifacts')
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(42)
    # Small test tensor: 1024 x 64 = 65536 elements
    weight = torch.randn(1024, 64).to(torch.bfloat16)

    artifact = compress_weight(weight, "test_layer", out_dir, verbose=True)
    print(f"\nTest artifact created at: {out_dir}/test_layer")
    print(f"  Elements: {artifact.num_elements}")
    print(f"  Original bytes: {artifact.original_bytes}")
    print(f"  Compressed bytes: {artifact.compressed_bytes}")
    print(f"  Ratio: {artifact.compression_ratio}")

    # Also save the original weight for verification
    weight_path = os.path.join(out_dir, "test_layer_original.pt")
    torch.save(weight, weight_path)
    print(f"  Original weight saved to: {weight_path}")

if __name__ == "__main__":
    main()
