"""End-to-end device-side Huffman decode: compress -> device decode -> verify -> report.

Usage:
    python scripts/run_device_decode.py [--artifact-dir DIR] [--generate] [--shape ROWS COLS]

If --generate is passed, creates a fresh test artifact with dummy weights.
Otherwise, uses the existing artifact at --artifact-dir.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def generate_test_artifact(out_dir: str, rows: int, cols: int) -> str:
    """Generate a compressed artifact from random BF16 weights."""
    import torch
    from df11_tt.compress.compressor import compress_weight

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(42)
    weight = torch.randn(rows, cols).to(torch.bfloat16)
    artifact = compress_weight(weight, "test_layer", out_dir, verbose=True)

    weight_path = os.path.join(out_dir, "test_layer_original.pt")
    torch.save(weight, weight_path)

    return os.path.join(out_dir, "test_layer")


def find_binary():
    """Locate the compiled df11_device_decode binary."""
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'device_decode', 'build', 'df11_device_decode'),
        os.path.join(os.path.dirname(__file__), '..', 'build', 'df11_device_decode'),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def run_device_decode(binary_path: str, artifact_dir: str, output_file: str = ""):
    """Run the C++ device decode binary and return its exit code."""
    env = os.environ.copy()
    if 'TT_METAL_HOME' not in env:
        env['TT_METAL_HOME'] = os.path.expanduser('~/tt-metal')
    if 'TT_METAL_RUNTIME_ROOT' not in env:
        env['TT_METAL_RUNTIME_ROOT'] = env['TT_METAL_HOME']

    cmd = [binary_path, artifact_dir]
    if output_file:
        cmd.append(output_file)

    print(f"\nRunning: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, env=env, capture_output=False)
    return result.returncode


def verify_against_python(artifact_dir: str, output_file: str):
    """Compare device-decoded BF16 output against Python CPU decode."""
    import torch
    import numpy as np
    from df11_tt.runtime.decoder import decode_artifact

    if not output_file or not os.path.exists(output_file):
        print("Skipping Python cross-verification (no output file)")
        return

    print("\n--- Python Cross-Verification ---")

    cpu_tensor = decode_artifact(artifact_dir, verify=True, verbose=True)
    cpu_raw = cpu_tensor.view(torch.int16).numpy().astype(np.uint16)

    device_raw = np.fromfile(output_file, dtype=np.uint16)

    if len(cpu_raw.flat) != len(device_raw):
        print(f"SIZE MISMATCH: cpu={len(cpu_raw.flat)}, device={len(device_raw)}")
        return

    mismatches = np.sum(cpu_raw.flat != device_raw)
    print(f"  Python vs Device comparison: {mismatches} mismatches out of {len(device_raw)} elements")
    if mismatches == 0:
        print("  VERIFIED: Python CPU decode matches device decode exactly")
    else:
        first_idx = np.where(cpu_raw.flat != device_raw)[0][0]
        print(f"  FAILED: First mismatch at index {first_idx}: "
              f"cpu=0x{cpu_raw.flat[first_idx]:04x}, device=0x{device_raw[first_idx]:04x}")


def print_metrics_summary(artifact_dir: str):
    """Print a summary from the device decode metrics JSON."""
    metrics_path = os.path.join(artifact_dir, "device_decode_metrics.json")
    if not os.path.exists(metrics_path):
        print("No metrics file found")
        return

    with open(metrics_path) as f:
        m = json.load(f)

    print("\n" + "=" * 60)
    print("DEVICE DECODE METRICS SUMMARY")
    print("=" * 60)
    print(f"  Layer:              {m['layer_name']}")
    print(f"  Elements:           {m['num_elements']:,}")
    print(f"  Compressed:         {m['compressed_bytes'] / 1e6:.2f} MB")
    print(f"  Output:             {m['output_bytes'] / 1e6:.2f} MB")
    print(f"  Compression ratio:  {m['compression_ratio']:.4f}x")
    print(f"  Bit-exact:          {'YES' if m['bit_exact'] else 'NO'}")
    print(f"  ---")
    print(f"  CPU decode:         {m['cpu_decode_ms']:.1f} ms ({m['cpu_bandwidth_mbs']:.1f} MB/s)")
    print(f"  Device decode:      {m['kernel_execution_ms']:.1f} ms ({m['device_bandwidth_mbs']:.1f} MB/s)")
    print(f"  DRAM upload:        {m['dram_upload_ms']:.1f} ms")
    print(f"  DRAM readback:      {m['dram_readback_ms']:.1f} ms")
    print(f"  Total wall clock:   {m['total_ms']:.1f} ms")
    if m['cpu_decode_ms'] > 0:
        ratio = m['cpu_decode_ms'] / m['kernel_execution_ms']
        print(f"  Device/CPU speed:   {ratio:.2f}x {'(device faster)' if ratio > 1 else '(CPU faster)'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DFloat11 device-side decode end-to-end")
    parser.add_argument("--artifact-dir", type=str, default=None,
                        help="Path to existing compressed artifact directory")
    parser.add_argument("--generate", action="store_true",
                        help="Generate a fresh test artifact from random weights")
    parser.add_argument("--shape", type=int, nargs=2, default=[1024, 64],
                        help="Shape of test weight tensor (rows cols)")
    parser.add_argument("--output-file", type=str, default="",
                        help="Path to write decoded BF16 binary for cross-verification")
    parser.add_argument("--skip-python-verify", action="store_true",
                        help="Skip Python cross-verification")
    args = parser.parse_args()

    binary = find_binary()
    if binary is None:
        print("ERROR: Cannot find df11_device_decode binary.")
        print("Build it first: cd device_decode && cmake -B build -S . "
              "-DCMAKE_CXX_COMPILER=clang++-20 -DCMAKE_C_COMPILER=clang-20 "
              "-DCMAKE_PREFIX_PATH=\"$TT_METAL_HOME/build/lib/cmake;"
              "$TT_METAL_HOME/build_Release/share/cmake\" && cmake --build build")
        return 1

    print(f"Using binary: {binary}")

    if args.generate:
        project_root = os.path.join(os.path.dirname(__file__), '..')
        out_dir = os.path.join(project_root, 'outputs', 'test_artifacts')
        artifact_dir = generate_test_artifact(out_dir, args.shape[0], args.shape[1])
    elif args.artifact_dir:
        artifact_dir = args.artifact_dir
    else:
        default = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'artifacts', 'lm_head_weight')
        if os.path.exists(default):
            artifact_dir = default
        else:
            print("ERROR: No artifact directory specified. Use --artifact-dir or --generate")
            return 1

    artifact_dir = os.path.abspath(artifact_dir)
    print(f"Artifact directory: {artifact_dir}")

    output_file = args.output_file
    if output_file:
        output_file = os.path.abspath(output_file)

    rc = run_device_decode(binary, artifact_dir, output_file)

    if rc != 0:
        print(f"\nDevice decode FAILED with exit code {rc}")
        return rc

    if output_file and not args.skip_python_verify:
        verify_against_python(artifact_dir, output_file)

    print_metrics_summary(artifact_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
