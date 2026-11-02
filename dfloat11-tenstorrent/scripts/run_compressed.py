#!/usr/bin/env python3
"""Run compressed TT inference: decode artifact -> upload -> TT linear -> metrics.

Usage:
    python scripts/run_compressed.py \
        --artifact-dir outputs/artifacts \
        --layer-target lm_head.weight \
        --report outputs/metrics/compressed_run.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.utils import get_logger, Timer, tensor_checksum
from df11_tt.utils.hashing import verify_bit_exact
from df11_tt.compress.artifact import load_artifact
from df11_tt.runtime.decoder import decode_artifact
from df11_tt.eval.metrics import MetricsReport
from df11_tt.eval.reporters import save_metrics, print_summary

log = get_logger("run_compressed")

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", str(Path.home() / "tt-metal"))
sys.path.insert(0, TT_METAL_HOME)


def find_artifact_path(artifact_dir, layer_name):
    safe_name = layer_name.replace(".", "_")
    candidate = Path(artifact_dir) / safe_name
    if candidate.exists():
        return str(candidate)
    for manifest_name in ["compression_summary.json", "manifest.json"]:
        mpath = Path(artifact_dir) / manifest_name
        if mpath.exists():
            with open(mpath) as f:
                data = json.load(f)
            for name, info in data.items():
                if layer_name in name or safe_name in name:
                    if isinstance(info, dict):
                        return info.get("artifact_path", "")
                    return info
    raise FileNotFoundError(f"No artifact found for '{layer_name}' in {artifact_dir}")


def run_compressed_tt_linear(
    artifact_path, activation, layer_name,
    original_weight=None, device_id=0, num_runs=3,
    warmup_runs=1, verify=True, verbose=True,
):
    import ttnn

    with Timer() as t_decode:
        decoded_weight = decode_artifact(artifact_path, verify=verify, verbose=verbose)

    bit_exact = None
    if original_weight is not None:
        bit_exact = verify_bit_exact(original_weight, decoded_weight)
        if verbose:
            log.info(f"  Bit-exact verification: {'PASS' if bit_exact else 'FAIL'}")

    out_f, in_f = decoded_weight.shape
    pad_out = (32 - out_f % 32) % 32
    pad_in = (32 - in_f % 32) % 32
    if pad_out > 0 or pad_in > 0:
        w_padded = torch.zeros(out_f + pad_out, in_f + pad_in, dtype=torch.bfloat16)
        w_padded[:out_f, :in_f] = decoded_weight
        decoded_weight = w_padded

    weight_T = decoded_weight.t().contiguous()
    device = ttnn.open_device(device_id=device_id)

    try:
        with Timer() as t_upload:
            tt_weight = ttnn.from_torch(
                weight_T, device=device, dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if verbose:
            log.info(f"  Weight upload to TT: {t_upload.elapsed_ms:.1f}ms")

        act_padded = activation
        if act_padded.shape[-1] != decoded_weight.shape[1]:
            act_padded = torch.zeros(*activation.shape[:-1], decoded_weight.shape[1], dtype=torch.bfloat16)
            act_padded[..., :activation.shape[-1]] = activation

        tt_activation = ttnn.from_torch(
            act_padded, device=device, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        for i in range(warmup_runs):
            _ = ttnn.linear(tt_activation, tt_weight)
            ttnn.synchronize_device(device)

        op_times = []
        for i in range(num_runs):
            with Timer() as t_op:
                output = ttnn.linear(tt_activation, tt_weight)
                ttnn.synchronize_device(device)
            op_times.append(t_op.elapsed_ms)

        avg_op_ms = sum(op_times) / len(op_times)
        result_torch = ttnn.to_torch(output)
        if verbose:
            log.info(f"  TT op avg: {avg_op_ms:.2f}ms over {num_runs} runs")

        ttnn.deallocate(tt_weight)
        ttnn.deallocate(tt_activation)
        ttnn.deallocate(output)
    finally:
        ttnn.close_device(device)

    artifact_meta, _, _, _ = load_artifact(artifact_path)
    report = MetricsReport(
        model_name="Llama-3.2-1B-Instruct",
        mode="compressed",
        layer_name=layer_name,
        original_bytes=artifact_meta.original_bytes,
        compressed_bytes=artifact_meta.compressed_bytes,
        compression_ratio=artifact_meta.compression_ratio,
        bytes_saved=artifact_meta.original_bytes - artifact_meta.compressed_bytes,
        decode_ms=t_decode.elapsed_ms,
        upload_ms=t_upload.elapsed_ms,
        tt_op_ms=avg_op_ms,
        total_layer_ms=t_decode.elapsed_ms + t_upload.elapsed_ms + avg_op_ms,
        compressed_e2e_ms=t_decode.elapsed_ms + t_upload.elapsed_ms + avg_op_ms,
        bit_exact=bit_exact,
    )
    return report, result_torch


def run_compressed_cpu_linear(
    artifact_path, activation, layer_name,
    original_weight=None, num_runs=3, verify=True, verbose=True,
):
    with Timer() as t_decode:
        decoded_weight = decode_artifact(artifact_path, verify=verify, verbose=verbose)

    bit_exact = None
    if original_weight is not None:
        bit_exact = verify_bit_exact(original_weight, decoded_weight)
        if verbose:
            log.info(f"  Bit-exact: {'PASS' if bit_exact else 'FAIL'}")

    op_times = []
    for i in range(num_runs):
        with Timer() as t_op:
            output = torch.matmul(activation.float(), decoded_weight.t().float()).to(torch.bfloat16)
        op_times.append(t_op.elapsed_ms)

    avg_op_ms = sum(op_times) / len(op_times)
    artifact_meta, _, _, _ = load_artifact(artifact_path)

    report = MetricsReport(
        model_name="Llama-3.2-1B-Instruct",
        mode="compressed",
        layer_name=layer_name,
        original_bytes=artifact_meta.original_bytes,
        compressed_bytes=artifact_meta.compressed_bytes,
        compression_ratio=artifact_meta.compression_ratio,
        bytes_saved=artifact_meta.original_bytes - artifact_meta.compressed_bytes,
        decode_ms=t_decode.elapsed_ms,
        upload_ms=0.0,
        tt_op_ms=avg_op_ms,
        total_layer_ms=t_decode.elapsed_ms + avg_op_ms,
        compressed_e2e_ms=t_decode.elapsed_ms + avg_op_ms,
        bit_exact=bit_exact,
    )
    return report, output


def main():
    parser = argparse.ArgumentParser(description="Run compressed TT linear benchmark")
    parser.add_argument("--artifact-dir", default="outputs/artifacts")
    parser.add_argument("--layer-target", default="lm_head.weight")
    parser.add_argument("--report", default="outputs/metrics/compressed_run.json")
    parser.add_argument("--extracted-dir", default="outputs/extracted")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    artifact_path = find_artifact_path(args.artifact_dir, args.layer_target)
    log.info(f"Using artifact: {artifact_path}")

    original_weight = None
    orig_path = Path(args.extracted_dir) / f"{args.layer_target.replace('.', '_')}.pt"
    if orig_path.exists():
        original_weight = torch.load(str(orig_path), map_location="cpu", weights_only=True)
        if original_weight.dtype != torch.bfloat16:
            original_weight = original_weight.to(torch.bfloat16)

    artifact_meta, _, _, _ = load_artifact(artifact_path)
    in_features = artifact_meta.shape[-1] if len(artifact_meta.shape) >= 2 else artifact_meta.shape[0]

    batch, seq_len = 1, 32
    pad_in = (32 - in_features % 32) % 32
    activation = torch.randn(batch, seq_len, in_features + pad_in, dtype=torch.bfloat16) * 0.1

    if args.cpu_only:
        report, output = run_compressed_cpu_linear(
            artifact_path, activation, args.layer_target,
            original_weight=original_weight,
            num_runs=args.num_runs, verify=not args.no_verify,
        )
    else:
        report, output = run_compressed_tt_linear(
            artifact_path, activation, args.layer_target,
            original_weight=original_weight,
            device_id=args.device_id, num_runs=args.num_runs,
            warmup_runs=args.warmup_runs, verify=not args.no_verify,
        )

    report.extra["output_checksum"] = tensor_checksum(output)
    print_summary(report)
    save_metrics(report, args.report)


if __name__ == "__main__":
    main()
