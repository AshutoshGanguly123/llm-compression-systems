#!/usr/bin/env python3
"""Run baseline TT inference: load weights normally, execute linear, collect metrics.

This script demonstrates the baseline (uncompressed) path:
  model weights -> ttnn.from_torch -> TT device -> ttnn.linear -> metrics

Usage:
    python scripts/run_baseline.py \
        --report outputs/metrics/baseline_run.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.utils import get_logger, Timer, tensor_checksum
from df11_tt.eval.metrics import MetricsReport
from df11_tt.eval.reporters import save_metrics, print_summary

log = get_logger("run_baseline")

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", str(Path.home() / "tt-metal"))
sys.path.insert(0, TT_METAL_HOME)


def run_baseline_tt_linear(
    weight: torch.Tensor,
    activation: torch.Tensor,
    layer_name: str,
    device_id: int = 0,
    num_runs: int = 3,
    warmup_runs: int = 1,
    verbose: bool = True,
) -> MetricsReport:
    """Run baseline TT linear operation with dense (uncompressed) weights.

    Args:
        weight: BF16 weight tensor [out_features, in_features]
        activation: BF16 input activation [batch, seq_len, in_features]
        layer_name: name of the layer being tested
        device_id: TT device ID
        num_runs: number of timed runs
        warmup_runs: warmup iterations
        verbose: print details

    Returns:
        MetricsReport with baseline metrics
    """
    import ttnn

    original_bytes = weight.nelement() * 2
    weight_T = weight.t().contiguous()  # [in_features, out_features] for linear

    device = ttnn.open_device(device_id=device_id)

    try:
        if verbose:
            log.info(f"Baseline: uploading weight {tuple(weight_T.shape)} to TT device {device_id}")

        with Timer() as t_upload:
            tt_weight = ttnn.from_torch(
                weight_T,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if verbose:
            log.info(f"  Weight upload: {t_upload.elapsed_ms:.1f}ms")

        with Timer() as t_act_upload:
            tt_activation = ttnn.from_torch(
                activation,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if verbose:
            log.info(f"  Activation upload: {t_act_upload.elapsed_ms:.1f}ms")

        for i in range(warmup_runs):
            _ = ttnn.linear(tt_activation, tt_weight)
            ttnn.synchronize_device(device)

        if verbose:
            log.info(f"  Warmup: {warmup_runs} runs complete")

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
            log.info(f"  Output shape: {tuple(result_torch.shape)}")

        ttnn.deallocate(tt_weight)
        ttnn.deallocate(tt_activation)
        ttnn.deallocate(output)

    finally:
        ttnn.close_device(device)

    report = MetricsReport(
        model_name="Llama-3.2-1B-Instruct",
        mode="baseline",
        layer_name=layer_name,
        original_bytes=original_bytes,
        compressed_bytes=original_bytes,
        compression_ratio=1.0,
        bytes_saved=0,
        tt_op_ms=avg_op_ms,
        total_layer_ms=t_upload.elapsed_ms + avg_op_ms,
        baseline_e2e_ms=t_upload.elapsed_ms + avg_op_ms,
        bit_exact=True,
    )

    return report, result_torch


def run_baseline_cpu_linear(
    weight: torch.Tensor,
    activation: torch.Tensor,
    layer_name: str,
    num_runs: int = 3,
    verbose: bool = True,
) -> MetricsReport:
    """CPU-only baseline for comparison (no TT device needed)."""
    original_bytes = weight.nelement() * 2

    op_times = []
    for i in range(num_runs):
        with Timer() as t_op:
            output = torch.matmul(activation.float(), weight.t().float()).to(torch.bfloat16)
        op_times.append(t_op.elapsed_ms)

    avg_op_ms = sum(op_times) / len(op_times)
    if verbose:
        log.info(f"CPU baseline: avg {avg_op_ms:.2f}ms over {num_runs} runs")

    report = MetricsReport(
        model_name="Llama-3.2-1B-Instruct",
        mode="baseline",
        layer_name=layer_name,
        original_bytes=original_bytes,
        compressed_bytes=original_bytes,
        compression_ratio=1.0,
        bytes_saved=0,
        tt_op_ms=avg_op_ms,
        total_layer_ms=avg_op_ms,
        baseline_e2e_ms=avg_op_ms,
        bit_exact=True,
    )
    return report, output


def main():
    parser = argparse.ArgumentParser(description="Run baseline TT linear benchmark")
    parser.add_argument("--report", default="outputs/metrics/baseline_run.json")
    parser.add_argument("--layer-name", default="lm_head.weight")
    parser.add_argument("--extracted-dir", default="outputs/extracted")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--cpu-only", action="store_true", help="Run CPU-only baseline")
    parser.add_argument("--dummy", action="store_true", help="Use dummy weights")
    args = parser.parse_args()

    if args.dummy:
        log.info("Generating dummy weights for baseline test")
        dim = 2048
        vocab_size = 128256
        weight = torch.randn(vocab_size, dim, dtype=torch.bfloat16) * 0.02
        layer_name = "lm_head.weight"
    else:
        weight_path = Path(args.extracted_dir) / f"{args.layer_name.replace('.', '_')}.pt"
        if not weight_path.exists():
            log.error(f"Weight file not found: {weight_path}")
            log.info("Run extract_weights.py first, or use --dummy")
            sys.exit(1)
        weight = torch.load(str(weight_path), map_location="cpu", weights_only=True)
        layer_name = args.layer_name

    if weight.dtype != torch.bfloat16:
        weight = weight.to(torch.bfloat16)

    # Pad weight dims to be tile-aligned (multiples of 32) for TT
    out_features, in_features = weight.shape
    pad_out = (32 - out_features % 32) % 32
    pad_in = (32 - in_features % 32) % 32
    if pad_out > 0 or pad_in > 0:
        weight_padded = torch.zeros(out_features + pad_out, in_features + pad_in, dtype=torch.bfloat16)
        weight_padded[:out_features, :in_features] = weight
        weight = weight_padded
        log.info(f"Padded weight to tile-aligned: {tuple(weight.shape)}")

    batch, seq_len = 1, 32
    activation = torch.randn(batch, seq_len, weight.shape[1], dtype=torch.bfloat16) * 0.1

    if args.cpu_only:
        report, output = run_baseline_cpu_linear(
            weight, activation, layer_name,
            num_runs=args.num_runs,
        )
    else:
        report, output = run_baseline_tt_linear(
            weight, activation, layer_name,
            device_id=args.device_id,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
        )

    report.extra["output_checksum"] = tensor_checksum(output)
    print_summary(report)
    save_metrics(report, args.report)


if __name__ == "__main__":
    main()
