"""Reporting utilities: save metrics to JSON, print summaries, compare runs."""

import json
from pathlib import Path

from ..utils.logging import get_logger
from .metrics import MetricsReport, ComparisonReport

log = get_logger("reporters")


def save_metrics(report: MetricsReport, path: str):
    """Save a metrics report to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    log.info(f"Saved metrics to {path}")


def print_summary(report: MetricsReport):
    """Print a human-readable summary of a metrics report."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  METRICS REPORT: {report.mode.upper()}")
    print(f"{sep}")
    print(f"  Model:            {report.model_name}")
    print(f"  Layer:            {report.layer_name}")

    if report.mode == "compressed" or report.compression_ratio > 0:
        print(f"  Original bytes:   {report.original_bytes:,}")
        print(f"  Compressed bytes: {report.compressed_bytes:,}")
        print(f"  Compression:      {report.compression_ratio:.3f}x "
              f"(saved {report.bytes_saved:,} bytes, "
              f"{(1 - report.compressed_bytes / max(report.original_bytes, 1)) * 100:.1f}%)")
        print(f"  Decode time:      {report.decode_ms:.2f} ms")
        print(f"  Upload time:      {report.upload_ms:.2f} ms")
        print(f"  TT op time:       {report.tt_op_ms:.2f} ms")
        print(f"  Total layer:      {report.total_layer_ms:.2f} ms")

    if report.baseline_e2e_ms > 0:
        print(f"  Baseline E2E:     {report.baseline_e2e_ms:.2f} ms")
    if report.compressed_e2e_ms > 0:
        print(f"  Compressed E2E:   {report.compressed_e2e_ms:.2f} ms")
    if report.tokens_per_sec is not None:
        print(f"  Tokens/sec:       {report.tokens_per_sec:.2f}")
    if report.ttft_ms is not None:
        print(f"  TTFT:             {report.ttft_ms:.2f} ms")
    if report.bit_exact is not None:
        print(f"  Bit-exact:        {'PASS' if report.bit_exact else 'FAIL'}")
    if report.token_match is not None:
        print(f"  Token match:      {'PASS' if report.token_match else 'FAIL'}")
    print(f"{sep}\n")


def compare_runs(baseline_path: str, compressed_path: str) -> ComparisonReport:
    """Compare baseline and compressed run metrics."""
    with open(baseline_path) as f:
        baseline = MetricsReport(**{k: v for k, v in json.load(f).items()
                                    if k in MetricsReport.__dataclass_fields__})
    with open(compressed_path) as f:
        compressed = MetricsReport(**{k: v for k, v in json.load(f).items()
                                      if k in MetricsReport.__dataclass_fields__})

    overhead_ms = compressed.total_layer_ms - baseline.total_layer_ms
    overhead_pct = (overhead_ms / max(baseline.total_layer_ms, 0.001)) * 100

    comparison = ComparisonReport(
        baseline=baseline,
        compressed=compressed,
        overhead_ms=overhead_ms,
        overhead_pct=overhead_pct,
    )

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  COMPARISON REPORT")
    print(f"{sep}")
    print(f"  Compression ratio:  {compressed.compression_ratio:.3f}x")
    print(f"  Bytes saved:        {compressed.bytes_saved:,}")
    print(f"  Decode overhead:    {compressed.decode_ms:.2f} ms")
    print(f"  Upload overhead:    {compressed.upload_ms:.2f} ms")
    print(f"  Total overhead:     {overhead_ms:.2f} ms ({overhead_pct:.1f}%)")
    if compressed.bit_exact is not None:
        print(f"  Bit-exact:          {'PASS' if compressed.bit_exact else 'FAIL'}")
    print(f"{sep}\n")

    return comparison
