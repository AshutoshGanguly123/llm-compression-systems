#!/usr/bin/env python3
"""Compare baseline and compressed run metrics.

Usage:
    python scripts/compare_runs.py \
        --baseline outputs/metrics/baseline_run.json \
        --compressed outputs/metrics/compressed_run.json \
        --report outputs/metrics/comparison.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.eval.reporters import compare_runs
from df11_tt.utils import get_logger

log = get_logger("compare_runs")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and compressed metrics")
    parser.add_argument("--baseline", default="outputs/metrics/baseline_run.json")
    parser.add_argument("--compressed", default="outputs/metrics/compressed_run.json")
    parser.add_argument("--report", default="outputs/metrics/comparison.json")
    args = parser.parse_args()

    if not Path(args.baseline).exists():
        log.error(f"Baseline metrics not found: {args.baseline}")
        sys.exit(1)
    if not Path(args.compressed).exists():
        log.error(f"Compressed metrics not found: {args.compressed}")
        sys.exit(1)

    comparison = compare_runs(args.baseline, args.compressed)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(comparison.to_dict(), f, indent=2)

    log.info(f"Saved comparison report to {args.report}")


if __name__ == "__main__":
    main()
