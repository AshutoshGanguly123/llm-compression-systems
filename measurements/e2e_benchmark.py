#!/usr/bin/env python3
"""Host-side e2e timing scaffold for compression path stages."""
from __future__ import annotations
import argparse, json, os, time, resource, tracemalloc
from pathlib import Path

def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux ru_maxrss is KiB
    return float(usage) / 1024.0

def stage(name: str, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "stage": name,
        "wall_s": wall,
        "tracemalloc_peak_mb": peak / (1024 * 1024),
        "rss_peak_mb": peak_rss_mb(),
        "result": out,
    }

def fake_decode(n_elems: int = 1_000_000):
    # CPU stand-in: allocate and touch memory proportional to a small decode
    buf = bytearray(n_elems)
    for i in range(0, n_elems, 4096):
        buf[i] = 1
    return {"elems": n_elems}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["decode_once", "repeated_decode"], default="decode_once")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--elems", type=int, default=1_000_000)
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    report = {"scenario": a.scenario, "hardware": "host_scaffold", "stages": [], "device_stages": "skipped_no_tt"}
    report["stages"].append(stage("load_stub", lambda: {"ok": True}))
    if a.scenario == "decode_once":
        report["stages"].append(stage("decode", lambda: fake_decode(a.elems)))
    else:
        for i in range(a.repeats):
            report["stages"].append(stage(f"decode_{i}", lambda: fake_decode(a.elems)))
    report["stages"].append({"stage": "transfer", "status": "skipped_no_tt"})
    report["stages"].append({"stage": "compute", "status": "skipped_no_tt"})
    report["note"] = (
        "Scaffold timings only. Replace fake_decode with df11 Huffman decode and "
        "enable TT transfer/compute on a TT node; compare to ~115s PoC on 262M elems."
    )
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
