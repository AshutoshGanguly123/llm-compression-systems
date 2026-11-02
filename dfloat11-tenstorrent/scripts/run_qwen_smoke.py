#!/usr/bin/env python3
"""One-shot Qwen generation on TT via QwenTTRunner (tt_transformers).

Example::

    export TT_METAL_HOME=/path/to/tt-metal
    export PYTHONPATH=$TT_METAL_HOME:$PWD/src
    python scripts/run_qwen_smoke.py --prompt "Say hello in one line."

    python scripts/run_qwen_smoke.py --prompts-json data/qwen_poc_prompts.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from df11_tt.serve import QwenTTRunner
from df11_tt.utils import get_logger

log = get_logger("run_qwen_smoke")


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "configs/qwen_poc.yaml")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompts-json", type=Path, default=None, help="JSON with {\"prompts\": [\"...\", ...]}")
    parser.add_argument("--tt-metal-home", default=None)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument(
        "--no-fabric",
        action="store_true",
        help="Disable fabric (single-device meshes only; overrides config runner.fabric)",
    )
    args = parser.parse_args()

    tt_home = args.tt_metal_home or os.environ.get("TT_METAL_HOME") or str(Path.home() / "tt-metal")
    cfg = _load_config(args.config) if args.config.is_file() else {}
    runner_cfg = cfg.get("runner", {})
    gen_cfg = cfg.get("generation", {})

    hf_model = args.hf_model or runner_cfg.get("hf_model") or os.environ.get(
        "HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"
    )

    prompts: list[str] = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompts_json:
        with open(args.prompts_json, encoding="utf-8") as f:
            data = json.load(f)
        prompts.extend(data.get("prompts", []))
    if not prompts:
        prompts = ["Give a one-sentence summary of what a transformer is."]

    phys = runner_cfg.get("physical_device_ids")
    if phys is not None and not isinstance(phys, list):
        phys = None

    use_fabric = bool(runner_cfg.get("fabric", True))
    if args.no_fabric:
        use_fabric = False

    log.info("Loading model %s ...", hf_model)
    runner = QwenTTRunner(
        tt_metal_home=tt_home,
        hf_model=hf_model,
        mesh_rows=int(runner_cfg.get("mesh_rows", 1)),
        mesh_cols=runner_cfg.get("mesh_cols"),
        instruct=bool(runner_cfg.get("instruct", True)),
        max_seq_len=int(runner_cfg.get("max_seq_len", 4096)),
        max_batch_size=int(runner_cfg.get("max_batch_size", 1)),
        data_parallel=int(runner_cfg.get("data_parallel", 1)),
        paged_attention=bool(runner_cfg.get("paged_attention", True)),
        page_block_size=int(runner_cfg.get("page_block_size", 32)),
        page_max_num_blocks_per_dp=int(runner_cfg.get("page_max_num_blocks_per_dp", 1024)),
        use_prefetcher=bool(runner_cfg.get("use_prefetcher", False)),
        num_layers=runner_cfg.get("num_layers"),
        physical_device_ids=phys,
        fabric=use_fabric,
    )

    try:
        max_new = int(gen_cfg.get("max_new_tokens", 64))
        temperature = float(gen_cfg.get("temperature", 0.0))
        top_p = float(gen_cfg.get("top_p", 0.08))
        top_k = int(gen_cfg.get("top_k", 32))
        stop_at_eos = bool(gen_cfg.get("stop_at_eos", True))

        bs = runner.max_batch_size
        global_idx = 0
        for off in range(0, len(prompts), bs):
            chunk = prompts[off : off + bs]
            results = runner.generate_batch(
                chunk,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_at_eos=stop_at_eos,
            )
            for r in results:
                print(f"--- [{global_idx}] ---")
                print(f"PROMPT: {r.prompt!r}")
                print(r.text)
                print(f"(decode iterations: {r.num_decode_iterations})")
                global_idx += 1
    finally:
        runner.close()


if __name__ == "__main__":
    main()
