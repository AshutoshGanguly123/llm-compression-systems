#!/usr/bin/env python3
"""Minimal HTTP PoC: load Qwen (tt_transformers) on TT mesh and serve JSON generation.

Run from a tt-metal Python environment with devices visible::

    export TT_METAL_HOME=/path/to/tt-metal
    export PYTHONPATH=$TT_METAL_HOME:$PWD/src
    python scripts/serve_qwen_poc.py --config configs/qwen_poc.yaml

POST /generate with body::

    {"prompt": "Hello", "max_new_tokens": 64, "temperature": 0.0, "top_p": 0.08, "top_k": 32}

Response::

    {"text": "...", "prompt": "...", "num_decode_iterations": N}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from df11_tt.serve import QwenTTRunner
from df11_tt.utils import get_logger

log = get_logger("serve_qwen_poc")

_runner: Optional[QwenTTRunner] = None
_gen_defaults: dict[str, Any] = {}


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "service": "qwen_tt_poc"}).encode())
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "invalid JSON"}).encode())
            return

        prompt = body.get("prompt")
        if not prompt or not isinstance(prompt, str):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "missing string field 'prompt'"}).encode())
            return

        max_new = int(body.get("max_new_tokens", _gen_defaults.get("max_new_tokens", 64)))
        temperature = float(body.get("temperature", _gen_defaults.get("temperature", 0.0)))
        top_p = float(body.get("top_p", _gen_defaults.get("top_p", 0.08)))
        top_k = int(body.get("top_k", _gen_defaults.get("top_k", 32)))
        stop_at_eos = bool(body.get("stop_at_eos", _gen_defaults.get("stop_at_eos", True)))

        assert _runner is not None
        try:
            result = _runner.generate(
                prompt,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_at_eos=stop_at_eos,
            )
        except Exception as e:
            log.exception("generate failed")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
            return

        out = {
            "text": result.text,
            "prompt": result.prompt,
            "num_decode_iterations": result.num_decode_iterations,
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())


def main() -> None:
    global _runner, _gen_defaults

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "configs/qwen_poc.yaml")
    parser.add_argument("--host", default=None, help="Bind address (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Port (overrides config)")
    parser.add_argument("--tt-metal-home", default=None, help="Override TT_METAL_HOME")
    args = parser.parse_args()
    tt_home = args.tt_metal_home or os.environ.get("TT_METAL_HOME")
    if not tt_home:
        tt_home = str(Path.home() / "tt-metal")

    cfg = _load_config(args.config) if args.config.is_file() else {}
    runner_cfg = cfg.get("runner", {})
    serve_cfg = cfg.get("serve", {})
    _gen_defaults = cfg.get("generation", {})

    hf_model = runner_cfg.get("hf_model") or os.environ.get("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    host = args.host or serve_cfg.get("host", "0.0.0.0")
    port = int(args.port or serve_cfg.get("port", 8765))

    phys = runner_cfg.get("physical_device_ids")
    if phys is not None and not isinstance(phys, list):
        phys = None

    log.info("Loading TT model %s (this may take several minutes)...", hf_model)
    _runner = QwenTTRunner(
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
        fabric=bool(runner_cfg.get("fabric", True)),
    )

    server = ThreadingHTTPServer((host, port), Handler)
    log.info("Listening on http://%s:%s  POST /generate", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
    finally:
        server.server_close()
        _runner.close()


if __name__ == "__main__":
    main()
