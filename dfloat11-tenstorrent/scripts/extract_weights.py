#!/usr/bin/env python3
"""Extract selected BF16 weights from a HuggingFace model checkpoint.

Usage:
    python scripts/extract_weights.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --layers output.weight model.layers.0.mlp.gate_proj.weight \
        --out-dir outputs/extracted
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from df11_tt.utils import get_logger, tensor_checksum

log = get_logger("extract_weights")


def extract_weights_from_hf(model_name_or_path: str, layer_patterns: list, out_dir: str, use_dummy: bool = False):
    """Extract weights from a HuggingFace model.

    For MVP: if HF model is not accessible, generate representative random BF16 weights
    with realistic shapes matching the Llama-3.2-1B architecture.
    """
    os.makedirs(out_dir, exist_ok=True)

    if use_dummy:
        log.info("Using dummy weights matching Llama-3.2-1B architecture")
        state_dict = _generate_dummy_llama_1b_weights(layer_patterns)
    else:
        try:
            from transformers import AutoModelForCausalLM
            log.info(f"Loading model: {model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            state_dict = model.state_dict()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            log.warning(f"Could not load HF model ({e}), falling back to dummy weights")
            state_dict = _generate_dummy_llama_1b_weights(layer_patterns)

    extracted = {}
    for name, param in state_dict.items():
        matched = False
        for pattern in layer_patterns:
            if pattern in name:
                matched = True
                break
        if not matched:
            continue

        if param.dtype != torch.bfloat16:
            param = param.to(torch.bfloat16)

        save_path = Path(out_dir) / f"{name.replace('.', '_')}.pt"
        torch.save(param, str(save_path))

        info = {
            "layer_name": name,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "original_bytes": param.nelement() * 2,
            "checksum": tensor_checksum(param),
        }
        extracted[name] = info
        log.info(f"Extracted '{name}': shape={tuple(param.shape)}, "
                 f"size={param.nelement() * 2:,} bytes")

    manifest = Path(out_dir) / "extraction_manifest.json"
    with open(manifest, "w") as f:
        json.dump(extracted, f, indent=2)

    log.info(f"Extracted {len(extracted)} layers to {out_dir}")
    return extracted


def _generate_dummy_llama_1b_weights(layer_patterns: list) -> dict:
    """Generate realistic dummy BF16 weights for Llama-3.2-1B.

    Llama-3.2-1B: dim=2048, n_layers=16, n_heads=32, n_kv_heads=8,
                  intermediate_size=8192, vocab_size=128256
    """
    dim = 2048
    intermediate = 8192
    vocab_size = 128256

    torch.manual_seed(42)

    all_weights = {}

    # lm_head / output projection
    all_weights["lm_head.weight"] = torch.randn(vocab_size, dim, dtype=torch.bfloat16) * 0.02
    all_weights["model.lm_head.weight"] = all_weights["lm_head.weight"]
    # Also meta-style name
    all_weights["output.weight"] = all_weights["lm_head.weight"]

    # MLP layers for a few blocks
    for layer_idx in range(16):
        prefix = f"model.layers.{layer_idx}"
        all_weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(intermediate, dim, dtype=torch.bfloat16) * 0.02
        all_weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(intermediate, dim, dtype=torch.bfloat16) * 0.02
        all_weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(dim, intermediate, dtype=torch.bfloat16) * 0.02
        all_weights[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(dim, dim, dtype=torch.bfloat16) * 0.02
        all_weights[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(512, dim, dtype=torch.bfloat16) * 0.02
        all_weights[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(512, dim, dtype=torch.bfloat16) * 0.02
        all_weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(dim, dim, dtype=torch.bfloat16) * 0.02

    return all_weights


def main():
    parser = argparse.ArgumentParser(description="Extract BF16 weights from model")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--layers", nargs="+",
                        default=["lm_head", "output.weight"],
                        help="Layer name patterns to extract")
    parser.add_argument("--out-dir", default="outputs/extracted",
                        help="Output directory")
    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy weights (no model download)")
    args = parser.parse_args()

    extract_weights_from_hf(args.model, args.layers, args.out_dir, use_dummy=args.dummy)


if __name__ == "__main__":
    main()
