# DFloat11 on Tenstorrent: Friday MVP

**Lossless BF16 weight compression for Tenstorrent inference using DFloat11-style Huffman exponent coding.**

## What This Is

A proof-of-concept showing that DFloat11-style compressed weight artifacts can be:
1. Created offline from BF16 model weights
2. Decoded just-in-time on the TT machine host CPU
3. Uploaded to the TT device via `ttnn.from_torch`
4. Used in `ttnn.linear` compute with **bit-exact correctness**

The architecture validates the path: `compressed artifact -> host decode -> TT tensor upload -> TT compute -> metrics`

## Clone this repository (GitHub)

Large generated paths are **not** committed (GitHub size limits, hygiene). After `git clone`, you recreate them locally.

**Not in git (each machine regenerates):**

- `venv/` — do not commit; use Tenstorrent’s `tt-metal` Python when possible (`setup_env.sh`).
- `outputs/` — extracted `*.pt` weights (including **`lm_head_weight.pt`** / **`model_lm_head_weight.pt`**), Huffman **`exponent_bitstream.bin`**, **`sign_mantissa.npy`**, metrics JSON, PDF reports, test artifacts.
- `device_decode/build/` — compile with `scripts/build_device_decode.sh`.

**HuggingFace models** (e.g. Qwen in `configs/qwen_poc.yaml`, or real `extract_weights.py` runs) live in the **HF cache** (`~/.cache/huggingface/hub`), not in this repo. Cloners run generation or extraction once to download.

### Prerequisites

- **Tenstorrent stack:** clone and build [tt-metal](https://github.com/tenstorrent/tt-metal), set `TT_METAL_HOME` (see `setup_env.sh`).
- **Python:** 3.10+ with `torch`, `ttnn` (from tt-metal) for device scripts; for **CPU-only** tests use a local venv and `pip install -r requirements.txt` (includes `transformers` for HF extraction).

### Steps after `git clone`

```bash
git clone https://github.com/<your-org>/foais_project_dfloat11.git
cd foais_project_dfloat11

# Recommended on a TT machine: use tt-metal's venv (provides ttnn, torch, etc.)
source setup_env.sh

# CPU-only dev without TT: create a venv and install deps
# python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Regenerate extracted BF16 tensors + compressed artifacts (~525MB disk for dummy lm_head)
bash scripts/bootstrap_local_artifacts.sh
# Or real Llama weights (HF token if gated):
# bash scripts/bootstrap_local_artifacts.sh --hf meta-llama/Llama-3.2-1B-Instruct

# Optional: C++ device decode helper used by scripts/run_device_decode.py
# bash scripts/build_device_decode.sh

# Then follow Quick Start below (baseline / compressed / compare), tests, or Qwen PoC.
```

## Key Results

| Metric | Value |
|--------|-------|
| Compression ratio | **1.509x** (33.7% size reduction) |
| Exponent compression | **3.07x** (only 19 unique exponents) |
| Bit-exact reconstruction | **PASS** |
| TT output equivalence | **PASS** (baseline == compressed) |
| TT op time (baseline) | 1.41 ms |
| TT op time (compressed) | 1.37 ms |
| Host decode time (lm_head, 262M elements) | ~115s (Python Huffman) |

The decode overhead is expected for a pure-Python implementation. The DFloat11 paper uses GPU-accelerated block-parallel decoding; a future TT-Metalium kernel would eliminate this bottleneck.

## Quick Start

```bash
# 1. Activate the environment
source setup_env.sh

# 2. Extract weights (use --dummy for no model download)
python scripts/extract_weights.py --dummy --layers lm_head --out-dir outputs/extracted

# 3. Compress weights
python scripts/compress_weights.py \
    --extracted-dir outputs/extracted \
    --artifact-dir outputs/artifacts \
    --layers lm_head.weight

# 4. Run baseline TT benchmark
python scripts/run_baseline.py \
    --dummy --device-id 0 \
    --report outputs/metrics/baseline_run.json

# 5. Run compressed TT benchmark
python scripts/run_compressed.py \
    --artifact-dir outputs/artifacts \
    --layer-target lm_head.weight \
    --extracted-dir outputs/extracted \
    --device-id 0 \
    --report outputs/metrics/compressed_run.json

# 6. Compare results
python scripts/compare_runs.py \
    --baseline outputs/metrics/baseline_run.json \
    --compressed outputs/metrics/compressed_run.json \
    --report outputs/metrics/comparison.json
```

## Run Tests

```bash
source setup_env.sh

# CPU-only tests (codec, artifact, runtime)
python -m pytest tests/test_codec_roundtrip.py tests/test_artifact_schema.py tests/test_runtime_decode.py tests/test_model_patch_smoke.py -v

# TT device smoke tests
python -m pytest tests/test_tt_linear_smoke.py -v
```

## Qwen on Tenstorrent (tt_transformers PoC)

End-to-end causal LM inference uses the same stack as `tt-metal/models/tt_transformers/demo/simple_text_demo.py` (mesh, `Generator`, optional fabric). Use **`source setup_env.sh`** so Python comes from **`$TT_METAL_HOME/python_env`** (not the repo’s lightweight `venv/`).

```bash
cd /path/to/foais_project_dfloat11
source setup_env.sh
# Optional: huggingface-cli login if the checkpoint is gated

# One-shot generation (vanilla PoC: 1 device in qwen_poc.yaml)
python scripts/run_qwen_smoke.py --config configs/qwen_poc.yaml --prompt "Hello."
# If single-chip fabric init fails after a board reset: add --no-fabric

# Or a small prompt list (batched according to max_batch_size in yaml)
python scripts/run_qwen_smoke.py --prompts-json data/qwen_poc_prompts.json

# JSON HTTP API (optional; blocking load at startup)
python scripts/serve_qwen_poc.py --config configs/qwen_poc.yaml
# curl -s localhost:8765/generate -H 'Content-Type: application/json' \
#   -d '{"prompt":"Say hi in three words.","max_new_tokens":32}'
```

Defaults are in `configs/qwen_poc.yaml`. The runner is `src/df11_tt/serve/qwen_tt_runner.py`.

- **PoC target:** **one device** (`mesh_cols: 1`) for the simplest path. tt_transformers requires `n_kv_heads % mesh_width == 0`; Qwen2.5-1.5B has **two KV heads**, so mesh width **1 or 2** only — **not four**.
- **Physical devices:** set `runner.physical_device_ids` (e.g. `[0, 1]` for two chips) so the mesh uses a **contiguous** pair and avoids bad fabric subsets.
- **Fabric:** `runner.fabric: true` by default. For **single-chip** only, set `fabric: false` if ethernet/fabric init still fails after a **board reset** (multi-device runs keep fabric on).
- **DFloat11 / paper line:** this PoC proves **full-model TT generation** with stock weights. **Phase B** is to hook the existing compressed path (`compressed_linear`, `model_patch`) at **`lm_head`** once baseline decode is stable — no four-chip requirement for that demo.

**If `open_mesh_device` fails** with an **active ethernet core** timeout (even with `--no-fabric`), the stack is asking for a **board reset / power cycle** and a consistent tt-firmware bundle — that is a hardware bring-up issue, not the Python PoC logic.

## Architecture

```
project/
  setup_env.sh              # Environment activation script
  configs/                  # Model and eval YAML configs
  scripts/
    bootstrap_local_artifacts.sh  # After clone: extract + compress into outputs/
    build_device_decode.sh        # CMake build for device_decode/
    extract_weights.py      # FR-2: Extract BF16 weights from model
    compress_weights.py     # FR-3: Offline Huffman compression
    run_baseline.py         # FR-1: Baseline TT linear benchmark
    run_compressed.py       # FR-8: Compressed runtime path
    compare_runs.py         # FR-9: Metrics comparison
    run_qwen_smoke.py       # Qwen / tt_transformers one-shot generation
    serve_qwen_poc.py       # Minimal HTTP server for Qwen TT generation
  src/df11_tt/
    config.py               # Central configuration
    utils/
      logging.py            # Structured logging
      timing.py             # Context-manager timer
      hashing.py            # SHA-256 tensor checksums
    compress/
      bitops.py             # BF16 sign/exponent/mantissa decomposition
      huffman.py            # Huffman codec for exponent stream
      artifact.py           # Compressed artifact save/load
      compressor.py         # Offline compression pipeline
    runtime/
      decoder.py            # FR-5: Host-side CPU decoder
      tt_adapter.py         # FR-6: TT device upload adapter
      compressed_linear.py  # FR-7: Compressed linear wrapper
      model_patch.py        # FR-8: Model patching utilities
    eval/
      metrics.py            # Metrics data classes
      reporters.py          # JSON/console reporting
      prompts.py            # Standard eval prompts
    serve/
      qwen_tt_runner.py     # TT mesh + tt_transformers Generator wrapper
  tests/
    test_codec_roundtrip.py    # 12 tests: bitops, huffman, full roundtrip
    test_artifact_schema.py    # 5 tests: file format, metadata fields
    test_runtime_decode.py     # 7 tests: decode correctness, CPU linear
    test_tt_linear_smoke.py    # 2 tests: TT device upload + compute
    test_model_patch_smoke.py  # 4 tests: layer identification + patching
```

## How DFloat11 Compression Works

BF16 has 16 bits: `[sign (1)] [exponent (8)] [mantissa (7)]`

DFloat11 exploits the fact that BF16 model weights have **very low exponent entropy** -- most weights cluster around a few exponent values. For a typical LLM layer:

1. **Split** each BF16 value into sign, exponent, mantissa
2. **Huffman-encode** the exponent stream (achieves ~3x on exponents alone)
3. **Store** sign+mantissa as raw uint8 payload (1 sign bit + 7 mantissa bits packed)
4. At runtime, **decode** exponents and **reassemble** bit-exact BF16 values

The overall compression ratio depends on exponent entropy:
- Random BF16: ~1.1x (uniform exponents, little to compress)
- Real model weights: **1.3-1.6x** (highly clustered exponents)

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Host-side decode | Avoids TT-Metalium custom kernel risk for MVP |
| Layer-level artifacts | Simpler than block/tile streaming |
| Row-major BF16 format | Simplest path from codec to `ttnn.from_torch` |
| One operator family | Patch lm_head first, MLP fallback, standalone fallback |
| Python Huffman | Correct and debuggable; C/GPU version is optimization |

## Environment

- **Hardware**: 4x Blackhole TT devices
- **Python**: 3.10 (tt-metal venv)
- **PyTorch**: 2.7.1
- **ttnn**: from tt-metal build

## Future Work

1. **C/Cython Huffman decoder** for 100-1000x decode speedup
2. **TT-Metalium device-side decode** kernel
3. **Block-level streaming** instead of full-layer decode
4. **Decode/compute overlap** across layers
5. **TT tile-native compressed format**
6. **Full model integration** with real token generation
