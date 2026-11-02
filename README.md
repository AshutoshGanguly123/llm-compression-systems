# llm-compression-systems

Measurement-first work on **efficient LLM weight representation and serving paths**: FP16 mantissa sensitivity under lm-eval, and a DFloat11-style Huffman path on Tenstorrent / TT-Metal.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Systems claim

Compression is only useful if the **full path** wins:

**compressed storage → decode → host/device transfer → model compute → user-visible latency**

This repo measures that path on a Tenstorrent PoC and separately maps how many FP16 mantissa bits you can drop before quality collapses.

## Track A — DFloat11 / Tenstorrent (`dfloat11-tenstorrent/`)

Offline Huffman coding of low-entropy BF16 exponents → host decode → `ttnn` upload → linear compute.

| Metric | Value | Takeaway |
|--------|------:|----------|
| Compression ratio | **1.509×** (33.7% smaller) | Real size win on `lm_head` |
| Exponent compression | **3.07×** | Exponent entropy is the lever |
| Bit-exact reconstruction | **PASS** | No silent numeric drift |
| TT output equivalence | **PASS** | Compressed path == baseline compute |
| TT op time (base / compressed) | 1.41 ms / 1.37 ms | Device compute not the bottleneck |
| **Host Huffman decode (262M elems)** | **~115 s** | **Dominant cost today** |

**Sell the judgment, not just the ratio:** until decode is accelerated or amortized across many inferences, compression can *lose* end-to-end even when ratios look good. That bottleneck is the RE3 systems question this PoC surfaces.

## Track B — FP16 mantissa cliff (`fp16-mantissa-sensitivity/`)

Zero N LSBs of FP16 mantissas in **Llama 3.1 8B Instruct**; evaluate hellaswag / piqa / arc_easy / winogrande / boolq.

| Bits zeroed | hellaswag | Story |
|------------:|----------:|-------|
| 0 (baseline) | 0.598 | — |
| 5 | 0.595 | ≈ free |
| 8 | 0.531 (**−11%**) | quality cliff begins |
| 9 | 0.326 (**−45%**) | collapse |

**Practical read:** several mantissa bits are accuracy-idle on short English benchmarks; the cliff is sharp past ~8 bits. This is an ablation, not a deployable codec.

## Method (short)

1. Mantissa: in-place LSB clear + EleutherAI lm-eval.
2. DFloat11 TT: split BF16 → Huffman exponents → Python host decode → `ttnn.from_torch` + `ttnn.linear` → metrics JSON.

## Reproducibility

Each track has its own requirements / setup scripts. Weights and TT artifacts are **not** committed.

## Limitations

- Host-side Python Huffman decode is a PoC — not a production GPU/TT kernel.
- Not yet a full-model MoE + streaming serving stack with TTFT / tok/s bakeoffs.
- Mantissa study ≠ deployment format; TT numbers are board/driver specific.

## RE3 roadmap

1. Accelerate decode (C++/CUDA/TT-Metal) or pipeline decode with compute.
2. Measure **load time, peak host/device memory, TTFT, sustained tok/s** vs BF16/FP8 baselines.
3. Chart when compression becomes net-positive after amortization.
4. Extend beyond single-op `lm_head` to full generative paths.

## E2E measurement harness

Protocol and host scaffold: [`measurements/`](measurements/). TT transfer/compute still require a TT node; do not treat scaffold timings as the 115s PoC replacement.

## Reference papers

[`reference_research_papers/`](reference_research_papers/)

## License

MIT. Upstream model licenses apply to downloaded weights.
