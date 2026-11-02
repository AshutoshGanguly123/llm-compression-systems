# llm-compression-systems

Experiments on **efficient LLM weight representation and serving paths**: (1) FP16 mantissa sensitivity under lm-eval, (2) DFloat11-style Huffman exponent coding on Tenstorrent / TT-Metal with bit-exact reconstruct and host-decode latency accounting.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Tracks

| Track | Path | What it measures |
|-------|------|------------------|
| FP16 mantissa ablation | [`fp16-mantissa-sensitivity/`](fp16-mantissa-sensitivity/) | Accuracy vs mantissa LSBs zeroed (Llama 3.1 8B) |
| DFloat11 on Tenstorrent | [`dfloat11-tenstorrent/`](dfloat11-tenstorrent/) | Offline Huffman compress → host decode → TT upload → linear compute |

## Headline results

**Mantissa (lm-eval):** ≤5 bits zeroed ≈ baseline; 8 bits hurts hellaswag (~−11%); 9 bits collapses (~−45% hellaswag). See track README.

**DFloat11 TT PoC (`lm_head`):**

| Metric | Value |
|--------|------:|
| Compression ratio | **1.509×** (33.7% size reduction) |
| Exponent compression | **3.07×** (19 unique exponents) |
| Bit-exact reconstruction | PASS |
| TT output equivalence | PASS (baseline == compressed) |
| TT op time baseline / compressed | 1.41 ms / 1.37 ms |
| Host decode (Python Huffman, 262M elems) | ~115 s |

## Method (short)

1. **Mantissa:** clear N LSBs of FP16 mantissa in weights; evaluate hellaswag, piqa, arc_easy, winogrande, boolq.
2. **DFloat11 TT:** split BF16 into sign/exponent/mantissa; Huffman-code low-entropy exponents; decode on host CPU; `ttnn.from_torch` + `ttnn.linear`; compare metrics JSON.

## Reproducibility

Each track has its own `requirements.txt` / `setup_env.sh` and scripts. Model weights are **not** committed (HF cache / TT artifacts regenerated locally).

## Limitations / non-claims

- Host-side Python Huffman decode is the bottleneck; not a production GPU/TT device kernel.
- Not a full-model MoE compression + streaming serving stack yet — this repo is measurement-first systems research.
- Mantissa study ≠ deployment codec; DFloat path is a PoC for bit-exact TT compute after decode.
- Numbers are from specific hardware/software pins documented in each track; expect variance across boards/drivers.

## Layout

```
fp16-mantissa-sensitivity/   # ablation + committed summary JSON + plots
dfloat11-tenstorrent/        # TT DFloat11 PoC (code, tests, docs)
```

## License

MIT (see `LICENSE`). Upstream model/dataset licenses still apply to downloaded assets.
