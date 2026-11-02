# llm-compression-systems

Measurement-first work on **efficient LLM weight representation and serving**: FP16 mantissa sensitivity under lm-eval, and a DFloat11-style Huffman path on Tenstorrent / TT-Metal.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Why this exists

Compression only helps if the **full path** wins:

compressed storage → decode → host/device transfer → model compute → user-visible latency

This repo measures that path on a Tenstorrent PoC and maps how many FP16 mantissa bits you can drop before quality collapses.

## Track A — DFloat11 / Tenstorrent (`dfloat11-tenstorrent/`)

Offline Huffman coding of low-entropy BF16 exponents → host decode → `ttnn` upload → linear compute.

| Metric | Value | Takeaway |
|--------|------:|----------|
| Compression ratio | **1.51×** (≈33.7% smaller) | Size win on `lm_head` |
| Exponent compression | **3.07×** | Exponent entropy is the lever |
| Bit-exact reconstruction | **PASS** | No silent numeric drift |
| TT output equivalence | **PASS** | Compressed path == baseline compute |
| TT op time (base / compressed) | 1.41 ms / 1.37 ms | Device compute is not the bottleneck |
| Host Huffman decode (262M elems) | **~115 s** | Dominant cost today |

**1.51× is a compression ratio, not a speedup.** End-to-end this PoC is slower today because ~115 s of host-side Python Huffman decode dwarfs a ~1.4 ms device op. Until decode is accelerated or amortized, compression can lose wall-clock even when ratios look good.

## Track B — FP16 mantissa cliff (`fp16-mantissa-sensitivity/`)

Zero N LSBs of FP16 mantissas in Llama 3.1 8B Instruct; evaluate hellaswag / piqa / arc_easy / winogrande / boolq.

| Bits zeroed | hellaswag | Note |
|------------:|----------:|------|
| 0 (baseline) | 0.598 | — |
| 5 | 0.595 | ≈ free |
| 8 | 0.531 (−11%) | cliff begins |
| 9 | 0.326 (−45%) | collapse |

Several mantissa bits look accuracy-idle on short English benchmarks; the cliff is sharp past ~8 bits. Ablation only — not a deployable codec.

## Next

See [`ROADMAP.md`](ROADMAP.md): accelerate decode, measure load time / peak memory / TTFT / tok/s, break-even after amortizing host decode. Protocol scaffold: [`measurements/`](measurements/).

## Limitations

- Host Python Huffman decode is a PoC, not a production kernel.
- Not yet a full-model generative serving bakeoff.
- TT numbers are board/driver specific.

## References

[`reference_research_papers/`](reference_research_papers/) — arXiv links only.

## License

MIT. Upstream model licenses apply to downloaded weights.
