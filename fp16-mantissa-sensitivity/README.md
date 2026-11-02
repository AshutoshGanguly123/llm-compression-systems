# FP16 Mantissa Sensitivity (Llama 3.1 8B Instruct)

Progressively zero least-significant FP16 mantissa bits (0–9) in model weights and measure downstream accuracy with EleutherAI `lm-evaluation-harness`.

## Claim

Many FP16 mantissa LSBs are unused for short-benchmark accuracy: up to ~5–7 bits can be zeroed with small degradation; bit 8–9 collapses quality.

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=...   # gated Llama access
python download_model.py
python experiment_runner.py
python visualize_results.py
```

## Metrics (committed `results/summary_report.json`)

Model: `meta-llama/Llama-3.1-8B-Instruct`, dtype FP16.

| Bits zeroed | hellaswag | piqa | arc_easy | winogrande | boolq |
|------------:|----------:|-----:|---------:|-----------:|------:|
| 0 (baseline)| 0.598 | 0.803 | 0.822 | 0.736 | 0.854 |
| 5 | 0.595 | 0.804 | 0.824 | 0.745 | 0.856 |
| 8 | 0.531 (−11%) | 0.779 | 0.801 | 0.741 | 0.849 |
| 9 | 0.326 (−45%) | 0.648 | 0.634 | 0.589 | 0.640 |

Plots: `visualizations/`.

## Limitations

- Ablation only (not a deployable codec). Does not measure tok/s, RAM, or KV-cache.
- Five short English benchmarks; no long-context or generative quality suite.
- Weight casting is in-place for measurement; not TensorRT/MoE integration.
