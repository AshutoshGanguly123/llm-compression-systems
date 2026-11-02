# End-to-end compression measurement protocol

## Decisive question
Under what deployment conditions does DFloat11-style compression help user-visible latency/memory?

## Scenarios
1. **Decode-once-at-startup**: decode artifact → keep BF16/TT tensors resident → many inferences.
2. **Repeated decode**: decode per load / per shard stream (worst case for host Huffman).

## Metrics (record in JSON)
| Stage | Fields |
|-------|--------|
| Load | wall_s, host_peak_rss_mb |
| Decode | wall_s, host_peak_rss_mb |
| Transfer | wall_s, device_peak_mb |
| Compute | wall_s (op or generate) |
| Serving | TTFT_s, tok_per_s, steady_state |

## Bottleneck policy
Optimize the **measured** dominant stage first. Current PoC points to **host Huffman decode (~115s / 262M elems)** → try compiled host decoder before device kernels.

## Runner
```bash
python measurements/e2e_benchmark.py --scenario decode_once --report measurements/results/latest.json
```
On machines without TT hardware, the script still records host-side load/decode timings and marks device stages as `skipped`.
