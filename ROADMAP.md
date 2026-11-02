# ROADMAP — toward an end-to-end systems result

## Goal

Show when DFloat11-style (or mantissa-truncated) weights improve **user-visible** serving metrics, not only compression ratio.

## Next measurements

1. Load time (cold start) with/without compressed artifacts.
2. Peak host RAM and device memory during decode + upload + compute.
3. Time to first token and sustained tokens/s on a fixed prompt set.
4. Break-even curve: number of inferences needed to amortize the ~115 s host decode (current PoC).

## Next engineering

1. Replace Python Huffman decode with a fast path (vectorized C++ / device kernel).
2. Overlap decode of layer *i+1* with compute of layer *i*.
3. Hook compressed linear into a full TT generative demo once baseline decode is stable.

No fabricated numbers — land measurements in `dfloat11-tenstorrent/` reports when run.


## Instrumentation landed

See `measurements/E2E_PROTOCOL.md` and `measurements/e2e_benchmark.py`. Replace the host stub with real Huffman decode on TT nodes; do not quote scaffold timings as production results.
