# Roadmap — end-to-end systems measurements

## Goal

Show when DFloat11-style (or mantissa-truncated) weights improve **user-visible** serving metrics, not only compression ratio.

## Next measurements

1. Cold-start load time with/without compressed artifacts.
2. Peak host RAM and device memory during decode + upload + compute.
3. Time to first token and sustained tokens/s on a fixed prompt set.
4. Break-even: how many inferences amortize the ~115 s host decode in the current PoC.

## Next engineering

1. Replace Python Huffman decode with a fast path (vectorized C++ / device kernel).
2. Overlap decode of layer *i+1* with compute of layer *i*.
3. Hook compressed linear into a full TT generative demo once baseline decode is stable.

Land numbers in `dfloat11-tenstorrent/` when run. Scaffold: `measurements/E2E_PROTOCOL.md`.
