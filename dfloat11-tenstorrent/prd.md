# PRD: DFloat11 on Tenstorrent Friday MVP

**Status:** Draft v1  
**Owner:** Research / Systems PoC  
**Target demo date:** Friday  
**Primary audience:** Codex / Claude Code / engineering collaborator  
**Priority:** Ship a working end-to-end prototype, not a fast one

---

## 1. Executive summary

Build the smallest possible end-to-end proof that **DFloat11-style lossless weight compression can be integrated into the Tenstorrent ecosystem**.

The Friday MVP is **not** “full paper parity” and **not** “full on-device TT decode.” The Friday MVP is:

1. Run a small, officially supported LLM baseline on a Tenstorrent machine.
2. Compress a selected set of BF16 weights offline using a DFloat11-style codec.
3. At inference time, **load compressed weights on demand**, **decode them on the TT machine host CPU**, move them to the TT device, run the corresponding TT compute, and free the temporary buffers.
4. Measure correctness and performance deltas versus baseline.

This proves the core infrastructure path:

`compressed artifact -> just-in-time decode -> TT tensor upload -> TT compute -> metrics`

That is enough to validate the architecture and create a clean platform for later research on:
- TT-native device-side decoding
- layer/block streaming
- decode/compute overlap
- TT-tile-aware compressed formats
- full-model compressed serving

---

## 2. Problem statement

Large BF16 models are heavy to store, move, and stage. DFloat11 shows that BF16 weights have low-entropy structure, especially in exponent bits, and can be compressed losslessly with about **30% size reduction** while preserving **bit-for-bit identical outputs** after reconstruction. The DFloat11 paper also emphasizes that practical inference requires **online decompression**, compact LUTs, and block-level execution strategy rather than naive full-model decompression. 

For Tenstorrent, we currently do **not** have reusable infrastructure for:
- offline DFloat11-compatible weight artifacts
- runtime just-in-time compressed weight loading
- swapping a TT model layer from static dense weights to compressed-on-storage weights
- measuring the tradeoff between transfer volume and runtime decode overhead

We need a **working PoC in two days**. That means the project must optimize for **implementation risk**, not elegance.

---

## 3. Product goal

### Primary goal
Demonstrate that a real model path running on Tenstorrent can consume **losslessly reconstructed BF16 weights sourced from compressed artifacts on demand**.

### Secondary goals
- Produce a repo that is easy to extend toward TT-native decode later.
- Produce metrics that tell us whether this direction is worth optimizing.
- Keep the system simple enough for codegen agents to build quickly.

### Non-goals for Friday
- Full DFloat11 paper reproduction
- Matching paper throughput numbers
- True TT on-device Huffman decode
- Streaming many layers concurrently
- Network transport stack
- Full-model compression coverage
- TT tile-native compressed storage format
- Decode/compute overlap

---

## 4. Friday MVP definition

The Friday MVP is considered successful if **all** of the following are true:

1. A supported small LLM runs on a TT machine in a known-good baseline path.
2. At least **one real model layer or layer family** is replaced by a compressed-weight runtime path.
3. The compressed path reconstructs BF16 weights **bit-exactly** on CPU.
4. The reconstructed weights are uploaded to the TT device and used in TT compute.
5. The model or model slice executes successfully end-to-end.
6. Baseline and compressed runs emit a metrics report.
7. The codebase is modular enough to later replace host decode with a TT-Metalium decoder.

### Strongly preferred MVP integration target
Integrate compression for **exactly one operator family** first:

1. **Preferred target:** `lm_head` or another isolated final projection layer used in real generation.
2. **Fallback target:** one MLP linear layer family in a single transformer block.
3. **Safety fallback:** a standalone TT `linear` benchmark using real model weights and representative activations.

The project should attempt targets in that order.

---

## 5. User stories

### Research engineer
“I want to prove that compressed weights can be served into TT compute without rewriting the whole model stack.”

### Systems engineer
“I want clean boundaries between offline compression, runtime loading, decode, TT transfer, and execution so I can optimize them separately later.”

### Future kernel developer
“I want a host-decode implementation whose interfaces match a future TT-Metalium device-decode path.”

---

## 6. Guiding product principles

1. **Ship reality, not mythology.** A slower working PoC beats a perfect diagram.
2. **One strategy only.** No layer-vs-block debate for Friday. Pick one and ship it.
3. **Preserve exactness where possible.** Compression/decompression must be bit-exact on the host.
4. **Minimize integration surface.** Patch one operator family first.
5. **Keep future doors open.** APIs must allow later device-side decode.

---

## 7. Scope

## In scope
- Baseline TT inference run for a small supported model
- Offline BF16 weight compression using DFloat11-style exponent Huffman coding
- CPU host decode on the TT machine
- Upload decoded BF16 tensors to TT device via `ttnn.from_torch`
- Use decoded weights in TT compute via `ttnn.linear` or the equivalent model path
- Metrics collection and comparison
- Tests for codec correctness and runtime integration
- Clear repo README and reproducible scripts

## Out of scope
- Full model-wide compression coverage
- True device-side Huffman decode on Tensix
- Custom TT-Metalium fused decode+matmul kernel
- Distributed serving
- Multi-request scheduler
- Overlapped transfer/compute pipeline
- Compression-aware TT tile packing
- Production deployment hardening

---

## 8. Recommended technical strategy

## Strategy choice for Friday
**Use host-side decode on the TT machine, layer-by-layer.**

Do **not** implement on-device Huffman decode for the Friday MVP.

### Why this is the right cut
- It preserves the interesting system property: **weights are stored compressed and materialized only on demand**.
- It gives a real end-to-end path on TT hardware.
- It avoids the highest-risk part: variable-length Huffman decode inside TT-Metalium custom kernels.
- It still lets us measure compression ratio, decode cost, and TT execution cost.

### Why not device-side decode yet
A true TT-native decoder would require low-level TT-Metalium work: program creation, custom kernel creation, runtime args, and data movement via APIs such as `noc_async_read`, all inside a custom Tensix kernel. TT’s docs expose these low-level APIs and programming examples, but that is the hard path, not the Friday path. DFloat11 itself also relies on **compact hierarchical LUTs**, a **two-phase GPU kernel**, and **block-level decompression** for efficiency, which is a significant implementation burden for a 2-day PoC.

---

## 9. Model selection requirements

The implementation must choose the **smallest officially supported model that is already runnable on the available TT environment**.

### Selection order
1. **Use the smallest supported model already known to run locally.**
2. If Meta-gated access is already available, prefer **Llama 3.2 1B**.
3. Otherwise choose any smaller supported / already-cached TT-compatible model or demo model that avoids access friction.

### Important constraint
Do **not** block the project on model access approvals or server infrastructure.

### Baseline preference
Prefer a **TT model demo / known-good model path** over a new inference-server deployment, because the official TT model metrics are collected from model demos and may differ from vLLM server behavior.

---

## 10. Architecture overview

## 10.1 High-level architecture

### Baseline path
`model weights -> TT model loader -> TT tensors -> TT inference -> metrics`

### Compressed path (Friday MVP)
`selected BF16 weights -> offline compression artifact -> runtime load compressed artifact -> CPU decode on TT host -> torch BF16 tensor -> ttnn.from_torch -> TT compute -> free temporary weight tensor -> metrics`

## 10.2 Component boundaries

### A. Offline compression tool
Responsible for:
- extracting selected BF16 weights
- compressing exponent stream with Huffman coding
- storing sign/mantissa raw payload
- saving metadata for exact reconstruction

### B. Runtime artifact loader
Responsible for:
- locating compressed artifacts by layer name
- loading metadata and payloads
- exposing a simple `decode()` method

### C. Host decoder
Responsible for:
- exact reconstruction of BF16 tensor from compressed artifact
- emitting a CPU `torch.bfloat16` tensor with original shape

### D. TT adapter
Responsible for:
- uploading the decoded torch tensor to TT
- converting to TT tensor layout and dtype
- deallocating temporary device buffers when safe

### E. Model integration layer
Responsible for:
- swapping selected dense weights with the compressed runtime path
- calling decode only when the layer is invoked

### F. Evaluation harness
Responsible for:
- baseline and compressed runs
- warmup behavior
- metrics aggregation
- correctness checks

---

## 11. Concrete MVP design

## 11.1 Compression granularity
For Friday, use **layer-level granularity**.

That means:
- each selected layer gets one compressed artifact
- at runtime, when the layer is called, the whole layer weight is decoded
- the decoded tensor is uploaded to TT, used once, then released

### Why layer-level and not block-level
- simpler artifact management
- simpler runtime API
- simpler debugging
- enough to prove the architecture

### Explicit tradeoff
This is slower and less memory-efficient than block/tile streaming, but much easier to build correctly.

## 11.2 Weight format
For Friday, keep the artifact format **row-major BF16-oriented**, not TT tile-native.

### Why
- simplest possible path from existing CPU codec code
- avoids inventing a TT-specific packed weight format immediately
- lets us reuse bit-exact host-side reconstruction logic

### Consequence
The runtime will decode into a CPU BF16 tensor and rely on TT APIs to convert/upload into TT layout.

## 11.3 Layer integration priority
Implementation order:
1. `lm_head` or final output projection if exposed cleanly
2. one MLP projection layer in one transformer block
3. one isolated TT `linear` using real model weights

The codebase must support all three, even if only one is fully integrated by Friday.

---

## 12. Functional requirements

## FR-1 Baseline runner
The project must include a baseline runner that:
- loads the chosen TT-supported model or model demo path
- runs at least one deterministic test prompt or representative input
- emits baseline metrics to JSON and human-readable stdout

## FR-2 Weight extraction
The project must include a tool to extract selected BF16 weights from the source model/checkpoint or live model state.

Required output:
- layer name
- shape
- dtype
- original byte size
- raw tensor checksum or hash

## FR-3 Offline compressor
The project must include an offline compression script that, for each selected layer:
- accepts a BF16 tensor
- extracts sign, exponent, mantissa information
- Huffman-encodes the exponent stream
- stores enough metadata for exact reconstruction
- writes a compressed artifact to disk

## FR-4 Compressed artifact format
Each compressed artifact must contain at minimum:
- layer name
- shape
- dtype (`bfloat16`)
- exponent symbol table / decode table / codec metadata
- encoded exponent bitstream
- raw sign+mantissa payload
- checksum of original tensor
- compressed byte count
- optional codec version field

Recommended formats:
- `.npz` for arrays/payloads
- optional `.json` sidecar for metadata

## FR-5 Host decoder
The runtime must include a host decoder that:
- loads a compressed artifact
- reconstructs the exact BF16 tensor on CPU
- returns `torch.bfloat16`
- verifies checksum in debug mode

## FR-6 TT upload adapter
The runtime must include a TT adapter that:
- takes a decoded CPU tensor
- uploads it to the selected TT device
- returns a TT tensor suitable for `ttnn.linear` or the chosen model path
- supports explicit cleanup/deallocation

## FR-7 Compressed layer wrapper
The runtime must include a wrapper object for a compressed layer.

Required behavior:
- accepts layer identifier + artifact path
- decodes on first use or every use depending on configuration
- uploads to TT
- executes the TT op
- frees temporary buffers when configured
- records timing metrics

## FR-8 Model integration
The project must integrate at least one compressed layer into a real TT execution path.

Acceptable forms:
- patched `lm_head`
- patched MLP projection
- standalone TT linear benchmark using real model weights and real hidden-state-like activations

## FR-9 Metrics reporting
The project must emit the following metrics where applicable:
- original bytes
- compressed bytes
- compression ratio
- host decode time per invocation
- host-to-device transfer time per invocation
- TT op time per invocation
- total layer latency
- baseline end-to-end latency
- compressed-path end-to-end latency
- tokens/sec if real generation is integrated
- TTFT if real generation is integrated

## FR-10 Correctness checks
The project must include checks for:
- bit-exact CPU reconstruction versus original BF16 tensor
- baseline TT output versus compressed TT output for the patched layer or path
- deterministic token match for fixed prompt if generation path is integrated

## FR-11 CLI / scripts
The project must provide simple commands for:
- extract
- compress
- run baseline
- run compressed
- evaluate

---

## 13. Non-functional requirements

## NFR-1 Time-to-first-success
A new engineer should be able to run the minimal happy path with a short README and a few commands.

## NFR-2 Modularity
Offline codec, host decode, TT upload, and model integration must be separate modules.

## NFR-3 Debuggability
The project must support a verbose/debug mode that prints layer names, shapes, byte counts, timing, and checksums.

## NFR-4 Minimal dependencies
Avoid unnecessary frameworks. Use existing TT Python stack plus PyTorch, NumPy, and the current codec code.

## NFR-5 Reproducibility
Metrics must be saved to disk in machine-readable form.

---

## 14. Explicit design decisions

### Decision 1
**Decode on the TT machine host CPU, not on Tensix.**

### Decision 2
**Use layer-level artifact granularity.**

### Decision 3
**Use row-major BF16 artifacts for Friday.**

### Decision 4
**Patch one operator family only.**

### Decision 5
**Report poor performance honestly if it works.**

### Decision 6
**Network transport is simulated only by local-file byte counts.**
No real remote serving is required for the Friday MVP.

---

## 15. Detailed runtime flow

## 15.1 Offline stage
1. Load source model or selected layer weights.
2. Convert selected weights to CPU `torch.bfloat16` if needed.
3. For each selected layer:
   - flatten tensor in row-major order
   - split into sign / exponent / mantissa components
   - build exponent Huffman codec
   - encode exponent stream
   - store sign+mantissa payload
   - write artifact + metadata
4. Write manifest mapping layer names to artifact paths and sizes.

## 15.2 Baseline inference stage
1. Load baseline TT model path.
2. Warm up if needed.
3. Run chosen prompt/input.
4. Save output and metrics.

## 15.3 Compressed inference stage
1. Load the same model path.
2. Replace target layer(s) with compressed wrapper.
3. On layer invocation:
   - load compressed artifact
   - decode to CPU BF16 tensor
   - convert/upload to TT tensor
   - execute TT op
   - free temporary decoded host tensor if configured
   - free TT weight tensor if configured
4. Save output and metrics.

---

## 16. API requirements

## 16.1 Offline API
```python
compress_weight(
    weight: torch.Tensor,
    layer_name: str,
    out_dir: str,
    codec: str = "huffman_exponent",
) -> CompressionArtifact
```

## 16.2 Runtime decode API
```python
decode_artifact(
    artifact_path: str,
    verify: bool = False,
) -> torch.Tensor
```

## 16.3 TT upload API
```python
to_tt_weight(
    weight: torch.Tensor,
    device,
    dtype=None,
    layout=None,
):
    ...
```

## 16.4 Compressed layer API
```python
class CompressedLinearRuntime:
    def __init__(self, artifact_path, device, bias=None, config=None):
        ...

    def forward(self, x):
        # decode -> upload -> linear -> cleanup
        ...
```

## 16.5 Evaluation API
```python
run_baseline(config) -> MetricsReport
run_compressed(config) -> MetricsReport
compare_runs(baseline, compressed) -> ComparisonReport
```

---

## 17. Repository structure

```text
project/
  README.md
  pyproject.toml
  requirements.txt
  configs/
    model_default.yaml
    eval_default.yaml
  scripts/
    extract_weights.py
    compress_weights.py
    run_baseline.py
    run_compressed.py
    compare_runs.py
  src/
    df11_tt/
      __init__.py
      config.py
      utils/
        logging.py
        timing.py
        hashing.py
      compress/
        bitops.py
        huffman.py
        artifact.py
        compressor.py
      runtime/
        loader.py
        decoder.py
        tt_adapter.py
        compressed_linear.py
        model_patch.py
      eval/
        metrics.py
        reporters.py
        prompts.py
  tests/
    test_codec_roundtrip.py
    test_artifact_schema.py
    test_runtime_decode.py
    test_tt_linear_smoke.py
    test_model_patch_smoke.py
  outputs/
    artifacts/
    metrics/
    logs/
```

---

## 18. Metrics specification

## 18.1 Required metrics

### Compression metrics
- original bytes
- compressed bytes
- compression ratio
- bytes saved

### Decode-path metrics
- artifact load time
- CPU decode time
- TT upload time
- TT op time
- total patched-layer latency

### End-to-end metrics
- wall-clock end-to-end latency
- prefill latency if available
- decode step latency if available
- tokens/sec if generation is integrated
- TTFT if generation is integrated

### Correctness metrics
- bit-exact roundtrip pass/fail
- TT output equivalence/tolerance check
- final token match pass/fail

## 18.2 Metrics output format
- save JSON per run
- optional CSV summary for quick comparison
- always print concise console summary

Example JSON keys:
```json
{
  "model_name": "...",
  "mode": "baseline|compressed",
  "layer_name": "...",
  "original_bytes": 0,
  "compressed_bytes": 0,
  "compression_ratio": 0.0,
  "decode_ms": 0.0,
  "upload_ms": 0.0,
  "tt_op_ms": 0.0,
  "total_ms": 0.0,
  "tokens_per_sec": null,
  "ttft_ms": null,
  "bit_exact": true,
  "token_match": true
}
```

---

## 19. Correctness requirements

## 19.1 Codec correctness
For every compressed artifact:
- decoded tensor must be **bitwise identical** to original CPU BF16 tensor
- test should compare via raw 16-bit representation, not float tolerance

## 19.2 TT path correctness
For the patched layer/path:
- TT output from baseline and compressed path should be identical or within a tiny BF16 tolerance if the runtime route introduces unavoidable format handling differences
- for deterministic generation runs, generated token IDs should match baseline whenever the same TT compute path is used with the same decoded weights

## 19.3 Failure behavior
If decode checksum fails:
- abort the run in strict mode
- log warning and continue only in explicit debug mode

---

## 20. Implementation milestones

## Milestone A: baseline
**Goal:** a known-good TT model path runs and produces metrics.

Deliverables:
- baseline runner script
- prompt/input config
- saved baseline metrics

## Milestone B: offline codec
**Goal:** selected weights compress and roundtrip bit-exactly on CPU.

Deliverables:
- extraction script
- artifact writer
- codec tests

## Milestone C: TT linear smoke path
**Goal:** decoded weights upload to TT and execute one TT linear op.

Deliverables:
- TT adapter
- compressed linear wrapper
- standalone TT linear demo

## Milestone D: real model integration
**Goal:** at least one real model layer path uses the compressed runtime.

Deliverables:
- model patch logic
- compressed run script
- comparison metrics

## Milestone E: polish
**Goal:** repo is understandable and reproducible.

Deliverables:
- README
- example commands
- metrics summary
- demo notebook or screenshots optional

---

## 21. Friday acceptance criteria

The Friday demo is accepted if the following all hold:

1. **Baseline works.** A small TT-supported model or TT-backed model slice runs.
2. **Compression works.** Selected weights roundtrip bit-exactly on CPU.
3. **TT integration works.** Decoded weights are uploaded and used in TT compute.
4. **Patched path works.** At least one real layer or real model slice uses compressed weights on demand.
5. **Metrics exist.** Compression ratio and latency breakdown are reported.
6. **Repo is usable.** Another engineer can reproduce the run from README instructions.

### Nice-to-have acceptance criteria
- real end-to-end token generation with patched layer path
- token outputs match baseline exactly
- CSV/JSON comparison report

---

## 22. Risk register

## Risk 1: model availability / access-gated weights
**Risk:** chosen model cannot be downloaded or run in time.  
**Mitigation:** choose the smallest supported model already present in the environment; never block on access approvals.

## Risk 2: patching full model stack is too invasive
**Risk:** integrating compressed weights into the official TT model code is messy.  
**Mitigation:** patch `lm_head` first; fallback to one MLP linear; fallback again to standalone TT linear benchmark with real weights.

## Risk 3: runtime overhead looks terrible
**Risk:** host decode + upload is much slower than baseline.  
**Mitigation:** accept it. Friday goal is functional proof, not speed win.

## Risk 4: TT layout mismatch
**Risk:** decoded CPU tensor shape/layout mismatches TT expectations.  
**Mitigation:** keep CPU decode row-major BF16 and rely on TT upload/layout conversion path for Friday.

## Risk 5: custom TT-Metalium kernel temptation
**Risk:** the team loses a day trying to write a real on-device Huffman decoder.  
**Mitigation:** ban this from MVP. Device-side decode is stretch only.

---

## 23. Stretch goals

These are optional and should only be attempted if the MVP is already working.

### Stretch 1: cache decoded weights for a single request
Useful to separate decode cost from repeated-use cost.

### Stretch 2: patch more than one layer
Useful for testing scaling behavior.

### Stretch 3: TT-Metalium smoke test kernel
Create a tiny custom data-movement kernel that reads bytes from device DRAM to L1 and writes them back, purely to validate kernel launch plumbing.

### Stretch 4: simple fixed-width on-device toy decoder
Implement a toy non-Huffman or byte-aligned decode kernel only to test the end-to-end TT-Metalium control path.

### Stretch 5: tile-aware artifact format
Move toward TT-native storage later.

---

## 24. Explicitly postponed research questions

These are intentionally postponed beyond Friday:
- Should storage granularity be layer, block, or tile?
- Should we compress row-major weights or TT-tiled weights?
- Can decode be fused with TT matmul?
- Can decode happen in DRAM reader kernels with circular buffers?
- Can we overlap decode with compute across layers?
- How should shared weights or tied embeddings be handled optimally?
- What is the best on-device decode algorithm for Tensix RISC-V?

---

## 25. Engineering notes for future TT-native decode

Future device-side work will likely require:
- TT-Metalium `CreateProgram` / `CreateKernel` host setup
- runtime args for buffer addresses and sizes
- data-movement kernels using `noc_async_read` and related primitives
- careful L1 / SRAM budgeting for decode tables and work buffers
- likely a simplified or block-structured decode scheme rather than naive bit-serial Huffman traversal

This is **not** part of the Friday MVP.

---

## 26. Suggested execution plan for the next two days

## Day 1 morning
- pick runnable TT model baseline
- confirm one deterministic input path works
- identify easiest patch target (`lm_head` first)

## Day 1 afternoon
- implement extraction + compression + roundtrip tests
- implement standalone TT linear smoke benchmark with decoded weights

## Day 2 morning
- patch real model target layer
- run baseline vs compressed comparison
- save metrics

## Day 2 afternoon
- clean CLI
- write README
- package outputs and demo script

---

## 27. Minimal demo script expectations

A single command such as below should exist by the end:

```bash
python scripts/run_compressed.py \
  --model-config configs/model_default.yaml \
  --eval-config configs/eval_default.yaml \
  --layer-target lm_head \
  --artifact-dir outputs/artifacts \
  --report outputs/metrics/compressed_run.json
```

And a baseline equivalent:

```bash
python scripts/run_baseline.py \
  --model-config configs/model_default.yaml \
  --eval-config configs/eval_default.yaml \
  --report outputs/metrics/baseline_run.json
```

---

## 28. What “done” looks like in one sentence

A real TT-backed model path runs with at least one layer served from a compressed DFloat11-style artifact, decoded just in time on the TT machine host, uploaded to the device, executed successfully, and compared against a vanilla TT baseline with saved metrics.

---

## 29. Direct instruction block for a coding agent

Build this project as a **Friday MVP**, not a research-grade final system.

Priority order:
1. Get a baseline TT model path running.
2. Implement offline compression + exact CPU roundtrip.
3. Implement TT upload + standalone compressed linear execution.
4. Integrate one real model layer with on-demand host decode.
5. Produce metrics and README.

Do **not** spend time on:
- TT-native Huffman decode
- full-model coverage
- performance optimization
- network stack
- fancy abstractions beyond what is needed for the above

If full model integration becomes fragile, ship the fallback path using a real extracted model layer and TT linear benchmark, but preserve the same runtime APIs so future full-model integration is straightforward.
