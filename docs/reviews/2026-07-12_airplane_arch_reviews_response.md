# Review Response: "Airplane" Architecture Reviews (2026-07-12)

**Status:** Dispositioned
**Inputs:** Two external review docs received 2026-07-12 — "Airplane review Arch"
(engine/topology review) and "Airplane extended arch" (quantization/memory techniques)
**Output:** Amendments A5–A7 folded into `docs/specs/DS5_Project_Spec_v0.3.md`;
work packs in `docs/work-packs/2026-07-12-review-incorporation/`
**Governing decisions unchanged:** ADR-001..005
**Policy:** Spec §12 (external reviews are inputs, not authorities)

## Summary judgment

More useful than the 2026-07-10 Gemini review. Two ideas are adopted (per-expert
sensitivity-driven quantization; a Metal-4 kernel-feature spike), one implicit
principle is made explicit (zero-copy weight mapping). The central topology proposal
of the first doc — attention and routing on Node A with per-layer round trips to
stateless expert workers — is rejected on arithmetic. The DeltaNet section of the
second doc describes a different model than DS5's target.

## Dispositions — "Airplane extended arch"

### Adopted

**A5 — Two-axis expert quantization manifest (M1 data, M4 consumption).**
The review's core point is sound: expert precision should be assigned per expert from
measurement, not uniformly per tier by fiat. DS5 already measures axis 1 (routing
frequency — the M1 235B telemetry capture). Adopt axis 2: per-expert quantization
sensitivity from an offline `llama.cpp` imatrix/KLD calibration pass over the
router-calibration corpus. Both axes land in a machine-readable `expert_stats.json`
consumed by the placement simulator and, at M4, by the placement/quant manifest
(per-expert precision column). Producing the mixed-precision 235B artifact stays
within ADR-002: `llama-quantize` with a manifest-driven tensor-override list is
"reusing community tooling offline," not a custom quantization pipeline. Work pack WP-1
(tooling + schema) and WP-2 (manifest policy).

**A6 — Metal 4 / M5-generation GPU feature spike (bounded, pre-M2-kernel-work).**
Worth one time-boxed investigation: native bf16 arithmetic, MSL tensor/cooperative-
matrix intrinsics, and any native sub-byte/quantized type support relevant to the
Q8_0 → Q4 → I-quant kernel sequence. Constraint: raw MSL features only. MPSGraph is
rejected (it is exactly the "someone else's ML library" that ADR-002 excludes from the
runtime), and the review's ANE-offload claim is dubious — the ANE is not schedulable
for this workload; treat that claim as unverified. Work pack WP-3.

**A7 — Arena-per-request control-plane state (minor).**
Non-hot-path session/request state on Node A uses arena allocators torn down
wholesale per request. The decode hot path remains zero-allocation regardless
(spec §7.1). One-line spec addition.

### Rejected

**R4 — Gated DeltaNet / hybrid-attention memory optimization.**
Wrong model. Qwen3-235B-A22B-Instruct-2507 uses full GQA attention in all 94 layers;
the linear-attention hybrid the review describes is the Qwen3-Next family. There are
no DeltaNet layers to give a constant-memory state buffer, so the promised ~75% KV
saving does not exist for this target. This is the second external review to carry
wrong model constants — the §7.4 GGUF load-time verification guard exists for exactly
this failure mode. No design change.

**R5 — Stream cold experts from NVMe per token (8–12 resident experts only).**
As stated, violates two binding rules: the steady-state decode hot path must be
UMA-resident, and blocking NVMe reads in steady-state decode must be zero (spec §10).
With top-8 routing over 128 experts per layer, a 90%-streamed expert pool puts NVMe
on the critical path of nearly every token. DS5's existing tiered residency
(hot/warm/cool/cold with prefetch and promotion, p95 cold-miss stall budget) is the
compliant version of the same 80/20 insight. No change beyond A5's measured inputs.

## Dispositions — "Airplane review Arch"

### Adopted

**A8 — Zero-copy weight mapping stated as principle.**
GGUF tensor data is mmap'd and pointer-cast in place (alignment-checked); the loader
never copies weight bytes through the heap. This was implicit in the M2 GGUF-parser
plan; it is now spec §7 language binding the implementation.

### Convergent (no change)

- **Raw framed TCP, no HTTP/gRPC:** already DS5 — M0 transport landed with framed
  messages; packet contracts are frozen in ADR-005. The review's proposed frame header
  is a rougher sketch of what already exists.
- **"Silent link when both experts are local":** trivially true of DS5's placement;
  the M1 decode-sim's parameterized remote-expert rate quantifies it.

### Rejected

**R6 — "Brain & Muscle" hub-and-spoke execution pipeline.**
The review proposes: Node A computes prefill, all attention, all routing, and owns all
KV; workers are stateless expert-matmul servers; every MoE layer does an
A→worker→A round trip. Rejected on arithmetic and on locked decisions:

- *Latency:* 94 MoE layers × 2 boundary crossings = 188 mesh crossings per token,
  serialized. At an optimistic 0.3ms RTT that is ~56ms/token of pure transport —
  a <18 tok/s ceiling before any compute — versus DS5's layer-ownership design
  (B owns one layer span, C the other) where a token crosses the mesh O(1) times.
- *Memory/bandwidth:* A (48GB, 307GB/s) cannot hold 94 layers of attention weights
  plus the full KV working set while also being the lowest-bandwidth node; DS5
  assigns KV ownership to the workers by layer for this reason.
- *Locked decision:* B/C-local router mirrors are the only routing path (ADR-002 #1);
  centralizing routing on A reintroduces the route-through-A design that was
  explicitly cut.

The one durable insight — Node A as sampling/logit owner and scheduler — is already
DS5's role assignment for A.

**R7 — `@cImport` ggml.c for I-quant dequantization math.**
Violates ADR-002 (no ggml linked into the runtime). The risk it addresses is real and
already managed: oracle fixtures provide the correctness net, and ADR-002's review
trigger defines the vendoring fallback if I-quant kernels prove intractable. No change.

**R8 — "Dispatch layer N+1 while workers finish layer N."**
Incoherent as stated for autoregressive decode: layer N+1's input is layer N's output
within the same token. The sound versions — overlapping transport with compute across
expert dispatches within a layer, and prefetching predicted misses — are already
spec §7.5, designed from decode-sim traces.

## Net effect on plan

No milestone, gate, or ADR changes. M1 gains one additional offline data product
(per-expert sensitivity stats alongside the existing routing-skew capture); M2 kernel
design gains one input memo (Metal 4 spike); M4's manifest format gains a per-expert
precision column with a measured-provenance rule. Work is packaged for low-cost agents
in `docs/work-packs/2026-07-12-review-incorporation/`.
