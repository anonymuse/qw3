# ADR-006: MPP (`matmul2d`) Scope Under ADR-002's Zero-Dependency Rule

**Status:** Accepted
**Owner decision (2026-07-13):** Accept — MPP `matmul2d` is in scope. Explicit rationale
from the project owner: reading ADR-002's zero-dependency rule to exclude Apple's own
header-only shader templates (the same tier as `MTLTensor`/MSL intrinsics the runtime
already assumes) would be an unnecessarily narrow interpretation of "no linked ML
library" — it is not what the rule was written to prevent.
**Date:** 2026-07-13
**Depends on:** ADR-002 (kernel strategy), ADR-005 (interface freeze)
**Input:** `docs/findings/metal4-kernel-features.md` (WP-3), §7.4 "ADR-002 boundary note"
and Q2; review disposition A6 (`docs/reviews/2026-07-12_airplane_arch_reviews_response.md`)

## Question

ADR-002 forbids linking ggml, llama.cpp, or MLX into the runtime — "someone else's ML
library" is out, full stop. WP-3's Metal 4 memo found that Apple's Metal Performance
Primitives (MPP) `tensor_ops` library, specifically `matmul2d`, is the only public route
to the M5 Neural Accelerator, worth ~2-4x on prefill/batch GEMMs (decode is unaffected —
it stays memory-bandwidth-bound at M=1). The memo flagged, but did not resolve, whether
`matmul2d` counts as "linking someone else's ML library" or as "using the platform API,"
and asked for owner sign-off before M2 kernel work assumes an answer either way.

## Decision

**MPP `tensor_ops`/`matmul2d` is in scope for the M2 prefill/batch kernel path.**

Rationale for treating it as platform API, not a linked library:

1. **Distribution mechanism.** MPP ships as header-only MSL shader templates compiled
   directly into our own `.metal` source, the same way `simdgroup_matrix` intrinsics
   already used elsewhere in the kernel set are compiled in. There is no separate
   runtime, compiled artifact, or framework beyond `Metal.framework` itself — already a
   load-bearing dependency for every kernel DS5 has, including the ones written from
   scratch. Nothing new is linked.
2. **Level of abstraction.** MPP exposes a fixed-shape primitive (`matmul2d_descriptor`
   with explicit m/n/k, explicit execution-scope participation rules) that we call from
   inside a kernel we still write, control the dispatch of, and validate against oracle
   fixtures ourselves. It does not choose an algorithm, fuse ops across kernel
   boundaries, manage its own graph, or make placement/scheduling decisions on our
   behalf.
3. **Precedent already set.** A6's disposition (2026-07-12) rejected MPSGraph by name as
   "exactly the 'someone else's ML library' that ADR-002 excludes." MPSGraph is a
   graph-compilation framework: you hand it a computation graph and it decides how to
   execute it, black-box. MPP is the opposite shape — a primitive you invoke inside code
   you wrote, comparable to calling a compiler intrinsic. The two are not the same tier,
   and treating them the same would also require reclassifying `simdgroup_matrix` as
   "vendored," which nobody proposed.
4. **No interface impact.** Per WP-3 Q2/implications table, `matmul2d` usage is an
   implementation detail entirely inside kernel bodies. It does not require changing the
   frozen kernel API or dtype set (ADR-005) and is not proposed here as an ADR-005
   amendment.

**Scope of the acceptance is narrow and matches the memo's findings, not a blanket
approval of Metal 4 tensor features:**

- In scope: `MTLTensor` + `matmul2d` (macOS 26, f16-in/f16-or-f32-accumulate-out) for
  **prefill and batched-expert GEMM shapes only** — Q8_0 (later Q4/I-quant) matmul and
  the fused expert MLP, where inputs are dequantized into threadgroup memory first and
  the shipped `matmul2d` call replaces a hand-rolled tiled GEMM loop.
- Out of scope, unchanged from WP-3: MPSGraph and MPS kernels (still forbidden by
  ADR-002, per A6); macOS-27-beta features (fp8/fp4/int2 element types, native
  quantized-tensor scale planes) — not load-bearing for M2 regardless of this ADR;
  cooperative tensors — "investigate later," no ruling needed yet since nothing in M2
  requires them.
- Decode-path kernels are untouched by this decision — MPP does not apply to M=1 GEMV
  and the memo found no exception to that.

## The tradeoff, stated plainly

Accepting MPP buys ~2-4x measured prefill/batch-GEMM headroom on M5-generation hardware
for a marginal-cost integration (a primitive call inside kernels we already own and
validate), at the cost of a slightly less absolute reading of "zero dependency" —
critics could point to MPP as "Apple's matmul library" rather than "raw Metal," even
though it ships as compiled-in shader source with no separate binary. Rejecting it keeps
"zero dependency" airtight and simple to state in the publishable artifact, at the cost
of leaving 2-4x of measured prefill headroom on the table and forcing a hand-rolled
`simdgroup_matrix` GEMM to chase performance MPP gets for less code — with no effect on
the decode-throughput number (G4) that is DS5's actual headline metric either way.

## Rejected alternative

**Reject MPP; MPP out of scope alongside MPSGraph/MPS.** M2's prefill path would use
`simdgroup_matrix`-based hand-rolled GEMM only, forgoing the measured ~2-4x prefill
headroom, in exchange for a maximally strict zero-dependency framing ("we call
`Metal.framework` and write every matrix operation ourselves, full stop"). Rejected by
project owner as an unnecessarily narrow reading of ADR-002 — see decision rationale
above. Decode-path design, ADR-002's core commitments, and ADR-005's frozen interfaces
would have been unchanged either way, so this was a pure performance-vs-framing
tradeoff, not a structural one.

## Review triggers

Revisit if a macOS 27 GA changes MPP's distribution model (e.g., it moves into a
separately versioned/linked framework rather than header-only shader templates), or if
prefill/batch-GEMM performance is not actually load-bearing for any M2-M4 milestone (in
which case the question is moot and this ADR can be marked superseded).

## Amendments

*(none yet)*
