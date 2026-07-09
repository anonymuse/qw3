# ADR-002: From-Scratch Zig + Metal Kernel Strategy

**Status:** Accepted
**Date:** 2026-07-08
**Depends on:** ADR-001 (model selection)

## Decision

The DS5 runtime implements all inference math from scratch in Zig + Metal. No ggml,
llama.cpp, or MLX code is linked into the runtime. Zero-dependency systems credibility is
part of the publishable artifact.

Three load-bearing consequences, accepted deliberately:

1. **Oracle-fixture testing replaces "correctness mode" routing.** We do not build the
   route-everything-through-Node-A correctness mode described in the placement spec.
   Instead, reference implementations (llama.cpp, HF transformers) are used **offline as
   oracles** to dump golden per-layer activations, router top-8 IDs, and gate weights for
   Qwen3-30B-A3B on deterministic prompts. Every Zig/Metal kernel is validated
   tensor-by-tensor against these fixtures (`tests/fixtures/`). B/C-local router mirrors
   are the only routing code path from day one.

2. **Quantization format sequencing.** Kernels are built in order of block-format
   complexity: `Q8_0` first (bring-up), then `Q4_0`/`Q4_K`, then I-quants (`IQ3_S`,
   `IQ2_M/XS/XXS`) only when 235B placement demands them. DS5 parses GGUF directly in
   Zig and reuses community/`llama-quantize` artifacts — **there is no custom
   quantization pipeline** (v0.2 epic E-006 is deleted until evidence requires it).

3. **Honest timeline.** First correct distributed 235B token is a weeks-4-to-6 outcome,
   not week 1. The week-1 publishable artifact is the Phase 0 viability finding
   (measured links + measured routing skew + placement simulation).

## Rejected alternatives

- **Vendor ggml-metal / link llama.cpp:** fastest to a running model, but the resulting
  artifact is "an orchestration layer around llama.cpp," which undercuts the narrow-engine
  thesis. Rejected by project owner.
- **MLX-based:** same objection, plus less control over memory layout and transport.

## Review triggers

Revisit if M2 (single-node 30B-A3B correctness) is not achieved within ~3 weeks of kernel
work, or if I-quant kernel correctness proves intractable solo. The fallback is vendoring
ggml kernels behind the existing manifest/transport/orchestration layer, which this
repo's structure keeps possible (kernels are isolated in `src/kernels/`).
