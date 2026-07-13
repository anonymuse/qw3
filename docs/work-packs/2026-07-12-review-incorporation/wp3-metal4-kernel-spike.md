# WP-3: Metal 4 / M5-Generation GPU Feature Spike

**Adopts:** A6 (Metal 4 kernel-feature investigation)
**Owns:** `docs/findings/metal4-kernel-features.md` (new)
**Must not touch:** `src/`, ADRs, spec, other packs' files
**Time-box:** Research memo only — no prototype code

## Goal

Give M2 kernel work (RMSNorm → Q8_0 dequant+matmul → RoPE → GQA attention →
router/top-8 → fused expert MLP) an evidence-based memo on which Metal 4 and
M5-generation GPU features to design around, and which claims from the 2026-07-12
external review to discard. Web research with citations; every claim labeled
verified (with source) or unverified.

## Questions to answer

1. **bf16:** Native bf16 arithmetic support in MSL on M5-generation GPUs — types,
   conversion costs, matmul viability vs fp16 for DS5's activation dtype choices
   (check against the frozen dtype set in `src/shared/contracts.zig` / ADR-005).
2. **Tensor/cooperative-matrix intrinsics:** Current MSL tensor ops
   (MTLTensor, cooperative matrices, simdgroup matrix ops) — availability on M5,
   applicability to Q8_0 dequant+matmul and the fused expert MLP, constraints
   (tile shapes, dtypes).
3. **Sub-byte/quantized types:** Any *native* Metal 4 support for sub-byte or
   block-quantized formats relevant to Q4/I-quant kernels, vs hand-written
   bit-unpacking in MSL. The review claimed native support — verify or debunk.
4. **ANE:** Confirm (briefly) that the Neural Engine is not schedulable for custom
   decode-loop kernels; one paragraph, kill the review's ANE-offload suggestion
   cleanly.
5. **What llama.cpp/MLX do on M5** — for context only (informing, not vendoring):
   which of these features their Metal paths use as of mid-2026.

## Constraints

- Raw MSL / Metal API features only. MPSGraph and MPS kernels are out — ADR-002
  excludes other people's ML libraries from the runtime, and the memo should say so
  where relevant rather than evaluating them.
- Recommendations must map to the M2 kernel list and the frozen kernel API
  (ADR-005) — a feature that would force an interface change needs an explicit
  "requires ADR-005 amendment" flag.

## Acceptance

- Memo answers all five questions with citations (Apple docs/WWDC sessions/release
  notes preferred; community sources labeled as such).
- Ends with a one-page "M2 kernel design implications" section: use / don't use /
  investigate-later per feature.
- No claim from the external review is repeated without independent verification.
