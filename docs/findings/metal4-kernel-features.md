# Findings: Metal 4 / M5-Generation GPU Features for DS5 Kernels

**Work pack:** WP-3 (`docs/work-packs/2026-07-12-review-incorporation/wp3-metal4-kernel-spike.md`), adopting A6 from the 2026-07-12 airplane review response
**Date:** 2026-07-12
**Type:** Research memo — no code, no prototypes
**Feeds:** M2 kernel design (RMSNorm → Q8_0 dequant+matmul → RoPE → GQA attention → router/top-8 → fused expert MLP)
**Scope constraint:** Raw MSL / Metal API features only. MPSGraph and MPS kernels are excluded by ADR-002 and are not evaluated here (noted where relevant).

**Claim labeling:** Every load-bearing claim is marked **[VERIFIED]** with a source, or **[UNVERIFIED]**. Sources are classed *Apple* (docs, WWDC sessions, release notes, Apple ML Research) or *Community* (papers, repos, blogs). Per the pack rules, no claim from the external review is repeated without independent verification.

---

## Background: what shipped when

- **Metal 4** (WWDC25, macOS 26): new command model, and — relevant here — `MTLTensor` as a first-class resource type alongside buffers/textures, plus the shader-side **Metal Performance Primitives (MPP)** `tensor_ops` library (`matmul2d`, convolution, reductions). [VERIFIED — Apple: [What's New in Metal](https://developer.apple.com/metal/whats-new/), [MPP Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)]
- **M5 generation** (M5 late 2025; M5 Pro / M5 Max 2026): first Apple GPUs with a dedicated **Neural Accelerator in each GPU core** — dedicated matmul hardware, distinct from the ANE. M1–M4 GPUs ran all matrix math on the general FP32 ALU pipeline (`simdgroup_matrix` improved utilization but had no dedicated hardware behind it). [VERIFIED — Apple: [Tech Talk 111432, "Accelerate your machine learning workloads with the M5 and A19 GPUs"](https://developer.apple.com/videos/play/tech-talks/111432/); Community: [tzakharko, "Investigating the GPU Neural Accelerators on Apple A19/M5"](https://tzakharko.github.io/apple-neural-accelerators-benchmark/), [Apple newsroom M5 announcement](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)]
- **WWDC26 / macOS 27 (beta as of this memo)**: extended tensor dtypes (fp8, fp4, int2) and multi-plane quantized tensors with block-wise scale factors. [VERIFIED — Apple: [WWDC26 session 330, "Optimize custom machine learning operations with Metal tensors"](https://developer.apple.com/videos/play/wwdc2026/330/)]

**Environment note for DS5:** cluster nodes are M5 Pro (A) / M5 Max (B, C); the dev laptop is an M5 Air. All are M5-generation and have GPU Neural Accelerators. Features gated on macOS 27 are **beta-only as of 2026-07** and must not be load-bearing for M2. [VERIFIED for M5 Pro/Max Neural Accelerators — Apple: [What's New in Metal](https://developer.apple.com/metal/whats-new/); base-M5 laptop parity assumed from the M5 announcement — minor residual risk, flagged, but Apple's Tech Talk covers "M5 and A19 GPUs" generally.]

---

## Q1 — bf16: native arithmetic on M5-generation GPUs

**MSL type support: yes.** MSL has a native `bfloat` type (declarations, arithmetic, conversions, `bfloat4` vectors, texture/buffer IO) — introduced in MSL 3.1 (2023-era toolchains) and present in the current spec. This is a GPU/MSL fact; note it is *unrelated* to host-side Clang, where `__bf16` is still rejected on Apple targets ("__bf16 is not supported on this target"). [VERIFIED — Apple: [Metal Shading Language Specification v4.1 (PDF)](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf); host-side limitation per [Apple Developer Forums thread 726201](https://developer.apple.com/forums/thread/726201). Community corroboration of the 2023 introduction: [HN, "Bfloat16 support coming to Apple's Metal and PyTorch"](https://news.ycombinator.com/item?id=36575443)]

**Neural Accelerator support: no (or not exposed).** The M5/A19 Neural Accelerator hardware paths measured to date are **FP16 (FP16 or FP32 accumulate)** and **INT8 (INT32 accumulate)**. bf16 is a "notable omission — it is unclear whether the first-generation Neural Accelerator hardware lacks dedicated support for this format or whether it is not yet exposed in the Metal Shading Language." [VERIFIED as a community measurement — [tzakharko benchmark](https://tzakharko.github.io/apple-neural-accelerators-benchmark/). Apple's own materials list fp16/int8 (and, in macOS 27, fp8/fp4/int2) tensor dtypes and never advertise accelerated bf16 matmul — [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/).]

Two secondary observations:

- llama.cpp's Metal 4 tensor-API path does accept bf16 tensors, but had to gate bf16 off below macOS 26.1 due to framework incompatibilities — i.e., bf16 *works* through the tensor API on current OSes but is the least mature dtype in that stack. Whether it runs on the Neural Accelerator or is converted/emulated is **[UNVERIFIED]**. [Community: [llama.cpp PR #16634](https://github.com/ggml-org/llama.cpp/pull/16634)]
- Conversion costs are trivial in either direction: bf16↔f32 is a 16-bit truncate/extend; bf16→f16 is a narrowing conversion with range clamping. Per-element converts in a bandwidth-bound kernel are effectively free relative to memory traffic. [VERIFIED as format arithmetic — bf16 is the top 16 bits of f32 by construction; [bfloat16 format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)]

**Verdict for DS5.** The frozen dtype set (`src/shared/contracts.zig` `HiddenDtype`: f16 | bf16 | f32; ADR-005) is compatible with reality: keep **f16 as the matmul/compute dtype** (only 16-bit dtype with verified hardware matmul), keep **f32 accumulate** everywhere, and treat **bf16 as a wire/interchange dtype only**, converted at kernel boundaries. Choosing bf16 as the *compute* dtype for matmul-heavy kernels would forfeit the Neural Accelerator on current evidence. No ADR-005 amendment required.

---

## Q2 — Tensor / cooperative-matrix intrinsics

Three tiers exist in current MSL, all usable from raw shaders (no MPSGraph/MPS involvement):

1. **`simdgroup_matrix`** (Metal 2.3+, all Apple-Silicon GPUs): 8×8 simdgroup matrix ops, half/float. On M1–M4 this is the *only* matrix path and runs on general ALUs; on M5 it does **not** use the Neural Accelerator. Portable, well-understood, and the fallback path llama.cpp keeps for pre-M5 devices. [VERIFIED — Apple: [MSL Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf); Community: [llama.cpp PR #16634](https://github.com/ggml-org/llama.cpp/pull/16634) (tensor API disabled pre-M5 because it regressed vs simdgroup path)]

2. **`MTLTensor` + MPP `tensor_ops` (`matmul2d`)** (Metal 4, macOS 26): tensors bound to shaders; `matmul2d_descriptor(m, n, k)` fixes the output tile per threadgroup; execution scopes are explicit (`execution_simdgroup`, `execution_simdgroups<N>`, threadgroup) and **all threads in the scope must participate or behavior is undefined**. This is the only public route to the M5 Neural Accelerator. [VERIFIED — Apple: [MPP Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf), [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/), [Tech Talk 111432](https://developer.apple.com/videos/play/tech-talks/111432/)]

3. **Cooperative tensors** (register-resident tensor fragments): hold matmul results in thread registers, avoiding a threadgroup-memory round trip (the FlashAttention pattern). Early M5 measurements said cooperative tensors could not be *inputs* to matmul; WWDC26 states macOS 26+ allows passing them directly as matmul inputs with compatibility checking. Treat the exact capability matrix as version-dependent. [VERIFIED both statements — Community: [tzakharko](https://tzakharko.github.io/apple-neural-accelerators-benchmark/); Apple: [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/)]

**Constraints that matter for kernel design:**

- **Dtypes:** hardware-accelerated matmul is fp16 (fp16/fp32 out) and int8 (int32 out) on M5. fp8/fp4/int2 tensor dtypes exist only from macOS 27 (beta). [VERIFIED — [tzakharko](https://tzakharko.github.io/apple-neural-accelerators-benchmark/); [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/)]
- **Tile shapes:** the accelerator wants large tiles — "at least 32×32 or even 32×64" chunks to reach peak; per-core peak is ~1024 FP16 FLOPS/cycle (~2× for INT8). Estimated ~70 TFLOPS FP16 on M5 Max. [Community measurement — [tzakharko](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)]
- **The decode-shape problem:** single-token decode is GEMV (M=1). A 32×32-tile matmul engine cannot be filled by M=1 work; the accelerator's headline wins are in compute-bound *prefill*. Apple's own MLX numbers make this concrete: ~4× faster time-to-first-token on M5 vs M4, but only 19–27% faster token *generation* — decode remains memory-bandwidth-bound. This is the single most important fact in this memo for DS5, whose G4 target is decode throughput. [VERIFIED — Apple: [Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU"](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)]
- **Accumulator/alignment fine print is under-documented.** Community reverse-engineering of the M4 tensor path (no accelerator, same API) found fp16 matmul accumulates in ≥fp32 (matching NVIDIA convention) but Apple's spec doesn't state it, and found undocumented inner-stride alignment rules (e.g., 32-element alignment for float host-bound tensors). Expect to validate numerics against oracle fixtures rather than trust documentation. [Community — [Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on the Apple M4 Max GPU (arXiv 2606.12765)](https://arxiv.org/pdf/2606.12765)]

**ADR-002 boundary note.** MPP `tensor_ops` is a header-only MSL template library compiled *into our own shaders* — Apple positions it as part of the Metal 4 shading language surface, not a runtime ML library. Reading ADR-002's exclusion ("no ggml/llama.cpp/MLX linked") and WP-3's constraint ("MPSGraph and MPS kernels are out"), MPP is **in scope**: it is the raw-MSL route to the hardware, more analogous to `simdgroup_matrix` intrinsics than to MPSGraph. This memo proceeds on that reading but flags it for owner sign-off, since "no one else's ML code in the runtime" could be read more strictly.

**Applicability:** Q8_0 dequant+matmul and the fused expert MLP in their *prefill/batch* forms map directly onto `matmul2d` (dequantize into threadgroup memory or scale planes, then tensor-op the GEMM — exactly llama.cpp's shipped pattern). Their *decode* (GEMV) forms do not benefit on current evidence.

---

## Q3 — Sub-byte / quantized types: verify or debunk the review's "native support" claim

**Verdict: partially true, and false in the way that matters for M2.**

What actually exists, by OS version:

| Capability | Status | Source |
|---|---|---|
| int8 / int4 tensor element types | macOS 26 (shipping) | [VERIFIED — Apple: [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/), [Metal machine-learning passes docs](https://developer.apple.com/tutorials/data/documentation/metal/machine-learning-passes.md)] |
| fp8 (E4M3), fp4, int2 tensor element types | macOS 27 **beta** | [VERIFIED — Apple: [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/); Community: [Rigel (arXiv 2606.12765)](https://arxiv.org/pdf/2606.12765) — fp8/fp4 paths require the macOS 27 + Xcode 27 toolchain] |
| Multi-plane quantized tensors: element plane + block-wise **scale plane** (e.g., 32×1 blocks; `MetalFloat8UE8M0` power-of-two scales), with `matmul2d` dequantizing automatically | macOS 27 **beta** | [VERIFIED — Apple: [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/), [What's New in Metal](https://developer.apple.com/metal/whats-new/) ("new quantized tensor formats and support for scale factors … native support in Metal Performance Primitives … Neural Accelerators on the M5 Pro and M5 Max")] |
| **GGUF block formats (Q8_0, Q4_0, Q4_K, IQ*) as stored** | **No native support at any OS version** | See below |

Why the last row is a "no":

- GGUF blocks are *interleaved* scale+data records — Q8_0 is a 34-byte record (one f16 scale + 32 int8 values, 8.5 bits/weight); K-quants use 256-element super-blocks with 6-bit sub-scales; I-quants add codebooks. [VERIFIED — Community: [llama.cpp quantization docs/DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/7.3-quantization-techniques)]
- Metal's quantized tensors want *separate planes* (contiguous element data + contiguous scale data), and the documented scale formats are MX-style (UE8M0 power-of-two, fp8-class) — not GGUF's f16 block scales. Whether an f16-typed scale plane is accepted is **[UNVERIFIED]** (worth one experiment in M2). Even in the best case, using native quantized matmul on GGUF weights requires an **offline repack** of each tensor into planes, and K-quant/I-quant sub-scale and codebook structures have no native mapping at all.
- The proof by counterexample: llama.cpp's shipped Metal-4 tensor path — written by the people most motivated to exploit native quantized types — still **dequantizes into threadgroup memory first**, then runs the f16 tensor op, noting register-direct dequant only as a future OS 26.2+ possibility. [VERIFIED — Community: [llama.cpp PR #16634](https://github.com/ggml-org/llama.cpp/pull/16634)]

**So: the review's claim of "native sub-byte support" is true for element *types* (int4 now; fp4/int2 in beta) and for MX-style block scaling (beta), but false for the thing DS5 actually loads — GGUF block formats.** Hand-written bit-unpacking in MSL remains the required approach for the Q8_0 → Q4 → I-quant kernel sequence (ADR-002 §2 sequencing unchanged). The native quantized-tensor path becomes interesting only as a macOS-27-era, offline-repacked, prefill-oriented optimization — and repacking would be a load-time transform, which must respect spec §7.6 (zero-copy mmap) or be explicitly carved out.

---

## Q4 — ANE: not schedulable; the review's offload suggestion is dead

The Apple Neural Engine is a separate fixed-function block with **no public kernel-level programming interface**: Core ML is the only public route, and it is a black-box scheduler that decides at runtime whether an op runs on CPU, GPU, or ANE — you cannot force ANE placement, cannot write custom ANE kernels, and cannot schedule it inside a custom decode loop. Everything that "runs on the ANE" outside Core ML does so via reverse-engineered private APIs (`_ANEClient`, `_ANECompiler`, MIL programs), which are unsupported, fragile across OS updates, and categorically outside DS5's engineering envelope. [VERIFIED — Community: [Orion: Characterizing and Programming Apple's Neural Engine (arXiv 2603.06728)](https://arxiv.org/html/2603.06728v1) ("CoreML is the only public interface to the ANE … a black-box scheduler"; developers "cannot force ANE execution"); [Apple Neural Engine: Architecture, Programming, and Performance (arXiv 2606.22283)](https://arxiv.org/pdf/2606.22283); [maderix/ANE private-API work](https://github.com/maderix/ANE); [Draw Things engineering on ANE in a custom stack](https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in)] Additionally, with M5 the strategic direction is explicit: Apple put matmul acceleration *into the GPU cores* precisely so custom Metal pipelines get it without the ANE — the Neural Accelerator supersedes ANE-offload for GPU-resident inference. [VERIFIED — Apple: [Tech Talk 111432](https://developer.apple.com/videos/play/tech-talks/111432/)] **The review's ANE-offload suggestion is rejected: no public schedulability, no custom kernels, black-box dispatch incompatible with a deterministic decode loop, and the hardware answer to the underlying wish already exists on the GPU.**

---

## Q5 — What llama.cpp and MLX do on M5 (context only, per ADR-002 not vendored)

- **llama.cpp** (Metal backend): merged initial Metal-4 tensor-API support ([PR #16634](https://github.com/ggml-org/llama.cpp/pull/16634), ggerganov). Pattern: matrix-*matrix* multiplication reworked to `matmul2d`; **quantized weights are dequantized into threadgroup memory, then tensor-op'd in f16**; f16/bf16/Q4_0-class paths exist; bf16 gated to macOS ≥26.1. Enabled only on M5-class devices (tensor path regressed slightly on M2–M4, which keep the `simdgroup_matrix` path); env kill-switch `GGML_METAL_DISABLE_TENSOR_API`. Reported gains: ~2× on larger-model prefill-heavy runs, ~19–23% end-to-end; community reports up to ~3.65× at ~20K-token prefill. Decode (GEMV/matvec) still runs the classic fused dequant+dot kernels. [VERIFIED — Community: PR above; [LM Studio issue #2040](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2040) (tensor path gated to M5 + macOS ≥26.2 in their runtime)]
- **MLX**: uses TensorOps/MPP to target the Neural Accelerators directly; Apple's own numbers on M5 vs M4: up to **4.06× faster TTFT (prefill, compute-bound)** but only **19–27% faster generation (decode, bandwidth-bound)**. MLX's quantized path remains its own affine scheme (uint32-packed groups, fused dequant+matmul kernels like `affine_qmm_t`) — i.e., MLX also does *not* rely on native sub-byte tensor formats for its shipped quantized inference. [VERIFIED — Apple: [Apple ML Research MLX/M5 post](https://machinelearning.apple.com/research/exploring-llms-mlx-m5); Community: [MLX issue #2693 (Metal 4 / M5 tensor-API adoption)](https://github.com/ml-explore/mlx/issues/2693), [TurboQuant-on-MLX writeup](https://medium.com/@antonrozanov/turboquant-on-mlx-4-6x-kv-cache-compression-with-custom-metal-kernels-9cdee3f7d2a2)]

**The consistent lesson from both:** tensor ops are a prefill/GEMM story; decode is bandwidth-bound and stays on hand-written fused dequant kernels. Both projects' quantized decode paths look like what ADR-002 already prescribes for DS5.

---

## M2 kernel design implications (use / don't use / investigate later)

DS5's G4 target is **decode throughput** (>12 tok/s at 8K–32K). On M5-class hardware, decode is memory-bandwidth-bound (614 GB/s on nodes B/C) and single-token GEMV cannot fill Neural Accelerator tiles. Therefore M2's center of gravity is unchanged by Metal 4: **hand-written, bandwidth-optimal MSL kernels with fused dequantization, f16 compute, f32 accumulate, validated against oracle fixtures.** Tensor ops are a bounded prefill-side add-on, not a redesign.

| Feature | Decision | Rationale / conditions |
|---|---|---|
| MSL `bfloat` arithmetic | **Use (boundaries only)** | Wire/interchange dtype per `contracts.zig`; convert to f16/f32 at kernel entry. Not a compute dtype: no verified accelerated matmul. No ADR-005 change. |
| f16 compute, f32 accumulate | **Use** | Only 16-bit dtype with verified Neural Accelerator matmul; f32 accumulate matches oracle-fixture tolerance expectations. |
| `simdgroup_matrix` (8×8) | **Use** | Baseline matrix path for attention/matmul inner loops where tensor ops don't apply; portable to every node incl. dev laptop. |
| Hand-written fused Q8_0 (later Q4/IQ) dequant+GEMV in plain MSL | **Use** | The decode path. Native quantized tensors cannot consume GGUF blocks; this is also what llama.cpp/MLX ship. Preserves ADR-002 sequencing and the frozen kernel API. |
| `MTLTensor` + MPP `matmul2d` (f16, macOS 26) | **Use (prefill/batch GEMM only)** | Prefill Q8_0 matmul and batched expert MLP: dequant to threadgroup mem → `matmul2d`. Up to ~4× prefill headroom on M5. Requires ~32×32+ tiles; all-threads-participate rule; validate accumulator numerics vs fixtures (docs are silent). Flag: MPP-in-scope reading of ADR-002 needs owner sign-off (see Q2). Fits behind the frozen kernel API (internal implementation detail) — no ADR-005 change. |
| Cooperative tensors (FlashAttention-style register residency) | **Investigate later** | Attractive for prefill GQA attention; capability matrix (matmul-input support) is OS-version-dependent. Decode attention doesn't need it. |
| INT8 tensor matmul (int8×int8→int32) for Q8_0 without dequant | **Investigate later** | ~2× fp16 throughput, but requires quantizing *activations* → changes numerics vs oracle. Only with fixture-tolerance proof, and only if prefill matters enough. |
| Native quantized tensor formats / scale planes (macOS 27) | **Investigate later (post-M2)** | Beta OS; MX-style scales ≠ GGUF f16 block scales ([UNVERIFIED] whether f16 scale planes exist); requires offline repack conflicting with §7.6 zero-copy mmap unless carved out. Revisit at M4 if I-quant prefill is a measured bottleneck. |
| fp8 / fp4 / int2 tensor dtypes | **Don't use (for M2–M4)** | macOS 27 beta only; no GGUF correspondence; quality rules (§6) pin router/KV tensors at FP16/Q8-class anyway. |
| bf16 as matmul compute dtype | **Don't use** | Not hardware-accelerated on M5 (community-verified omission); f16 + f32-accumulate covers the range needs of these kernels. |
| MPSGraph / MPS kernels | **Don't use** | Excluded by ADR-002; out of scope per WP-3. Recorded here only for completeness. |
| ANE offload | **Don't use** | Not schedulable for custom decode kernels via any public API (Q4). Review claim rejected. |

**Per-kernel summary (M2 list):**

- **RMSNorm, RoPE, router/top-8:** plain MSL, elementwise/reduction, bandwidth-bound — no Metal 4 feature applies. f32 accumulate for the RMSNorm reduction.
- **Q8_0 dequant+matmul:** decode = fused dequant GEMV (plain MSL); prefill = dequant-to-threadgroup + `matmul2d` f16 (use); int8-native path investigate-later.
- **GQA attention:** decode = simdgroup-level fused kernel; prefill = candidate for `matmul2d`/cooperative-tensor FlashAttention (investigate later).
- **Fused expert MLP:** decode = per-expert fused GEMV chain (plain MSL); batched/prefill expert GEMMs = `matmul2d` (use).

**Interface impact:** none of the "use" items requires changing the frozen kernel API or dtype set — tensor-op usage is an implementation detail inside kernels. The only future item that could touch load-path contracts is the macOS-27 repack idea (spec §7.6 zero-copy rule), flagged as **requires explicit spec/ADR review** if ever pursued. No ADR-005 amendment is proposed.
