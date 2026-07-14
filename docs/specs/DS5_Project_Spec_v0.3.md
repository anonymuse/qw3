# DS5 Project Specification v0.3: Qwen3-235B-A22B Target

**Document type:** Project specification
**Status:** Active — supersedes `imported_v0.2/DS5_Project_Spec_v0.2_Qwen3_235B_A22B.md`
**Date:** 2026-07-10
**Companion:** `DS5_Execution_Plan_v0.3.md` (same planning generation)
**Governing decisions:** ADR-001 (model), ADR-002 (kernels), ADR-003 (bring-up),
ADR-004 (aux hardware), ADR-005 (interface freeze)
**Review inputs:** `docs/reviews/2026-07-10_gemini_arch_review_response.md`,
`docs/reviews/2026-07-12_airplane_arch_reviews_response.md` (amendments A5–A8)

---

## 1. Project statement and thesis

DS5 is a bespoke local distributed inference system for running **Qwen3-235B-A22B**
on a three-node Apple Silicon Thunderbolt 5 mesh, built as a narrow, model-specific
runtime rather than a general-purpose inference stack.

> **Thesis:** A narrow runtime specialized to one sparse MoE model, one hardware
> topology, and one long-horizon local-agent workload can produce a better
> quality/latency/control tradeoff than a general inference runtime on the same cluster.

DS5 is a **partitioned-model** system, not an ensemble: one model's experts and layers
are distributed across nodes with exact routing semantics preserved. No node holds a
complete model; no logit fusion across independent models occurs anywhere in the design.

## 2. Goals (refined, in priority order)

| # | Goal | Measured by |
|---|---|---|
| G1 | **Correctness before speed.** Every kernel and every distributed path matches oracle fixtures; distributed output equals single-node output deterministically | M2/M3 gates; zero intentional top-8 deviations |
| G2 | **An honest viability finding.** Measured links + measured routing skew + placement simulation produce a go/no-go ceiling decomposition — publishable even if negative | `docs/findings/f001` (end of week 1) and successors |
| G3 | **Zero-dependency runtime.** All inference math from scratch in Zig + Metal; no ggml/llama.cpp/MLX linked (offline oracle use only) | ADR-002; `src/kernels/` isolation preserved as the fallback seam |
| G4 | **235B distributed decode at practical context.** >12 tok/s target at 8K–32K, evaluated honestly against measurements, not asserted | M4 benchmarks vs f001 projection |
| G5 | **Reproducibility from day one.** Every binary emits run-metadata JSON; benchmarks re-runnable to <10% variance | Benchmark Spec §5; M0 gate style throughout |

Goal order is load-bearing: no goal may be advanced by weakening a higher-numbered
gate (e.g., no throughput work that bends routing semantics).

## 3. Non-goals and deferrals

| Item | Status | Reason |
|---|---|---|
| Logits ensembling / test-time model fusion (PackLLM, STM) | **Rejected** | Category mismatch: DS5 partitions one model; there are no independent replicas to fuse (review response R1) |
| Kernel-level transport (IOKit driver, XNU bypass) | **Rejected** | Measure-first rule; contingency ladder in §8 caps at `Network.framework` (review response R2) |
| Speculative decoding (drafter on Node A, linear or tree) | Deferred post-M4 | Sequencing: correctness gates first; adopt only on measured verify-batch scaling |
| Agentic harness, PRM/output gating, tool-call decoding | Deferred (v0.2 Phase 5) | After M4 numbers exist |
| 64K+ context | Deferred | 8K/32K must be stable first |
| 128K–262K+ context, 1M context | Research-gated / out of scope | Requires sparse-attention/KV-compression work not scheduled |
| Custom quantization pipeline | Deleted (v0.3) | GGUF ingestion of existing community/`llama-quantize` artifacts |
| Dense baseline, route-through-A correctness mode | Cut (ADR-002/003) | Oracle fixtures replace both |
| Multi-user serving, non-Apple backends, model-plugin abstractions | Out of scope | Narrowness is intentional |
| Placement/prefetch optimizations (co-activation grouping, replicated-attention expert-parallel hybrid, router-driven prefetch) | Deferred to Phase 2 | Proven but out of V1 scope; trigger conditions in `docs/backlog/DS5_Phase2_Optimization_Backlog.md` |

V1 topology is layer-parallel (placement spec §3): B owns layers 0–46, C owns 47–93,
A is control plane / sampling / lm_head / cold-expert host. The layer-parallel vs
expert-parallel decision and the deferred optimizations above are recorded in the
Phase-2 backlog.

## 4. Hardware assumptions

| Node | Role | Class | Memory | Bandwidth | Static-weight cap |
|---|---|---|---:|---:|---:|
| A | Control plane, scheduler, tokenizer/sampling owner, KV page-table owner, expert-placement controller | M5 Pro | 48GB UMA | 307GB/s | 33.6GB |
| B | Decode worker | M5 Max | 48GB UMA | 614GB/s | 33.6GB |
| C | Decode worker | M5 Max | 48GB UMA | 614GB/s | 33.6GB |

Budget rule (per node): 70% static weights (33.6GB), 30% runtime reserve (14.4GB) for
KV pages, Metal heaps, staging/ring buffers, page tables, and OS overhead. Cluster
static cap: 100.8GB. The loader refuses manifests exceeding per-node caps without
explicit override. Link performance is **measured, never assumed** — M0 (landed,
commit 9eb940c) provides RTT/bandwidth/jitter per node pair from the real mesh.

Dev note: the development laptop (M5 MacBook Air, 24GB) is not a cluster node; cluster
runs follow `docs/runbook.md`.

## 5. Model constants (authoritative)

| Dimension | Value |
|---|---:|
| Total / activated parameters | 235B / 22B |
| Layers | 94 |
| Experts / activated | 128 / top-8 |
| Hidden size | 4096 |
| Attention | GQA — 64 Q heads, 4 KV heads, head dim 128 |
| Native context | 262,144 |
| First performance context | 8K–32K |

These constants are compiled into the runtime per ADR-005 and **verified against GGUF
metadata at load; mismatch is a refusal, not a warning.** External documents (including
architecture reviews) have carried wrong constants for this model; GGUF metadata plus
the HF config are the only accepted sources.

Bring-up model: **Qwen3-30B-A3B-Instruct-2507** (sole bring-up model, ADR-003), same
architecture family, validated against oracle fixtures (`tests/fixtures/`).

## 6. Architecture overview

- **Partitioning:** Experts and layers of the single 235B model are placed across B/C
  per a placement manifest, with tiered expert residency (hot/warm/cool/cold) driven by
  **measured** routing skew (M1 telemetry capture), not assumed skew. Node A holds no
  decode-critical expert weights; it may execute remote experts as overflow (M4).
- **Routing:** B/C-local router mirrors are the only routing path. Top-8 semantics are
  never altered — no capacity factors, no routing shortcuts, no approximations.
- **Transport:** Length-prefixed framed messages over TCP via the raw libc layer
  (`src/shared/sys.zig`). Activation packets, one-packet-per-destination, checksums
  (M3). See §8 for the contingency ladder.
- **KV:** Layer-owned KV pages, global page table on A, quantized/old-page policy per
  the KV spec.
- **Correctness:** Oracle fixtures (DS5T format, `src/shared/fixture.zig`, frozen
  tolerance rule) generated offline by `tools/make_fixtures.py` from reference
  implementations. Reference tools are never linked into the runtime.
- **Quantization:** Community/`llama-quantize` GGUF artifacts only — no custom
  quantization pipeline. For 235B (M4), per-expert precision in the placement/quant
  manifest is assigned from **two measured axes**: routing frequency (M1 telemetry)
  and quantization sensitivity (offline imatrix/KLD calibration), recorded in
  `expert_stats.json`. Precision assignments must trace to measured stats, never to
  assumed skew. Router/gate tensors FP16/Q8 and KV-sensitive tensors Q8/Q6-class
  remain binding (v0.2 quality rules). (Adopted 2026-07-12, A5.)

## 7. Runtime engineering principles (new in v0.3)

Adopted from the 2026-07-10 review where sound; stated here so they bind implementation.

1. **No hot-path allocation.** All steady-state decode memory is allocated at load from
   the fixed §4 budgets (fixed-buffer/arena allocators). The decode loop performs zero
   heap allocation and zero Metal heap growth.
2. **Ring-buffered KV pages.** Context pruning and page recycling advance pointers over
   preallocated pages; `free()` never appears in the decode path.
3. **Sparse wire payloads.** Nothing crosses the mesh at full width when a top-k
   projection suffices: routed-expert activations only, top-8 IDs + gate weights, and
   top-K logits (K ≤ 64) if sampling ever moves remote from the final layer. Decode-sim
   (M1) quantifies every payload class before M3 freezes packet formats.
4. **Comptime specialization, load-time verification.** Model constants are comptime
   per ADR-005; the loader cross-checks GGUF metadata and refuses mismatches (§5).
5. **Latency hiding by pipelining, measured first.** Overlap of transport and compute
   is designed from decode-sim traces (M1), not from assumed RTTs. Within-token layer
   dependencies are sequential; overlap happens across expert dispatches and prefetch,
   never by pretending layer N+1 can start before layer N completes.
6. **Zero-copy weight mapping.** GGUF tensor data is mmap'd and pointer-cast in place
   (alignment-checked); the loader never copies weight bytes through the heap.
   (Adopted 2026-07-12, A8.)
7. **Arena-per-request control state.** Non-hot-path session/request state on Node A
   uses arena allocators torn down wholesale per request; the decode hot path remains
   allocation-free per (1). (Adopted 2026-07-12, A7.)

## 8. Transport strategy ladder

| Rung | Mechanism | Status |
|---|---|---|
| 1 | TCP over TB5 via raw libc (`sys.zig`) | **Current.** M0 measured; M1 decode-sim decides if this is the binding constraint |
| 2 | `Network.framework` custom framing | Contingency only — requires decode-sim evidence that rung 1's RTT bounds the token ceiling below the >12 tok/s target |
| 3 | Kernel-level driver (IOKit/DriverKit "raw DMA") | **Never.** Not viable on modern macOS; rejected in review response R2 |

Any move to rung 2 is an ADR, gated on a finding, and keeps the framing/packet
contracts of ADR-005 unchanged.

## 9. Milestones

Inherited from `DS5_Execution_Plan_v0.3.md` (authoritative for gates and windows):
M0 mesh reality (**landed**) → M1 viability model + f001 → M2 single-node 30B-A3B
engine core → M3 distributed correctness → M4 235B placement + runtime → M5 findings.

Week-1 compressed push (2026-07-09..): day 1 shipped the ADR-005 interface freeze,
DS5T fixture loader, CPU reference context, and the synthetic fixture set + oracle
generator (verified vs HF to 7e-7). Week goal: 30B-A3B distributed across two nodes
correct (M2+M3 gates) plus the f001 235B viability finding; 235B execution is stretch,
never at the cost of correctness gates.

## 10. Acceptance criteria

### Functional
- Qwen3-235B-A22B-Instruct-2507 metadata parsed; 94 layers mapped to B/C per manifest.
- Top-8 routing matches reference within fixture tolerance; **zero intentional
  semantic deviations**.
- Distributed token output equals single-node output under deterministic settings.
- GGUF constant-verification refusal path tested (deliberately wrong metadata fixture).

### Performance
| Metric | Target |
|---|---:|
| Decode throughput, 8K context | >12 tok/s after optimization |
| Decode throughput, 32K context | >12 tok/s target, benchmarked honestly |
| Blocking NVMe reads in steady-state decode | 0 |
| Decode-loop heap allocations (steady state) | 0 (§7.1) |
| Per-node static weights | ≤33.6GB |
| Benchmark reproducibility | <10% variance across 3 runs |

### Quality
- Hot experts quantized higher-fidelity than cold tiers; router/gate tensors FP16/Q8;
  KV-sensitive tensors Q8/Q6-class unless evals justify lower.
- Findings (`docs/findings/`) publishable even when negative — a well-decomposed
  "not viable at >12 tok/s" is an acceptable project outcome (G2).

## 11. Binding execution constraints

Unchanged from v0.3 execution plan: no kernel optimization before placement simulation
proves memory feasibility; no routing shortcuts before top-k parity is validated; no
generic model-plugin abstractions; no 128K+ context work until 8K/32K are stable;
loader refuses over-cap manifests without explicit override.

## 12. External review policy (new in v0.3)

External architecture reviews are inputs, not authorities. Each review gets a
disposition document in `docs/reviews/` (adopt / defer / reject with reasons) before
any spec change; claims about model architecture are checked against GGUF/HF-config
ground truth; claims about prior art are verified firsthand before citation. Locked
decisions (ADRs) change only through the ADR process, never directly from a review.
