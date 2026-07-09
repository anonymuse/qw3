# DS5 Execution Plan v0.3 — Coding-First Restructure

**Status:** Active plan
**Date:** 2026-07-08
**Supersedes:** `imported_v0.2/DS5_Execution_Plan_Input_v0.2_Qwen3_235B_A22B.md` (phase
structure only; its gates and DoD are inherited where referenced)
**Governing decisions:** ADR-001 (model), ADR-002 (kernels), ADR-003 (bring-up),
ADR-004 (aux hardware)

## Changes from v0.2

1. Phases 1+2 merged; dense baseline cut (ADR-003).
2. Router-telemetry capture added to Phase 0 — the v0.2 Phase 0 table omitted the
   thesis-critical measurement. Expert tiering assumes routing skew; load-balanced MoE
   training argues against it. Measured before any tiering machinery is built.
3. Route-through-A correctness mode dropped in favor of oracle fixtures (ADR-002).
4. Quant pipeline (epic E-006) deleted; GGUF ingestion of existing artifacts instead.
5. Deferred entirely: speculative drafter, agentic harness (v0.2 Phase 5), 64K+ context,
   Mac minis, RTX box, Thinking-2507 variant.
6. Benchmark harness is not a late epic: every binary emits run-metadata JSON
   (Benchmark Spec §5) from day one.

## Milestones

| Milestone | Window | Deliverable | Gate |
|---|---|---|---|
| **M0 — Mesh reality** | Days 1–2 | Repo scaffold; `ds5 node` daemon; `ds5 bench link` (RTT/bandwidth/jitter per node pair); model downloads kicked off; JSON results in `bench/results/` | Reproducible link numbers from the real 3-node mesh (<10% variance across 3 runs) |
| **M1 — Viability model** | Days 3–5 | `ds5 bench decode-sim` (94-layer per-token traffic replay over real links, parameterized miss rate); Python placement simulator vs real safetensors metadata with 33.6GB caps; routing-skew report from 235B telemetry capture | `docs/findings/f001`: projected tok/s ceiling decomposition + go/no-go vs >12 tok/s |
| **M2 — Single-node engine core** | Week 2 | Zig GGUF parser; Metal glue; kernels in dependency order: RMSNorm → Q8_0 dequant+matmul → RoPE → GQA attention → router/top-8 → fused expert MLP; each vs golden fixtures | Qwen3-30B-A3B produces reference-matching tokens on one M5 Max |
| **M3 — Distributed correctness** | Week 3 | 30B-A3B split across B/C over M0 transport: activation packets, one-packet-per-destination rule, checksums, KV pages | Distributed output == single-node output, deterministic under fixed seed (inherits v0.2 Phase 1+2 gates) |
| **M4 — 235B placement + runtime** | Weeks 4–6 | Placement/quant manifests; I-quant dequant kernels; tiered expert residency; remote-expert execution on A; 8K/32K benchmarks | First distributed 235B decode; honest numbers vs >12 tok/s target |
| **M5 — Findings** | Ongoing | `docs/findings/` write-ups, diagrams, README | f001 ships end of week 1; final finding publishable even if negative |

## Hour-zero background tasks

Start before/alongside M0 coding (long download lead times):

1. Download Qwen3-30B-A3B-Instruct-2507 GGUF (Q8_0, ~32GB) — oracle + bring-up artifact.
2. Download Qwen3-235B-A22B-Instruct-2507 GGUF (~Q2-class, ~85GB) — telemetry capture only.
3. When (2) lands: run llama.cpp (offline tool, never linked) with mmap on one M5 Max
   over the router-calibration corpus, capturing per-layer expert-usage distribution.

`tools/download_models.sh` does (1) and (2). Run it on a cluster node with disk to spare,
not the dev laptop.

## Execution constraints (inherited from v0.2, still binding)

- No Qwen3 kernel optimization before placement simulation proves memory feasibility.
- No routing shortcuts before top-k parity is validated. Never alter top-8 semantics.
- No generic model-plugin abstractions.
- No 128K+ context work until 8K/32K are stable.
- Loader refuses manifests exceeding 33.6GB/node static weights without explicit override.
