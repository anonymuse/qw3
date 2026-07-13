# DS5 Phase-2 Optimization Backlog

**Document type:** Deferred-work backlog
**Status:** Active — items are NOT scheduled for V1
**Date:** 2026-07-13
**Governing spec:** `docs/specs/DS5_Project_Spec_v0.3.md`
**Placement reference:** `docs/specs/imported_v0.2/DS5_Model_Runtime_Placement_Spec_v0.2_Qwen3_235B_A22B.md`
**Origin:** Architecture assessment of the "narrow engine / antirez" question (2026-07-13),
following the review dispositions in `docs/reviews/`

## Purpose

Captures model- and hardware-specific optimizations that are sound and proven but
deliberately **out of V1 scope**, so they are not lost and can be picked up when their
trigger conditions are met. Each item states *when* to revisit, not just *what*. None
of these may be started before its trigger fires; V1 ships without them.

Guardrail (unchanged from spec §3): these are the "proven, cheap, model-specific"
tier. Anything academic or unproven — speculative decoding, logit ensembling/fusion,
learned prefetch predictors, dynamic online re-placement, PRM/output gating — stays
rejected or deferred beyond this backlog, not promoted into it.

## V1 baseline this backlog assumes

- **Topology: layer-parallel (placement spec §3).** B owns layers 0–46 (attention +
  KV + resident experts), C owns 47–93, A is control plane / sampling / lm_head /
  cold-expert host. Chosen for V1 because single-stream decode is read-bandwidth-bound
  with large headroom (~50 tok/s paper ceiling for sequential two-node traversal vs the
  >12 tok/s target), so the ~50% per-node idle of single-stream pipelining is free
  margin rather than a problem worth added complexity to reclaim.
- **Cold experts on Node A by bandwidth asymmetry** (placement spec §4.1/§5.3) is
  *already adopted*, not deferred. Rationale now recorded so it isn't re-litigated:
  A's 307 GB/s vs the workers' 614 GB/s makes it a poor decode worker but a fine home
  for rarely-read weights, and keeps A's slower bandwidth off the synchronous critical
  path (a symmetric 3-way decode split would make A the Amdahl pace-setter).

### Near-term hygiene (not Phase-2, do at M4 manifest work)

- **H-1 — Resolve the layer-parallel vs expert-parallel ambiguity in the placement
  spec.** §3 describes layer-parallel; §4.2's cold-start ("B even expert IDs, C odd")
  is expert-parallel residue. Treat §4.2's odd/even split as **superseded** by the
  layer-parallel V1 choice and rewrite it as a per-layer hotness-rank residency order
  within each node's owned layer range. This is a doc-consistency fix, not new
  architecture; fold it into the M4 manifest pass so the implementation targets one
  topology.

## Deferred Phase-2 items

| ID | Item | Trigger to revisit | Primary dependency |
|---|---|---|---|
| P2-OPT-1 | Co-activation-aware expert grouping | At/after M3 correctness, before M4 placement freeze | Telemetry extended to pairwise co-activation counts |
| P2-OPT-2 | Replicated-attention expert-parallel hybrid | Only if measured decode margin falls below target after overhead | M1 decode-sim comparison + M4 real numbers |
| P2-OPT-3 | Router-driven cold-expert prefetch (simple) | Only if M4 cold-miss stalls exceed the p95 budget | Working tiered residency + promotion path (M4) |

---

### P2-OPT-1 — Co-activation-aware expert grouping

**What.** Placement currently ranks experts independently via `S(l,e)` (placement spec
§4.3: hit-rate + gate-mass + outlier). That ignores *which experts fire together*.
Build the per-layer expert co-activation matrix from routing telemetry and min-cut
partition each layer's experts across B/C so frequently co-activated experts share a
node. This directly reduces the two-destination case of the one-packet-per-node rule
(placement spec §8.2) — fewer tokens that must hit both workers in the same layer.

**Why deferred, not V1.** Pure optimization on top of a working distributed decode;
it changes placement quality, not correctness. Meaningless until M3 proves distributed
output is correct and M1 telemetry exists to partition on.

**Why in-scope (not academic).** Offline measure-then-partition; a standard graph
min-cut at manifest-build time, zero runtime cost. No model behavior changes — routing
semantics and top-8 are untouched; only *where* experts physically sit changes.

**Scope when picked up.**
- Extend the expert-stats capture (WP-1 `tools/expert_stats/`,
  `docs/specs/schemas/expert_stats.schema.json`) to record pairwise co-activation
  counts per layer alongside the existing per-expert counts.
- Add a partition step to the placement simulator that consumes the co-activation
  matrix and emits a B/C assignment minimizing cross-node co-fire, subject to the
  per-node 33.6GB static cap.
- Feed the result into the M4 quant/placement manifest
  (`docs/specs/DS5_Quant_Manifest_v0.1.md`) as the node-assignment source, replacing
  hotness-rank-only assignment.

**Boundary.** Static, offline partition only. No online/dynamic re-placement.

---

### P2-OPT-2 — Replicated-attention expert-parallel hybrid topology

**What.** An alternative to layer-parallel for the case where single-stream layer
pipelining leaves too much bandwidth on the table. Because the attention backbone is
only ~6.7B params (~3.4GB at Q4), replicate it on **both** B and C and split only the
experts of every layer across them. Decode then uses both nodes' memory bandwidth on
the experts (the ~96% of bytes that matter) every layer, with attention/KV local on
both — no KV round-trips.

**Trigger — this is the escape hatch, not a default.** Pursue only if M1 decode-sim, or
real M4 numbers, show layer-parallel decode throughput dropping below the >12 tok/s
target once measured TB5 round-trip latency and per-layer overheads are included. If
layer-parallel keeps its headroom (expected on paper), do not build this.

**Why in-scope (not academic).** Expert/tensor parallelism with a replicated small
backbone is a standard MoE distribution pattern; the only model-specific insight is
that Qwen3's attention is small enough to replicate cheaply. Routing semantics
unchanged.

**Scope when picked up.**
- Decode-sim variant modeling per-layer expert fan-out to both workers with replicated
  attention, vs the layer-parallel baseline, on measured link latency.
- Attention-weight replication in the loader/manifest (both workers hold the full
  ~6.7B backbone; experts split per layer).
- Transport: per-layer activation broadcast + result gather, still honoring
  one-packet-per-destination.

**Boundary.** Attention replicated, experts split. No cross-node attention/KV sharding
(that reintroduces the coherence cost this design avoids).

---

### P2-OPT-3 — Router-driven cold-expert prefetch (simple)

**What.** Hide cold-miss NVMe latency by prefetching experts likely to be needed before
the decode loop blocks on them. Two proven signals only: (a) the statically-hot set
prefetched at load, and (b) measured layer-to-layer correlation — experts hot at layer
L predict likely experts at L+1 — used to issue NVMe reads during earlier-layer compute.

**Trigger.** Only if M4 telemetry shows cold-miss stalls exceeding the configured p95
stall budget (spec §10; placement spec §9). If residency + aggressive cold-tier quant
keep misses under budget, prefetch is unnecessary complexity — skip it.

**Why in-scope (not academic).** Static-hot prefetch and correlation-table prefetch are
standard cache-warming; they touch scheduling, not model math, and fail safe (a wrong
prefetch wastes bandwidth, never changes output).

**Scope when picked up.**
- Build the layer-to-layer expert correlation table offline from telemetry.
- Add a prefetch scheduler on Node A that issues promotion reads ahead of predicted
  need, within the promotion-block rules (placement spec §9: 64MiB blocks, no blocking
  read on the decode hot path).

**Boundary — hard line.** No *learned* predictor (a model that infers next-layer
experts from the hidden state). That is the academic/unproven version and stays out of
this backlog entirely. Simple static + correlation only.

## Review

Revisit this backlog at each milestone gate (M1 f001, M3, M4). Promote an item to a
work pack only when its trigger has actually fired and its dependency exists — not
speculatively.
