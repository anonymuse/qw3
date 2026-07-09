# ADR-003: Qwen3-30B-A3B Is the Sole Bring-Up Model

**Status:** Accepted
**Date:** 2026-07-08
**Depends on:** ADR-001, ADR-002

## Decision

Qwen3-30B-A3B-Instruct-2507 is the only bring-up model on the critical path. The dense
32B/70B baseline (v0.2 Phase 1) is cut; v0.2 Phases 1 and 2 merge into a single
"distributed small-MoE correctness" milestone (M2/M3 in the v0.3 execution plan).

## Rationale

- Same family as the final target: identical router semantics (top-k softmax gating),
  GQA attention, RMSNorm, GGUF tensor layout. Every kernel and every line of MoE
  scheduling code transfers to 235B directly.
- Small enough (~30B total, ~3B active) to run whole on one M5 Max at Q8, giving a
  trusted single-node reference for the distributed split — which is what the dense
  phase existed to provide.
- Transport/KV validation (the other purpose of the dense phase) happens by splitting
  30B-A3B across Nodes B/C in M3.

Dense models may still be run ad hoc via llama.cpp as benchmark comparators; they are
not a DS5 runtime target.

## Consequences

- v0.2 Benchmark Spec Phase 1 gates (token correctness, KV replay, transport checksums,
  memory cap) are inherited unchanged by the merged milestone — they gate M3.
- Saves an estimated 1–2 weeks of solo schedule.
