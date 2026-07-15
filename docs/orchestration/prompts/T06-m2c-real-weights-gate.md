# T06 — M2c: Qwen3-30B-A3B real-weights gate (the M2 milestone gate)

**Model:** Sonnet, with orchestrator reviewing the gate result. **Branch:**
`t06-real-30b` off `integration` (requires T04+T05 merged). **Hardware:**
runs on a machine holding the 30B Q8_0 GGUF (~32GB) — a cluster M5 Max
(48GB) fits it in memory; the dev M5 Air (24GB) can still run it mmap-backed
(slower, fine for a gate). Coordinate with the project owner for access; if
only the dev machine is available, use it.

## Pre-req artifacts (owner-provided)

1. GGUF at `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/*Q8_0*.gguf`.
2. Real-model oracle fixtures: run `tools/make_fixtures.py --hf <HF_MODEL_DIR>
   --out tests/fixtures/qwen30b/ --layers 0,23,47 --e2e` (see `--help`; needs
   HF transformers CPU fp32 on a box with RAM — the RTX/64GB offline box per
   ADR-004, or layer-streamed on a Max). If these fixtures don't exist yet,
   generating them IS part of this task; per-layer tensors for first/middle/
   last layers + end-to-end logits for the 5 prompts suffice.

## Gate (v0.2 Phase-2 gates, inherited by M2)

- `ds5 run` on the 30B GGUF, both backends, 5 oracle prompts:
  greedy tokens match the oracle EXACTLY for 64 generated tokens per prompt,
  and final-step logits within ADR-005 §4 tolerances.
- Config sanity: parsed `ModelConfig` == ADR-001 §2 table
  (48 layers for 30B-A3B — read the real value from metadata, don't assume;
  128 experts, top-8, GQA 32/4 for 30B — VERIFY against artifact, A-07).
- Router parity on real weights: for prompt p0, dump per-layer expert IDs and
  compare to oracle — 100% match required (this validates A-07's claim that
  community GGUFs preserve router tensors).
- Memory: peak RSS logged; loader must be mmap-backed (no full-file read).

## When it diverges (it will)

Use T04's trace hook against the real-model per-layer fixtures. Localize to
first divergent layer/op, then follow DEBUG-divergence.md. Usual suspects at
this stage: GGUF tensor-name mapping (ADR-005 §5 table), head-dim/rope-theta
metadata mismatches, q/k per-head norm wiring, expert bank offsets
(3-D tensor stride math), f16 scale handling at real-weight magnitudes.

## Definition of done

Gate results written to `docs/findings/m2-gate.md` (pass/fail per item, diffs,
peak RSS, tok/s observed — honest numbers). All suites still green. If the
gate fails after 1.5 days of focused debugging, write up exactly where the
divergence localizes and hand back to the orchestrator — that writeup is the
deliverable then.

## Forbidden

Loosening tolerances or "close enough" token matches; skipping the router
parity check; hardcoding any model constant that metadata provides.
