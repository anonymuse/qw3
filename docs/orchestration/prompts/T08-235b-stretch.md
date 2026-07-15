# T08 — 235B push (STRETCH — only after the T07 gate passes)

**Model:** Sonnet. **Branches:** `t08a-placement`, `t08b-iq2-kernels` (two
executors, can run parallel). Gate to even start: T07 passed AND the T03
placement simulator says the chosen quant mix fits AND the owner's telemetry
JSON exists (routing skew is thesis-critical — assumption A-04).

## T08a — Placement manifests from real data

Inputs: T03's `tools/placement_sim.py` + committed metadata cache, owner's
telemetry JSON (per-layer expert-usage), measured mesh JSONs in
`bench/results/`. Produce: `manifests/model/qwen3-235b-3node.zon` — layer
ranges per node + per-tensor quant assignment per Placement Spec §6 (router/
gate tensors ≥Q8 — ADR-001 rule 3), static bytes per node ≤33.6GB proven by
the simulator run committed alongside. Update `docs/findings/f001` skew
section with the real telemetry distribution. No runtime code.

## T08b — IQ2-class dequant kernels

Scope: `matmul` and `expertMlpSwiglu` support for `iq2_xxs`/`iq2_xs`/`iq2_s`
(+`q4_k` if the placement mix needs it) in CPU provider + Metal shaders,
same PORTING-doc pattern as kernels A/B/C. The block layouts are documented
in ggml (read `docs/decisions/ADR-002-kernel-strategy.md` — reading ggml
source as REFERENCE is allowed, linking/vendoring it is not; cite the file
+ commit you transcribed layouts from). Validation: extend
`tools/make_fixtures.py` to emit iq2-quantized weight fixtures (it can use
llama.cpp's `llama-quantize` offline to produce reference blocks, or
`gguf-py`); dequant-matmul must match f32 reference within ADR-005 §4 quant
tolerances. CPU first, Metal second — a CPU-only landing is acceptable scope
cut (GPU IQ2 becomes its own task).

## First 235B decode attempt (orchestrator-led, not an executor task)

Only when: T08a manifest exists, T08b kernels fixture-pass, T07 runbook
proven on real B/C, artifacts downloaded on both workers. Commands per
`docs/runbook-m3.md` with the 235B manifest, `--steps 16`, telemetry on.
ANY outcome (including OOM or 0.3 tok/s) is a valid f001 data point — record
honestly in `docs/findings/f001`.

## Forbidden

Starting before the gates above; quantizing router/gate below Q8; NVMe
streaming in the steady-state decode path (promotion only — Placement Spec
§9); vendoring ggml code.
