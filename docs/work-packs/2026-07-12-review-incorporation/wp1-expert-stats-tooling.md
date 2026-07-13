# WP-1: Per-Expert Stats Capture (Frequency + Sensitivity)

**Adopts:** A5 (two-axis expert quantization), data-collection half
**Owns:** `tools/expert_stats/` (new), `docs/runbooks/expert-stats-capture.md` (new),
`docs/specs/schemas/expert_stats.schema.json` (new)
**Must not touch:** `src/`, ADRs, spec, other packs' files

## Goal

Define and tool the `expert_stats.json` artifact: one record per (layer, expert) of
Qwen3-235B-A22B carrying (a) routing frequency from the M1 telemetry capture and
(b) quantization sensitivity from an offline llama.cpp imatrix/KLD calibration pass.
This is the measured input that placement simulation (M1) and the M4 quant manifest
consume. Cluster/model-dependent steps ship as a runbook, not as executed results.

## Deliverables

1. **Schema** — `docs/specs/schemas/expert_stats.schema.json` (JSON Schema):
   - header: model id, GGUF file hash, corpus id + hash, capture tool versions,
     git commit, capture date;
   - per (layer, expert): `activation_count`, `activation_fraction`,
     `mean_gate_weight`, sensitivity fields (per candidate quant type: KLD or
     imatrix-derived error proxy), and a `sources` block saying which run produced
     each field. Frequency and sensitivity arrive from different passes — the schema
     must allow partial records with explicit provenance.
2. **Merge/validate tool** — `tools/expert_stats/merge_stats.py`:
   - inputs: routing-telemetry dump (the M1 capture format — read
     `DS5_Execution_Plan_v0.3.md` hour-zero task 3 for context) and llama.cpp
     imatrix/KLD output files;
   - output: validated `expert_stats.json`; refuses on model-shape mismatch
     (94 layers × 128 experts) or missing provenance;
   - unit-testable with small synthetic inputs — include `tools/expert_stats/tests/`
     fixtures that run on the dev laptop with the repo `.venv`.
3. **Runbook** — `docs/runbooks/expert-stats-capture.md`: exact llama.cpp commands
   (`llama-imatrix`, KLD comparison flags) to run on a cluster node against the
   downloaded 235B GGUF over the router-calibration corpus; expected runtimes and
   disk needs; where outputs land; how to invoke the merge tool. llama.cpp is an
   offline oracle only (ADR-002) — never linked, never in the runtime path.

## Acceptance

- `python tools/expert_stats/merge_stats.py --help` works in `.venv`; tests pass
  locally with synthetic fixtures.
- Schema validates the synthetic merged output.
- Runbook is executable top-to-bottom by an operator with no context beyond it.
- Verify llama.cpp's current imatrix/KLD CLI flags against upstream docs — do not
  trust the review's tool claims.
