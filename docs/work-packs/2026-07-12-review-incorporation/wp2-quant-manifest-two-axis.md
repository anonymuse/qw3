# WP-2: Two-Axis Quant/Placement Manifest Policy

**Adopts:** A5 (two-axis expert quantization), manifest-policy half
**Owns:** `docs/specs/DS5_Quant_Manifest_v0.1.md` (new),
`docs/specs/schemas/quant_manifest.schema.json` (new)
**Must not touch:** `src/`, ADRs, spec v0.3 body, other packs' files
**Depends on:** WP-1's `expert_stats.schema.json` field names (coordinate via the
schema file once WP-1 lands it; if racing, define against the WP-1 pack text)

## Goal

Specify the M4 placement/quant manifest format: which quant type each (layer, expert)
gets, on which node it resides, in which residency tier — with every precision
assignment traceable to `expert_stats.json`. Documentation and schema only; the Zig
loader implements refusal rules at M4.

## Deliverables

1. **Manifest spec** — `docs/specs/DS5_Quant_Manifest_v0.1.md`:
   - per-expert record: node, residency tier (hot/warm/cool/cold), quant type
     (from the ADR-002 sequence: Q8_0, Q4_0/Q4_K, IQ3_S, IQ2_M/XS/XXS), and a
     provenance pointer into `expert_stats.json` (stats-file hash + record);
   - assignment policy: precision is a function of (frequency axis, sensitivity
     axis) with explicit thresholds left as named parameters (values come from f001
     data, not this doc); document the policy shape and the constraint lattice:
     - router/gate tensors FP16/Q8 always; KV-sensitive tensors Q8/Q6-class
       (v0.2 quality rules, binding);
     - per-node static totals must fit 33.6GB caps — manifest carries computed
       per-node byte totals;
     - a "no stats, no downgrade" rule: an expert without measured stats defaults
       to the highest-fidelity tier that fits, never to an aggressive I-quant;
   - loader refusal rules (spec §5/§7.4 style): shape mismatch vs GGUF metadata,
     over-cap totals without explicit override, missing/hash-mismatched stats
     provenance, quant types the kernel set doesn't yet implement;
   - the `llama-quantize` production path: manifest → tensor-type override list →
     offline requantization command (document the mechanism; verify current
     llama-quantize override syntax against upstream — do not trust review claims).
2. **Schema** — `docs/specs/schemas/quant_manifest.schema.json` validating the above,
   with one small synthetic example manifest embedded in the spec doc that validates.

## Acceptance

- Example manifest validates against the schema (show the command).
- Every rule in the spec doc cites its source (v0.2 quality rules, spec v0.3 §4/§5/§7,
  ADR-002 quant sequencing) — no new policy invented beyond the A5 adoption.
- Byte-total arithmetic for the example manifest is shown and correct.
