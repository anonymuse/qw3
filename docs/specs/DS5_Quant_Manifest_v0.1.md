# DS5 Quantization & Placement Manifest Specification v0.1

**Document type:** Specification (documentation + schema only; no runtime code)
**Status:** Active
**Date:** 2026-07-12 (§8.2 re-verified against the real downloaded GGUF artifact
2026-07-13 — see §8.2 for evidence; no other section changed)
**Adopts:** A5 (two-axis expert quantization), manifest-policy half — per
`docs/work-packs/2026-07-12-review-incorporation/wp2-quant-manifest-two-axis.md`
**Sources (binding):** DS5 Project Spec v0.3 §4 (memory caps), §5 (model constants),
§6 (quantization, A5 adoption), §11 (execution constraints); DS5 Project Spec v0.2
§7.3 (quality rules, carried binding into v0.3 §6); ADR-002 (quant format sequencing)
**Schema:** `docs/specs/schemas/quant_manifest.schema.json`
**Companion:** `docs/specs/schemas/expert_stats.schema.json` (WP-1; field names in
this document are aligned to that schema as landed 2026-07-12)
**Implements at:** M4 — the Zig loader enforces the refusal rules in §7; this
document specifies the format and policy only.

---

## 1. Purpose

The placement/quant manifest is the single artifact that tells the M4 loader, for
every routed expert of Qwen3-235B-A22B:

- **which node** (B or C) holds its weights,
- **which residency tier** (hot/warm/cool/cold) it occupies,
- **which quantization type** its weights use, and
- **where that decision came from** — a provenance pointer into a specific
  `expert_stats.json` capture (file hash + record index).

Per Spec v0.3 §6 (adoption A5), precision is a function of **two measured axes** —
routing frequency (M1 telemetry) and quantization sensitivity (offline imatrix/KLD
calibration) — and every assignment must trace to measured stats, never to assumed
skew. This document defines the record format (§3), the assignment policy shape and
constraint lattice (§4), the memory-cap accounting (§5), a synthetic validating
example (§6), the loader refusal rules (§7), and the offline `llama-quantize`
production path (§8).

## 2. Fixed model shape and caps (restated, not redefined)

| Constant | Value | Source |
|---|---:|---|
| Layers | 94 | Spec v0.3 §5 |
| Experts per layer | 128 | Spec v0.3 §5 |
| Activated experts (top-k) | 8 | Spec v0.3 §5 |
| Per-node static-weight cap | 33.6 GB = 33,600,000,000 bytes | Spec v0.3 §4; v0.2 §4 (48GB × 0.70) |
| Cluster static cap | 100.8 GB | Spec v0.3 §4 |

Per Spec v0.3 §5, these constants are verified against GGUF metadata at load;
mismatch is a refusal, not a warning (§7 R1 below). All byte→GB conversions in this
document are decimal (1 GB = 10⁹ bytes), matching the v0.2 §4 cap arithmetic.

Derived constant used for the size estimates in §6: the per-expert FFN shape. Each
routed expert comprises three projection matrices — gate and up (hidden 4096 →
moe-FFN 1536) and down (1536 → 4096) — i.e. 3 × 4096 × 1536 = **18,874,368
elements per expert**. `moe_intermediate_size = 1536` is taken from the HF config
for Qwen3-235B-A22B (GGUF metadata and the HF config are the only accepted sources
per Spec v0.3 §5); the loader re-derives exact tensor sizes from GGUF metadata at
load, and the manifest's size fields are advisory estimates (§5.2).

## 3. Manifest format

A manifest is one JSON document validating against
`docs/specs/schemas/quant_manifest.schema.json`.

### 3.1 Top level

| Field | Type | Meaning |
|---|---|---|
| `manifest_version` | string | `"0.1"` for this spec |
| `model_id` | string | e.g. `"Qwen3-235B-A22B-Instruct-2507"` |
| `model_shape` | object | `num_layers` (94), `num_experts_per_layer` (128), `top_k` (8) — field names identical to `expert_stats.schema.json` `model_shape`; cross-checked against GGUF at load (§7 R1) |
| `expert_stats_source` | object | Identity of the stats capture this manifest was built from: `stats_file_hash` (SHA256 of the `expert_stats.json` file itself), plus `model_hash`, `corpus_id`, `capture_date`, `git_commit` copied from that file's `header` (same field names as the WP-1 schema) |
| `policy_parameters` | object | The named thresholds of §4.2 with the values used to build this manifest (`null` until f001 supplies them) |
| `nodes` | object | Exactly `B` and `C` (Node A holds no decode-critical expert weights, Spec v0.3 §6) |
| `quality_attestation` | object | Statements that the §4.3 binding rules were applied (informational; the loader re-checks, it does not trust) |

### 3.2 Per-node object

| Field | Type | Meaning |
|---|---|---|
| `role` | string | `"decode_worker"` |
| `static_cap_bytes` | integer | 33,600,000,000, fixed by Spec v0.3 §4 |
| `experts` | array | Per-expert placement records (§3.3) |
| `computed_static_total_bytes` | integer | Σ `quantized_size_bytes` over `experts`; must equal the recomputed sum and fit the cap (§5, §7 R2) |

### 3.3 Per-expert placement record

| Field | Type | Meaning |
|---|---|---|
| `layer` | int 0–93 | Layer index. Field name matches `expert_stats.schema.json` (`layer`, not `layer_idx`) |
| `expert_index` | int 0–127 | Expert index within the layer (matches WP-1 field name) |
| `residency_tier` | enum | `hot` \| `warm` \| `cool` \| `cold` (Spec v0.3 §6 tiering) |
| `quant_type` | enum | One of the ADR-002 sequence: `Q8_0`, `Q4_0`, `Q4_K`, `IQ3_S`, `IQ2_M`, `IQ2_XS`, `IQ2_XXS` |
| `quantized_size_bytes` | integer | Estimated weight bytes for this expert at `quant_type` (GGML block arithmetic, §6.2); loader recomputes from GGUF, manifest value used for pre-load cap check |
| `stats_provenance` | object | See below |

`stats_provenance`:

| Field | Type | Meaning |
|---|---|---|
| `stats_file_hash` | string | SHA256 of the `expert_stats.json` used; must equal the top-level `expert_stats_source.stats_file_hash` and the hash of the file presented at load (§7 R3) |
| `expert_record_index` | int 0–12031 | Index into the stats file's `experts` array. Per the WP-1 schema the array is indexed `layer * 128 + expert_index`; the loader verifies the record at this index carries matching `layer`/`expert_index` (§7 R3) |
| `assignment_basis` | enum | `measured` (stats record exists with both axes populated) or `no_stats_default` (§4.3 rule 3) |
| `frequency_percentile` | number 0–100, or null | This expert's routing-frequency percentile derived from the stats file's `activation_fraction` distribution (higher = hotter). Null iff `assignment_basis = no_stats_default` |
| `sensitivity_class` | enum or null | `high` \| `medium` \| `low`, derived from the stats file's `quantization_sensitivity` metrics via the §4.2 thresholds. Null iff `no_stats_default` |
| `sensitivity_metric` | string or null | Which measured field drove the class, as a path into the WP-1 record, e.g. `"quantization_sensitivity.Q4_K_M.kld"` |

## 4. Assignment policy: two measured axes

### 4.1 Policy shape

Precision for expert *(l, e)* is a pure function of two inputs read from
`expert_stats.json`:

- **Frequency axis:** `activation_fraction` for *(l, e)*, converted to a percentile
  over all 94 × 128 = 12,032 experts. Determines `residency_tier`.
- **Sensitivity axis:** the per-quant-type degradation metrics
  (`quantization_sensitivity.<TYPE>.kld` / `.ppl_ratio` / `.imatrix_importance`
  in the WP-1 schema), reduced to a `sensitivity_class`.

```
tier(l,e)  = tier_of(frequency_percentile; F_HOT, F_WARM, F_COOL)
class(l,e) = class_of(sensitivity_metric;  S_HIGH, S_LOW)
quant(l,e) = LATTICE[tier][class]          (§4.2 table)
then: apply binding rules (§4.3), then fit-check (§5); on over-cap the
      lowest-frequency, lowest-sensitivity experts step down one position in the
      ADR-002 sequence and the fit-check repeats.
```

### 4.2 Named threshold parameters and the constraint lattice

Threshold **values are not defined in this document** — they come from f001
measured data (Spec v0.3 G2: measured links + measured routing skew). This spec
fixes only their names and semantics:

| Parameter | Semantics |
|---|---|
| `F_HOT` | Frequency percentile at/above which an expert is `hot` |
| `F_WARM` | Percentile at/above which (and below `F_HOT`) an expert is `warm` |
| `F_COOL` | Percentile at/above which (and below `F_WARM`) an expert is `cool`; below is `cold` |
| `S_HIGH` | Sensitivity-metric value at/above which class is `high` |
| `S_LOW` | Value at/below which class is `low`; between `S_LOW` and `S_HIGH` is `medium` |
| `sensitivity_metric` | Which WP-1 field the S thresholds apply to (e.g. `quantization_sensitivity.Q4_K_M.kld`) |

Constraint lattice (cell values are illustrative defaults finalized with f001 data;
the **monotonicity and floor/ceiling constraints below are binding**):

| tier \ class | high | medium | low |
|---|---|---|---|
| hot | Q8_0 | Q8_0 | Q4_K |
| warm | Q8_0 | Q4_K | Q4_K |
| cool | Q8_0 | Q4_K | Q4_0 |
| cold | Q8_0 | IQ3_S | IQ2_M |

Binding lattice constraints:

1. **Monotonicity:** fidelity is non-decreasing along both axes — a hotter tier
   never receives a lower-fidelity type than a colder tier at the same class, and
   a more sensitive class never receives lower fidelity than a less sensitive
   class at the same tier. (Implements v0.3 §10 Quality: "hot experts quantized
   higher-fidelity than cold tiers".)
2. **Sensitivity ceiling:** `high`-sensitivity experts are never below Q8_0
   regardless of tier — an expert that measurably degrades under Q4 stays Q8_0
   even when cold.
3. **I-quant floor:** IQ2-class types (`IQ2_M/XS/XXS`) are only ever assigned to
   `cold`-tier experts of `low` class, and only when 235B placement demands them
   (ADR-002 consequence 2: I-quant kernels are built last, only under placement
   pressure).

### 4.3 Binding rules (sourced, not invented)

1. **Router/gate tensors are FP16 or Q8-class, always.** (v0.2 §7.3 "Router/gate
   tensors remain FP16 or Q8"; carried binding in v0.3 §6.) These are per-layer
   tensors (`blk.<L>.ffn_gate_inp.weight` in GGUF naming), not per-expert records;
   they never appear in `nodes.*.experts` and the loader enforces their type
   globally (§7 R5).
2. **KV-sensitive tensors are Q8/Q6-class** unless evals justify lower (v0.2 §7.3;
   v0.3 §6, §10). Also layer-level (attention K/V projections), enforced by the
   loader outside the per-expert records. Per-expert records cannot override
   either rule.
3. **No stats, no downgrade.** An expert with no measured record — missing from
   the stats file, present with neither axis populated, or unusable because of a
   provenance hash mismatch — is marked `assignment_basis: "no_stats_default"`
   and receives the **highest-fidelity type that fits the §5 cap**, stepping down
   the ADR-002 sequence only as far as the fit-check requires and **never into an
   I-quant** (`IQ3_S`/`IQ2_*` are excluded for such experts; the schema enforces
   the exclusion). Source: v0.3 §6 — assignments trace to measured stats, "never
   to assumed skew"; WP-2 pack (A5 manifest-policy half).
4. **Hot experts are quantized at higher fidelity than cold tiers** (v0.3 §10
   Quality; v0.2 §7.3) — subsumed by lattice constraint 1, restated because it is
   an acceptance criterion.

## 5. Memory-cap accounting

### 5.1 Rule

Per Spec v0.3 §4: 33.6 GB static weights per node; the loader refuses manifests
exceeding per-node caps without explicit override (also v0.3 §11). The manifest
carries computed per-node byte totals:

```
computed_static_total_bytes(node) = Σ quantized_size_bytes over nodes.<node>.experts
```

Validity requires, per node, `computed_static_total_bytes ≤ static_cap_bytes`
(= 33,600,000,000). The manifest total covers **routed-expert weights placed by
this manifest**; attention/embedding/router weights are accounted against the same
33.6 GB cap by the loader from GGUF metadata — the manifest total is a lower bound
the loader augments, and the §7 R2 refusal applies to the loader's full recomputed
figure as well as to the manifest's own arithmetic.

### 5.2 Estimated vs. actual sizes

`quantized_size_bytes` values are computed from GGML block layouts (shown in §6.2)
and are estimates for pre-load checking. The loader recomputes exact totals from
GGUF tensor metadata at load; a manifest whose declared node totals disagree with
the sum of its own records is refused (§7 R2). Disagreement between an estimate
and the GGUF-actual size is re-checked against the cap but is not by itself a
refusal.

## 6. Synthetic example manifest

A deliberately tiny 9-expert example (a real M4 manifest places all 94 × 128
experts). It exercises all four tiers, four quant types, both nodes, and one
`no_stats_default` expert. This exact JSON is validated against the schema by the
command in §6.3.

### 6.1 Example

```json
{
  "manifest_version": "0.1",
  "model_id": "Qwen3-235B-A22B-Instruct-2507",
  "model_shape": {
    "num_layers": 94,
    "num_experts_per_layer": 128,
    "top_k": 8
  },
  "expert_stats_source": {
    "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
    "model_hash": "sha256:1e2d3c4b5a6978877665544332211ffeeddccbbaa99887766554433221100ffe",
    "corpus_id": "ds5-calib-v1",
    "capture_date": "2026-07-12T09:00:00Z",
    "git_commit": "042ba21aa00000000000000000000000000000000"
  },
  "policy_parameters": {
    "F_HOT": null,
    "F_WARM": null,
    "F_COOL": null,
    "S_HIGH": null,
    "S_LOW": null,
    "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
  },
  "nodes": {
    "B": {
      "role": "decode_worker",
      "static_cap_bytes": 33600000000,
      "experts": [
        {
          "layer": 0,
          "expert_index": 0,
          "residency_tier": "hot",
          "quant_type": "Q8_0",
          "quantized_size_bytes": 20054016,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 0,
            "assignment_basis": "measured",
            "frequency_percentile": 98.5,
            "sensitivity_class": "high",
            "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
          }
        },
        {
          "layer": 1,
          "expert_index": 2,
          "residency_tier": "hot",
          "quant_type": "Q8_0",
          "quantized_size_bytes": 20054016,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 130,
            "assignment_basis": "measured",
            "frequency_percentile": 97.2,
            "sensitivity_class": "medium",
            "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
          }
        },
        {
          "layer": 5,
          "expert_index": 4,
          "residency_tier": "warm",
          "quant_type": "Q4_K",
          "quantized_size_bytes": 10616832,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 644,
            "assignment_basis": "measured",
            "frequency_percentile": 22.5,
            "sensitivity_class": "medium",
            "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
          }
        },
        {
          "layer": 10,
          "expert_index": 7,
          "residency_tier": "cool",
          "quant_type": "Q4_0",
          "quantized_size_bytes": 10616832,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 1287,
            "assignment_basis": "measured",
            "frequency_percentile": 45.0,
            "sensitivity_class": "low",
            "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
          }
        },
        {
          "layer": 30,
          "expert_index": 9,
          "residency_tier": "warm",
          "quant_type": "Q8_0",
          "quantized_size_bytes": 20054016,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 3849,
            "assignment_basis": "no_stats_default",
            "frequency_percentile": null,
            "sensitivity_class": null,
            "sensitivity_metric": null
          }
        }
      ],
      "computed_static_total_bytes": 81395712
    },
    "C": {
      "role": "decode_worker",
      "static_cap_bytes": 33600000000,
      "experts": [
        {
          "layer": 2,
          "expert_index": 1,
          "residency_tier": "hot",
          "quant_type": "Q8_0",
          "quantized_size_bytes": 20054016,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 257,
            "assignment_basis": "measured",
            "frequency_percentile": 96.8,
            "sensitivity_class": "high",
            "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
          }
        },
        {
          "layer": 8,
          "expert_index": 5,
          "residency_tier": "warm",
          "quant_type": "Q4_K",
          "quantized_size_bytes": 10616832,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 1029,
            "assignment_basis": "measured",
            "frequency_percentile": 18.3,
            "sensitivity_class": "medium",
            "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
          }
        },
        {
          "layer": 15,
          "expert_index": 3,
          "residency_tier": "cold",
          "quant_type": "IQ3_S",
          "quantized_size_bytes": 8110080,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 1923,
            "assignment_basis": "measured",
            "frequency_percentile": 5.2,
            "sensitivity_class": "medium",
            "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
          }
        },
        {
          "layer": 20,
          "expert_index": 6,
          "residency_tier": "cold",
          "quant_type": "IQ3_S",
          "quantized_size_bytes": 8110080,
          "stats_provenance": {
            "stats_file_hash": "sha256:9f2c1a7e5b3d8046c2e91f70a4b6d835e7c0912f4a6b8d3c5e7f9012a4b6c8d0",
            "expert_record_index": 2566,
            "assignment_basis": "measured",
            "frequency_percentile": 3.1,
            "sensitivity_class": "medium",
            "sensitivity_metric": "quantization_sensitivity.Q4_K_M.kld"
          }
        }
      ],
      "computed_static_total_bytes": 46891008
    }
  },
  "quality_attestation": {
    "router_gate_rule": "FP16/Q8-class, layer-level, loader-enforced (spec 4.3 rule 1)",
    "kv_sensitive_rule": "Q8/Q6-class, layer-level, loader-enforced (spec 4.3 rule 2)",
    "no_stats_no_downgrade": "1 expert (B: layer 30, expert 9) defaulted to Q8_0 (spec 4.3 rule 3)"
  }
}
```

Consistency notes: `expert_record_index` follows WP-1's `layer * 128 +
expert_index` convention (layer 1, expert 2 → 130; layer 20, expert 6 → 2566).
Every `measured` assignment sits inside the §4.2 lattice; the `no_stats_default`
expert is Q8_0 with null axes; `policy_parameters` thresholds are null pending
f001.

### 6.2 Byte-total arithmetic

Per-expert element count (§2): 3 × 4096 × 1536 = **18,874,368 elements**.
GGML block layouts (from the upstream ggml block structs):

| Type | Block size | Bytes/block | Bytes/element | Per-expert bytes |
|---|---:|---:|---:|---:|
| Q8_0 | 32 | 34 (fp16 scale + 32 × int8) | 1.0625 | 18,874,368 / 32 × 34 = **20,054,016** |
| Q4_K | 256 | 144 (2 × fp16 + 12 scales + 128 quants) | 0.5625 | 18,874,368 / 256 × 144 = **10,616,832** |
| Q4_0 | 32 | 18 (fp16 scale + 16 packed) | 0.5625 | 18,874,368 / 32 × 18 = **10,616,832** |
| IQ3_S | 256 | 110 | 0.4296875 | 18,874,368 / 256 × 110 = **8,110,080** |

Node B (5 experts):

```
  3 × Q8_0  = 3 × 20,054,016 = 60,162,048
+ 1 × Q4_K  =                  10,616,832
+ 1 × Q4_0  =                  10,616,832
─────────────────────────────────────────
  total     =                  81,395,712 bytes = 0.0814 GB ≤ 33,600,000,000 ✓
```

Node C (4 experts):

```
  1 × Q8_0  =                  20,054,016
+ 1 × Q4_K  =                  10,616,832
+ 2 × IQ3_S = 2 ×  8,110,080 = 16,220,160
─────────────────────────────────────────
  total     =                  46,891,008 bytes = 0.0469 GB ≤ 33,600,000,000 ✓
```

Cluster example total: 81,395,712 + 46,891,008 = **128,286,720 bytes ≈ 0.128 GB**
(9 of 12,032 experts). Scale check on why the policy exists: 12,032 experts at
uniform Q8_0 (20,054,016 B) would be ≈ 241.3 GB — far over the 67.2 GB B+C expert
budget; even uniform Q4-class (10,616,832 B) is ≈ 127.7 GB. The mixed lattice with
cold-tier I-quants is what makes the caps reachable, which is why ADR-002 schedules
I-quant kernels "only when 235B placement demands them" — it will.

### 6.3 Validation command

The example above is extracted verbatim from this document and validated with
`jsonschema` (draft-07), plus an independent re-sum of the byte totals:

```bash
python tools/validate_quant_manifest_example.py   # or the inline python -c form
```

(The check performed: extract the §6.1 fenced JSON, `Draft7Validator.validate()`
against `docs/specs/schemas/quant_manifest.schema.json`, then recompute each
node's `computed_static_total_bytes` from its records. Command and output are
recorded in the WP-2 completion report.)

## 7. Loader refusal rules

Style and force follow Spec v0.3 §5 ("mismatch is a refusal, not a warning") and
§11 ("loader refuses over-cap manifests without explicit override"). The M4 Zig
loader implements these; they are normative here.

| # | Condition | Action & source |
|---|---|---|
| R1 | **Shape mismatch.** `model_shape` (94/128/8) disagrees with the GGUF metadata of the model being loaded, or any record's `layer`/`expert_index` is out of range for the GGUF | Refuse. v0.3 §5 (constants verified against GGUF; refusal, not warning) |
| R2 | **Over-cap or inconsistent totals.** Any node's recomputed static total (manifest experts + loader-accounted shared tensors) exceeds `static_cap_bytes`; or `computed_static_total_bytes` ≠ the sum of that node's records | Refuse unless an explicit operator override flag is passed; any override is logged in run-metadata JSON. v0.3 §4, §11; G5 |
| R3 | **Missing/hash-mismatched provenance.** The `expert_stats.json` presented at load hashes differently from `expert_stats_source.stats_file_hash`; or an `expert_record_index` is absent from the stats file; or the record at that index has different `layer`/`expert_index`; or a `measured` record's referenced `sensitivity_metric` field is absent from the stats record | Refuse. v0.3 §6 (assignments must trace to measured stats; A5). A missing stats record is acceptable only when the manifest already declares that expert `no_stats_default` with a §4.3-rule-3-compliant type |
| R4 | **Unimplemented quant type.** Any `quant_type` not in the kernel set actually built at load time (ADR-002 sequencing: Q8_0 first, then Q4_0/Q4_K, I-quants last) | Refuse — never silently substitute another type. ADR-002 consequence 2; v0.3 G1 |
| R5 | **Binding-rule violation.** Router/gate or KV-sensitive tensor types in the GGUF artifact below the §4.3 floors; or a `no_stats_default` expert carrying an I-quant type | Refuse. v0.2 §7.3 (binding via v0.3 §6) |

Refusals emit a structured reason into the run-metadata JSON (v0.3 G5). The GGUF
constant-verification refusal path gets a deliberate wrong-metadata test fixture
(v0.3 §10 Functional).

## 8. `llama-quantize` production path

DS5 has **no custom quantization pipeline** (ADR-002; v0.3 §3): quantized GGUF
artifacts are produced offline with upstream `llama-quantize`, and the manifest
drives that tool through tensor-type overrides.

### 8.1 Verified override syntax (upstream llama.cpp, checked 2026-07-12)

From the upstream tool documentation and maintainer discussion (ggml-org/llama.cpp
`tools/quantize/README.md`; discussion #12741, "Experimenting with custom quants
using `--tensor-type`"):

- `--tensor-type PATTERN=TYPE` — quantize tensor(s) whose name matches `PATTERN`
  (exact name or regex) to ggml type `TYPE`; **repeatable**; applies only to
  tensors with ≥ 2 dimensions. Upstream examples: `--tensor-type attn_v=q5_k`;
  layer-selective `--tensor-type "\.([0-9]|1[01257]|31)\.attn_v=q4_k"`.
- `--output-tensor-type TYPE`, `--token-embedding-type TYPE` — dedicated overrides
  for the output head and token-embedding tensors.
- `--imatrix FILE` — importance-matrix input (required in practice for I-quants).
- Positional form: `llama-quantize [flags] input.gguf output.gguf FTYPE
  [nthreads]`, where `FTYPE` (e.g. `Q4_K_M`) is the default for tensors not
  matched by any override.

Verified against the upstream repository on 2026-07-12, not taken from review
claims (work-pack README rule 5; v0.3 §12).

### 8.2 Granularity constraint (load-bearing, verified against the real artifact)

**Confirmed 2026-07-13** by parsing the GGUF header (magic/KV/tensor-info sections;
no full-weight read needed) of the actual downloaded
`Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL` artifact (`tools/download_models.sh`
item 2; unsloth dynamic quant, both shards, 1,131 tensors total). This supersedes
the "verified against upstream naming, re-verify at M4" caveat in the prior draft
of this section — the re-verification is done, on this exact artifact:

- `general.architecture = qwen3moe`, `qwen3moe.block_count = 94`,
  `qwen3moe.expert_count = 128`, `qwen3moe.expert_used_count = 8` — matches §2
  exactly.
- For **all 94 layers** (`blk.0` … `blk.93`), the three routed-expert projections
  are single 3D tensors with the expert axis fused into the tensor shape, one
  `ggml_type` per tensor:
  - `blk.<L>.ffn_gate_exps.weight` — shape `(4096, 1536, 128)`
  - `blk.<L>.ffn_up_exps.weight` — shape `(4096, 1536, 128)`
  - `blk.<L>.ffn_down_exps.weight` — shape `(1536, 4096, 128)`
  - router: `blk.<L>.ffn_gate_inp.weight` — shape `(4096, 128)`, type `F32`
    (per-layer, not per-expert, and not fused with the routed projections)
- A regex search over all 1,131 tensor names for any per-expert pattern
  (`exps.<idx>.`, `ffn_(gate|up|down).<idx>.`, etc.) matched **zero** tensors.
  There is no GGUF convention in this artifact — nor in the llama.cpp
  `qwen3moe` architecture that produced it — that exposes individual experts as
  separate tensors.
- Confirms the original finding: because `--tensor-type` (§8.1) matches whole
  named tensors, offline requantization granularity is **per (layer,
  projection), not per expert.**
- Incidental corroboration: this dynamic quant already realizes *different*
  `ggml_type`s per layer within the same projection (e.g. `ffn_down_exps.weight`
  is `Q3_K` at layer 0, `Q4_K` at layer 1, `Q3_K` at layer 4 — `ffn_gate_exps`
  and `ffn_up_exps` are uniformly `Q2_K` throughout in this particular UD build).
  This is an existing, shipping example of exactly the per-(layer, projection)
  granularity §8.2 assumes, produced by a different tool (unsloth's calibration
  pipeline) using the same underlying `--tensor-type`-style mechanism — independent
  evidence the granularity constraint is real, not a llama.cpp quirk unique to a
  hand-built manifest.

No newer GGUF convention exposing per-expert tensors was found on the artifact
actually in use for DS5. (Community repos vary in *which* `ggml_type` they pick
per layer, not in *whether* the tensor is fused — the fusion is an architecture
property of the `qwen3moe` GGUF writer in llama.cpp, not a per-quantizer choice.)

Consequences:

1. The manifest stays per-expert: placement and residency genuinely are
   per-expert — expert rows of the fused tensors are sliced across B/C at load,
   which block-quantized layouts permit because GGML blocks never span rows.
2. The manifest→override generator **aggregates each layer's expert assignments
   upward**: the override for `blk.<L>.ffn_*_exps` uses the *highest-fidelity*
   type assigned to any expert of layer L on either node. Upgrade-only
   aggregation is the only direction consistent with §4.3 (never downgrade below
   what the two-axis policy assigned).
3. True per-expert mixed precision within one layer would require splitting expert
   tensors at conversion time or slicing per-expert from multiple quantized
   artifacts at load. Both are M4-decision material and out of scope here; the
   format does not assume either (per-expert `quant_type` records what policy
   wants; cap checks use the types actually realized in the artifact, which after
   aggregation may be ≥ the policy minimum — never below it).

#### 8.2.1 Aggregation algorithm (concrete design)

`expert_stats.json` and the schema stay per-expert regardless of GGUF tensor
layout (WP-1 field names are unchanged by this section). Only the **manifest
realization step** — turning a per-expert manifest into the `--tensor-type`
override list of §8.3 — performs the roll-up, and it is a pure, order-independent
reduction with one input (the manifest's per-expert `quant_type` records) and one
output per (layer, projection).

**Fidelity order.** A total order over the ADR-002 sequence, highest first:
`Q8_0 > Q4_K > Q4_0 > IQ3_S > IQ2_M > IQ2_XS > IQ2_XXS`. (ADR-002 groups `Q4_0`
and `Q4_K` as one sequencing tier — built together, before I-quants — but does not
rank one above the other; this document ranks `Q4_K` above `Q4_0` for aggregation
purposes because K-quant blocks carry finer per-superblock scaling at the same
nominal bit width, so realizing a layer at `Q4_K` when only `Q4_0` was assigned is
still an upgrade, never a downgrade, satisfying §4.3 rule 3 either way. This
ranking is a document-local convention for the aggregation reduction, not an
ADR-002 amendment.)

**Reduction.** For each layer `L` (0–93):

```
projection_type(L) = max_fidelity(
    quant_type(record) for record in all_expert_records(L)   # all 128,
                                                               # union of nodes B and C
)
```

Because a per-expert record's `quant_type` in this schema applies uniformly to
that expert's gate/up/down weights (§3.3 — one field, not three), the same
reduced value is used for all three `--tensor-type` overrides of layer `L`
(`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`); §8.3's example groups layers by
this single per-layer value rather than computing it three times.

**Interaction with "no stats, no downgrade" (§4.3 rule 3).** The reduction is
`max`, so it is monotonic in every input: raising any single expert's assigned
type can only raise or hold `projection_type(L)`, never lower it. Two
consequences fall directly out of this:

- A `no_stats_default` expert is, by rule 3, assigned the highest-fidelity type
  that fits the cap (never an I-quant). If that expert's assigned type is the max
  in its layer, the whole layer realizes at that type — every other expert in the
  layer receives *at least* what the two-axis policy assigned it, and the
  `no_stats_default` expert receives *at least* its rule-3 floor. Aggregation
  cannot push a `no_stats_default` expert below its floor, because aggregation
  never lowers anything.
- Conversely, a `measured` expert can end up realized at a *higher* fidelity than
  its own record specifies, because a layer-mate needed more. This is the
  intended and only permitted direction of drift (§4.3 preamble: "never downgrade
  below what the two-axis policy assigned"); the manifest's per-expert
  `quant_type` field remains the source of truth for *what the policy wanted*,
  and is not overwritten by the realized value.

**Cap-check consequence (ties to §5.2).** Because realized bytes are `≥` the
per-expert estimate sum whenever aggregation forces an upgrade,
`computed_static_total_bytes` in the manifest is a **lower bound**, not the
number that will actually land on disk/in memory. §7 R2 already requires the
loader to refuse on the *loader's recomputed* total from GGUF metadata, not the
manifest's own arithmetic — that rule is what catches aggregation-driven growth;
no new refusal rule is needed, but the manifest generator should log the
per-layer aggregated type next to the per-expert requested types so an operator
can see where and why totals grew before the loader's cap check runs.

### 8.3 Manifest → override list → command

Offline pipeline (dev/ops machine, never the runtime):

1. Group `nodes.*.experts` by `layer`; take the max-fidelity `quant_type` per
   layer (§8.2 rule 2).
2. Emit one `--tensor-type` regex per (quant type → layer set) matching the fused
   expert tensors of those layers.
3. Add the §4.3 binding-rule overrides (router/gate, KV-sensitive floors).
4. Run `llama-quantize` from an F16/BF16 source GGUF with an imatrix.

Example command generated from the §6.1 manifest (layers 0,1,2,30 → Q8_0;
layers 5,8 → Q4_K; layer 10 → Q4_0; layers 15,20 → IQ3_S):

```bash
llama-quantize \
  --imatrix imatrix-qwen3-235b-ds5-calib-v1.gguf \
  --token-embedding-type q8_0 \
  --output-tensor-type q8_0 \
  --tensor-type "blk\.(0|1|2|30)\.ffn_(gate|up|down)_exps=q8_0" \
  --tensor-type "blk\.(5|8)\.ffn_(gate|up|down)_exps=q4_k" \
  --tensor-type "blk\.10\.ffn_(gate|up|down)_exps=q4_0" \
  --tensor-type "blk\.(15|20)\.ffn_(gate|up|down)_exps=iq3_s" \
  --tensor-type "ffn_gate_inp=q8_0" \
  --tensor-type "attn_k=q8_0" \
  --tensor-type "attn_v=q8_0" \
  Qwen3-235B-A22B-Instruct-2507-F16.gguf \
  Qwen3-235B-A22B-Instruct-2507-DS5-mixed.gguf \
  Q4_K_M 8
```

The concrete tensor-name patterns must be re-checked against the actual GGUF
tensor listing before the run; execution on the real 235B artifact ships as an M4
runbook (work-pack README rule 3 — cluster/download-dependent work is
operator-executed later).

## 9. Source map (acceptance: every rule cites its source)

| Rule in this document | Source |
|---|---|
| §2 model shape 94/128/8; GGUF verification refusal | Spec v0.3 §5 |
| §2, §5 caps: 33.6 GB/node, 100.8 GB cluster, 70/30 split | Spec v0.3 §4; v0.2 §4 |
| §4 two-axis policy, measured-stats traceability, hot/warm/cool/cold tiers | Spec v0.3 §6 (adoption A5) |
| §4.2 quant-type sequence and I-quants-last | ADR-002 consequence 2 |
| §4.3 rules 1–2 (router/gate FP16/Q8; KV Q8/Q6) | Spec v0.2 §7.3 (binding via v0.3 §6, §10) |
| §4.3 rule 3 (no stats, no downgrade) | Spec v0.3 §6 ("never to assumed skew") + WP-2 pack (A5 manifest-policy half) |
| §4.3 rule 4 (hot fidelity > cold) | Spec v0.3 §10 Quality; v0.2 §7.3 |
| §7 refusal style, over-cap override, run-metadata logging | Spec v0.3 §5, §11, G5 |
| §7 R4 no silent substitution | ADR-002; v0.3 G1 |
| §8 no custom quant pipeline; artifacts via `llama-quantize` | ADR-002; v0.3 §3, §6 |
| §8.1 override syntax | Upstream ggml-org/llama.cpp `tools/quantize/README.md` and discussion #12741 (verified 2026-07-12) |
| §8.2 fused expert tensor naming (`blk.<L>.ffn_*_exps.weight`) | Confirmed against the real downloaded artifact (`Qwen3-235B-A22B-Instruct-2507-UD-Q2_K_XL`, both shards, all 94 layers, verified 2026-07-13); originally sourced from upstream llama.cpp usage (2026-07-12) |
| Field-name alignment (`layer`, `expert_index`, `model_shape`, header fields, `experts[layer*128+expert_index]`) | `docs/specs/schemas/expert_stats.schema.json` (WP-1, landed 2026-07-12) |
