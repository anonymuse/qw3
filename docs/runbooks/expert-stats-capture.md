# Runbook: Expert Stats Capture (Routing Frequency + Quantization Sensitivity)

**Work pack:** WP-1 (adopts amendment A5, two-axis expert quantization — data half)
**Produces:** `expert_stats.json` conforming to `docs/specs/schemas/expert_stats.schema.json`
**Consumed by:** M1 placement simulator, `docs/findings/f001`, M4 quant manifest (WP-2)
**Boundary (ADR-002):** llama.cpp is an **offline oracle only** — run it on a cluster
node to produce data files; it is never linked into, or invoked by, the DS5 runtime.

CLI flags below were verified against upstream llama.cpp docs
(`tools/imatrix/README.md`, `tools/perplexity/README.md`,
`tools/quantize/README.md` at `github.com/ggml-org/llama.cpp`) on 2026-07-12.
Re-verify with `--help` on the exact commit you build — flags have churned
historically (e.g. imatrix default output moved from `.dat` to `.gguf`).

---

## 0. Prerequisites

| Item | Requirement |
|---|---|
| Machine | One cluster M5 Max node (48GB UMA), not the dev laptop |
| Disk | See per-step estimates; budget **≥ 400GB free** on NVMe for the full pass |
| Model | Qwen3-235B-A22B-Instruct-2507 GGUF artifacts (see §1) — download via `tools/download_models.sh`, already covers the Q2-class telemetry artifact |
| Corpus | The router-calibration corpus at `bench/corpus/` (see its README) |
| llama.cpp | Built from source at a **pinned commit you record** (see §1) |
| Repo | This repo checked out; `.venv` available for the merge step (§5 can also run on the dev laptop after copying outputs back) |

### Build llama.cpp (offline oracle)

```sh
git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
cd /tmp/llama.cpp
git rev-parse HEAD > LLAMA_COMMIT.txt        # record for provenance — required by merge tool
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
# Binaries: build/bin/llama-imatrix, build/bin/llama-perplexity, build/bin/llama-quantize
```

### Record provenance hashes (required — the merge tool refuses without them)

```sh
MODEL=/path/to/Qwen3-235B-A22B-Instruct-2507-<QUANT>.gguf
CORPUS=/path/to/router-calibration-corpus.txt
shasum -a 256 "$MODEL"    # -> model_hash ("sha256:<hex>")
shasum -a 256 "$CORPUS"   # -> corpus_hash ("sha256:<hex>")
```

Sanity-check the GGUF metadata before burning a day of compute — 94 layers,
128 experts, top-8 (spec §5; external docs have carried wrong constants):

```sh
python /tmp/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py "$MODEL" \
  | grep -E "block_count|expert_count|expert_used_count"
# expect: block_count = 94, expert_count = 128, expert_used_count = 8
```

If those three values are anything else, **stop** — wrong artifact.

---

## 1. Choose artifacts per pass

| Pass | Artifact | Why |
|---|---|---|
| Routing telemetry (§2) | Q2-class 235B GGUF (~85GB) | Routing is computed from FP router/gate tensors even in low-bit quants; fits mmap-streamed on one node. Matches execution-plan hour-zero task 3. |
| imatrix (§3) | Highest-precision GGUF you can stream — Q8_0 (~250GB) preferred over Q2-class | imatrix quality tracks base-model fidelity. Q8_0 does not fit in 48GB; llama.cpp mmap streams from NVMe. Slow but offline. |
| KLD baseline + comparisons (§4) | Baseline: same Q8_0 (as "reference"); comparisons: each candidate quant | True FP16 (~470GB) is impractical on one node; using Q8_0 as the KLD reference is a documented compromise — record it in the run log. |

---

## 2. Pass A — Routing-frequency telemetry (M1 capture)

**What it measures:** per (layer, expert) activation counts and gate weights over
the router-calibration corpus. This is execution-plan hour-zero task 3.

**Honest tooling note (verified 2026-07-12):** stock llama.cpp has **no CLI flag
that dumps per-expert routing counts**. The M1 telemetry capture instruments
llama.cpp's eval path (an eval-callback or small local patch logging the
`top_k` expert IDs + gate weights per layer per token — capture instrumentation
is owned by the M1 workstream, not this pack). Whatever the instrument is, its
output must be reduced to the telemetry JSON contract below; the merge tool
consumes only this contract.

### Telemetry JSON contract (input to merge_stats.py)

```json
{
  "source_id": "telemetry-2026-07-XX-run1",
  "model_id": "Qwen3-235B-A22B-Instruct-2507",
  "model_hash": "sha256:<hex of the GGUF used>",
  "corpus_id": "router-calibration-corpus-v1",
  "corpus_hash": "sha256:<hex>",
  "num_layers": 94,
  "num_experts_per_layer": 128,
  "top_k": 8,
  "total_tokens": 123456,
  "layers": [
    {
      "layer": 0,
      "activation_counts": [/* 128 non-negative ints, one per expert */],
      "gate_weight_sums":  [/* 128 floats: sum of gate weights over activations */]
    }
    /* ... exactly 94 entries, layers 0..93 ... */
  ]
}
```

Rules the merge tool enforces (refusal, exit 2): exactly 94 layer entries ×
128 counts; `activation_counts[e] <= total_tokens`; all header/hash fields
present; shape fields exactly 94/128/8.

**Runtime estimate (unverified until first run):** Q2-class 235B decode on one
M5 Max streams ~85GB of weights; expect low single-digit tok/s. For a ~100K-token
corpus budget **8–24 hours**. Disk: model only (~85GB) plus a few MB of output.

**Output location:** `bench/results/expert_stats/telemetry-<date>-runN.json`

---

## 3. Pass B — imatrix (per-expert importance proxy)

**What it measures:** mean squared activation magnitudes per weight position,
accumulated over the corpus. For Qwen3-MoE GGUFs the expert FFN weights are
fused 3D tensors (`blk.<L>.ffn_gate_exps.weight`, `ffn_up_exps`,
`ffn_down_exps` — one slice per expert), so the imatrix carries per-expert
activation data that we reduce to `imatrix_importance` per (layer, expert).

### Run (flags verified against tools/imatrix/README.md)

```sh
/tmp/llama.cpp/build/bin/llama-imatrix \
  -m "$MODEL_Q8" \
  -f "$CORPUS" \
  -o imatrix-235b-<date>.gguf \
  --output-format gguf \
  -ngl 99 \
  --chunks 200
```

Flag notes:
- `-m` / `-f` are the two mandatory flags (model, calibration text).
- `-o` output file; default is `imatrix.gguf`. GGUF is the current default
  format; `--output-format dat` selects the legacy binary.
- `-ngl 99` offloads all layers Metal-side; with a streamed Q8_0 the run is
  NVMe-bound regardless.
- `--chunks 200` caps processed 512-token chunks — size to your corpus
  (200 chunks ≈ 102K tokens). Omit to process the whole file.
- `--in-file existing.gguf` merges previous imatrix runs if you capture the
  corpus in slices.
- `--show-statistics` (run afterwards, cheap) prints per-tensor summaries —
  useful sanity check, not the extraction path.

**Runtime estimate (unverified):** imatrix is prompt-processing (prefill), much
faster per token than decode, but a streamed 250GB Q8_0 is NVMe-bound: budget
**4–12 hours** for ~100K tokens. Disk: model (~250GB) + imatrix file (tens of MB).

### Reduce to per-expert sensitivity JSON

The imatrix GGUF stores per-tensor sums of squared activations plus counts.
Reduce each fused expert tensor to one scalar per expert slice (mean of that
expert's slice) with the `gguf` Python package (in llama.cpp's `gguf-py`, or
`pip install gguf` in a scratch venv — **not** the repo `.venv`):

```python
# reduce_imatrix.py — run next to the imatrix output; writes sensitivity JSON
# in the contract below. ~30 lines: for each blk.<L>.ffn_*_exps tensor in the
# imatrix GGUF, reshape sums to (num_experts, -1), divide by counts, take the
# per-expert mean across the three FFN tensors -> imatrix_importance[L][e].
```

(The exact tensor naming inside imatrix GGUF files should be confirmed against
the file itself with `gguf_dump.py` — the layout has changed across llama.cpp
versions. This is why the reduction script lives with the operator, pinned to
the recorded llama.cpp commit, not in this repo.)

### Sensitivity JSON contract (input to merge_stats.py, one file per quant type)

```json
{
  "source_id": "imatrix-2026-07-XX-run1",
  "quant_type": "Q4_K_M",
  "model_id": "Qwen3-235B-A22B-Instruct-2507",
  "model_hash": "sha256:<must match the telemetry pass's model line>",
  "llama_cpp_commit": "<git rev-parse HEAD from the build>",
  "num_layers": 94,
  "num_experts_per_layer": 128,
  "top_k": 8,
  "experts": [
    {
      "layer": 0, "expert_index": 0,
      "imatrix_importance": 12.34,
      "kld": 0.012, "kld_stderr": 0.0004, "ppl_ratio": 1.02
    }
    /* partial coverage allowed — records may omit experts or metric fields,
       but each record needs at least one metric */
  ]
}
```

Allowed `quant_type` values: `Q8_0`, `Q4_K_M`, `Q4_K_S`, `IQ3_S`, `IQ2_M`
(the ADR-002 kernel sequence). Note: `model_hash` must equal the telemetry
file's `model_hash` or the merge refuses — both axes must describe the same
artifact.

---

## 4. Pass C — KLD per candidate quant type

**What it measures:** how much each candidate quantization distorts the output
distribution, via Kullback–Leibler divergence against a reference model's
logits over the same corpus.

**Honest scope note:** llama.cpp's KLD is a **whole-model** metric — it does not
attribute divergence to individual experts. Per-expert `kld` fields in the
sensitivity JSON can only be populated by per-expert override experiments
(quantize one expert group down, re-measure — combinatorially expensive) and
are expected to stay absent in the first capture. The per-expert axis-2 signal
is `imatrix_importance` (§3); whole-model KLD per quant type validates the
candidate set and feeds `kld`/`ppl_ratio` at whatever granularity you measured.
Absent fields are fine — the schema and merge tool accept partial records.

### Step 1 — record reference logits (flags verified against tools/perplexity/README.md)

```sh
/tmp/llama.cpp/build/bin/llama-perplexity \
  -m "$MODEL_Q8" \
  -f "$CORPUS" \
  --kl-divergence-base logits-base-235b-<date>.bin
```

**Disk warning:** the logits file is huge — upstream README cites 11 GiB
(LLaMA 2) to 37 GiB (LLaMA 3) for standard corpora; Qwen3's ~151K vocab makes
it larger per token. For a ~100K-token corpus budget **50–150GB (estimate)**.
Keep the corpus modest.

### Step 2 — produce candidate quants (flags verified against tools/quantize/README.md)

```sh
/tmp/llama.cpp/build/bin/llama-quantize \
  --imatrix imatrix-235b-<date>.gguf \
  "$MODEL_SOURCE" candidate-IQ3_S.gguf IQ3_S
```

(Repeat per candidate type. `--include-weights` / `--exclude-weights` exist for
tensor-targeted imatrix application but cannot be combined with each other.)

### Step 3 — measure KLD per candidate

```sh
/tmp/llama.cpp/build/bin/llama-perplexity \
  -m candidate-IQ3_S.gguf \
  -f "$CORPUS" \
  --kl-divergence-base logits-base-235b-<date>.bin \
  --kl-divergence
```

Record from the printed report: mean KLD ± stderr, PPL ratio. Enter them into
that quant type's sensitivity JSON (§3 contract) — at model granularity, either
replicated per expert with a `sources` note or held in the run log until
per-expert measurements exist.

**Runtime estimate (unverified):** each perplexity pass is prefill-speed over
the corpus; NVMe-streaming dominates. Budget **2–8 hours per candidate**.

---

## 5. Merge and validate

Copy the telemetry JSON (§2) and sensitivity JSONs (§3/§4) to the repo, then:

```sh
cd <repo-root>
source .venv/bin/activate
python tools/expert_stats/merge_stats.py \
  --telemetry bench/results/expert_stats/telemetry-<date>-run1.json \
  --sensitivity bench/results/expert_stats/sens-Q4_K_M-<date>.json \
  --sensitivity bench/results/expert_stats/sens-IQ3_S-<date>.json \
  --output bench/results/expert_stats/expert_stats.json
```

- `--git-commit` defaults to `git rev-parse HEAD`; pass explicitly if merging
  outside a checkout.
- Exit 0: output written, one record per (layer, expert) — 12,032 records —
  with per-field provenance. Exit 2 (`refused: ...` on stderr): shape mismatch,
  missing provenance, or hash mismatch; **no output is written**. Fix the
  input, do not force.
- The output is checked against
  `docs/specs/schemas/expert_stats.schema.json` constraints (full JSON Schema
  validation additionally runs if `jsonschema` is installed; the built-in
  checks always run).

Smoke-test the tool anywhere with the synthetic fixtures:

```sh
source .venv/bin/activate
python -m unittest discover tools/expert_stats/tests -v
```

---

## 6. Where outputs land

| Artifact | Location | Committed? |
|---|---|---|
| Telemetry JSON | `bench/results/expert_stats/telemetry-<date>-runN.json` | Yes (small) |
| Sensitivity JSONs | `bench/results/expert_stats/sens-<QUANT>-<date>.json` | Yes (small) |
| Merged `expert_stats.json` | `bench/results/expert_stats/expert_stats.json` | Yes — the M1/M4 input |
| imatrix GGUF | cluster node scratch; back up alongside results | Optional (tens of MB) |
| KLD base logits `.bin` | cluster node scratch | **No** (50GB+); delete after §4 |
| Candidate quant GGUFs | cluster node scratch | No; keep the winners for M4 |
| Run log (commands, timings, llama.cpp commit, compromises) | `bench/results/expert_stats/RUNLOG-<date>.md` | Yes |
