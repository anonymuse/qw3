# M2c gate — Qwen3-30B-A3B real-weights (T06)

**Status: PASS** per ADR-005 Amendment 2 (2026-07-17) — originally recorded
PARTIAL PASS (2026-07-16) pending an acceptance rule for bf16-oracle
comparisons; see §9 for the re-scored disposition. Mechanical checks (load,
run, config sanity, mmap memory) pass on both backends. Oracle-comparison
checks are mixed under the original brief's bar: 3 of 5 prompts pass
greedy-token-exact-match; 2 diverge at a specific token. CPU and Metal
backends agree with each other to 4+ significant figures throughout —
this is not a backend-specific bug. Router parity is 13/15 exact
(token/layer combinations checked — see §5; earlier drafts miscounted
14/15); the single divergent pattern (token 0 only, layers 23/47 only)
plausibly explains the downstream token divergence.

Branch: `t06-real-30b`, merged with `main` (includes PR #21's conflict-marker
fix). `zig build test` 74/74, `zig build test-gpu` 81/81, both green
throughout this gate — never modified to make progress.

Hardware: cluster Node B (`max-1`, Apple M5 Max, 48GB RAM), driven over SSH
from the dev laptop (Node D) with the project owner's explicit authorization
for both SSH access and the checkpoint download below.

---

## 1. Gate item results

| # | Gate item | Result | Evidence |
|---|---|---|---|
| 1 | `ds5 run` on 30B GGUF, backend=cpu, loads and executes end-to-end | **PASS** | §2 |
| 2 | `ds5 run` on 30B GGUF, backend=metal, loads and executes end-to-end | **PASS** | §2 |
| 3 | Greedy tokens match oracle exactly, 64 tokens, 5 oracle prompts | **PARTIAL: 3/5** | §4 |
| 4 | Final-step logits within ADR-005 §4 e2e tolerance (5e-2/5e-2) | **FAIL: 0/5** | §4 |
| 5 | Config sanity: parsed `ModelConfig` == ADR-001 §2 / ADR-005 §5 table | **PASS** | §3 |
| 6 | Router parity on real weights, prompt p0, 100% expert-ID match vs oracle | **13/15 token/layer combos** (fail under the brief's 100% bar; re-scored PASS-diagnostic in §9) | §5 |
| 7 | Memory: peak RSS logged, loader mmap-backed (no full-file read) | **PASS** | §2 |

Nothing here loosens a tolerance or skips a check by fiat. Items 3/4/6 are
reported exactly as measured, including the passing subset, per the
forbidden-list instruction not to claim a blanket pass or fail without
evidence.

---

## 2. `ds5 run` end-to-end, both backends, memory

Model: `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf`
(32,483,932,576 bytes, GGUF magic verified).

All 5 oracle prompts ran to completion on both `--backend cpu` and
`--backend metal`, 64 greedy decode steps each (10 runs total). No crashes,
no NaN-driven aborts, no out-of-range token ids. Loader is mmap-backed
(`sys.mmapFileRead`, `src/gguf/gguf.zig`'s `Model.open`) — confirmed
structurally (no `read()` over the file body) and empirically (peak RSS
~6GB on a 32GB file in an earlier ad hoc CPU run this session, far below a
full-file-read footprint).

---

## 3. Config sanity (ADR-001 §2 / ADR-005 §5)

Parsed via `gguf.Model.config()`, reading every field from GGUF metadata —
no hardcoded constants.

```
n_layers = 48            n_experts = 128        vocab_size = 151936
hidden_dim = 2048        top_k = 8               rms_eps = 0.000001
n_q_heads = 32            expert_ffn_dim = 768    rope_theta = 10000000
n_kv_heads = 4  (GQA 32/4)                          max_ctx = 262144
head_dim = 128
```

All fields match ADR-001 §2 / ADR-005 §5's 30B-A3B-Instruct-2507 table
exactly. **PASS.**

---

## 4. Oracle-comparison results (gate items 3 & 4)

Oracle fixtures generated via `tools/make_fixtures.py --hf <HF safetensors
checkpoint> --out tests/fixtures/qwen3-30b-a3b --layers 0,23,47 --n-new 64
--e2e` (real Qwen3-30B-A3B-Instruct-2507 HF checkpoint, 56.9 GiB, downloaded
to max-1 with the project owner's explicit sign-off — the GGUF alone is
insufficient for this step; `make_fixtures.py` requires the original
unquantized safetensors). `--n-new 64` matches the gate's 64-token
requirement exactly (the tool's own default is 8; regenerated at the
project owner's direction to match spec instead of a reduced-scope
substitute).

| Prompt | Backend | Greedy tokens | First divergence | Final-logit max\_abs\_diff | Logit tolerance (atol=rtol=0.05) |
|---|---|---|---|---|---|
| p0_capital | cpu | **exact** (64/64) | — | 1.7710 | fail (149174/151936 elems) |
| p0_capital | metal | **exact** (64/64) | — | 1.7711 | fail (149174/151936 elems) |
| p1_count | cpu | **exact** (64/64) | — | 0.6810 | fail (41184/151936 elems) |
| p1_count | metal | **exact** (64/64) | — | 0.6810 | fail (41185/151936 elems) |
| p2_code | cpu | **exact** (64/64) | — | 0.2821 | fail (9673/151936 elems) |
| p2_code | metal | **exact** (64/64) | — | 0.2821 | fail (9673/151936 elems) |
| p3_json | cpu | diverges | **token 48/64** | 32.1622 | fail (151045/151936 elems) |
| p3_json | metal | diverges | **token 48/64** | 32.1622 | fail (151045/151936 elems) |
| p4_reason | cpu | diverges | **token 7/64** | 31.0963 | fail (149264/151936 elems) |
| p4_reason | metal | diverges | **token 7/64** | 31.0962 | fail (149263/151936 elems) |

Observations:
- **CPU and Metal agree with each other to 4+ significant figures on every
  prompt**, including on the two that diverge from the oracle. This rules
  out a backend-specific (Metal dispatch / kernel) bug — whatever's
  happening is common to both, consistent with something upstream of the
  kernel-provider split (most likely the router-selection sensitivity in
  §5, or accumulated fp32 differences that affect both providers
  identically since GPU router selection is host-computed via the same
  code path as CPU per T05's design).
- The 3 passing prompts (p0-p2) still fail the strict *logit* tolerance
  despite exact token matches — the argmax decision is robust to a
  numerical difference that the raw logit values are not. Divergence
  magnitude correlates loosely with generation length/complexity (p2, the
  shortest/most-constrained completion, has both the smallest logit
  diff and using the fewest low-probability continuations).
- The 2 failing prompts diverge at very different points (token 7 vs. 48
  of 64) with the post-divergence continuations going in genuinely
  different semantic directions — this is a real correctness gap, not a
  rounding formality, for those two prompts specifically.

---

## 5. Router parity (gate item 6)

Per spec: dump per-layer expert IDs for prompt p0 at layers 0/23/47 (the
same layers make_fixtures.py traces), compare to oracle.

Method: temporary env-var-gated instrumentation added to
`src/engine/forward.zig` (dump `self.router_ids` right after `routerTopK`,
gated on `self.pos == 0` to capture only the prefill pass) and
`src/main.zig` (dump the final-step logits row). Both reverted via
`git checkout --` before this branch's final state — confirmed via
`git status`/`git diff` showing no residual changes, and both `zig build
test` (74/74) and `zig build test-gpu` (81/81) re-verified green after the
revert.

Result, by token (prompt p0 has 5 tokens, top\_k=8 experts each), compared
as sets since order doesn't affect the downstream weighted-sum MoE output:

| Layer | token 0 | token 1 | token 2 | token 3 | token 4 |
|---|---|---|---|---|---|
| 0 | match | match | match | match | match |
| 23 | **DIFFER** (1 of 8 experts) | match | match | match | match |
| 47 | **DIFFER** (1 of 8 experts) | match | match | match | match |

13 of 15 token/layer combinations match exactly (the table above has two
DIFFER cells; earlier drafts of this doc, PR #31's recap, and the HANDOFF
scoreboard miscounted 14/15 — also independently flagged by Codex review).
The two divergences are
both isolated to **token 0** (the very first prompt token) at the two
deeper traced layers (23, 47) — layer 0 (shallow, least accumulated
numerical difference) is clean, and tokens 1-4 are clean at every traced
layer. In both cases exactly one expert differs out of 8 selected (layer
23: oracle picks expert 80, the real run picks 120; layer 47: oracle picks
55, the real run picks 20) — a single-swap pattern, not wholesale
disagreement.

The oracle's own fixture generator asserts no near-tie exists at the
top-k boundary for these exact cases (`assert_no_router_neartie`, tol=1e-6
on post-softmax probability gap) — so this is not a case where the
Python oracle itself flags ambiguity. The divergence therefore likely
stems from fp32 computation-order differences between the Python
(`torch.softmax` over a single dense matmul) and Zig (dequantized-weight
GEMM + separate softmax) paths being large enough — plausibly O(1e-4) to
O(1e-3) relative, typical for large fp32 matmuls under different summation
order — to cross a boundary the oracle's stricter 1e-6 check didn't
anticipate needing to guard against cross-implementation, only
within-oracle ambiguity.

**Not conclusively root-caused within this gate's scope** (see §6).

---

## 6. Localization and recommended follow-up

This gate did not go further than identifying *where* the divergence
starts; per the T06 brief, full root-causing beyond a gate's reasonable
scope is an explicit hand-back point, not a requirement to exhaust here.

What's established:
1. The failure mode is a **single differing MoE expert selection** at
   specific (token, layer) points, isolated to deep layers and not
   present at layer 0.
2. It is **not tolerance-related for p3/p4** — real token-level
   divergence with semantically different continuations.
3. It is **not a CPU/Metal backend bug** — both backends agree with each
   other far more tightly than either agrees with the oracle.
4. It **does not reproduce for p0-p2** at the token level, only at the
   raw-logit level — suggesting the effect is real but usually
   sub-threshold for argmax, and only occasionally (p3 at step 48, p4 at
   step 7) large enough to flip a decision.

Recommended next step for whoever picks this up: extend
`tools/make_fixtures.py`'s per-layer trace fixtures to prompts p3 and p4
specifically (currently only p0 is traced), then use the trace-hook
pattern from `test_forward.zig`/`test_gpu_forward.zig` to find the first
layer where p3's or p4's hidden states diverge beyond the "layer" op
tolerance (2e-3 abs / 1e-2 rel per ADR-005 §4) — that will show whether
the divergence is still purely router-selection-driven at deeper layers,
or whether something else compounds alongside it. A secondary, cheaper
experiment: dump the actual gating probabilities (not just top-8 ids) at
layer 23/47 for p0's token 0 to quantify exactly how close the 8th/9th
expert's scores are — if it's a hair's-breadth margin, that's strong
evidence this is fundamentally a numerics-precision issue rather than a
logic bug, and may warrant an ADR-level discussion about whether e2e
tolerance or router-parity strictness needs revisiting for deep-MoE
models at this scale, rather than continued kernel debugging.

---

## 7. What did NOT happen (forbidden-list compliance)

- No fixture tolerance was loosened or bypassed.
- No router-parity or logit-tolerance check was skipped or claimed passing
  without evidence — failures are reported as failures, with exact counts.
- No model constant was hardcoded; every `ModelConfig` field in §3 came
  from live GGUF metadata parsing.
- `src/shared/contracts.zig`, `src/kernels/cpu/kernels_{a,b,c}.zig`, and
  all `PORTING-*.md` files were not touched.
- The temporary router/logit-dump instrumentation (§5) was reverted before
  this branch's final state; `git status` is clean relative to `main` plus
  PR #21 aside from this file and `bench/results/` run-metadata JSON.
- `zig build test` (74/74) and `zig build test-gpu` (81/81) stayed green
  throughout, re-verified after the instrumentation revert.

---

## 8. Outstanding items for whoever picks up T07 [resolved 2026-07-17 — see §9]

- T07 (M3 distributed) should **not** be unblocked on this gate's current
  result — greedy-token correctness at 3/5 and logit-tolerance at 0/5 is
  short of "pass," even though the mechanical/config/memory checks and
  the CPU/Metal parity story are strong. Recommend treating §6's
  root-cause as a explicit pre-T07 task, not a background concern.
- Real 3-node `ds5 bench link` (not loopback) still hasn't been run —
  separate from this gate, but also blocking for T07's distributed work.
- `docs/orchestration/HANDOFF.md`'s task DAG and scoreboard need updating
  to reflect this gate's actual (partial) result rather than the
  T05-era "T06 BLOCKED, awaiting GGUF" framing, which is now stale.

---

## 9. Disposition under ADR-005 Amendment 2 (2026-07-17)

ADR-005 §4's footnote already made the e2e logit tolerance diagnostic-only
against a bf16 HF oracle ("only greedy tokens gate") but never defined the
greedy acceptance rule for that non-weight-matched case — and the T06 brief's
items 4 and 6 imported hard bars (logit tolerance, router 100%) that the ADR
only ever established for weight-matched comparisons. Amendment 2
(`docs/decisions/ADR-005-interface-freeze.md`) supplies the missing rule: a
near-tie-guarded greedy gate. This section re-scores the affected items under
it, using this doc's measurements plus the root-cause investigation
(`docs/findings/m2-router-divergence-localization.md`, PR #31).

| # | Gate item | Re-scored | Basis |
|---|---|---|---|
| 3 | Greedy tokens vs bf16 oracle | **PASS** | 0 mismatches at guarded positions (oracle top-2 margin ≥ 0.5); 2 invoked near-tie exclusions in 249 context-identical decisions (0.8% ≤ 5%), both with the engine picking the oracle's own #2 token at margins 0.00702 / 0.00255; sample floor met (249 ≥ 200). |
| 4 | E2E logit tolerance vs bf16 oracle | **DIAGNOSTIC** (not a gate) | Never a hard gate for this oracle class per the pre-existing §4 footnote; the brief's wording dropped that carve-out. Recorded baseline band at context-identical positions: 0.28–1.77 max-abs. p3/p4's 32.2/31.1 were measured at post-divergence (context-mismatched) positions; future runs measure at the last context-identical position instead. |
| 6 | Router parity vs bf16 oracle | **PASS (diagnostic)** | 13/15 sampled combos exact. Both swaps benign per Amendment 2's signature: single-expert; oracle-side displaced probability mass 2.0e-6 (layer 23) and ~6.1e-3 (layer 47) ≤ 2e-2; swapped experts' probabilities ≤ 0.05; no recurring pattern. Sampling caveat (also raised by Codex review): the original audit covered 3 of 48 layers; PR #31's `--block-trace` extended router coverage to all 48 layers at p3/p4's critical decode steps (p4: single-expert swaps at layers 9/12/34/46; p3: zero swaps), all consistent with the benign signature. The hard router gate remains the §2 fixture gate (weight-matched, near-tie-guarded), which passes at 100% IDs / ~1e-7 gate weights. |
| 8 (new) | CPU-vs-Metal cross-backend e2e — hard, weight-matched by construction | **PASS** | Greedy tokens identical on all 5 prompts including both oracle-divergent ones; logits agree to 4+ significant figures (§4 table), far inside the §4 e2e row (5e-2/5e-2). |

Items 1/2/5/7 are unchanged (PASS). **Overall: T06 = PASS.** T07's DAG
dependency on this gate is satisfied. T07 remains gated on the independent
real 3-node `ds5 bench link` run (HANDOFF §2 hardware item 2). Note for T07:
its distributed-vs-single-node comparison is weight-matched by construction
and therefore gates on the hard §4 rows directly — Amendment 2's statistical
rule applies only to non-weight-matched (e.g. bf16 HF) oracles.
