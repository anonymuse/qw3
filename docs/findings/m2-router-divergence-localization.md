# M2c gate follow-up: router divergence root-cause localization (T06)

**Status: root cause localized to Q8_0 weight-quantization error, not a
coding bug.** Two diagnostics were run against the real Qwen3-30B-A3B GGUF
on Node B (`max-1`): (1) the actual post-softmax probability margins at the
two known router swaps in p0 (layers 23/47, token 0), on both the oracle and
real sides; (2) per-layer localization of where p3's and p4's hidden state /
router selection first diverges from oracle at the specific decode step that
produces their first mismatched token. Both diagnostics converge on the same
explanation: the router's *input* (the hidden state feeding it) drifts away
from the full-precision oracle by roughly the magnitude expected from Q8_0
weight quantization, compounding gradually with depth; occasionally that
drift is large enough to flip a router top-k decision. Neither diagnostic
found evidence of a logic bug in the router or kernel implementations.

Branch: `t06-router-localization`, off `t06-real-30b` (PR #29). No frozen
file was touched in the final state (`src/engine/forward.zig` was
temporarily instrumented and reverted — see §5). `zig build test` 74/74,
`zig build test-gpu` 81/81, unchanged.

---

## 1. Recap of what PR #29 established

(`docs/findings/m2-gate.md`, read in full before this investigation) — 3/5
oracle prompts (p0/p1/p2) match the oracle's greedy tokens exactly across 64
generated tokens; 2/5 (p3/p4) diverge at generated-token index 48 and 7
respectively (0-indexed; confirmed by direct diff of `~/gate-results/*.tokens.txt`
against the oracle manifest's `greedy_tokens`, both here independently and
consistent with the gate doc's "48/64" and "7/64" language). CPU and Metal
backends agree with each other far more tightly than either agrees with the
oracle, ruling out a backend-specific bug. Router parity for p0 at layers
0/23/47 is 13/15 token/layer combinations exact (13, not the 14 the gate
doc's prose originally stated — its own table shows two DIFFER cells, since
corrected); the two exceptions are both
at token 0, layers 23 and 47 (a single expert swapped out of the top-8 each
time): layer 23 oracle picks expert 80 vs real picks 120; layer 47 oracle
picks expert 55 vs real picks 20.

---

## 2. Diagnostic 1 — router probability margin at the p0 swap points

### Method

**Oracle side**: extended `tools/make_fixtures.py` with a `--block-trace`
mode (§4) that, among other things, dumps the FULL 128-expert post-softmax
distribution per traced layer (`LayerTrace.probs`, which `router_topk()`
already computes internally — `emit_layer_cases` just never persisted more
than the top-8 before now). Ran it for p0 (`--trace-prompt p0_capital
--trace-append 0`), layers 23/47.

**Real (Zig) side**: temporary env-var-gated instrumentation in
`src/engine/forward.zig` (`DS5_ROUTER_PROBE_DIR` / `DS5_ROUTER_PROBE_LAYERS`,
active only during the prefill call, `self.pos == 0`). Right after the real
`K.routerTopK` call (unmodified, top_k=8, the actual production path), it
makes a **second, diagnostic-only** call to the same frozen `routerTopK`
kernel (`src/kernels/cpu/kernels_c.zig`, not edited) with `top_k =
n_experts` (128) and `norm_topk_prob = false`. Because every expert is
"selected" in that call, `out_weights` is exactly the raw per-expert softmax
probability (the picked-set sums to 1 regardless of the norm flag), sorted
descending by construction of the kernel's own top-k selection loop. This
reads the frozen kernel differently, not modifies it, and writes to
scratch buffers separate from the ones the real MoE compute uses, so it does
not perturb the model's actual output.

Cross-validated: the new `--block-trace` oracle tool reproduces the
already-committed `l{0,23,47}_block.output.ds5t` fixtures bit-for-bit
(`max_abs_diff = 0.0`), and the real-side probe reproduces the exact expert
swap already reported in PR #29 (120 vs 80 at layer 23, 20 vs 55 at layer
47) — so both new instruments agree with the already-trusted prior work
before being used for anything new.

### Result

| Layer | Oracle pick (rank in oracle order) | Real pick (rank in oracle order) | Oracle boundary gap (rank7−rank8) | Real boundary gap (rank7−rank8) | Rank shift |
|---|---|---|---|---|---|
| 23 | expert 80, rank 7, p=7.7906e-3 | expert 120, rank 8, p=7.7887e-3 | **1.953e-6** | 3.607e-6 | 1 (adjacent) |
| 47 | expert 55, rank 6, p=2.5517e-2 | expert 20, rank 9, p=1.9442e-2 | **5.244e-4** | 5.412e-3 | 7 (NOT adjacent) |

**Layer 23 is a genuine coincidental near-tie**: both oracle's and real's own
softmax distributions place experts 80 and 120 within ~2-4e-6 of each other
at the rank-7/rank-8 boundary — a hair's-breadth, and the two
implementations land on opposite sides of it. This is squarely in the "≈1e-6,
unavoidable numerics" regime the investigation brief asked to distinguish.

**Layer 47 is NOT a near-tie.** The gap between oracle's pick and real's pick
is 5.4e-3–6.1e-3 (depending which side's ranking you measure it on) — roughly
1000x larger than layer 23's — and the rank shift is 7, not 1: on the real
side, oracle's pick (expert 55) has fallen all the way to rank 13, while
real's pick (expert 20) has risen to rank 5. This is a real, if modest (~0.6
percentage points of probability mass), reordering of several experts, not a
coin-flip at a single boundary.

### Why layer 47 differs from layer 23: hidden-state drift, not router-kernel noise

To understand *why* layer 47's margin is so much larger, I extended the
`--block-trace` dump to capture the full per-layer residual stream (`x_out`)
for all 48 layers of p0's prefill, and added a matching real-side dump
(`DS5_LAYER_TRACE_DIR` / `DS5_LAYER_TRACE_AT_POS`, gated on the forward
call's starting position) that downloads `self.x` after every layer's block
update via the existing generic `ctx.download` (backend-agnostic — same code
path Metal already uses for its logits download).

Comparing oracle vs real residual streams for p0 token 0, layer by layer
(ADR-005 "layer" tolerance is atol=2e-3/rtol=1e-2 for reference, though see
the caveat below):

| Layer | oracle `|x|` max | diff max | note |
|---|---|---|---|
| 0 | 2.47 | 0.0023 | tiny — only embedding-table dequant so far |
| 1 | 120.18 | 0.558 | **jumps ~240x** the instant one full Q8_0-quantized attn+MoE block has run |
| 2–45 | ~800–1300 | ~2.2–2.4 | roughly flat, ~8% of RMS, for 44 layers |
| 46 | 849 | 9.41 | **jumps ~4x** |
| 47 | 243 | 22.7 | **jumps ~2.4x more** |

(Full 48-row table in the investigation scratch output; the pattern is the
useful part.) The jump from layer 0 to layer 1 lines up exactly with the
point where Q8_0-quantized weight matrices (attn_q/k/v/o, all three expert
banks — everything except norms and the router weight, which stay f32 per
ADR-005 §1/§5) first get multiplied into the computation; layers 2–45 hold a
roughly constant *relative* perturbation (consistent with quantization noise
propagating through RMSNorm-renormalized residual blocks without obviously
compounding further); then layers 46–47 show a distinct several-fold
acceleration right where the layer-47 router swap happens.

**Important methodological caveat**: this "layer" comparison is oracle
(full-precision fp32 HF weights) vs. the *production* pipeline (real Q8_0
GGUF) — not two equal-precision implementations. The gate's own per-op
fixtures (which DO bake Q8_0 into the oracle's own weights before comparing,
to isolate kernel correctness from quantization error) show the router
kernel matches its own quantization-aware oracle to 100% top-8-id-exactness
and ~1e-7/1e-8 gate-weight precision on the synthetic set (`zig build test`
output: `router case l0_router: ids 68/68 exact, gate max_abs_diff
5.9604645e-8`). So the **kernel logic itself is not the source of
divergence** — the divergence is entirely attributable to the router's
*input* differing, and that input differs because the real pipeline is
running on genuinely quantized weights and the gate's oracle (by design,
consistent with how PR #29's own oracle was already built) is not.

This refines PR #29's working hypothesis. §5/§6 of `docs/findings/m2-gate.md`
attributed the divergence to "fp32 computation-order differences... plausibly
O(1e-4) to O(1e-3) relative" — i.e., framed it as noise between two
equal-precision implementations. The evidence here says the dominant
contributor is **Q8_0 weight-quantization error** (an intentional, expected
property of the production pipeline being gated, of a materially larger
magnitude — order 1e-3 to 1e-2 relative — than pure fp32 summation-order
noise, order 1e-7), compounding gradually across 47 layers of depth until it
occasionally (layer 47 here) crosses a router decision boundary by a
non-trivial margin.

---

## 3. Diagnostic 2 — localizing p3's and p4's divergence

### Method

Per-prompt "critical step" identification: for each of p3/p4, diffed the
real engine's own generated tokens (`~/gate-results/p{3,4}_*_cpu.tokens.txt`)
against the oracle's `greedy_tokens` (manifest) to confirm the exact
divergence index (0-indexed): **p3 diverges at generated[48]**, **p4 at
generated[7]** (both counted programmatically here, matching PR #29's
"48/64"/"7/64"). Working through `main.zig`'s driver loop (prefill +
single-token decode calls, `next = argmax(...)` feeding the next call),
generated[k] (k≥1) is produced by the decode call that places
generated[k-1] at absolute position `prompt_len + k - 1`. So:

- **p3** (prompt_len=7): the decode call that produces generated[48] starts
  at absolute position **54**, processing input token generated[47].
- **p4** (prompt_len=14): the decode call that produces generated[7] starts
  at absolute position **20**, processing input token generated[6].

Since real matches oracle exactly up through generated[47] (p3) / generated[6]
(p4) by definition of "first divergence," oracle's own token sequence can be
used directly to reconstruct the exact input to this one critical call —
`tools/make_fixtures.py --block-trace --trace-prompt p3_json --trace-append
48` (and `--trace-append 7` for p4), reusing the already-cached
`greedy_tokens` from the existing manifest rather than re-running
`greedy_decode` (see §4). This produces oracle per-layer traces (residual
stream + router picks) for **all 48 layers at every position**, from which
the LAST position (54 for p3, 20 for p4) is the one that matters.

On the real side, ran `ds5 run --backend cpu --greedy --steps 48` (p3) /
`--steps 7` (p4) with the same `DS5_LAYER_TRACE_DIR`/`DS5_LAYER_TRACE_AT_POS`
instrumentation as Diagnostic 1, gated to fire only on the one decode call
whose starting position matches the target (54 / 20) — capturing that one
call's per-layer residual stream and top-8 router picks. Confirmed both real
runs reproduce the documented token sequences (real's own greedy tokens
through this point exactly match oracle's, as expected).

### Result: p4 — divergence is scattered across depth, gradual, no single culprit layer

| | |
|---|---|
| Router expert-set mismatches (vs oracle, this one critical step) | **layers 9, 12, 34, 46** — each a single-expert swap |
| Hidden-state diff\_max trend | grows roughly monotonically: 0.0006 (layer 0) → 0.02 (layer 10) → 0.07 (layer 15) → 0.19 (layer 38) → 0.67 (layer 47) |
| "background" diff\_max (median, layers 1-19) | 0.0198 |
| Layers exceeding 3× that background | 14, 15, 25, 26, 34-47 (a mild second-half acceleration, not one sharp jump) |

No single layer stands out as "the" cause — four different layers each
independently swap one expert (out of 8) for a very similar-magnitude
neighbor, and the hidden-state drift accumulates gradually and fairly
smoothly across the full 48-layer depth of this one call. This is a
*distributed*, not a localized, divergence.

### Result: p3 — no router swap in the critical step at all; pure accumulated-logit near-tie

Even more informative: for p3's critical step (predicting generated[48]),
**the router expert set matches oracle exactly at all 48 layers of this
specific forward call** — zero swaps. The residual-stream diff still grows
gradually with depth (background diff\_max ~0.007 at layers 1-19, growing to
~0.19-0.28 by layers 44-46), matching the same slow Q8_0-quantization-drift
signature as p0 and p4's non-swap layers, but never yet enough by itself to
flip a router decision *at this step*. The divergence must therefore stem
from the accumulated numerical drift acting directly on the final logits —
either from this step's own continuous drift, from earlier decode steps'
(1-47) router swaps or drift baked into the 54 prior KV-cache entries this
step's attention reads, or both; distinguishing those precisely would need
instrumenting every one of the preceding 47 decode steps individually, which
was judged out of scope for the time available (a legitimate partial
result — see the investigation brief's own allowance for this).

### The decisive confirmation: both failures are razor-thin FINAL-LOGIT near-ties

Using the already-generated `p3_json.logits.ds5t` / `p4_reason.logits.ds5t`
e2e fixtures (full per-position oracle logits, not just the final row) and
the real engine's actual chosen tokens from `~/gate-results/`, extracting
the oracle's logit row at exactly the critical position:

| Prompt | Oracle top-1 (correct) | Oracle top-2 | **Real engine's actual pick** | Margin (top-1 − top-2 logit) |
|---|---|---|---|---|
| p3_json, pos 54 | token 5501, logit 24.0264 | token 5097, logit 24.0194 | **token 5097 (= oracle's #2!)** | **0.00702** |
| p4_reason, pos 20 | token 1352, logit 27.8511 | token 5944, logit 27.8486 | **token 5944 (= oracle's #2!)** | **0.00255** |

In both cases, out of a 151,936-token vocabulary, the real engine's actual
output is *exactly* the oracle's own second-ranked candidate, separated from
the correct answer by a logit margin of ~0.003–0.007 — about 0.01-0.03% of
the ~24-28 logit scale at that position. This is about as clean a
"coincidental near-tie, tipped by accumulated quantization noise" signature
as this kind of investigation could hope to find: not a wrong answer by a
wide margin, but the model's own two best candidates being close enough that
an INT8-quantization-scale perturbation, compounded over 7-14 prompt tokens
× up to 54 generation steps × 48 layers, was enough to swap which one wins.

---

## 4. `tools/make_fixtures.py` extension

Added `--block-trace` (with `--trace-prompt NAME` and `--trace-append N`),
generalizing the per-op trace beyond the previously-hardcoded `prompts[0]`
(p0_capital only). Unlike the existing `--layers` per-op path (which runs
the full `emit_layer_cases` battery — rmsnorm/rope/attention/router/
expert_mlp/layer sub-fixtures, expensive per layer on a 30B model since it
re-quantizes full expert banks for each), `--block-trace` is a lightweight
dump of just the residual stream (`x_out`) and router state (`ids`,
`gate_w`, full `probs`) for every requested layer, using `LayerTrace`
fields `model_forward` already populates. Since `model_forward` runs every
layer regardless of which are traced, tracing all 48 costs the same as
tracing 3, so it always defaults to all layers.

`--trace-append N` extends the traced sequence with the first N tokens of
the prompt's own oracle greedy continuation, needed to reach decode-depth
divergences like p3's (token 48) and p4's (token 7), not just the prefill.
If an `--e2e` fixture manifest already exists in `--out` (as it does here,
from PR #29's gate run), those exact cached `greedy_tokens` are reused
instead of re-running `greedy_decode` (avoiding ~40+ redundant O(n²)
recompute-from-scratch passes for p3's 48-token depth).

This is committed (not thrown away) — it's a direct, reusable answer to a
gap the original gate's own §6 flagged ("extend make_fixtures.py's per-layer
trace fixtures to prompts p3/p4 specifically... currently only p0 is
traced").

---

## 5. Temporary instrumentation and revert

`src/engine/forward.zig` was temporarily modified (both diagnostics'
real-side dumps, all env-var gated and no-ops when the vars are unset) and
reverted via `git checkout --` before this branch's final state, exactly
matching PR #29's own precedent. Confirmed via `git status`/`git diff`
against `t06-real-30b` showing no residual changes outside this findings
file and `tools/make_fixtures.py`. `zig build test` (74/74) and `zig build
test-gpu` (81/81) both re-verified green after the revert.

`src/main.zig` was not touched this time (unlike PR #29's logits-dump
instrumentation, all diagnostic dumping here was driven entirely through
env vars read inside `forward.zig`, invoked via the existing `ds5 run` CLI
with no new flags needed).

---

## 6. Assessment: inherent precision limit, not a fixable bug

**This is a numerics/quantization-sensitivity characteristic of the
production pipeline, not a coding bug**, for the following reasons:

0. **The two actual token-level failures are both razor-thin final-logit
   near-ties where the real engine lands on exactly the oracle's own
   second-best answer** (§3): margins of 0.0025-0.0070 out of a ~24-28 logit
   scale, at the single position that actually flips. This is the single
   strongest data point in this investigation — it is very hard to reconcile
   with a code-level bug (which would have no particular reason to produce
   the oracle's own runner-up rather than something unrelated) and easy to
   reconcile with "accumulated quantization noise occasionally crosses an
   already-close decision."
1. **The router kernel is verified exact given matched-precision inputs.**
   The per-op synthetic fixtures (which bake Q8_0 into the oracle's own
   weights, isolating kernel correctness from quantization error) show
   100% top-8-id match and ~1e-7/1e-8 weight precision. Whatever is
   happening is not a bug in `routerTopK`'s selection or tie-break logic.
2. **CPU and Metal agree with each other far more tightly than either
   agrees with the oracle** (PR #29's own finding, unchanged) — rules out a
   backend/kernel-dispatch bug.
3. **The divergence's depth-profile is exactly the expected signature of
   weight-quantization error propagating through a residual network**: it
   appears the instant the first Q8_0-quantized weight matrix is used
   (layer 0→1), stays roughly proportional to the residual stream's own
   (rapidly-growing) magnitude through the middle layers, and is not tied to
   any particular op, shape, or code path that would suggest a logic error.
4. **The specific swap margins range from razor-thin (2e-6) to modest
   (5-6e-3, ~0.6 probability points)** — never large. Nothing resembles "the
   oracle's top pick and the real top pick disagree by tens of percentage
   points," which is the signature a genuine bug (wrong dtype read, transposed
   weight, off-by-one indexing) would produce.
5. Greedy decoding is fundamentally sensitive to compounding: once ANY
   single token in a 64-token autoregressive chain flips, everything after
   it is free to diverge completely (as p3/p4 show) — this is expected
   behavior of the *decoding strategy* under any nonzero source of numerical
   difference between two implementations of "the same" model at different
   effective precision, not evidence the underlying model is unstable in a
   way that would matter for non-greedy use.

**What would be worth escalating, if anything**: not a bug fix, but a
possible ADR-005-level discussion (explicitly out of scope for this
investigation to decide) about whether router-parity strictness / logit
tolerance for deep 128-expert MoE models at this scale should be
relaxed to acknowledge that an INT8-quantized production pipeline
diverging from a full-precision oracle by an occasional top-k boundary flip,
at a magnitude proportional to normal quantization error, is expected
rather than a gate failure — as opposed to continuing to chase kernel-level
explanations that this investigation did not find evidence for. A possible
(not implemented, not evaluated here) mitigation if tighter oracle-parity
is actually a hard requirement: keeping a small number of the deepest
layers' expert banks (or just the router-adjacent ones) at a higher-precision
quant (f16 or f32) instead of Q8_0, to reduce (not eliminate) the late-layer
acceleration observed in §2.

---

## 7. What did NOT happen (forbidden-list compliance)

- No tolerance was loosened, no router/kernel logic was changed.
  `src/shared/contracts.zig`, `src/kernels/cpu/kernels_{a,b,c}.zig`, and all
  `PORTING-*.md` files were not touched.
- `src/engine/forward.zig`'s temporary instrumentation was reverted; `git
  status`/`git diff` confirm a clean state relative to `t06-real-30b` aside
  from this file and `tools/make_fixtures.py`.
- `zig build test` (74/74) and `zig build test-gpu` (81/81) stayed green.
- T07 / `docs/orchestration/HANDOFF.md`'s task-DAG status was not touched —
  that decision remains with the project owner.
- This investigation stayed on Node B (`max-1`) only.
