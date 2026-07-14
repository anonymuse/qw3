# T04 — M2a: CPU end-to-end forward pass on the synthetic model

**Model:** Sonnet. **Branch:** `t04-cpu-forward` off `integration` (requires
T01 GGUF parser and T02 attention merged; check `git log --oneline integration`
first and stop if either is missing — tell the orchestrator).

## Goal

A single-process, CPU-only, f32 forward pass producing logits that match the
oracle end-to-end fixtures for the synthetic 4-layer model. This de-risks all
model-wiring logic before any GPU or real-weight work.

## Read first

`docs/decisions/ADR-005-interface-freeze.md` §6 — the forward-pass recipe is
already written down step by step (embeddings → per layer: rmsnorm → qkv →
per-head q/k rmsnorm → rope → kvAppend → gqaAttention → o_proj → residual →
rmsnorm → router → expertMlpSwiglu → residual; final norm → lm_head). Also:
`src/shared/contracts.zig` (everything), `src/kernels/cpu/` (all providers +
`ctx.zig`), `src/shared/fixture.zig`, `tests/fixtures/synthetic/manifest.json`
(`prompts` array = 5 deterministic prompts with token ids and reference
logits/greedy tokens; per-layer tensors exist for tracing).

## Deliverables

- `src/engine/forward.zig`: `Engine` struct generic over (Ctx, kernels
  provider namespace) — comptime parameter, NOT runtime vtable — that owns
  per-layer KV caches and scratch buffers and implements the ADR-005 §6
  recipe. Weight source: the GGUF `Model` (T01) OR a fixture-backed weight
  table for the synthetic model — define a thin `Weights` interface with a
  fixture-backed impl now; GGUF-backed impl is trivial once the synthetic
  GGUF exists (T01 test emits one — reuse it if present).
- `ds5 run --model PATH --prompt-tokens "1,2,3" --steps N --greedy` CLI
  subcommand printing per-step argmax token ids (engine smoke path).
- `src/test_forward.zig`: for each of the 5 fixture prompts: run the full
  forward, compare final logits to `pN_*.logits.ds5t` within manifest
  tolerance AND greedy argmax tokens exactly. Plus a **trace mode** test
  hook: compare the residual stream after every layer to the per-layer
  fixture tensors, reporting the FIRST layer/op where divergence exceeds
  tolerance (this hook is the M2c debugging tool — build it now).

## Definition of done

All 5 prompts: logits within tolerance, greedy tokens exact. `zig build test`
green including your test root. Report per-prompt max diffs.

## Debugging protocol (when logits diverge)

Use the trace hook to find the first divergent layer/op; then diff that op's
isolated fixture case (they all passed standalone, so the bug is in wiring:
buffer reuse, aliasing, position bookkeeping, or layout). Follow
`docs/orchestration/prompts/DEBUG-divergence.md`. Do not touch kernel
internals — they are fixture-proven; if you believe a kernel is wrong anyway,
stop and report to the orchestrator with the trace evidence.

## Forbidden

Editing contracts, kernels, or fixtures; GPU code (that is T05); tokenizers
(prompts are pre-tokenized ids in the manifest — no tokenizer exists yet and
none is needed for M2).
