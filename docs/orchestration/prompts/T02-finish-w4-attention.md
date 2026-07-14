# T02 — Finish GQA attention kernels (workstream W4)

**Model:** Haiku. **Branch:** `w4-kernels-b` off `d1-interface-freeze` — check
whether it exists first (`git branch -a`); a prior agent wrote the CPU
reference and was killed before committing the Metal shader ("Now the KV-layout
analysis note"). If the branch or files exist, finish them; else start fresh.

## Read first (mandatory)

1. `src/shared/contracts.zig` — `AttnArgs`, `KvAppendArgs`, KV layout comment
   (f32 `[n_kv_heads, max_ctx, head_dim]`, K/V separate buffers, one pair per
   layer), `assertKernelApi`.
2. `docs/decisions/ADR-005-interface-freeze.md` §§1,4,6.
3. Pattern to copy exactly: `src/kernels/cpu/kernels_a.zig` +
   `src/kernels/shaders/kernels_a.metal` + `PORTING-kernels-a.md` (all merged
   on `integration` — cherry-pick reading from there if not on your base).
4. `docs/orchestration/HANDOFF.md` §5 landmines (MSL attribute rule!).

## Deliverables

- `src/kernels/cpu/kernels_b.zig`: `gqaAttention` over `*CpuCtx`
  (`src/kernels/cpu/ctx.zig`) with the frozen signature. Note `kvAppend`
  already ships in kernels_a.zig — do not duplicate it; your fixture tests may
  call it to fill caches. Semantics: causal, query token t at absolute
  position pos+t attends to cache positions 0..pos+t inclusive; GQA group =
  q_head / (n_q_heads / n_kv_heads); scores·scale → f32 softmax → weighted V
  sum. f32 accumulation throughout.
- `src/kernels/shaders/kernels_b.metal`: `gqa_attention_f32` (source only, NOT
  compiled in your worktree — W2's glue compiles at runtime). One threadgroup
  per (token, q_head) with a threadgroup-shared softmax is the expected v1
  shape; `precise::` math for exp/sqrt.
- `src/kernels/shaders/PORTING-kernels-b.md`: dispatch contract for the glue —
  params struct layout (POD, 4-byte aligned), binding indices, grid geometry,
  glue-side assertions. Follow PORTING-kernels-a.md's format.
- `src/test_kernels_b.zig` standalone test root.

## Validation gate

Fixtures in `tests/fixtures/synthetic/` (loader: `src/shared/fixture.zig`,
manifest `manifest.json`): iterate ALL cases with op `attn` — both
`*_attn_prefill` and `*_attn_decode` variants (inputs: q, k_cache, v_cache +
params incl. pos; output tensor per case). Every case must pass manifest
tolerances. Add unit tests: single-token-vs-hand-computation, causality (a
future K/V slot filled with garbage must not affect output), GQA group mapping
(perturb one KV head, verify only its 16-query-head group moves — synthetic
config is 4Q/2KV so group = 2).

`zig test src/test_kernels_b.zig` all green; `zig build test` untouched and
green. Report fixture pass counts and max diffs per case, exact commands,
anything ambiguous you resolved.

## Scope-cut rule

None available — attention correctness is on the M2 critical path. If truly
stuck after a day, report precisely where numbers diverge (first token, first
head, first position) and stop.

## Forbidden

Editing contracts/fixtures; compiling Metal; touching kernels_a/c files.
