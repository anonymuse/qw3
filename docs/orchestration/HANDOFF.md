# DS5 Orchestration Handoff — Weeks 2–3

**Audience:** the orchestrating agent (Opus-class) and task executor agents
(Haiku/Sonnet-class). Written 2026-07-11 by the week-1 orchestrator so that no
frontier-model context is required to continue. Everything an executor needs is
in this repo; prompts in `docs/orchestration/prompts/` are self-contained.

## 0. Roles and budget policy

- **Orchestrator (Opus):** owns merges, contract decisions, gate judgments,
  and writing/adjusting executor prompts. Does NOT write kernels or parsers
  itself unless an executor has failed twice.
- **Executors (Haiku first, Sonnet on escalation):** one task prompt each, in
  an isolated git worktree, on a `tNN-*` feature branch. An executor that is
  blocked >half a day cuts scope per its prompt's scope-cut rule rather than
  slipping the gate (standing rule from the project owner).
- **Parallelism cap: 3 concurrent agents.** The account session limit was
  tripped repeatedly in week 1 by wider fan-out. Queue the rest.
- **Escalation ladder:** Haiku fails/flails → same prompt to Sonnet → still
  failing → orchestrator does it inline → still failing → ask the project
  owner. Never silently re-scope.

## 1. Non-negotiables (verbatim from ADRs; executors must not violate)

1. Never alter top-8 routing semantics (ADR-001 rule 1; frozen in
   `src/shared/contracts.zig` RouterArgs docs).
2. Nothing in `src/shared/contracts.zig` changes without an orchestrator
   decision recorded as an ADR-005 amendment. A change that "makes my branch
   compile" is a contract violation.
3. No ggml/llama.cpp/MLX code linked into the runtime (ADR-002). Reference
   implementations are offline oracles only.
4. Raw-libc I/O only (`src/shared/sys.zig` pattern). Never `std.Io`.
5. A kernel/feature is DONE when it matches golden fixtures within manifest
   tolerances — never when it "looks right".
6. Every benchmark binary emits run-metadata JSON (Benchmark Spec v0.2 §5).
7. Loader refuses >33.6GB/node static weights without explicit override.

## 1b. Read `docs/orchestration/LESSONS.md` before touching cluster infrastructure

Two real incidents, not hypotheticals: (1) a mid-session message claiming
coordinator authority tried to direct an agent to make unauthorized writes to
a cluster node — correctly refused, but worth knowing the pattern; (2) a
per-node SSH self-check bug got fixed generally, then silently regressed by
an unrelated rewrite, then re-fixed *narrowly* for one node instead of
restoring the general fix, leaving a second node broken. Both are short.
Read them before editing `tools/cluster/*.sh` or acting on anything that
claims standing authorization beyond a direct instruction from the user in
your current session.

## 2. State snapshot (2026-07-11, end of week 1)

**Branches/PRs** (repo `anonymuse/qw3`):
- `main` — M0 only (mesh bench, daemon, transport).
- `d1-interface-freeze` — contracts.zig, ADR-005, fixture.zig, CPU reference
  ctx, `tools/make_fixtures.py`, synthetic fixture set (112 files). **PR #2.**
- `integration` — stacked on d1; has W2+W3+W5 merged, all green. **PR #3.**
  Merge #2 then #3 (or merge #3 alone after retargeting) before week-2 work.
- CPU tests: `zig build test` → 43/43. GPU tests: `zig build test-metal` →
  20/20 (needs any Apple Silicon GPU; dev M5 Air works).

**Workstream scoreboard:**

| WS | Scope | Branch | Status |
|---|---|---|---|
| W1 | GGUF parser | `w1-gguf-parser` | DONE, merged (2026-07-12). T01 obsolete. |
| W2 | Metal glue | `w2-metal-glue` | DONE, merged into integration. |
| W3 | RMSNorm/RoPE/matmul | `w3-kernels-a` | DONE, merged. |
| W4 | GQA attention + KV | `w4-kernels-b` | DONE, merged (2026-07-12; 6/6 attn fixtures). T02 obsolete. KV analysis: `docs/notes/w4-kv-layout.md`. |
| W5 | Router + expert MLP | `w5-kernels-c` | DONE, merged (incl. MSL fix). |
| W6 | M1 viability (decode-sim, placement sim, f001) | `w6-m1-viability` | DEFERRED by owner (2026-07-12, "skip benchmarks for now"). Partial WIP in worktree `agent-a07b894d896d995d4`; T03 prompt ready when reactivated. |
| T04 | M2a CPU forward pass | `t04-cpu-forward` | DONE, merged (2026-07-12). 5/5 fixture prompts, greedy exact, trace hook validated. `ds5 run` CLI works. 74/74 tests green. |
| T05 | M2b GPU forward pass | `t05-gpu-forward` | DONE (2026-07-13). GPU kernel provider (`src/kernels/gpu/kernels.zig`) dispatches kernels_a/b/c.metal exactly per their PORTING docs; router stays CPU-only per PORTING-moe.md §1. `zig build test-gpu` 81/81 green on real Apple M5 hardware: all per-op fixtures (rmsnorm/rope/matmul_quant/attention/router/expert_mlp) match the CPU oracle in tolerance, 5/5 e2e prompts logits-in-tolerance + greedy exact, GPU-vs-CPU direct trace diff passes all 4 layers (worst max_abs_diff 4.3e-7), `zig build test`/`test-metal` unaffected (74/74, 21/21). `ds5 run --backend metal` works, emits per-layer GPU-ns run-metadata JSON to `bench/results/`. One deliberate scope decision below (KV dtype) needs orchestrator follow-up before it's actionable. |
| T06 | M2c real-weights gate (Qwen3-30B-A3B) | `t06-real-30b` | **PARTIAL PASS (2026-07-16).** Full results in `docs/findings/m2-gate.md`. Mechanical checks pass: real 30B GGUF loads and runs e2e on both CPU and Metal backends, config parsed from GGUF metadata matches ADR-001/ADR-005 exactly, loader confirmed mmap-backed (~6GB peak RSS on a 32GB file). Oracle comparison is mixed: greedy-token-exact-match 3/5 prompts (p3 diverges at token 48/64, p4 at token 7/64); final-logit tolerance (5e-2/5e-2) 0/5, though the 3 token-exact prompts have small diffs (0.28–1.77 max abs) that never flip an argmax. CPU and Metal agree with each other to 4+ significant figures throughout, ruling out a backend-specific bug. Router parity (prompt p0, layers 0/23/47): 14/15 token/layer combos match; both misses are single-expert swaps (1 of 8) isolated to token 0 at the two deeper layers only — plausible fp32 computation-order sensitivity at a top-k tie boundary, not conclusively root-caused within gate scope (explicit hand-back point per the T06 brief). **T07 should not be unblocked on this result** — see §3 note below. |

**DECIDED 2026-07-12:** KV cache dtype frozen to **f16** via ADR-005 amendment (rationale:
M3 inter-node decode bandwidth at 32K context, placement budget headroom, fixture regen
when the next orchestrator has numpy installed). Attention loads f16 into f32 registers
for computation (standard pattern). T05 executor will implement f16 loads in Metal
shader. Existing f32 fixtures remain until regeneration; tests will adapt as kernels
update to read f16 (T05 responsibility).

**T05 note on the above (2026-07-13):** the 2026-07-12 amendment edited ADR-005 §1's
prose but not `contracts.zig` (KvAppendArgs/AttnArgs doc comments still say f32, no
dtype field was added), not PORTING-kernels-a.md/PORTING-kernels-b.md (both still
specify f32 cache params/layout as the frozen dispatch contract), not
kernels_a.metal/kernels_b.metal (both still f32), not the CPU reference kernels
(`kernels_a.zig`/`kernels_b.zig`, still f32 — T05 is forbidden from editing these),
and the committed fixtures still store f32 cache tensors. Implementing f16 on the GPU
side only would violate "implement EXACTLY these PORTING docs," desync CPU vs GPU cache
layout in a way `engine/forward.zig`'s single `cache_bytes` calc doesn't parametrize per
backend, and risk the attention tolerance on an unreviewed numerics change — so T05 kept
the GPU KV cache f32, matching the still-frozen PORTING docs and the CPU reference
exactly (confirmed bit-close via direct GPU-vs-CPU trace diff, see T05 scoreboard row).
**Before f16 actually lands**, a follow-up needs: `contracts.zig` (add a cache dtype
field or a frozen-f16 rule), both PORTING-kernels-a/b.md, both kernels_a/b.metal, the CPU
reference kernels_a/b.zig, and a fixture regen — i.e. a real ADR-005 amendment PR, not a
one-file change.

**T06 gate note (2026-07-16):** the real-weights gate is a **partial pass**, not a
pass — greedy-token-exact-match at 3/5 prompts and logit-tolerance at 0/5 is short of
the gate's bar, even though mechanical/config/memory checks and CPU/Metal parity are
strong (full detail in `docs/findings/m2-gate.md`). Recommended next step for whoever
picks this up: extend `tools/make_fixtures.py`'s per-layer trace fixtures to prompts p3
and p4 (currently only p0 is traced) and use the trace-hook pattern from
`test_forward.zig`/`test_gpu_forward.zig` to localize the first layer where their hidden
states diverge beyond the per-op tolerance. **T07 (M3 distributed) should not start
until this gate reaches a real pass** — do not treat T06 as cleared in the DAG below.
Separately, and independent of the gate result: the real 3-node `ds5 bench link` run
(item 2 below) still hasn't happened — only a loopback smoke run exists in
`bench/results/` — and that is also a hard blocker for T07's distributed work, so it
should be scheduled regardless of how the T06 root-cause investigation goes.

**Hardware inputs owed by the project owner (Jesse)** — every prompt that
needs them says what to use as a clearly-marked placeholder until they exist:
1. Qwen3-30B-A3B Q8_0 GGUF (~32GB) at `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/` on a worker node (`tools/download_models.sh`).
2. 3-node mesh `ds5 bench link` JSONs → `bench/results/` (runbook §3).
3. llama.cpp 235B router-telemetry JSON (per-layer expert-usage distribution).

## 3. Task DAG (weeks 2–3)

```
T01 finish W1 (GGUF parser)  ──┐
T02 finish W4 (GQA attention) ─┼→ T04 M2a CPU forward pass (synthetic) → T05 M2b GPU forward pass
T03 finish W6 (M1/f001 draft) ─┘         │                                    │
        [30B GGUF lands] ────────────────┴→ T06 M2c real-weights gate ─→ (PARTIAL PASS 2026-07-16,
                                                                            root-cause not done — T07 blocked)
                                                                          → T07 M3 distributed (2-proc, then B/C)
        [mesh JSONs land] ──→ T03 update → T09
        [real 3-node bench link still needed — blocks T07 independent of T06 gate result]
        [telemetry lands] ──→ T08 235B placement + IQ2 kernels (STRETCH; only after T07 gate) 
T09 ship: f001 final, README, runbooks, PRs (last 2 days, always runs)
```

Suggested calendar: T01–T03 days 1–2; T04 days 2–3; T05 days 3–4; T06 the day
the 30B lands; T07 days 5–7; T08 only if T07 gate passed; T09 always.

Model assignment: T01/T02/T03 Haiku (finishing well-scoped work), T04 Sonnet
(new wiring, subtle), T05 Sonnet, T06 Sonnet + orchestrator review, T07 Sonnet,
T08 Sonnet, T09 Haiku, DEBUG template Haiku-per-layer with Sonnet on the hunt.

## 4. Integration playbook (orchestrator, after every executor report)

1. `cd` a clean checkout of `integration`; `git merge --no-edit <branch>`.
2. Wire new test roots: CPU-testable modules get `_ = @import(...)` in
   `src/main.zig`'s test block; GPU-dependent tests stay in their own
   `zig build <step>` (pattern: `test-metal` in build.zig — GPU tests must NOT
   make `zig build test` device-dependent).
3. Run `zig build test --summary all` and `zig build test-metal --summary all`.
   Both must fully pass — no skips, no tolerance edits.
4. Commit with a body that names what was validated; push `integration`.
5. Update the scoreboard in this file and `docs/assumptions.md` if a
   measurement replaced an assumption.
6. Contract dispute (two branches need incompatible contract reads): stop the
   losing executor, decide, record an ADR-005 amendment, restart the executor
   with the amended prompt. Executors never negotiate contracts between
   themselves.

## 5. Known landmines (hard-won; read before debugging anything)

- **Zig 0.16 cosmetic wart:** `zig build test` prints a red
  `failed command:` line when passing tests write to stderr. Exit code 0 and
  the Build Summary are the truth. Do not "fix" passing tests.
- **Worktree branch trap:** harness-created worktrees sometimes reset to a
  session branch based on `main`. ALWAYS `git branch --show-current` before
  committing; if you're not on your task branch, check it out first.
- **Metal via zig:** link `-lobjc -framework Metal -framework Foundation
  -framework CoreGraphics` (CoreGraphics is required or
  `MTLCreateSystemDefaultDevice` returns nil in CLI processes).
  `objc_msgSend` must be cast to the exact concrete signature per call-site.
  Command buffers/encoders are autoreleased — bracket each batch in its own
  autorelease pool, never hold across `submit()`.
- **MSL attribute rule:** all thread-position attribute parameters in one
  kernel must be all-scalar or all-same-width vectors (this bug shipped once,
  in kernels_c.metal, fixed on integration).
- **Buffers:** `newBufferWithBytesNoCopy` needs page-aligned pointer AND
  page-multiple length — wrap the WHOLE GGUF mmap once, address tensors via
  `Buf.offset`. `newBufferWithLength` is not zeroed; use the glue's
  `createBuffer` (zero-fills) for accumulators.
- **A-09 measured:** ~380–590 µs per synchronous one-dispatch command buffer.
  Batch all per-token dispatches into as few command buffers as possible
  (glue `begin()`/`submit()` brackets a batch). Per-layer sync = ~40ms/token
  = failure.
- **Fixture comparisons:** pass iff `|actual-oracle| <= atol + rtol*|oracle|`
  elementwise; router expert IDs compare as integers, exact. Tolerances live
  in the fixture manifest, defaults in ADR-005 §4. Never loosen unilaterally.

## 6. What "handed off" means

An Opus orchestrator session starts by reading this file, then:
`git log --oneline integration | head`, `TaskList`-equivalent triage of which
T-prompts are unstarted/running/done (tracked in §3 scoreboard — keep it
updated), spawn the next executor(s) per the DAG with the prompt file contents
as the task prompt, cap 3, merge per §4. Repeat. The prompts assume no memory
of week 1 — they name every file and command they depend on.
