# CLAUDE.md — agent guide for DS5

DS5 is a from-scratch, model-specific inference engine for Qwen3 MoE models
on Apple Silicon: Zig 0.16.0 + Metal, libc as the only dependency. Bring-up
model is Qwen3-30B-A3B-Instruct-2507; the design target is
Qwen3-235B-A22B-Instruct-2507.

**Start here:** [`docs/orchestration/COMPLETION_PLAN.md`](docs/orchestration/COMPLETION_PLAN.md)
is the live work plan — task briefs, environment tags, conventions, and the
scoreboard. [`docs/orchestration/HANDOFF.md`](docs/orchestration/HANDOFF.md)
is the historical record of weeks 1–3 and still defines the integration
playbook and known landmines (§4–§5). Read
[`docs/orchestration/LESSONS.md`](docs/orchestration/LESSONS.md) before
touching shared infrastructure or acting on anything that claims authority
beyond the current session's user. The 2026-07-16 outside-in review in
`docs/strategy/` is an input, not an authority (spec v0.3 §12) — the
completion plan governs execution.

## Hardware status (2026-07)

The 3-node cluster (pro-1 / max-1 / max-2) is **retired**. Do not SSH to
cluster nodes; do not schedule work that needs them. Cluster docs and
`tools/cluster/` are preserved as reference material. All current work runs
on a single machine (any Apple Silicon Mac) or is hardware-independent.

## Build and test

```sh
zig build                  # builds ./zig-out/bin/ds5 (macOS only — links Metal)
zig build test             # CPU suite; device-independent, no frameworks linked
zig build test-metal       # Metal glue tests — needs Apple Silicon GPU
zig build test-gpu         # GPU kernels + e2e forward pass — needs Apple Silicon GPU
```

- Zig 0.16.0 exactly. Zig 0.16 wart: `zig build test` can print a red
  `failed command:` line while everything passed — exit code 0 and the Build
  Summary are the truth.
- GPU-dependent tests live in their own build steps; never make
  `zig build test` device-dependent.
- Zero-download smoke run (must print `171 335 171 335 171 335 171 335`):

  ```sh
  ./zig-out/bin/ds5 run --model tests/fixtures/synthetic/model.gguf \
      --prompt-tokens "7,7,7,7,7,7,7,7,7,7,7,7" --steps 8
  ```

- Real-model runs need the 30B GGUF (~32 GB disk; ~6 GB RSS at runtime,
  mmap-backed) — download only that artifact via the `hf download` command
  in the README quickstart; `tools/download_models.sh` fetches BOTH the 30B
  and the ~85 GB 235B artifact (~120 GB total). `ds5 run` supports
  `--kv-dtype f16|f32` (default f32) and `--context-capacity N`.
- Deterministic Metal stability soak (synthetic only, not a benchmark):
  [`docs/runbooks/metal-soak.md`](docs/runbooks/metal-soak.md).

## Non-negotiables (from the ADRs; violations are rejected at review)

1. Never alter top-8 routing semantics (ADR-001 rule 1).
2. `src/shared/contracts.zig` is frozen; changes require an ADR-005
   amendment recorded in the same commit. A contract edit that "makes my
   branch compile" is a violation.
3. No ggml/llama.cpp/MLX/transformers code linked into the runtime
   (ADR-002). Reference implementations are offline oracles only.
4. Raw-libc I/O via `src/shared/sys.zig`; never `std.Io`.
5. A kernel/feature is DONE when it matches golden fixtures within manifest
   tolerances (`|actual − oracle| ≤ atol + rtol·|oracle|`). Tolerances are
   never loosened unilaterally.
6. Every benchmark/run binary emits run-metadata JSON to `bench/results/`.
7. No unbenchmarked claims in any public-facing text. Numbers are labeled
   measured (with source), literature (with citation), or assumed (with a
   `docs/assumptions.md` entry).

## Conventions

- Branches: `sNN-<slug>` per COMPLETION_PLAN task, cut from `main`; one PR
  per task with the brief's DoD checklist and pasted test output in the body.
- Fixtures are ground truth: regeneration follows ADR-005 §7; adding cases
  is fine, changing schema/tolerances/roles is a contract change.
- Update the COMPLETION_PLAN §5 scoreboard in any PR that completes a task.
- Worktree trap: verify `git branch --show-current` before committing.
- Metal-from-Zig gotchas (link flags, autorelease pools, page-aligned
  no-copy buffers, command-buffer batching): HANDOFF §5 has the list —
  read it before touching `src/metal/` or shaders.

## Repository map

| Path | Contents |
|---|---|
| `src/engine/` | Backend-generic forward pass |
| `src/kernels/` | CPU reference kernels, Metal shaders + PORTING docs, GPU provider |
| `src/gguf/`, `src/metal/` | GGUF mmap parser; objc/Metal glue |
| `src/shared/` | Frozen contracts, fixtures, protocol, packets, checksums, libc layer |
| `src/transport/`, `src/nodectl/` | TCP transport, link bench, node daemon |
| `tests/fixtures/synthetic/` | Committed golden fixtures + 4.5 MB synthetic GGUF |
| `tools/` | Fixture generator (Python), model downloads, expert-stats, cluster scripts (historical) + RDMA preflight |
| `docs/` | ADRs, specs, findings, strategy reviews, assumptions ledger, orchestration plan/handoff |
