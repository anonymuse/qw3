# T03 — M1 viability: decode-sim, placement simulator, f001 draft (W6)

**Model:** Haiku (Sonnet if the analysis section flails). **Branch:**
`w6-m1-viability` off `main` (kernel contracts not needed). A prior agent left
uncommitted WIP (`src/sim/decodesim.zig`, `src/main.zig` edits) in worktree
`.claude/worktrees/agent-a07b894d896d995d4` — salvage it if still present
(`git -C <path> diff` / copy files), else rewrite. It had also fetched real
235B safetensors metadata: 235.09B params, 36,945 tensors, 118 shards.

## Read first

`docs/specs/DS5_Execution_Plan_v0.3.md` (M1 row), ADR-001 §2 (model shape:
94 layers, 128 experts, top-8, GQA 64/4),
`docs/specs/imported_v0.2/DS5_Model_Runtime_Placement_Spec_v0.2_...md` §§5,6,8
(budgets, quant matrix, packet rule),
`docs/specs/imported_v0.2/DS5_Benchmark_and_Acceptance_Spec_v0.2_...md` §5
(run-metadata JSON schema — every binary emits it),
`src/transport/linkbench.zig` + `src/shared/jsonbuf.zig` (existing JSON/bench
patterns), `manifests/cluster/loopback.zon`.

## Deliverables

1. **`ds5 bench decode-sim`** (Zig, `src/sim/decodesim.zig`, wired into
   `src/main.zig`): simulates per-token decode traffic for a 94-layer MoE
   split across nodes. Per layer, per token: one activation packet per
   destination node (80-byte header from `src/shared/activation_packet.zig` +
   hidden-vector payload, 4096×f16 default) sent over the M0 transport
   (`src/transport/tcp.zig`, loopback by default). Parameters (flags):
   `--layers`, `--hidden`, `--miss-rate P` (probability a token needs a
   remote expert fetch, adding a configurable stall), `--routing-json PATH`
   (optional per-layer expert-usage distribution; uniform if absent),
   `--tokens N`, `--cluster PATH`. Output: run-metadata JSON to
   `bench/results/decode-sim-<epoch>.json` with per-token p50/p95/p99 latency
   decomposition (serialization, RTT, stall) and projected tok/s.
   **Loopback numbers must be labeled `"link_source": "loopback-placeholder"`
   in the JSON and the finding** — real mesh JSONs replace them when the
   owner runs the runbook.
2. **Placement simulator** (`tools/placement_sim.py`, stdlib + optional
   `huggingface_hub` for metadata): reads Qwen3-235B safetensors metadata
   (name→shape/dtype only; cache the index JSON into
   `tools/cache/qwen3-235b-index.json` and commit it), applies the quant
   matrix from Placement Spec §6 and a bytes-per-weight table, and answers:
   does the model fit 3 nodes at ≤33.6GB static each, per quant mix? Which
   mixes close the budget? Emits a table (markdown) + JSON. No downloads of
   weights — metadata only.
3. **`docs/findings/f001-viability.md` draft**: tok/s ceiling decomposition
   (UMA-bandwidth bound per token from A-01 estimates + measured link numbers
   when present + decode-sim outputs + placement result), go/no-go framing vs
   the >12 tok/s target, explicit "placeholder" markers on every unmeasured
   number, and a "replace with measured" checklist tied to the three hardware
   artifacts (mesh JSONs, telemetry JSON, 30B artifacts).

## Definition of done

`zig build test` green; `ds5 bench decode-sim --tokens 64 --quick`-style smoke
run works on loopback and writes valid JSON; placement_sim.py runs offline
from the committed cache; f001 draft complete with placeholders. Report: files,
commands, sample JSON snippet, and the placement verdict table.

## Forbidden

Downloading model weights; claiming any placeholder number as measured;
touching kernels/contracts. Scope-cut: drop `--routing-json` support (keep
uniform) if blocked — note it in f001.
