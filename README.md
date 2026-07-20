# DS5 — a model-specific distributed inference engine for Qwen3-235B-A22B

DS5 runs exactly one model — `Qwen3-235B-A22B-Instruct-2507` (sparse MoE, 235B
total / 22B active, 94 layers, 128 experts, top-8 routing) — on exactly one
topology: three Apple Silicon Macs on a Thunderbolt 5 mesh.

- **Node A** (M5 Pro, 48GB): control plane — scheduler, tokenizer, sampler,
  KV page table, expert placement policy, cold-expert inventory.
- **Nodes B/C** (M5 Max, 48GB): decode workers — layers 0–46 / 47–93,
  attention, KV pages, resident experts, local router mirrors.

This is deliberately **not** a general inference framework. The thesis: a
narrow runtime specialized to one MoE model, one hardware topology, and one
workload can beat a general runtime on the same cluster — and measuring
exactly where that holds (or breaks) is the publishable result, even if the
answer is negative.

The runtime is from-scratch Zig + Metal with libc as the only system
dependency. Reference implementations (llama.cpp, HF transformers) are used
strictly as *offline* oracles for golden-fixture tests — never linked in.

## Status: M2 single-node bring-up; M3 distributed work is blocked

Working today:

```sh
zig build && zig build test                              # CPU/unit suite
zig build test-metal && zig build test-gpu              # Apple GPU suites
./zig-out/bin/ds5 node --name a                                  # per-node daemon
./zig-out/bin/ds5 bench link --cluster manifests/cluster/lab.zon --self a
./zig-out/bin/ds5 health --host 10.5.0.1
./zig-out/bin/ds5 run --model model.gguf --prompt-tokens "1,2,3" \
  --backend metal --context-capacity 32 --kv-dtype f16
```

`ds5 bench link` measures per-pair RTT (by message size), bandwidth (by block
size), and sustained throughput, and writes machine-readable JSON with full
run metadata (git commit, versions, node health) to `bench/results/`.
See [docs/runbook.md](docs/runbook.md) for the 3-node procedure.

Evidence boundary as of 2026-07-16:

- On this branch's 2026-07-16 local M5 validation, the synthetic CPU and Metal
  fixture suites pass. The real Qwen3-30B-A3B Q8_0 CPU/Metal result is recorded
  in the still-open evidence pull requests below, not re-established by those
  synthetic suites.
- The real-weight gate is a **partial pass**, not a correctness clearance;
  [PR #29](https://github.com/anonymuse/qw3/pull/29) contains the gate and
  [PR #31](https://github.com/anonymuse/qw3/pull/31) investigates the remaining
  router difference and quantization-sensitive margins. Both are still open
  and have evidence/provenance review findings to resolve.
- The only committed link result is `loopback-smoke`. No real three-node
  Thunderbolt result has been committed, and no distributed forward path is
  implemented. LAN reachability is not Thunderbolt evidence.
- Apple now exposes RDMA over Thunderbolt 5 on supported macOS/hardware. The
  repository includes a read-only preflight, but no QW3 RDMA benchmark or
  transport implementation yet.

The current outside-in review, model/runtime landscape, and delivery plan are
in [the 2026-07-16 frontier local-inference strategy](docs/strategy/2026-07-16-frontier-local-inference-review.md).

## Layout

| Path | Purpose |
|---|---|
| `docs/decisions/` | ADRs. Start with ADR-001 (model), ADR-002 (kernel strategy) |
| `docs/specs/` | Active specs; `imported_v0.2/` is the planning baseline, v0.3 execution plan is current |
| `docs/assumptions.md` | Every unmeasured number the project currently leans on |
| `docs/findings/` | Measured results write-ups — the publishable artifact |
| `manifests/` | Deterministic configs: cluster, model, quant, placement |
| `src/shared/` | Wire protocol, activation packet, manifests, checksums, libc layer |
| `src/transport/` | TCP-over-Thunderbolt transport and `bench link` |
| `src/nodectl/` | Node daemon (discovery, health) |
| `src/sim/` | M1 simulator scaffold; viability implementation remains deferred |
| `src/orchestrator/` | Node A runtime scaffold (M3+) |
| `src/worker/` | Node B/C runtime scaffold (M3+) |
| `src/kernels/` | CPU reference and Metal kernels for M2 bring-up |
| `tools/` | Python/shell tooling: model downloads, oracle traces, telemetry |
| `bench/` | Benchmark harness, corpus, committed results |

## Milestones

M0 mesh reality → M1 viability model (`docs/findings/f001`) → M2 single-node
30B-A3B correctness → M3 distributed correctness → M4 235B runtime → M5
findings. Details: [docs/specs/DS5_Execution_Plan_v0.3.md](docs/specs/DS5_Execution_Plan_v0.3.md).

Ground rules (ADR-001): never alter top-8 routing semantics; NVMe never in the
steady-state decode path; router/gates stay FP16/Q8; no unbenchmarked claims.
