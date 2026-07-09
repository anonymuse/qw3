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

## Status: M0 (mesh reality)

Working today:

```sh
zig build && zig build test        # Zig 0.16.0
./zig-out/bin/ds5 node --name a                                  # per-node daemon
./zig-out/bin/ds5 bench link --cluster manifests/cluster/lab.zon --self a
./zig-out/bin/ds5 health --host 10.5.0.1
```

`ds5 bench link` measures per-pair RTT (by message size), bandwidth (by block
size), and sustained throughput, and writes machine-readable JSON with full
run metadata (git commit, versions, node health) to `bench/results/`.
See [docs/runbook.md](docs/runbook.md) for the 3-node procedure.

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
| `src/sim/` | Placement simulator + decode traffic replay (M1) |
| `src/orchestrator/` | Node A runtime (M3+) |
| `src/worker/` | Node B/C runtime (M2+) |
| `src/kernels/` | Metal shaders (M2+; empty until placement proves feasibility) |
| `tools/` | Python/shell tooling: model downloads, oracle traces, telemetry |
| `bench/` | Benchmark harness, corpus, committed results |

## Milestones

M0 mesh reality → M1 viability model (`docs/findings/f001`) → M2 single-node
30B-A3B correctness → M3 distributed correctness → M4 235B runtime → M5
findings. Details: [docs/specs/DS5_Execution_Plan_v0.3.md](docs/specs/DS5_Execution_Plan_v0.3.md).

Ground rules (ADR-001): never alter top-8 routing semantics; NVMe never in the
steady-state decode path; router/gates stay FP16/Q8; no unbenchmarked claims.
