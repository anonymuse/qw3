# DS5 Benchmark and Acceptance Spec v0.2: Qwen3-235B-A22B

**Document type:** Benchmark, verification, and acceptance specification  
**Status:** Planning baseline update  
**Date:** 2026-07-08  
**Target model:** Qwen3-235B-A22B-Instruct-2507

---

## 1. Purpose

This document defines the benchmarks required to convert the Qwen3-235B-A22B architecture decision into measurable execution gates.

No throughput, quality, or context-length claim is accepted until it is reproduced by this benchmark harness and recorded with model version, quant manifest, placement manifest, hardware measurements, and git commit.

---

## 2. Benchmark classes

| Class | Purpose | Required by phase |
|---|---|---|
| Hardware benchmark | Measure actual mesh, storage, memory, Metal overhead | Phase 0 |
| Dense baseline benchmark | Validate distributed execution independent of MoE complexity | Phase 1 |
| MoE correctness benchmark | Validate top-k routing and expert execution | Phase 2 |
| Qwen placement benchmark | Validate memory maps and quantization budget | Phase 3 |
| Qwen runtime benchmark | Measure final target throughput/latency | Phase 4 |
| Harness reliability benchmark | Measure tool-call and long-horizon reliability | Phase 5 |
| Regression benchmark | Prevent correctness/performance regressions | All phases after Phase 1 |

---

## 3. Phase gates

### Phase 0: Hardware validation

| Metric | Acceptance |
|---|---|
| Thunderbolt link throughput | Measured and recorded per link |
| Link latency | Measured and recorded per message size |
| NVMe sequential throughput | Measured with target block sizes |
| macOS cache-control behavior | Measured with selected file I/O path |
| Metal command overhead | Measured for small and fused kernels |
| UMA pressure behavior | Launch/load pressure profile recorded |

Exit criterion: all assumptions in the architecture doc are replaced by measured values or marked as unresolved risk.

### Phase 1: Dense distributed baseline

| Metric | Acceptance |
|---|---|
| Token correctness | Deterministic prompts match reference within tolerance |
| KV correctness | KV replay works under fixed seed |
| Transport correctness | Activation packets are checksummed and traceable |
| p95 decode latency | Recorded, not necessarily final target |
| Memory cap | Static load stays below configured cap |

Exit criterion: distributed engine is stable enough to isolate MoE-specific failures in later phases.

### Phase 2: Small MoE correctness

| Metric | Acceptance |
|---|---|
| Router top-k parity | 100% expert-ID parity against reference for test corpus |
| Gate-weight parity | Within numeric tolerance |
| Fused expert output | Within numeric tolerance |
| Placement map correctness | No layer/expert ownership mismatches |
| Local/remote packet rule | No per-expert packets |

Exit criterion: MoE implementation is correct before scaling to Qwen3-235B-A22B.

### Phase 3: Qwen3 placement prototype

| Metric | Acceptance |
|---|---|
| Manifest parse | Model dimensions and tensor shapes loaded |
| Per-node static memory | ≤33.6GB unless explicit test override |
| Quantization manifest | All tensor classes assigned explicit quant |
| Expert tier manifest | Hot/warm/cool/cold tiers generated |
| Cold inventory | Node A/NVMe cold backing works without decode dependency |

Exit criterion: Qwen3-235B-A22B can be represented by manifests and placed within the planning budget.

### Phase 4: Qwen3 runtime

| Metric | Target |
|---|---:|
| Throughput at 8K context | >12 tok/s target |
| Throughput at 32K context | >12 tok/s target after optimization |
| p95 decode latency | Recorded and tracked |
| p99 decode latency | Recorded and tracked |
| Blocking NVMe reads in steady-state decode | 0 |
| Cold-miss rate | Below configured stall threshold |
| Router semantic deviations | 0 intentional deviations |
| Tool-call JSON validity | ≥99% on constrained benchmark, target to refine after baseline |

Exit criterion: final model path is benchmarkable and stable enough to integrate with the harness.

### Phase 5: Agentic harness reliability

| Metric | Acceptance |
|---|---|
| Structured tool-call validity | Measured under constrained decoding |
| Multi-turn state replay | Deterministic under fixed run state |
| Error recovery | Defined recovery paths pass test corpus |
| Trace capture | Full replay artifact produced |
| Comparative runs | Dense fallback / smaller MoE / Qwen final compared |

Exit criterion: project can produce a publishable finding about local-agent reliability and the effect of model/runtime specialization.

---

## 4. Required benchmark corpus

| Corpus slice | Purpose |
|---|---|
| Short deterministic prompts | Token parity and decode correctness |
| Router calibration prompts | Expert hit-rate and gate-mass telemetry |
| Long-context retrieval | KV stability and attention quality |
| Summarization | Long sequence behavior and latency |
| Code-editing | Tool-call precision and local-agent workload |
| JSON/function-call generation | Constrained decoding reliability |
| Error recovery cases | Harness robustness |
| Cold-start prompts | Initial placement behavior |
| Adversarial routing prompts | Cold-miss stress |

---

## 5. Run metadata schema

Every benchmark run must record:

```yaml
run_id: string
date: string
git_commit: string
model:
  name: Qwen3-235B-A22B-Instruct-2507
  source_revision: string
  artifact_hashes: []
quant_manifest: string
placement_manifest: string
hardware:
  node_a: {chip: M5 Pro, memory_gb: 48, measured_bandwidth_gbps: null}
  node_b: {chip: M5 Max, memory_gb: 48, measured_bandwidth_gbps: null}
  node_c: {chip: M5 Max, memory_gb: 48, measured_bandwidth_gbps: null}
context_tokens: integer
prompt_class: string
decode_settings:
  temperature: float
  top_p: float
  seed: integer
metrics:
  tok_per_sec: float
  p50_latency_ms: float
  p95_latency_ms: float
  p99_latency_ms: float
  cold_miss_rate: float
  blocking_nvme_reads: integer
  max_node_memory_gb: float
```

---

## 6. Release criteria

A DS5 release candidate must include:

- reproducible benchmark results;
- model and quantization manifests;
- placement manifest;
- risk register update;
- known limitations;
- exact hardware measurement table;
- runbook for launching and reproducing results;
- archived traces for at least one passing run per benchmark class.

---

## 7. Rejection criteria

A build is rejected if any of the following occur without an explicit waiver:

- it changes Qwen top-k routing semantics;
- it requires NVMe for steady-state active-weight decode;
- it exceeds 33.6GB static memory on any node;
- it produces untraceable nondeterminism in correctness mode;
- it cannot reproduce benchmark runs under fixed seed;
- it silently falls back to another model;
- it claims context or throughput targets that were not benchmarked.
