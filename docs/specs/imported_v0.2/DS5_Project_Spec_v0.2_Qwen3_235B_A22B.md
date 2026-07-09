# DS5 Project Specification v0.2: Qwen3-235B-A22B Target

**Document type:** Project specification  
**Status:** Planning baseline update  
**Date:** 2026-07-08  
**Model decision:** Qwen3-235B-A22B is the DS5 project target model  
**Primary target variant:** Qwen3-235B-A22B-Instruct-2507

---

## 1. Project statement

DS5 is a bespoke local distributed inference system for running **Qwen3-235B-A22B** on a three-node Apple Silicon mesh using a narrow, model-specific runtime rather than a general-purpose inference stack.

The project thesis is:

> A narrow runtime specialized to one sparse MoE model, one hardware topology, and one long-horizon local-agent workload can produce a better quality/latency/control tradeoff than a general inference runtime on the same cluster.

---

## 2. Target outcome

The target outcome is a reproducible local DS5 runtime that can:

- load and execute Qwen3-235B-A22B-Instruct-2507;
- distribute decode work across Nodes B and C;
- use Node A as control plane, scheduler, drafter host, tokenizer/logit owner, global KV/page-table owner, and expert-placement controller;
- preserve exact Qwen MoE routing semantics;
- use mixed tensor-aware quantization;
- keep the steady-state decode hot path resident in UMA;
- use NVMe only for promotion, prefetch, long-context backing, and cold storage;
- expose benchmark data sufficient to decide whether the architecture is viable.

---

## 3. Scope

### 3.1 In scope

| Area | Included work |
|---|---|
| Model runtime | Qwen3-235B-A22B load, placement, scheduling, decode, prefill, sampling |
| Distributed execution | A/B/C node roles, transport protocol, activation/result packets, ring buffers |
| MoE mechanics | Router correctness, top-8 routing, expert placement, hot/warm/cool/cold tiers |
| Quantization | Tensor-class-aware quantization manifest and compatibility tests |
| KV management | Layer-owned KV pages, page table, quantized/old-page policy, context benchmarks |
| Metal kernels | Attention, dequantization, fused multi-expert matmul, logits path |
| Agentic harness interface | Streaming tokens, constrained tool-call decoding, trace capture, replay hooks |
| Benchmarking | Performance, correctness, memory, routing, tool-call reliability, reproducibility |
| Operations | Config manifests, launch scripts, telemetry, runbooks, failure capture |

### 3.2 Out of scope for v0.2

| Item | Reason |
|---|---|
| Full DeepSeek-V3/R1 runtime | Exceeds safe resident memory assumptions for DS5 |
| Kimi-K2 runtime | Exceeds safe resident memory assumptions for DS5 |
| Low-latency 128K-262K context promise | Requires later sparse attention/KV compression work |
| Multi-user serving platform | DS5 is initially local, single-user, single-cluster |
| CUDA or non-Apple backend | Project is Apple Silicon / Metal / Zig scoped |
| Arbitrary model plugin system | Narrowness is intentional |

---

## 4. Hardware assumptions

| Node | Role | Class | Memory | Bandwidth assumption | Static-weight cap |
|---|---|---|---:|---:|---:|
| Node A | Control plane | M5 Pro | 48GB UMA | 307GB/s | 33.6GB |
| Node B | Decode worker | M5 Max | 48GB UMA | 614GB/s | 33.6GB |
| Node C | Decode worker | M5 Max | 48GB UMA | 614GB/s | 33.6GB |
| Cluster | Distributed local inference | Full mesh | 144GB raw | Link assumptions to be measured | 100.8GB static cap |

Static cap rule:

```text
48GB * 0.70 = 33.6GB static weights per node
48GB * 0.30 = 14.4GB runtime reserve per node
144GB * 0.70 = 100.8GB cluster static-weight cap
```

Runtime reserve covers KV cache, Metal heaps, staging buffers, ring buffers, page tables, allocator fragmentation, OS overhead, and promotion scratch.

---

## 5. Model assumptions

| Dimension | Planning value |
|---|---:|
| Total parameters | 235B |
| Activated parameters | 22B |
| Layers | 94 |
| Experts | 128 |
| Activated experts | 8 |
| Hidden size | 4096 |
| Attention heads | 64 Q heads |
| KV heads | 4 KV heads |
| Head dimension | 128 |
| Native context | 262,144 tokens |
| First performance context | 8K-32K |
| Stretch context | 64K |
| Research context | 128K-262K |

---

## 6. Delivery phases

| Phase | Name | Primary purpose | Exit gate |
|---|---|---|---|
| Phase 0 | Hardware validation | Measure transport, storage, Metal overhead, memory pressure | Hardware assumptions replaced by measurements |
| Phase 1 | Dense baseline | Bring up distributed pipeline with dense 32B-70B | Correct tokens, stable KV, measured latency |
| Phase 2 | Small MoE correctness | Validate router, expert placement, fused kernels on Qwen3-30B-A3B or equivalent | Exact top-k and placement correctness |
| Phase 3 | Qwen3 placement prototype | Load partial/full Qwen3-235B-A22B manifests and verify memory maps | Static budget and routing telemetry validated |
| Phase 4 | Qwen3 runtime | Execute Qwen3-235B-A22B-Instruct-2507 decode at practical context | >12 tok/s target evaluated at 8K-32K |
| Phase 5 | Harness integration | Tool-call decoding, replay, reliability benchmark | Agentic benchmark produces reproducible findings |

---

## 7. Acceptance criteria

### 7.1 Functional acceptance

- Qwen3-235B-A22B-Instruct-2507 model metadata is parsed correctly.
- 94 layers are mapped to Node B and Node C according to the placement manifest.
- Top-8 routing decisions match a trusted reference within accepted tolerance.
- Token output agrees with a reference implementation under deterministic settings for short test prompts.
- Tool-call JSON can be constrained and replayed.

### 7.2 Performance acceptance

| Metric | Target |
|---|---:|
| Decode throughput at 8K context | >12 tok/s after optimization |
| Decode throughput at 32K context | >12 tok/s target, benchmarked honestly |
| p95 cold-miss stall | Below configured stall threshold |
| Resident hot-path blocking NVMe reads | 0 in steady-state decode |
| Per-node static model memory | ≤33.6GB |
| Router/top-k semantic deviations | 0 intentional deviations |

### 7.3 Quality acceptance

- Hot experts use higher-fidelity quantization than cold tiers.
- Router/gate tensors remain FP16 or Q8 during acceptance testing.
- K/V-sensitive tensors remain Q8/Q6-class unless evals justify lower precision.
- Quantization regression is measured on perplexity-like, tool-call, and long-context probes.

---

## 8. Required project documents

The following documents become required planning artifacts:

1. Architecture decision record for Qwen3-235B-A22B selection.
2. System architecture document.
3. Runtime and placement spec.
4. Quantization manifest spec.
5. KV cache and storage spec.
6. Transport protocol spec.
7. Benchmark and acceptance spec.
8. Risk register.
9. Execution plan backlog.
10. Runbook and observability spec.

---

## 9. Scope correction: JW4/Gemma

The earlier JW4 brief describes a Gemma-focused, two-node, learning-oriented engine. That scope is no longer the main DS5 target after the Qwen3-235B-A22B decision.

JW4 should be handled in one of two ways:

| Option | Meaning | Recommendation |
|---|---|---|
| Archive as precursor | JW4 remains a learning brief, not the project target | Preferred if DS5 is now primary |
| Split into separate track | JW4/Gemma remains a separate learning effort | Acceptable only if resourced independently |

Do not mix JW4/Gemma acceptance criteria with DS5/Qwen3 acceptance criteria in the same execution plan.
