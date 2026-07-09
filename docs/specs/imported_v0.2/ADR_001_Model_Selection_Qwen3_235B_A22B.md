# ADR-001: Select Qwen3-235B-A22B as the DS5 Target Model

**Document type:** Architecture decision record  
**Status:** Accepted  
**Date:** 2026-07-08  
**Decision owner:** DS5 architecture  
**Supersedes:** Dense-70B-final, Mixtral-8x22B-final, DeepSeek-V3/R1-final, Kimi-K2-final, JW4/Gemma-as-main-target assumptions

---

## 1. Decision

DS5 will use **Qwen3-235B-A22B** as the project target model.

The primary runtime target is:

```text
Qwen3-235B-A22B-Instruct-2507
```

The thinking variant is deferred:

```text
Qwen3-235B-A22B-Thinking-2507
```

The thinking variant is not part of the first stable execution path. It may be added after the non-thinking instruct model reaches correctness, placement, quantization, benchmark, and tool-call acceptance criteria.

---

## 2. Decision summary

| Decision field | Value |
|---|---|
| Final model family | Qwen3 |
| Final model | Qwen3-235B-A22B |
| First target variant | Qwen3-235B-A22B-Instruct-2507 |
| Later optional variant | Qwen3-235B-A22B-Thinking-2507 |
| Model class | Sparse Mixture-of-Experts |
| Total parameters | 235B |
| Activated parameters | 22B |
| Layers | 94 |
| Experts | 128 |
| Activated experts | 8 |
| Attention | GQA, 64 Q heads / 4 KV heads |
| Native context | 262,144 tokens |
| Practical first performance context | 8K-32K active context |
| Final cluster assumption | 3-node Apple Silicon DS5 mesh: A = M5 Pro, B/C = M5 Max |

Reference source for model properties: official Hugging Face model cards for `Qwen/Qwen3-235B-A22B-Instruct-2507` and `Qwen/Qwen3-235B-A22B-Thinking-2507`.

---

## 3. Rationale

### 3.1 Why Qwen3-235B-A22B

Qwen3-235B-A22B is the best-fit final model because it balances:

- large total model capacity;
- relatively low activated-parameter footprint;
- GQA-based KV efficiency;
- MoE topology suitable for expert placement and promotion;
- a final target large enough to justify a narrow DS5 runtime;
- a smaller active path than Mixtral 8x22B;
- a much smaller total residency requirement than DeepSeek-V3/R1 or Kimi-K2.

### 3.2 Why not dense 70B as final target

Dense 32B-70B remains useful for bring-up because it is simpler to distribute and validate. It is not the final target because a dense 70B decode path reads the dense model every token and leaves less bandwidth margin at 32K context once KV, dequantization, synchronization, and kernel overhead are included.

### 3.3 Why not Mixtral 8x22B as final target

Mixtral 8x22B remains a useful intermediate MoE candidate because it has a simpler expert topology. It is not the final target because its active parameter footprint is materially larger than Qwen3-235B-A22B's 22B active path and therefore has a weaker quality/throughput fit for the DS5 mesh.

### 3.4 Why not DeepSeek-V3/R1 or Kimi-K2 as final target

Full DeepSeek-V3/R1 and Kimi-K2-scale checkpoints exceed the safe resident-weight envelope for the 144GB DS5 cluster. The project will reuse the architectural ideas of Prefill/Decode Disaggregation, expert locality, hot expert residency, and promotion, but it will not rely on full active expert streaming from NVMe or on resident storage of DeepSeek/Kimi-scale full checkpoints.

---

## 4. Consequences

### 4.1 Required system updates

The runtime must support:

- 94-layer MoE scheduling;
- 128 experts per MoE layer;
- top-8 expert routing;
- exact preservation of Qwen routing semantics;
- B/C decode-worker ownership of layer ranges;
- locally mirrored router/gate tensors after correctness validation;
- mixed tensor-aware quantization;
- expert-hotness telemetry and placement maps;
- cold expert promotion without blocking the steady-state decode path;
- KV page management for 8K-32K first, with 64K+ as stretch.

### 4.2 Required project updates

The project spec must be revised from a generic or Gemma-learning-engine scope to a Qwen3-235B-A22B MoE runtime scope. The earlier Gemma/JW4 brief should either be archived as a precursor or explicitly reframed as a separate learning track.

### 4.3 Required benchmark updates

Benchmarks must measure:

- exact router/top-k correctness;
- MoE placement correctness;
- resident hot-expert hit rate;
- cold-miss rate and stall time;
- tool-call JSON correctness;
- 8K, 32K, and 64K context behavior;
- p50/p95/p99 decode latency;
- memory pressure against the 30% per-node runtime reserve.

---

## 5. Non-negotiable rules

1. **Do not alter model top-k routing to force locality.** Expert locality is achieved through placement and promotion, not by selecting lower-ranked local experts.
2. **Do not use NVMe as the steady-state active-weight path.** NVMe is for promotion, prefetch, cold backing, and KV/page backing.
3. **Do not quantize router/gate tensors aggressively.** Router/gate tensors remain FP16 or Q8 unless benchmark evidence proves otherwise.
4. **Do not claim >12 tok/s at 128K-262K context until sparse attention, KV compression, and paging are benchmarked.**
5. **Do not treat BitNet-style 1.58-bit conversion as a safe PTQ path for Qwen3 weights.** It is out of scope unless empirically validated as a separate research task.

---

## 6. Status of alternatives

| Alternative | Status | Use allowed |
|---|---|---|
| Dense 32B | Bring-up target | Yes |
| Dense 70B Q4/IQ4 | Distributed fallback and benchmark comparator | Yes |
| Qwen3-30B-A3B | Small MoE correctness target | Yes |
| Mixtral 8x22B | Intermediate MoE or comparator | Yes |
| DeepSeek-V3/R1 full checkpoint | Rejected final target | No, except as architectural study |
| Kimi-K2 full checkpoint | Rejected final target | No, except as architectural study |
| Gemma/JW4 target | Superseded for DS5 | Only as separate precursor project |

---

## 7. Decision review triggers

This ADR must be reviewed only if one of the following occurs:

- the hardware topology changes from 3 nodes to a larger memory pool;
- official Qwen3-235B-A22B model artifacts become unavailable or unsuitable;
- a smaller model achieves the project benchmark goals with materially lower complexity;
- an official Qwen successor has a better active/total footprint and compatible license;
- Phase 2 MoE correctness shows unacceptable quality loss under the required quantization envelope.
