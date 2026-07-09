# DS5 Model Runtime and Placement Spec v0.2: Qwen3-235B-A22B

**Document type:** Runtime, placement, quantization, and memory specification  
**Status:** Planning baseline update  
**Date:** 2026-07-08  
**Target model:** Qwen3-235B-A22B-Instruct-2507

---

## 1. Purpose

This document defines how the DS5 runtime should load, place, execute, quantize, route, page, promote, and measure Qwen3-235B-A22B across the three-node Apple Silicon cluster.

---

## 2. Runtime invariants

| Invariant | Required behavior |
|---|---|
| Exact routing | Preserve model top-8 routing semantics |
| Resident decode | Steady-state active expert path must be resident in UMA |
| Tensor-aware quantization | Quantization varies by tensor criticality |
| Node A not overloaded | Node A coordinates and owns policy; B/C carry decode data plane |
| No per-expert packets | Send one activation packet per destination node per layer |
| No hidden model substitution | Bring-up models are not the final model |
| Launch refusal | Runtime refuses manifests that exceed static caps unless explicitly overridden |

---

## 3. Layer placement

| Layer range | Owner | Responsibilities |
|---|---|---|
| Layers 0-46 | Node B | Attention, KV pages, local router mirror, resident experts, fused expert kernels |
| Layers 47-93 | Node C | Attention, KV pages, local router mirror, resident experts, fused expert kernels |
| Cross-layer state | Node A | Scheduler, page table, final logits, sampling, trace capture, policy |

Initial correctness mode may route through Node A for every layer. Performance mode should use local router/gate mirrors on B/C after correctness is established.

---

## 4. Expert tiering

### 4.1 Tier definitions

| Tier | Placement | Quantization | Policy |
|---|---|---|---|
| Hot | Resident on B/C | IQ3_S or Q4_K_S | Highest-hit experts; never NVMe-blocking |
| Warm | Resident on B/C | IQ2_M or Q3-class | Promote/demote by telemetry |
| Cool | Resident on B/C if budget allows | IQ2_XS | Lower-frequency experts |
| Cold | Node A + NVMe mirror | IQ2_XXS | Promotion source, not normal decode path |

### 4.2 Cold-start fallback

Before calibration data exists:

| Node | Cold-start expert assignment |
|---|---|
| Node B | Even-numbered resident expert IDs from the selected resident range |
| Node C | Odd-numbered resident expert IDs from the selected resident range |
| Node A | Cold/overflow experts and promotion inventory |

After calibration, replace static ID order with per-layer hotness rank.

### 4.3 Calibration scoring

For each layer `l` and expert `e`:

```text
S(l,e) = 0.70 * hit_rate(l,e)
       + 0.20 * gate_mass(l,e)
       + 0.10 * outlier_score(l,e)
```

Sort experts per layer by `S(l,e)` and place by tier rank.

---

## 5. Memory budget

### 5.1 Static cap

```text
Per node static cap = 48GB * 0.70 = 33.6GB
Per node runtime reserve = 48GB * 0.30 = 14.4GB
Cluster static cap = 144GB * 0.70 = 100.8GB
```

### 5.2 Worker budget target

| Component | Node B target | Node C target | Notes |
|---|---:|---:|---|
| Attention weights | ~3.35GB | ~3.35GB | Planning estimate for half model at Q8-class |
| Resident experts | ~28GB | ~28GB | Hot/warm/cool mix |
| Static subtotal | ~31.3GB | ~31.3GB | Below 33.6GB cap |
| Runtime reserve | ≥14.4GB | ≥14.4GB | KV, heaps, buffers, rings |

### 5.3 Node A budget target

| Component | Node A planning target | Notes |
|---|---:|---|
| Attention verification/prefill copy | ~6.70GB | May be trimmed after correctness |
| Embedding + lm_head | ~1.25GB | Prefer Q8/Q6 |
| Router/gates | ~0.05-0.10GB | FP16 or Q8 |
| Cold experts | ~14.6GB | IQ2_XXS planning tier |
| Drafter | ~2.2-4.2GB | Qwen-class 4B-7B candidate |
| Static subtotal | ~25-27GB | Below 33.6GB cap |

---

## 6. Quantization matrix

| Tensor class | Required/default quantization | Reason |
|---|---|---|
| Token embeddings | Q8 or Q6_K | Preserve lexical/tool fidelity |
| lm_head | Q8 or Q6_K | Preserve logits and constrained decode quality |
| Router/gate matrices | FP16 or Q8 | Routing errors are high impact |
| RMSNorm/scales/RoPE | FP16/FP32 | Small; avoid needless degradation |
| Q/K/V projections | Q8 for K/V; Q6-Q8 for Q | Long-context and retrieval sensitivity |
| O projection | Q6-Q8 | Some compression tolerance |
| Hot experts | IQ3_S or Q4_K_S | Preserve frequent expert quality |
| Warm experts | IQ2_M or Q3-class | Balance quality and memory |
| Cool experts | IQ2_XS | Lower-frequency compromise |
| Cold experts | IQ2_XXS | Storage/promotion source only |
| KV hot pages | FP16/BF16 or Q8_KV | Active quality path |
| KV old pages | Q8/Q4 KV or paged backing | Long-context scalability |

---

## 7. KV cache sizing

Planning formula:

```text
KV bytes = context_tokens * layers * 2 * kv_heads * head_dim * bytes_per_element
         = context_tokens * 94 * 2 * 4 * 128 * 2
         = context_tokens * 192,512 bytes
```

| Context | FP16 KV planning size | Status |
|---:|---:|---|
| 8K | ~1.58GB | Comfortable |
| 32K | ~6.31GB | Primary performance target |
| 64K | ~12.62GB | Stretch target |
| 128K | ~25.23GB | Requires KV quant/paging and careful attention path |
| 262K | ~50.47GB | Not a low-latency dense-attention target |

---

## 8. Transport packet rules

### 8.1 Activation packet

```c
struct Ds5ActivationPacket {
    uint16_t version;
    uint16_t layer_id;
    uint16_t source_node;
    uint16_t destination_node;
    uint16_t hidden_dtype;
    uint16_t expert_count;
    uint32_t sequence_id;
    uint64_t token_id;
    uint64_t trace_id;
    uint16_t expert_ids[8];
    float    gate_weights[8];
    // followed by aligned hidden vector payload
};
```

### 8.2 Rule

A layer may send at most one activation packet per destination node. It must not send one packet per expert.

---

## 9. Promotion and NVMe rules

| Rule | Required behavior |
|---|---|
| Decode hot path | Resident only |
| Promotion block | 64MiB default maximum |
| KV stripe | 16-64MiB |
| Sequential prefill shard | 128-256MiB |
| Blocking read during decode | Defect unless explicitly in fallback mode |
| Cold miss handling | Log, optionally stall under controlled fallback, then promote if policy threshold is met |

---

## 10. Telemetry requirements

Each decode run must capture:

- selected experts per layer;
- gate weights per selected expert;
- local vs remote expert execution;
- hot/warm/cool/cold tier hit;
- promotion/demotion events;
- blocking I/O events;
- per-layer latency;
- per-node memory pressure;
- kernel time, dequant time, softmax time, transport time;
- final token latency and throughput.

---

## 11. Correctness tests

| Test | Pass condition |
|---|---|
| Manifest parse | Model dimensions match expected Qwen3-235B-A22B metadata |
| Router parity | Top-8 expert IDs match trusted reference for deterministic prompts |
| Gate parity | Gate weights within tolerance |
| Layer split parity | B/C split output matches single-node reference within tolerance |
| Quantized output drift | Within accepted regression envelope |
| Transport checksum | Activation/result packets pass checksum and trace alignment |
| KV replay | Replay with saved KV produces deterministic tokens under fixed seed |
