# DS5 System Architecture v0.2: Qwen3-235B-A22B

**Document type:** System architecture specification  
**Status:** Planning baseline update  
**Date:** 2026-07-08  
**Model decision:** Qwen3-235B-A22B is the locked DS5 target model

---

## 1. Architecture summary

DS5 uses a three-node Apple Silicon mesh to run Qwen3-235B-A22B as a sparse MoE model. The architecture separates control-plane duties from decode data-plane duties:

- **Node A** is the control plane: scheduler, tokenizer, sampler/logits owner, drafter host, global KV/page-table owner, telemetry collector, placement policy owner, and cold-expert inventory manager.
- **Node B** is decode worker 1: owns layers 0-46, attention/KV for those layers, local router mirror, resident hot/warm/cool experts, and promotion cache.
- **Node C** is decode worker 2: owns layers 47-93, attention/KV for those layers, local router mirror, resident hot/warm/cool experts, and promotion cache.

The steady-state decode path should be resident in unified memory. NVMe is not used as the normal active-weight path.

---

## 2. Context diagram

```mermaid
flowchart LR
    User["Local operator"] --> Harness["Agentic harness
structured tool calls
durable state
replay"]
    Harness --> API["DS5 local inference API
streaming tokens
constrained decoding
trace capture"]
    API --> A["Node A control plane
scheduler
tokenizer/logits
drafter
KV page table
placement policy"]
    A <--> B["Node B decode worker
layers 0-46
attention/KV
resident experts"]
    A <--> C["Node C decode worker
layers 47-93
attention/KV
resident experts"]
    B <--> C["activation / expert-result exchange"]
    A --> StoreA["A storage
model manifests
cold experts
logs
KV backing"]
    B --> StoreB["B storage
promotion cache
KV backing"]
    C --> StoreC["C storage
promotion cache
KV backing"]
    Harness --> Bench["Benchmark harness
performance
quality
tool reliability"]
```

---

## 3. Deployment topology

```mermaid
flowchart TB
    subgraph NodeA["Node A - M5 Pro - 48GB UMA"]
        A1["Coordinator"]
        A2["Tokenizer / sampler / logits"]
        A3["Speculative drafter"]
        A4["Global KV page table"]
        A5["Router telemetry"]
        A6["Cold expert inventory"]
    end

    subgraph NodeB["Node B - M5 Max - 48GB UMA"]
        B1["Layers 0-46 attention"]
        B2["Layers 0-46 KV pages"]
        B3["Local router mirror"]
        B4["Resident hot/warm/cool experts"]
        B5["Promotion cache"]
    end

    subgraph NodeC["Node C - M5 Max - 48GB UMA"]
        C1["Layers 47-93 attention"]
        C2["Layers 47-93 KV pages"]
        C3["Local router mirror"]
        C4["Resident hot/warm/cool experts"]
        C5["Promotion cache"]
    end

    NodeA <-->|"Thunderbolt 5 mesh - measured in Phase 0"| NodeB
    NodeA <-->|"Thunderbolt 5 mesh - measured in Phase 0"| NodeC
    NodeB <-->|"Thunderbolt 5 mesh - measured in Phase 0"| NodeC
```

---

## 4. Logical component view

```mermaid
flowchart TD
    Config["Config + manifests"] --> Loader["Weight loader"]
    Loader --> Placement["Placement planner"]
    Placement --> ARuntime["Node A runtime"]
    Placement --> BRuntime["Node B runtime"]
    Placement --> CRuntime["Node C runtime"]

    ARuntime --> Scheduler["Token scheduler"]
    ARuntime --> Sampler["Tokenizer / logits / sampler"]
    ARuntime --> Drafter["Speculative drafter"]
    ARuntime --> KVIndex["Global KV page index"]
    ARuntime --> ExpertPolicy["Expert hotness + promotion policy"]

    BRuntime --> BAttention["Attention kernels 0-46"]
    BRuntime --> BExperts["Fused expert kernels"]
    BRuntime --> BKV["KV pages 0-46"]

    CRuntime --> CAttention["Attention kernels 47-93"]
    CRuntime --> CExperts["Fused expert kernels"]
    CRuntime --> CKV["KV pages 47-93"]

    Scheduler --> Transport["Transport protocol"]
    Transport --> BRuntime
    Transport --> CRuntime
    BRuntime --> Telemetry["Runtime telemetry"]
    CRuntime --> Telemetry
    Telemetry --> ExpertPolicy
    Telemetry --> Bench["Benchmark records"]
```

---

## 5. Runtime decode flow

```mermaid
sequenceDiagram
    participant H as Harness/API
    participant A as Node A Control
    participant B as Node B Layers 0-46
    participant C as Node C Layers 47-93
    participant S as Storage/NVMe

    H->>A: request + prompt state
    A->>A: tokenize / schedule / page-table lookup
    A->>B: hidden state + decode command
    loop layers 0-46
        B->>B: attention + local router mirror
        B->>B: select top-8 exact model experts
        B->>C: optional remote expert packet
        C-->>B: optional reduced expert output
        B->>B: fused local expert kernel + residual
    end
    B->>C: hidden state after layer 46
    loop layers 47-93
        C->>C: attention + local router mirror
        C->>C: select top-8 exact model experts
        C->>B: optional remote expert packet
        B-->>C: optional reduced expert output
        C->>C: fused local expert kernel + residual
    end
    C->>A: final hidden state
    A->>A: logits / constrained decode / sampling
    A-->>H: token + trace record
    A->>S: async promotion/prefetch/logging only
```

---

## 6. Node responsibilities

| Capability | Node A | Node B | Node C |
|---|---|---|---|
| Tokenizer | Primary | No | No |
| Sampler/logits | Primary initially | Optional later shard | Optional later shard |
| Drafter | Primary | No | No |
| Global scheduler | Primary | Worker endpoint | Worker endpoint |
| Layer attention | Optional validation copy | Layers 0-46 | Layers 47-93 |
| KV cache | Global page table | Pages 0-46 | Pages 47-93 |
| Router/gates | Authoritative copy; policy | Local mirror after validation | Local mirror after validation |
| Hot/warm experts | No, except cold/overflow | Resident | Resident |
| Cold experts | Inventory + compressed source | Promotion target | Promotion target |
| Telemetry | Aggregator | Producer | Producer |
| NVMe | Cold backing/logs/KV backing | Promotion/KV backing | Promotion/KV backing |

---

## 7. Data view

### 7.1 Persistent data classes

| Data class | Location | Durability | Notes |
|---|---|---|---|
| Raw model artifact | External/local model store | Persistent | Reference source; not mutated |
| Quantized weight shards | A/B/C local storage | Persistent | Generated from quant manifest |
| Placement manifest | A primary, B/C copy | Versioned | Defines layer, expert, quant, owner |
| KV page backing | A/B/C storage | Ephemeral or checkpointed | Used for long context and replay |
| Router telemetry | A logs | Persistent for calibration | Hit rate, gate mass, cold misses |
| Benchmark traces | A logs | Persistent | Reproducibility and regression tests |
| Tool traces | Harness store | Persistent | Used for tool reliability benchmark |

### 7.2 Placement manifest schema sketch

```yaml
model: Qwen3-235B-A22B-Instruct-2507
manifest_version: 0.2
nodes:
  A: {role: control_plane, static_cap_gb: 33.6}
  B: {role: decode_worker, layers: [0, 46], static_cap_gb: 33.6}
  C: {role: decode_worker, layers: [47, 93], static_cap_gb: 33.6}
tensors:
  router_gates: {quant: fp16_or_q8, owner: A, mirrors: [B, C]}
  lm_head: {quant: q8_or_q6, owner: A}
  attention:
    layers_0_46: {owner: B, quant: q8_or_q6}
    layers_47_93: {owner: C, quant: q8_or_q6}
experts:
  tier_policy: calibrated_hotness
  hot: {quant: iq3_s_or_q4_k_s, owners: [B, C]}
  warm: {quant: iq2_m, owners: [B, C]}
  cool: {quant: iq2_xs, owners: [B, C]}
  cold: {quant: iq2_xxs, owner: A, backing: nvme}
```

---

## 8. Security and governance view

| Area | Requirement |
|---|---|
| Local-only prompts | Prompt, tool, and state data remain local unless explicitly exported |
| Model artifacts | Verify checksums and quant manifest provenance |
| Tool credentials | Stored outside traces; redacted in benchmark replay artifacts |
| Logs | Trace logs must support redaction and retention policy |
| Reproducibility | Benchmark runs must record model version, quant manifest, placement manifest, git commit, and hardware measurements |
| Change control | Model target changes require a superseding ADR |

---

## 9. Operational view

| Operational signal | Owner | Action |
|---|---|---|
| Per-node static memory | Loader | Refuse launch if >33.6GB unless override is explicit |
| Runtime memory pressure | Node daemon | Throttle context/promotion; emit alert |
| Router/top-k mismatch | Correctness harness | Block release |
| Cold-miss burst | Placement policy | Promote/demote and update heatmap |
| Blocking NVMe read in decode | Runtime tracer | Treat as performance defect |
| Tool-call parse failure | Harness | Log, retry under constrained decode, benchmark |

---

## 10. Architecture principles

1. Narrow model-specific runtime over generic abstraction.
2. Correctness before performance.
3. Exact Qwen routing semantics before locality optimization.
4. Resident hot path before NVMe promotion tricks.
5. Measurement before throughput claims.
6. Mixed precision by tensor criticality, not uniform quantization.
7. Document every model/runtime assumption in versioned manifests.
