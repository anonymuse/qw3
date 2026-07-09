# DS5 Risk Register v0.2: Qwen3-235B-A22B

**Document type:** Risk register  
**Status:** Planning baseline update  
**Date:** 2026-07-08  
**Target model:** Qwen3-235B-A22B-Instruct-2507

---

## 1. Risk summary

| ID | Risk | Probability | Impact | Severity | Mitigation | Owner |
|---|---|---:|---:|---:|---|---|
| R-001 | Qwen3-235B-A22B placement exceeds per-node static cap after real quant artifacts | Medium | High | High | Build placement simulator before kernel work; refuse launch over 33.6GB | Runtime/loader |
| R-002 | Mixed IQ2/IQ3 expert quantization damages quality | Medium | High | High | Use calibration corpus; keep hot experts higher precision; compare against reference | Quant/eval |
| R-003 | Router/gate quantization changes top-8 expert selection | Medium | High | High | Keep router/gates FP16 or Q8; block release on top-k mismatch | MoE correctness |
| R-004 | Node A becomes a synchronous bottleneck | Medium | High | High | Start with Node A routing for correctness; move to B/C local mirrors for performance | Runtime architecture |
| R-005 | Cold expert miss bursts stall decode | High | High | Critical | Resident hot path; telemetry-driven placement; no steady-state blocking NVMe reads | Placement policy |
| R-006 | Per-expert network packetization creates latency collapse | Medium | Medium | Medium | One packet per destination node per layer | Transport |
| R-007 | Metal command-buffer overhead dominates small expert kernels | Medium | High | High | Fused multi-expert kernels; benchmark command overhead in Phase 0 | Kernels |
| R-008 | 64K+ context consumes bandwidth and memory margin | High | Medium | High | Treat 8K-32K as target; make 64K stretch; defer 128K+ | KV architecture |
| R-009 | macOS I/O behavior differs from Linux-style assumptions | Medium | Medium | Medium | Validate actual file I/O path in Phase 0; avoid unverified O_DIRECT assumptions | Storage |
| R-010 | Harness scope competes with engine scope | High | High | Critical | Phase gates; define hard stop for engine; keep publishable benchmark in plan | Project owner |
| R-011 | Earlier JW4/Gemma scope causes project confusion | High | Medium | High | Supersede/split JW4; all DS5 specs name Qwen3 target explicitly | Documentation |
| R-012 | Official model artifact changes or disappears | Low | High | Medium | Pin revision/hash; mirror artifacts; record source metadata | Model ops |
| R-013 | Thinking variant creates unacceptable latency | Medium | Medium | Medium | Defer Thinking-2507; first stabilize Instruct-2507 | Product/runtime |
| R-014 | Benchmark results are not reproducible | Medium | High | High | Record manifests, seed, commit, hardware measurements, traces | Benchmarking |
| R-015 | Tool-call reliability remains poor despite model size | Medium | High | High | Constrained decoding, harness recovery, trace-based evals | Harness |

---

## 2. Technical risk detail

### R-001: Placement budget risk

**Description:** Planning estimates place Node B/C static weights below 33.6GB, but real quantized artifacts, metadata, padding, alignment, or duplicated tensors may exceed the static cap.

**Mitigation:**

- Build a placement simulator that uses actual tensor metadata.
- Include padding, alignment, scales, and metadata in estimates.
- Fail fast at load time if any node exceeds cap.
- Keep a lower-precision fallback tier for cool experts.

### R-003: Router correctness risk

**Description:** Small numeric changes in router/gate tensors can change selected experts and create large output drift.

**Mitigation:**

- Keep router/gates FP16 or Q8.
- Compare top-8 IDs and gate weights against a trusted reference.
- Treat any intentional routing substitution as a release blocker.

### R-005: Cold miss risk

**Description:** The system cannot rely on NVMe to stream all active expert weights per token. A burst of cold misses can dominate latency.

**Mitigation:**

- Keep hot/warm/cool experts resident.
- Use Node A/NVMe as cold backing and promotion source only.
- Track miss rate, miss burst length, and stall time.
- Promote by calibrated hit-rate and gate mass.

### R-010: Engine/harness scope risk

**Description:** A full Qwen3-235B-A22B runtime is complex enough to consume the whole project, displacing the agentic harness and benchmark thesis.

**Mitigation:**

- Preserve phase gates.
- Treat dense and small-MoE bring-up as learning stages.
- Stop optimization once the benchmark can answer the project question.
- Keep harness reliability benchmark as a first-class deliverable.

---

## 3. Risk review cadence

| Event | Required review |
|---|---|
| End of each phase | Re-score all High/Critical risks |
| Model artifact pinning | Review R-001, R-002, R-012 |
| First Qwen3 load | Review R-001, R-004, R-007 |
| First 32K context run | Review R-005, R-008 |
| First harness benchmark | Review R-010, R-015 |
| Any model target change | Supersede ADR-001 and update this register |
