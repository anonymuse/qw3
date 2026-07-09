> **Status: ARCHIVED / SUPERSEDED.** This document conflicts with ADR-001 non-negotiables (Top-R gating alters top-k routing; BitNet ~2.06bpw PTQ; O_DIRECT does not exist on macOS; <8microsecond TB sync latency and 1M-token gates are unmeasured fantasy). Retained for history only. See docs/decisions/ and docs/specs/imported_v0.2/ for the active baseline.

# Gemini Project Spec	  
  
  
I confirm that we are officially standardizing on the **Qwen3-235B-A22B** Mixture-of-Experts (MoE) model for the **DS5 Bespoke Qwen Zig Engine** project.  
I confirm that we are officially standardizing on the **Qwen3-235B-A22B** Mixture-of-Experts (MoE) model for the **DS5 Bespoke Qwen Zig Engine** project.  
To reflect this architectural locking, the two primary files requiring immediate updates are the core **Project Specification (project_spec.md)** and the **System Architecture & Integration Blueprint (PROJECT_KICKOFF.md)**. These documents have been hyper-tailored to handle the distributed streaming constraints, ultra-low-bit quantization math, and asymmetric network topology of your 144GB Apple Silicon cluster.  
To reflect this architectural locking, the two primary files requiring immediate updates are the core **Project Specification (project_spec.md)** and the **System Architecture & Integration Blueprint (PROJECT_KICKOFF.md)**. These documents have been hyper-tailored to handle the distributed streaming constraints, ultra-low-bit quantization math, and asymmetric network topology of your 144GB Apple Silicon cluster.  
## 1. Updated Project Specification (project_spec.md)  
Markdown  
  
# Project Specification: Bespoke Qwen Zig Engine (DS5)  
## Target Architecture: Qwen3-235B-A22B (Asymmetric Scaling Path)  
  
### 1. System Overview & Target Topology  
The DS5 project is a distributed, model-specific open-weight MoE runtime written in zero-dependency Zig 1.0. It coordinates a heterogeneous 3-node Apple Silicon cluster connected via a bare-metal Thunderbolt 5 mesh fabric to run the Qwen3-235B-A22B model at low latency.  
  
#### Hardware Cluster Profile  
* **Node A (Coordinator / Anchor):** MacBook Pro M5 Pro (48GB Unified Memory | 307 GB/s)  
* **Node B (Compute Node 1):** MacBook Pro M5 Max (48GB Unified Memory | 614 GB/s)  
* **Node C (Compute Node 2):** MacBook Pro M5 Max (48GB Unified Memory | 614 GB/s)  
* **Network Interconnect:** Thunderbolt 5 Full Mesh (Latency Target: < 8μs sync time)  
* **Cluster Constraints:** Total 144GB physical RAM. Absolute, non-negotiable **30% per-node memory headroom rule**, limiting total static resident weights to **33.6 GB maximum per node**.  
  
---  
  
### 2. Model Architecture & Quantization Strategy  
The target workload is the **Qwen3-235B-A22B** Mixture-of-Experts architecture:  
* **Total Parameters:** 235 Billion  
* **Active Parameters Per Token:** 22 Billion (Sparse routing slice)  
* **Total Experts:** 128 routing experts  
  
#### Asymmetric Quantization Matrix  
* **Attention & Gating Layers:** Pinned in unified memory at high-fidelity **Q8_0 / FP8 Precision** to avoid performance degradation in multi-turn reasoning loops.  
* **MoE Expert MLPs:** Aggressively compressed to ultra-low-bit arrays using **IQ2_XXS / BitNet (~2.06 bits per weight)**.  
* **Memory Footprint Calculation:** $$235 \times 10^9 \text{ parameters} \times \sim0.2575 \text{ bytes/param} \approx 54\text{ GB to } 60\text{ GB static weight footprint}$$  
    Split across Compute Nodes B & C, this fits inside the combined 96GB allocation while maintaining the 30% safety headroom and leaving a massive buffer for rotating expert streams and KV-caches.  
  
---  
  
### 3. Core Runtime Innovations  
To mitigate the physical limitations of streaming a 235B parameter model across 48GB unified memory nodes, the runtime implements three core software innovations:  
  
#### Innovation A: Fused Gating & Routing Packet Protocol  
To minimize Thunderbolt network roundtrip overhead, Node A does not evaluate gating layer-by-layer. Instead, Node A's coordinator concurrently evaluates block routing sequences several layers ahead and broadcasts a highly compact, zero-copy packet over the Thunderbolt mesh:  
```zig  
const GatingPacket = struct {  
    layer_id: u8,  
    active_expert_ids: [8]u8,  
    weight_coefficients: [8]f16,  
    target_nodes: u2,  
};  
**Innovation B: Asynchronous Rolling Pre-fetch Buffer**  
The 94% most statistically active "hot" experts are permanently pinned in the M5 Max UMA pools. The remaining "cold" experts reside on local high-speed NVMe SSDs.  
* The Zig orchestrator computes a continuous *Routing Entropy Heatmap*.  
* If a cold expert's activation probability crosses a **>5% threshold** over an upcoming speculative token sequence, an asynchronous io.async command bypasses the filesystem (O_DIRECT) to stream the expert block into a rotating VRAM UMA buffer **7 to 8 tokens ahead of execution**, hiding I/O times completely behind live active shader compute.  
**Innovation C: Communication-Aware Top-R Gating (Topology-Predictive Routing)**  
Traditional Top-K routing is replaced with environment-aware **Top-R (Resource-Aware) Gating**. Node A's gating engine maintains an active telemetry matrix of NVMe read-head queue depths and PCIe bus utilization across Nodes B and C. If two experts have near-identical gating coefficients ($\alpha_1 = 0.42, \alpha_2 = 0.40$), but Expert 1 is "cold" (requiring NVMe streaming) and Expert 2 is currently "hot" in memory, the routing algorithm dynamically favors Expert 2 to eliminate hardware execution stalls.  
## 4. Hardware Safety & Kernel Execution  
* **Metal Kernel Fragmentation:** All large multi-expert calculations running on Nodes B and C must be fragmented into localized, non-blocking Metal command fragments. No matrix multiplication loop may saturate unified memory lanes long enough to breach the 5-second macOS hardware command buffer timeout threshold.  
* **Memory Lifecycles:** Enforce strict Zig ArenaAllocator structures that reset after every token-generation pass to eliminate leaks. All io.async calls MUST be paired with a defer task.cancel(io) catch pattern.  
## 5. PM Validation & Quality Assurance (The 4 Gates)  
1. **GATE 1 (System Integrity):** Seamlessly aggregate the 144GB unified pool; guarantee zero memory leaks via Arena resets under a continuous multi-hour loop.  
2. **GATE 2 (Compile & Static):** Ensure a zero-dependency Zig 1.0 compilation target across macOS Tahoe (v26.2).  
3. **GATE 3 (Network & RDMA):** Verify Thunderbolt socket buffers achieve synchronization latencies under < 8μs per block.  
**	4	GATE 4 (QA & Stress Test):** Fuzz a 1M token context window; achieve a steady **> 12 tokens/sec** using the asymmetric 2-bit MoE expert pipeline.  
---  
  
### 2. Updated Project Kickoff Document (`PROJECT_KICKOFF.md`)  
  
```markdown  
# Project Manifest & Architecture Integration: DS5 Engine  
  
This document establishes the scaffold and distributed directory configuration for implementing the Qwen3-235B-A22B engine across our heterogeneous hardware topology.  
  
  
### 1. Heterogeneous Cluster Directory Scaffolding  
  
```text  
├── build.zig                 # Unified multi-target Zig 1.0 build configuration  
├── src/  
│   ├── shared/               # Shared protocols, types, and tensor interfaces  
│   │   ├── protocol.zig      # Zero-copy Fused Gating packet definitions  
│   │   ├── tensorbus.zig     # Core interface abstraction layer (std.mem.copy -> RDMA)  
│   │   └── math.zig          # Fixed-point calculations and weight metadata structures  
│   │  
│   ├── node_a/               # THE BRAIN (MacBook Pro M5 Pro 48GB)  
│   │   ├── main.zig          # Coordinator entry point & Deterministic validation loop  
│   │   ├── tokenizer.zig     # High-throughput text tokenization loop  
│   │   ├── drafter.zig       # Speculative draft engine (Qwen-7B running at high-precision Q8)  
│   │   └── router.zig        # Telemetry-aware Top-R Gating and entropy heatmap calculator  
│   │  
│   └── node_b_c/             # THE COMPUTE WORKERS (MacBook Pro M5 Max 48GB x 2)  
│       ├── main.zig          # Worker execution loops and listener sockets  
│       ├── metal_runtime.zig # Non-blocking Metal compute shader pipeline orchestrator  
│       ├── expert_cache.zig  # Rolling NVMe SSD pre-fetch buffer manager (O_DIRECT)  
│       └── shaders/  
│           ├── qwen_gqa.metal# Custom Metal shaders for Grouped Query Attention  
│           └── moe_expert.metal # Compressed IQ2_XXS multi-expert matrix multiply kernels  
## 2. Structural Component Matrix  

| Module | Target Node | Systems Purpose |
| ------------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| src/shared/tensorbus.zig | Shared | Abstracts physical layer tensor transfers, seamlessly moving from single-node memory copies to Thunderbolt 5 mesh RDMA writes. |
| src/shared/protocol.zig | Shared | Quantizes and packages fused, multi-layer routing packets to decrease Thunderbolt fabric traffic frequency. |
| src/node_a/router.zig | Node A | Evaluates Top-R resource-aware gating routing by pairing mathematical coefficients with live hardware telemetry from the workers. |
| src/node_a/drafter.zig | Node A | Runs a small, high-fidelity draft model to predict future token paths and concurrently feed the worker pre-fetch pipelines. |
| src/node_b_c/expert_cache.zig | Node B/C | Intercepts speculative pre-fetch trees and issues asynchronous unbuffered disk reads to prepare "cold" experts before execution. |
| src/node_b_c/metal_runtime.zig | Node B/C | Splits heavy matrix workloads into tiny, isolated command buffer fragments to ensure constant operating system display and system stability. |
  
****3. Execution Pipeline & Bootstrap Protocol****  
1. **Orchestrator Boot:** node_a loads the model's structural attention blocks, high-fidelity gating weights, and tokenization dictionaries. It spins up the Routing Entropy Heatmap module.  
2. **Worker Registration:** node_b_c runtimes establish memory arenas, pin the 94% "hot" statistical experts into unified memory pools, and format the NVMe scratch disk to establish the O_DIRECT expert streaming cache.  
3. **Inference Ring:** Node A handles input text, streams context layers via TensorBus, executes the token draft loop, and transmits the GatingPacket trees. Compute nodes handle the active 22B token-sparse parameters, continuously pre-fetching outlier experts on background threads without pausing the primary hardware compute pipeline.  
