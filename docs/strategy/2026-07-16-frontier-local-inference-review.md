# QW3 / DS5 frontier local-inference strategy

**Decision memorandum for the project owner, engineering agents, and product-management agents**

**As of:** 2026-07-16

**Scope:** repository and history review, current open-model landscape, OpenAI open-weight local inference, Apple Silicon M5, Thunderbolt 5 clustering, Exo/MLX, serving-system research, and an executable delivery plan

## Executive decision

QW3 should become the **reproducible Apple Silicon inference lab**: the place an operator can trust to answer three questions that the rest of the local-LLM market usually blurs together:

1. Will this exact model and quantization fit safely on this Mac or cluster?
2. Is this runtime semantically and numerically correct for this exact artifact?
3. Does adding another Mac make inference faster, or merely make the model loadable?

DS5—the narrow, from-scratch Zig + Metal engine for a three-Mac topology—should remain the project's technical proof and research core. It should not be expanded immediately into an undifferentiated general-purpose runtime. Around it, QW3 should add a thin product and evidence layer that can benchmark DS5 against MLX-LM, Exo/JACCL, and llama.cpp on identical hardware and workloads.

This is a stronger and more defensible position than either extreme:

- staying a Qwen3-235B-only research artifact with a raw-token CLI and outdated transport assumptions; or
- abandoning the original thesis to become another wrapper around general runtimes.

Five decisions should be made now:

1. **Replace the TCP-only premise with a measured transport bakeoff.** Apple now exposes RDMA over Thunderbolt 5 on Apple Silicon in macOS 26.2+, and MLX/JACCL and Exo use it. The current assumption that user-space RDMA is unavailable is obsolete. Run TCP, direct RDMA/JACCL, and Exo baselines on the actual three-node mesh before freezing M3 transport.
2. **Split correctness from quantization quality.** The current T06 comparison uses Q8_0 engine weights against a full-precision Hugging Face oracle. That cannot cleanly prove an implementation bug or pass. Use a same-artifact reference for engine parity, then evaluate Q8/IQ/MXFP4 quality against BF16 in a separate gate.
3. **Reinstate the deferred M1 viability work.** The 235B placement thesis depends on real expert-use skew, real link performance, and realistic decode traffic. Those inputs were deliberately deferred, but they are more decision-critical than another kernel optimization.
4. **Run a model tournament before investing in M4.** Keep Qwen3-235B as the scientific baseline, but compare it with gpt-oss-120b and Qwen3.5-122B-A10B on reference runtimes. gpt-oss-120b is especially attractive because only 5.1B parameters are active per token, but its 60.8 GiB checkpoint is a tight—not clean—fit for the two 33.6 GB worker budgets: an even byte split leaves less than 1 GB per worker before shared tensors, uneven placement, runtime state, or cache.
5. **Ship a product-shaped proof.** Add tokenizer-backed text input, streaming OpenAI-compatible Chat/Responses endpoints, one-command cluster diagnostics, and shareable evidence cards. A public demo should not require users to supply token IDs.

The immediate objective is not “support every model.” It is to produce an honest, reproducible decision system and one memorable demonstration:

> Three Thunderbolt-connected Macs run an open reasoning model privately, and QW3 proves exactly what made it fit, whether it stayed correct, and whether the third Mac helped.

## 1. Mandate, method, and evidence standard

### 1.1 Questions this review answers

This review treats the codebase, its git history, open pull requests, project specifications, findings, and tests as the internal fact base. It then asks:

- Is the architecture still built on current Apple and distributed-inference facts?
- Does the current milestone sequence test the core thesis as early as possible?
- Which open models best fit the owned M5 hardware and the intended public story?
- Where should DS5 build from scratch, and where should it use external systems as baselines or product adapters?
- What would make the project faster to deliver, more credible, more useful, and more likely to spread?

### 1.2 Evidence grades

Claims in this memorandum use four evidence grades:

| Grade | Meaning | Use |
|---|---|---|
| **P1** | Peer-reviewed research or archival conference paper | System-design mechanism or established empirical result |
| **P2** | Official documentation, model card, source repository, hardware specification, or QW3 committed artifact | Current capability or project fact |
| **P3** | Vendor benchmark, preprint, or self-reported project result | Hypothesis to reproduce |
| **P4** | Practitioner report, forum thread, or issue report | Test-case generation and directional warning only |

Measured, simulated, estimated, and target values must remain distinct. In particular, a loopback result is not cluster evidence, a vendor throughput claim is not a QW3 measurement, and “active parameters” are not a weight-residency estimate.

### 1.3 What “all models” can responsibly mean

The open-model market changes faster than a static report can remain exhaustive. “All” should therefore mean a maintained taxonomy and qualification pipeline, not a one-time list of every checkpoint:

- deployment tier: edge, workstation, high-memory workstation, or cluster;
- architecture: dense, MoE, hybrid attention/state-space, multimodal;
- capability: text, vision, reasoning, tools/agents, structured output;
- license: Apache/MIT, modified permissive, or custom community license;
- format: BF16/FP8, GGUF, MLX affine, GPTQ/AWQ, MXFP4/NVFP4;
- runtime maturity: verified, conversion available, experimental, or unsupported;
- serving maturity: template, cache, tool, batching, streaming, and distributed correctness.

“Open weight” and “open source” should not be used as synonyms. Public weights do not imply an OSI-approved license, released training data, or reproducible training code.

## 2. Repository fact base

### 2.1 What has been built

The repository is a serious, rapidly executed research prototype rather than a paper design. Across 77 commits between 2026-07-08 and 2026-07-16, it progressed from scaffold to a real Qwen3-30B-A3B Q8_0 forward pass. The implementation now includes:

- a Zig 0.16 command-line application and raw-libc system layer;
- node daemon, health protocol, TCP-over-Thunderbolt transport, and link benchmark;
- frozen buffer/kernel contracts and generated golden fixtures;
- a GGUF parser and mmap-backed weight access;
- CPU reference kernels for normalization, RoPE, quantized matrix operations, GQA attention, routing, and MoE expert MLP;
- hand-written Metal kernels and a generic CPU/Metal forward engine;
- per-layer tracing and run-metadata output;
- cluster bootstrap and operations tooling;
- a real 30B model gate and a follow-up router-divergence investigation.

#### Commit-history arc

The 77-commit mainline contains 56 non-merge commits and 21 merge commits. Read chronologically, the effort is coherent even though the public status text fell behind:

| Period | Representative commits | What the team accomplished | Outside-in interpretation |
|---|---|---|---|
| 2026-07-08–09 | `dd57e1a`, `9eb940c`, `771a195` | Repository/M0 scaffold, node daemon, health and TCP link benchmark, frozen contracts, oracle-fixture generator | The project began with a narrow topology and contract-first discipline rather than a generic serving shell |
| 2026-07-10–12 | `1435dd9` through `91da845` | CPU reference kernels, Metal glue and shaders, exact router/MoE path, GGUF v3 mmap parser, GQA/KV path, A-09 synchronization measurement | Multiple bounded work packs converged into a credible low-level execution substrate |
| 2026-07-12–13 | `47aa511` through `7ca4dbb` | Architecture reviews and v0.3 specs, CPU end-to-end engine/CLI, expert-stat tooling, quantization manifest and fused-expert finding, f16 KV decision, optimization backlog | The work exposed the real thesis risks—placement, quantization storage, and synchronization—but several M1 measurements stayed deferred |
| 2026-07-14–15 | `c8d7c5a` through `bcdfc20` | GPU forward integration, f16 KV dispatch, three-node bootstrap/SSH/verification tooling, repeated operator fixes, T06 model staging | Delivery velocity was high, but cluster operations and evidence semantics accumulated faster than end-to-end distributed inference |
| 2026-07-16 | `b42e511` through `27cbf88` | Node-D formalization, real LAN manifest updates, GitHub auth setup, generalized verifier fixes, overnight prompts, T06 handoff/block status | Main ended with strong synthetic single-node evidence and improved operations, while real-model findings remained in open stacked PRs and T07 correctly stayed blocked |

The repeated small PRs and corrective cluster commits show healthy willingness to repair operational mistakes quickly. They also show why exact-SHA gates, one authoritative handoff, and evidence labels are now necessary: coordination complexity has become a first-order engineering concern.

The current test baseline on refreshed `origin/main` is strong:

- `zig build test --summary all`: **74/74 passed**;
- `zig build test-metal --summary all`: **21/21 passed** on local M5 hardware;
- `zig build test-gpu --summary all`: **81/81 passed** on local M5 hardware;
- expert-stat Python tests: **30 run: 29 passed, 1 skipped** because `jsonschema` was not installed;
- cluster shell scripts pass syntax validation.

The global formatting gate is not clean on current main: `zig fmt --check build.zig src` reports four pre-existing unformatted Zig files. This does not invalidate the passing tests, but it should be repaired before making formatting a CI requirement.

This is unusually good correctness discipline for a nine-day-old low-level inference project.

### 2.2 Integration truth at review start

The public state is fragmented:

- `origin/main` is at `27cbf88` as of this review.
- [PR #29](https://github.com/anonymuse/qw3/pull/29) contains the T06 real-weights gate and remains open. Review findings flag incomplete benchmark provenance, router sampling of only 15 of 240 token/layer combinations, and a reported-count error: two misses in 15 observations means 13/15 matches, not 14/15. [[provenance finding](https://github.com/anonymuse/qw3/pull/29#discussion_r3599248955)] [[coverage finding](https://github.com/anonymuse/qw3/pull/29#discussion_r3599248959)] [[count finding](https://github.com/anonymuse/qw3/pull/29#discussion_r3599248961)]
- [PR #31](https://github.com/anonymuse/qw3/pull/31) is stacked on #29 and contains a proposed quantization-divergence localization; it also remains open. A review finding shows that reused continuation tokens are not bound to checkpoint/tokenizer provenance, so the localization should be repaired and reproduced before it becomes an authoritative root cause. [[manifest-provenance finding](https://github.com/anonymuse/qw3/pull/31#discussion_r3599918527)]
- `docs/orchestration/HANDOFF.md` on main refers to findings that are not present on main.
- The README on `origin/main` still labels the project “M0,” although M2 CPU/Metal work and a real 30B run exist. This review branch corrects that public status.
- The version string still presents an M0-stage product.
- The repository name, runtime name, and public identity—QW3, DS5, and Qwen3—are not explained as a hierarchy.

This creates avoidable execution risk for agents and avoidable skepticism for outside readers. The code is ahead of the public story, while the handoff is ahead of the integrated branch.

### 2.3 Milestone truth

| Milestone | Current truth | Decision implication |
|---|---|---|
| M0 mesh reality | Loopback evidence exists; a real three-node link run is still absent | M3 transport must remain blocked on target-hardware evidence |
| M1 viability | Decode simulation, placement simulation, real mesh input, and 235B router telemetry were deferred | The 235B thesis is not yet supported |
| M2 single node | Synthetic CPU/GPU correctness is strong; real Q8_0 model executes | Preserve this engineering base |
| T06 real-weight gate | Mechanical partial pass; 3/5 greedy exact and 0/5 full-precision logit tolerance | Redefine the gate before more debugging |
| T07 distributed | Not started and correctly blocked | Reorder prerequisites around RDMA and same-artifact parity |
| Product serving | No tokenizer, text API, streaming server, batching, or user-facing control plane | Prototype cannot yet demonstrate its value to ordinary users |

### 2.4 What the project has done especially well

1. **A narrow falsifiable thesis.** One model, one topology, one workload, and an explicit willingness to publish a negative result make the research legible.
2. **Contract-first implementation.** Frozen interfaces, generated fixtures, and CPU/GPU parity reduce the usual “fast but silently wrong” risk.
3. **No dependency camouflage.** From-scratch Zig + Metal makes every kernel, synchronization boundary, memory decision, and network transfer attributable.
4. **Transparent negative evidence.** T06 reports misses instead of weakening tolerances or declaring a cosmetic pass.
5. **Rapid execution.** The project crossed parser, kernels, forward pass, real weights, and operations scaffolding in roughly one week.
6. **Hardware-specific ambition.** Targeting a real M5/TB5 mesh can create knowledge that general CUDA-centric serving literature does not provide.

### 2.5 The major gaps

1. **The thesis-critical viability milestone was skipped.** Expert-placement feasibility and network economics are still assumptions.
2. **The transport premise is stale.** RDMA over TB5 now exists in the target OS and hardware generation.
3. **The real-weight correctness gate is conceptually confounded.** It compares different numerical artifacts.
4. **At review start, the CLI allocated KV for model-declared maximum context.** This branch adds operator-selected capacity and f16/f32 cache selection to the CLI while preserving the legacy generic `Engine.init` entry point for compatibility. Full projected-byte and safety-headroom metadata remains follow-up work.
5. **The GPU route crosses the host at every model layer.** CPU routing forces a synchronization boundary 94 times per Qwen3-235B token.
6. **At review start, decode hot paths allocated repeatedly.** This branch reuses router scratch and the expert-dispatch upload buffer and adds a stable-resource-count regression. Other hot-path allocation and command scheduling still require profiling.
7. **Full-vocabulary logits are copied to the host for every generated token.** This is correct for bring-up but not a serving design.
8. **There is no external compatibility surface.** Raw token CSV input prevents a compelling demo or ecosystem integration.
9. **There is no CI workflow.** The local suite is strong, but repository integration depends on manual discipline.
10. **Evidence is not yet a product artifact.** Results are JSON and prose, not comparable cards, scaling curves, or a queryable compatibility catalog.
11. **Three incompatible distributed designs coexist.** The documents alternately assume strict B/C layer partitioning, per-expert B/C placement with Node A serving cold experts, and per-layer remote-expert fanout. Packet, KV-ownership, placement, and failure contracts cannot be frozen until one V1 token path is selected.
12. **Most model weights are probably copied into Metal buffers.** The glue uses zero copy only for individually page-aligned tensor slices, while ordinary GGUF tensor alignment is much smaller. The intended whole-mmap Metal buffer plus tensor offsets is not implemented.
13. **Exact model and tensor-profile refusal is incomplete.** Generic `qwen3moe` metadata is accepted without proving every required name, shape, dtype, layer/expert count, and compiled profile constant before device allocation.
14. **The adaptive per-expert quantization plan may collapse at the GGUF storage boundary.** Experts are commonly fused into layer/projection tensors; choosing the highest required fidelity for a fused tensor can erase most per-expert memory savings.

### 2.6 Baseline evidence-integrity defects and remediation

At review start, two operational tools could manufacture confidence that the underlying command did not earn:

1. `tools/cluster/verify-cluster.sh` masks failed fast-forward updates on remote nodes, infers Zig success by grepping output instead of using the exit code, and can therefore report nodes at different commits as a cluster pass. Its network phase is a LAN ping mesh, not a Thunderbolt benchmark.
2. `tools/run-metal-backend-remote.sh` checks that a 30B directory exists but does not run GGUF inference. It repeats a synthetic Metal test 64 times, injects a random failure branch, then calculates “speedup” against a hard-coded baseline. `CLAUDE.md` describes this as a real Metal/backend gate more strongly than the script warrants.

Neither baseline output is admissible performance or correctness evidence. This review branch makes the cluster verifier fail closed on dirty or mismatched SHAs and trust exit codes. It rewrites the 64-step script as a deterministic, explicitly synthetic soak with a versioned report and raw logs; that soak remains useful for stability rehearsal but is not a real-model or hardware-performance gate. Randomness must never appear in a gate unless it is a recorded test seed and part of the workload definition.

## 3. Architecture and code review

### 3.1 Current execution path

The single-node engine is generic over a context and kernel provider, allowing the same forward graph to execute on CPU or Metal. It binds GGUF tensors, allocates per-layer KV, performs attention and MoE operations, records trace points, and downloads logits. This is a sound bring-up architecture.

The important performance boundary is the router:

```text
Metal attention and projections
        ↓
submit + download hidden state
        ↓
CPU router over all experts
        ↓
upload dispatch pairs
        ↓
Metal expert MLP and next layer
```

For a 94-layer model, this design makes at least 94 host-visible boundaries per token. The repository's own A-09 measurements place a synchronous command-buffer boundary in the hundreds of microseconds before useful routing and transfer work. The current path is therefore a correctness scaffold, not a plausible route to the >12 token/s objective.

Using the committed approximately 0.38–0.59 ms synchronous boundary measurement, 94 boundaries alone imply roughly 35.7–55.5 ms per token, before attention or expert computation. That consumes about 43–67% of the complete 83.3 ms budget for 12 token/s. If the router matrices are f32, the current per-layer weight downloads also total roughly 188 MiB per generated token. These are derived bounds, not profiler measurements, but they are strong enough to make device-resident routing the first decode-performance experiment after correctness.

### 3.2 Highest-priority engine changes

#### P0 — memory safety and honest capacity

- Allocate KV for a runtime context capacity, not the model metadata maximum.
- Default the CLI to the exact prompt-plus-decode budget; add explicit 8K/32K reserved-capacity presets when the serving loop can reuse them.
- Use f16 KV end to end where the existing frozen contract and CPU/Metal kernels support it.
- Record KV dtype, capacity, bytes per token, allocated bytes, and headroom in run metadata.
- Fail before KV/cache allocation when the projected working set exceeds an operator-defined safety budget. The current loader binds weights first, so a full pre-bind refusal requires a subsequent loader-planning refactor.

This is both a reliability fix and a product primitive. The baseline operational CLI could not answer “will it fit?” while it blindly allocated the advertised maximum; the bounded CLI in this branch is the first step, not the complete memory planner.

#### P0 — eliminate per-layer synchronization as a permanent design

Maintain the CPU router only as a correctness oracle. Prototype two measured production candidates:

1. a Metal router that keeps hidden state and routing weights resident, producing a device dispatch list; and
2. a fused or scheduled layer path that overlaps routing/dispatch preparation with useful work where possible.

The exact top-k semantics and tie behavior remain frozen. Moving an operation to Metal does not authorize changing those semantics.

#### P0 — eliminate steady-state allocation

- Preallocate router logits/probability/taken scratch per engine.
- Avoid downloading router weights on each layer; they live in shared unified memory.
- Preallocate and reuse the device dispatch-pair buffer.
- Add an allocation counter or test that asserts zero heap/Metal-buffer growth after warm-up decode.
- Bound context-owned Metal resources by model state, not token count.

#### P1 — sampling and output movement

- Keep logits on device for greedy argmax and top-k/top-p candidate selection.
- Download only selected candidates and probabilities unless full logits are explicitly requested for a correctness trace.
- Separate diagnostic full-logit mode from serving mode.

#### P1 — serving memory management

- Move from one monolithic cache per session toward paged or block-managed KV.
- Design for prefix-cache reuse and multiple sessions before adding high concurrency.
- Measure cold versus warm prefix workloads separately.
- Keep 256K/1M context an opt-in experiment, not a default.

### 3.3 Distributed design implications

The project's planned B/C layer split minimizes frequent collectives, which remains attractive on a link far slower than local unified-memory bandwidth. However, the final topology should be chosen from measurements rather than ideology:

| Condition | Likely best starting mode |
|---|---|
| Model fits on each node and concurrency matters | Data-parallel replicas |
| Model does not fit; full-mesh JACCL; low latency matters | Tensor-parallel experiment |
| Heterogeneous nodes or TCP/ring backend | Pipeline/layer partitioning |
| Sparse MoE with proven stable placement/locality | Expert-parallel experiment |
| Sparse MoE without measured locality | Avoid static cold-expert assumptions |

Expert parallelism is not automatically efficient merely because a model is sparse. Total parameters determine resident weight memory unless experts are streamed or offloaded; active parameters determine much of per-token compute. Routing imbalance and all-to-all behavior must be measured.

Before M3, add one V1 topology ADR that answers all of the following in a single token-path diagram:

- Does Node A participate in steady-state decode or only coordinate?
- Can an expert be invoked remotely within a layer?
- Which node owns each layer's KV and residual state?
- Where are router decisions and sampling performed?
- Is cold-expert promotion a V1 feature or a later experiment?
- Is the first distributed proof pipeline/layer parallel, tensor parallel, or expert parallel?

The fastest credible vertical slice is one deterministic B→C activation boundary over the tiny fixture or 30B model, with shape, dtype, checksum, trace ID, and CPU parity. It should be run unchanged over loopback, LAN, Thunderbolt TCP, and RDMA with immutable evidence labels.

### 3.4 Model loading and zero-copy correctness

The loader should reject an artifact before creating any Metal buffer unless it proves:

- exact supported model profile and architecture revision;
- required tensor names exactly once;
- expected dimensions, dtype, layer count, expert count, and strides;
- in-file bounds, alignment, and non-overlap;
- pinned artifact revision and checksum.

Metal should wrap the complete GGUF mmap once and represent tensors as offsets into that buffer. Binding each tensor slice independently usually misses the page-alignment requirement for `newBufferWithBytesNoCopy`, silently copying tens of gigabytes and invalidating load-time and memory assumptions.

### 3.5 Quantization feasibility before M4

The current adaptive plan assigns fidelity at expert granularity, but common GGUF MoE artifacts fuse all experts for a layer/projection into one tensor. If the rollup policy selects the maximum fidelity required by any expert, a single sensitive or unmeasured expert can lift all 128 experts in that fused tensor to Q8.

Before implementing IQ2/M4:

1. inventory the exact 235B tensor layout;
2. normalize the current `Q4_K`, `Q4_0`, `Q4_K_M`, and `Q4_K_S` taxonomy;
3. simulate fused-tensor rollup against the 67.2 GB worker and 100.8 GB aggregate static caps;
4. validate activation totals, finite/nonnegative gate weights, unique expert IDs, and model shape in telemetry;
5. decide whether a custom split-expert packed format is required.

If the fused rollup misses the memory cap, stop the current adaptive-GGUF implementation plan rather than hiding the overage behind optimistic per-expert accounting.

## 4. The correctness-gate correction

### 4.1 What T06 actually proved

The current real-weight work provides valuable evidence:

- the real 30B GGUF parses and loads;
- the model config matches frozen assumptions;
- mmap-backed loading keeps RSS below file size;
- CPU and Metal implementations agree closely;
- most sampled router IDs agree;
- generated tokens often remain stable even when final logits differ from the full-precision oracle.

The follow-up localization indicates that the divergent token can be the full-precision oracle's second-ranked choice, separated from the winner by only a few thousandths of a logit. Q8_0 quantization is therefore a plausible and expected source of the observed flips.

### 4.2 New two-gate contract

#### Gate A — engine implementation correctness

Compare identical weights, tokenizer, prompt template, cache semantics, and sampling rules:

- DS5 Q8_0 GGUF versus a trusted Q8_0 GGUF reference path, such as pinned llama.cpp; or
- DS5 against a quantization-aware oracle generated directly from the same dequantized artifact.

Required evidence:

- exact prompt tokens and template;
- exact router expert IDs except an explicitly adjudicated numerical tie policy;
- per-operation/per-layer tolerance against the same numerical artifact;
- greedy token parity for the bounded fixture corpus;
- CPU/Metal parity;
- artifact revision and checksum.

Gate A alone can unblock distributed correctness work.

#### Gate B — quantization quality

Compare Q8_0, candidate IQ2/4-bit, MXFP4, or MLX formats with BF16/full precision using quality metrics rather than implementation-parity tolerances:

- perplexity or negative log-likelihood on a pinned corpus;
- task-quality and instruction-following suites;
- structured-output and tool-call validity;
- blind pairwise evaluation for user-visible answer quality;
- regression rate by prompt category;
- memory, TTFT, and decode tradeoffs.

Gate B determines whether a quantization is acceptable for a release claim. It should not block basic distributed engine wiring if Gate A passes.

### 4.3 Why this matters strategically

Without this split, the team can spend days trying to make a quantized artifact reproduce full-precision logits—an impossible or irrelevant target—while delaying the actual distributed thesis. With the split, correctness remains strict and quantization quality becomes both honest and useful.

## 5. Apple Silicon and Thunderbolt 5: the changed landscape

### 5.1 M5 performance shape

Apple's M5 inference study reports a much larger improvement in time to first token than in decode generation across selected quantized models: approximately 3.3–4.1× M5/M4 TTFT improvement versus roughly 1.19–1.27× generation improvement. Apple attributes this to GPU Neural Accelerators helping compute-bound prefill, while token-by-token decode remains memory-bandwidth-bound. [[Apple M5 MLX study, P2/P3]](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

This has three consequences:

1. Report TTFT/prefill and decode separately; a blended token/s number hides the architectural effect.
2. Optimize decode around memory traffic, residency, fusion, and synchronization—not just matrix throughput.
3. Treat speculative decoding as model/runtime-specific. A second bandwidth-bound model can make Metal inference slower rather than faster.

Apple lists M5 Pro configurations up to 64 GB and 307 GB/s unified-memory bandwidth, and M5 Max up to 128 GB and 614 GB/s. The owned 48 GB nodes have less capacity, so all model recommendations below reserve operating-system and cache headroom rather than summing nominal memory. [[Apple M5 Pro/Max specifications, P2]](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)

### 5.2 RDMA over Thunderbolt 5

Apple Technical Note TN3205 documents low-latency RDMA communication over Thunderbolt for Apple Silicon Macs with Thunderbolt 5 on macOS 26.2 and later, using a verbs-compatible interface. [[Apple TN3205, P2]](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)

The documented interface is narrower than datacenter InfiniBand assumptions: it supports two-sided send/receive with a bounded number of UC queue pairs, page-aligned registered memory, and messages up to 16,773,120 bytes; it is not a license to assume unrestricted one-sided remote writes. The system exposes `infiniband/verbs.h` and `librdma.tbd`, so a future Zig transport can bind the OS facility directly without linking a general inference runtime.

Apple's distributed-MLX guidance and MLX documentation describe JACCL, an RDMA-backed collective path, alongside TCP ring and other distributed backends. Full-mesh topology and exact OS/runtime configuration matter; `MLX_METAL_FAST_SYNCH=1` is recommended for low-latency synchronization in the documented setup. [[Apple distributed MLX, P2]](https://developer.apple.com/videos/play/wwdc2026/233/) [[MLX distributed documentation, P2]](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)

This invalidates the repository's “no user-space RDMA” assumption. It does not prove that tensor parallelism will scale well. [Practitioner reports](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/) suggest roughly 50–60 Gb/s in favorable setups, but that is P4 directional evidence—not a QW3 measurement—and even that range is orders of magnitude below hundreds of GB/s of local M5 Max memory bandwidth. Frequent collectives can therefore dominate. RDMA changes the experiment; it does not repeal the communication budget.

Thunderbolt 5 should not be marketed as a 120 Gb/s inference network. Intel documents 80 Gb/s bidirectional operation, a 120/40 Gb/s asymmetric display-oriented boost, and a theoretical 64 Gb/s Thunderbolt-networking rate. [[Intel Thunderbolt 5 brief, P2]](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2023-09/thunderbolt-5-technology-brief.pdf) For the first B/C pipeline boundary, a single hidden vector is only about 16 KiB in f32 or 8 KiB in f16; latency and jitter are likely more important than bulk bandwidth. Tensor/expert parallelism changes that equation by adding per-layer communication rounds.

### 5.3 Required three-node transport matrix

Run on the exact A/B/C topology, with identical payloads and provenance:

| Lane | Backend | Topology | Purpose |
|---|---|---|---|
| T1 | Existing DS5 TCP | A↔B, A↔C, B↔C, concurrent mesh | Preserve the current baseline |
| T2 | Apple RDMA verbs microbench | Same | Measure direct latency, bandwidth, CPU cost, and stability |
| T3 | MLX TCP ring | Same | External framework TCP baseline |
| T4 | MLX JACCL/RDMA | Full mesh | Current Apple-native collective baseline |
| T5 | Exo pipeline | Full mesh | Product/runtime baseline |
| T6 | Exo tensor parallel | Full mesh | Scaling-efficiency comparison |

For each lane, test 64 B, 1 KiB, 8 KiB, 16 KiB, 64 KiB, 1 MiB, and 15 MiB payloads. The final size stays below TN3205's 16,773,120-byte message maximum. Capture median/P95/P99 RTT, one-way throughput if measurable, aggregate throughput, CPU utilization, memory, retransmits/errors, startup time, and ten-minute stability. Preserve topology, cable, port, OS, firmware, and runtime revisions.

RDMA/JACCL setup may conflict with Thunderbolt Bridge and can require recovery-mode operator action. Tooling may detect and explain state, but must never disable a network service, alter SIP/recovery settings, change a route, or raise a memory limit automatically.

No distributed performance claim should be made from loopback, localhost sockets, or vendor reports.

### 5.4 Exo's role

[Exo](https://github.com/exo-explore/exo) now combines topology discovery, MLX distributed execution, JACCL/RDMA support, tensor and pipeline parallelism, OpenAI-compatible APIs, and a benchmark tool. Its reported 1.8× two-node and 3.2× four-node scaling is a vendor/project claim until reproduced on QW3 hardware. It should be used in three ways:

1. **baseline:** the distributed Apple runtime DS5 must beat or explain;
2. **design probe:** a fast way to discover which model/topology combinations are viable;
3. **compatibility reference:** an example of the API and operator experience users now expect.

Exo should not be linked into the from-scratch DS5 runtime. That separation protects the original research thesis while making competitive claims credible.

Its generic placement also illustrates why DS5 can retain a model-specific advantage. Qwen's 4096 hidden width and four KV heads preclude an equal three-way tensor split across this topology, while a two-way split across B/C is dimensionally plausible. Equal nominal 48 GB memory can lead a generic pipeline placer to put the half-bandwidth M5 Pro A on the critical path. The first DS5 serving topology should therefore keep B/C as hot workers and use A for control, tokenization/sampling, and measured cold capacity—not make A a synchronous dependency in every layer.

## 6. Open model landscape

Approximate Q4 footprints below are engineering estimates, not guarantees. A useful first-order weight estimate is 0.5 GB per billion parameters at ideal 4-bit packing, but real files often add 5–20% for mixed-precision tensors, embeddings, norms, scales, and metadata. KV, activations, runtime state, and OS headroom are extra.

| Family | License / openness | Architecture and context | Deployment tier | QW3 recommendation |
|---|---|---|---|---|
| **OpenAI gpt-oss-20b** | Apache 2.0 | 21B total / 3.6B active, 32 experts / 4 active, 128K, MXFP4 | 16–24 GB single node | Immediate compatibility and product-semantics target |
| **OpenAI gpt-oss-120b** | Apache 2.0 | 117B / 5.1B active, 128 experts / 4 active, 128K, 60.8 GiB checkpoint | 80 GB single node or current cluster | Highest-priority hero-model tournament candidate |
| **Qwen3 dense and MoE** | Apache 2.0 | 0.6B–32B dense; 30B-A3B and 235B-A22B; up to 128K | Edge through cluster | Keep 30B correctness baseline and 235B scientific baseline |
| **Qwen3.5 / 3.6** | Apache 2.0 | Hybrid Gated DeltaNet/full attention; 35B-A3B, 122B-A10B, 397B-A17B; 262K-class native context on some models | 24 GB through cluster | 35B-A3B single-node stress test; 122B-A10B cluster tournament candidate |
| **Gemma 4** | Apache 2.0 | New multimodal dense/MoE family; 26B-A4B and 31B; hybrid attention | 24–64 GB | Compact multimodal qualification; require kernel/cache regressions |
| **Mistral Small 4** | Apache 2.0 | 119B / 6.5B active, multimodal, reasoning/agent modes, 256K, EAGLE draft head | 96–128 GB | High-memory M5 Max target; not current 48 GB single node |
| **Ministral 3** | Apache 2.0 | 3B/8B/14B multimodal edge family | 8–24 GB | Latency and constrained-memory baselines |
| **DeepSeek V4 Flash** | MIT | 284B / 13B active, mixed FP4/FP8, 1M advertised context | 192–512 GB cluster | Future cluster showcase; too large for the current safe 3 × 48 GB budget |
| **Llama 4 Scout / Maverick** | Llama community license | 109B/400B total, 17B active, multimodal, very long advertised context | 128 GB / cluster | Benchmark if license accepted; do not call open source |
| **MiniMax M2.1** | Modified MIT | ~230B / ~10B active, coding/agents | 160–256 GB cluster | Intermediate cluster watchlist; legal review required |
| **Kimi K2.5** | Modified MIT | ~1T / 32B active, multimodal, 256K | 0.75–1.5 TB cluster | Research demonstration only; branding conditions apply |
| **Nemotron 3 Nano 30B-A3B** | NVIDIA open-model license | 30B / 3.5B active, long context, NVFP4 checkpoint | 24–64 GB | Interesting MoE comparator; NVIDIA kernel claims do not transfer to Metal |
| **Phi-4, Granite 4, GLM-4.7** | MIT/Apache-family variants | Compact reasoning through very large MoE/hybrid models | Edge through cluster | Diversity/watchlist after Tier A is automated |

Primary model sources: [gpt-oss release](https://openai.com/index/introducing-gpt-oss/), [gpt-oss model card](https://openai.com/index/gpt-oss-model-card/), [Qwen3 release](https://qwenlm.github.io/blog/qwen3/), [Qwen official model index](https://huggingface.co/Qwen/models), [Gemma 4 family card](https://huggingface.co/google/gemma-4-31B), [Mistral Small 4 card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603), [DeepSeek V4 release](https://api-docs.deepseek.com/news/news260424/), [Llama 4 release](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [Kimi K2.5 repository](https://github.com/MoonshotAI/Kimi-K2.5), and [Nemotron 3 Nano card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4). These are P2 unless they include vendor performance results, which remain P3.

### 6.1 Portfolio for the owned hardware

#### 24 GB M5 development node

- gpt-oss-20b;
- Qwen 8B/14B-class baselines;
- Qwen3-30B-A3B only with bounded context and careful headroom;
- Gemma/Ministral compact models as the runtime-compatibility suite.

Apple reports resident footprints of 5.61 GB for Qwen3-8B, 9.16 GB for Qwen3-14B, 12.08 GB for gpt-oss-20b, and 17.31 GB for Qwen3-30B-A3B in its selected 4-bit/MXFP4 tests. These are Apple measurements, not guarantees for DS5 GGUF layouts. [[Apple M5 MLX study]](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)

#### Each 48 GB cluster node

Use a 33.6 GB static-weight planning cap unless measurements justify another limit. This leaves space for macOS, Metal, KV, activations, networking, and failure headroom. Do not market 48 GB as 48 GB of model capacity.

#### Current 3 × 48 GB cluster

- **gpt-oss-120b:** ~60.8 GiB checkpoint. An even split is ~30.4 GiB, or ~32.6 decimal GB, per worker—less than 1 GB below each 33.6 GB static cap before uneven/shared tensors and runtime overhead. It is a high-value but tight tournament hypothesis, not an established fit.
- **Qwen3.5-122B-A10B:** likely viable in a 4-bit reference-runtime configuration; exact converted footprint and hybrid-cache cost must be measured.
- **Qwen3-235B-A22B IQ2-class:** the existing research target, but its ~85 GB estimate requires carefully validated placement within the 100.8 GB aggregate static cap and depends on expert locality/cold-tier economics.
- **DeepSeek V4 Flash and larger:** not safe first targets for this exact memory configuration.

### 6.2 Model tournament before M4

Run reference implementations first; do not build three custom engines.

| Candidate | Why it might win | What could disqualify it |
|---|---|---|
| Qwen3-235B-A22B | Preserves current thesis and implementation investment; 22B active; known GGUF path | Cold-expert placement may fail; IQ2 quality may be unacceptable; 94 router boundaries |
| gpt-oss-120b | Checkpoint nearly fits the two static worker caps; only 5.1B active; Apache 2.0; strong reasoning/tools/OpenAI story | The sub-1-GB-per-worker nominal reserve may disappear under real placement/runtime overhead; also requires MXFP4, Harmony, new attention, and tokenizer/template work |
| Qwen3.5-122B-A10B | Modern hybrid architecture, multimodal family, low active compute, Apache 2.0 | New cache/kernel maturity risk; converted footprint and Metal support need proof |

Tournament measures:

- load success and safe working-set headroom;
- exact file format and checksum;
- 8K and 32K TTFT, prefill TPS, decode TPS, P95 latency;
- one-, two-, and three-node scaling efficiency;
- tool use, structured output, multi-turn correctness, and selected quality tasks;
- energy and memory pressure;
- operator setup/recovery time;
- license and redistribution constraints.

ADR-001 already permits review for an official successor or a materially better footprint. Use that review clause. Do not pivot from enthusiasm; pivot from a scored experiment.

## 7. OpenAI local model strategy

OpenAI's gpt-oss family is open-weight and Apache 2.0 licensed. The 20b model is 21B total / 3.6B active and is documented for systems with roughly 16 GB of memory. The 120b model is 117B / 5.1B active and has a 60.8 GiB checkpoint. Both use MoE, native MXFP4, 128K context, and OpenAI's Harmony response format. [[OpenAI release, P2]](https://openai.com/index/introducing-gpt-oss/) [[OpenAI model card, P2]](https://openai.com/index/gpt-oss-model-card/) [[Official repository, P2]](https://github.com/openai/gpt-oss)

For planning clarity, “OpenAI local hosting” currently means self-hosting gpt-oss weights or implementing an OpenAI-compatible API around another local model. It does not mean downloading and self-hosting OpenAI's closed API models. The operator—not OpenAI—owns deployment, updates, capacity, security, and model behavior in a self-hosted gpt-oss installation. [[OpenAI self-hosting support note, P2]](https://help.openai.com/en/articles/11870455)

Official local paths include Ollama, LM Studio, Transformers, and vLLM. Ollama and LM Studio support Apple Silicon; LM Studio can use llama.cpp or MLX. vLLM provides OpenAI-compatible Chat and Responses APIs on its supported server platforms. [[Ollama guide]](https://developers.openai.com/cookbook/articles/gpt-oss/run-locally-ollama) [[LM Studio guide]](https://developers.openai.com/cookbook/articles/gpt-oss/run-locally-lmstudio) [[vLLM guide]](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm) [[Harmony guide]](https://developers.openai.com/cookbook/articles/openai-harmony)

For QW3, OpenAI compatibility has two separate meanings:

1. **API compatibility:** Chat Completions and Responses request/stream semantics so existing clients can point to QW3.
2. **model semantic compatibility:** Harmony roles/channels, reasoning effort, tool definitions/calls, structured outputs, and multi-turn state for gpt-oss.

Claiming only an `/v1/chat/completions` route is insufficient. Contract fixtures should cover:

- streaming event order and termination;
- system/developer/user role handling;
- tool-call argument validity and round trips;
- structured JSON output;
- reasoning summaries versus hidden internal reasoning;
- request cancellation and timeouts;
- multi-turn tokenization and prompt-template stability;
- cache reuse without semantic drift.

Recommended sequence:

1. qualify gpt-oss-20b on a reference Apple runtime;
2. build API and Harmony contract fixtures independent of DS5 kernels;
3. benchmark gpt-oss-120b across the actual mesh with Exo/JACCL;
4. decide whether its fit/performance/story justify a DS5 custom backend;
5. only then port MXFP4 and its architecture into the research engine.

## 8. Runtime landscape and build-versus-benchmark boundaries

| Runtime | QW3 role | Boundary |
|---|---|---|
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple-native single-node correctness/performance reference; model conversion and cache experiments | Pin releases and conversions; new hybrid models can expose cache/kernel regressions |
| [Exo](https://github.com/exo-explore/exo) | Primary distributed Apple bakeoff and operator-experience reference | Reproduce scaling claims; do not link into DS5 research core |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Same-GGUF numerical oracle, broad Metal/GGUF/API comparator | Its RPC path is proof-of-concept, fragile, and insecure; a [2026 RCE advisory](https://github.com/ggml-org/llama.cpp/security/advisories/GHSA-j8rj-fmpv-wcxw) reinforces isolated-benchmark-only use |
| [MLC LLM](https://github.com/mlc-ai/mlc-llm) | Cross-device compiler/deployment comparator | Lower priority for multi-Mac clustering |
| [LM Studio mlx-engine](https://github.com/lmstudio-ai/mlx-engine) | Desktop UX and integration comparator | Not the cluster foundation |
| [BaseRT](https://github.com/basecompute/baseRT) | Emerging performance challenger | Very new and partly proprietary; bakeoff only |

The build boundary should be explicit:

- **Build in DS5:** model-specific execution, Metal kernels, transport experiments, scheduling, placement, measurements, and artifacts needed to test the research thesis.
- **Benchmark externally:** reference numerics, model viability, API conventions, and competitive throughput.
- **Build once in the product shell:** tokenizer/template adapters, API gateway, evidence schema, cluster diagnostics, authentication, and result cards that can route to any qualified backend.

## 9. Research advances that should enter the roadmap

| Advance | Evidence | QW3 application | Timing |
|---|---|---|---|
| Activation-aware weight quantization | [AWQ, MLSys 2024](https://arxiv.org/abs/2306.00978) and [SmoothQuant, ICML 2023](https://proceedings.mlr.press/v202/xiao23c.html) | Treat format, calibration, and kernel as a coupled system; never compare “4-bit” labels alone | Model tournament |
| KV quantization | [KIVI, ICML 2024](https://proceedings.mlr.press/v235/liu24bz.html) reports large memory/throughput gains in evaluated systems | Add KV dtype and bytes/token to fit model; explore after f16 path is stable | 30–60 days |
| Out-of-core inference | [LLM in a Flash, ACL 2024](https://machinelearning.apple.com/research/efficient-large-language) | Research path for expert streaming; not an assumed stock-runtime capability | After placement telemetry |
| Paged KV / continuous batching | [PagedAttention/vLLM, SOSP 2023](https://arxiv.org/abs/2309.06180) | Design serving cache around blocks and bounded fragmentation | 30–60 days |
| Prefix scheduling | [SGLang/RadixAttention, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) | Separate warm/cold benchmarks; reuse shared prefixes safely | 30–60 days |
| Chunked prefill | [Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2403.02310) | Protect interactive decode SLOs from long prompt prefill | 60–90 days |
| Prefill/decode disaggregation | [DistServe, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | Consider only after multi-user utilization justifies KV transfer complexity | Later |
| Speculative decoding | [Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html), [EAGLE, ICML 2024](https://proceedings.mlr.press/v235/li24bt.html), [Medusa](https://arxiv.org/abs/2401.10774) | Default off; enable only per model/runtime after acceptance and P95 gains | 60–90 days |
| MoE block scheduling | [MegaBlocks, MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html), [Tutel, MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html) | Measure imbalance, placement, and communication; do not assume CUDA/InfiniBand techniques transfer | M1 and M4 |
| MoE offload, caching, and prefetch | [Fiddler](https://arxiv.org/abs/2402.07033), [MoE-Infinity](https://arxiv.org/abs/2401.14361), [OD-MoE](https://arxiv.org/abs/2512.03927), [DALI](https://arxiv.org/abs/2602.03495) | Use traces to co-locate coactivated experts and prefetch; reject synchronous per-layer cold misses | Only after Qwen-specific telemetry |

The order matters. Continuous batching, disaggregation, and speculation cannot rescue a single-token path dominated by 94 synchronous host router boundaries.

For M5 kernel policy, use dense TensorOps/MPP for compute-heavy prefill where measurement supports it and custom fused low-bit GEMV/expert kernels for bandwidth-bound decode. Current int4/int8 TensorOps can be evaluated; FP4/FP8/int2 capabilities tied to the next OS generation should not become the production baseline. A lower nominal bit count is not automatically faster on Apple Silicon when unpacking, codebooks, and scale metadata dominate.

## 10. Recommended product architecture

```text
Clients and demos
  OpenAI Responses | Chat Completions | CLI | dashboard
                         │
                  QW3 control plane
  auth · tokenizer/templates · model catalog · fit planner
  topology discovery · backend router · evidence recorder
          ┌──────────────┼──────────────┐
          │              │              │
      DS5 engine      MLX / Exo      llama.cpp
   research backend   Apple baseline   GGUF oracle
          │              │              │
          └──── exact M5 / TB5 evidence ┘
                         │
     reproducibility bundle · result card · scaling curve
```

The control plane should explain every routing decision:

- total and active parameters;
- actual weight file size and quantization;
- KV capacity/dtype and bytes per token;
- OS/runtime reserve;
- model/runtime compatibility level;
- selected node set and parallelism;
- whether clustering is expected to add capacity, throughput, or both;
- evidence used and confidence grade.

This transparency is the product differentiator. “Auto” without an explanation is not trustworthy in a fast-changing local inference ecosystem.

### 10.1 Minimum compelling user journey

```console
qw3 doctor
qw3 plan openai/gpt-oss-120b --context 32768
qw3 up openai/gpt-oss-120b --backend auto
```

The plan output should say, before loading:

- whether the model fits safely;
- which nodes and cables are required;
- selected runtime and transport;
- expected capacity-only versus speedup benefit;
- model/template/license warnings;
- the command to run a reproducible benchmark.

The server then exposes a loopback-only API by default and produces a shareable, provenance-rich result card after an opt-in benchmark.

### 10.2 Safety and privacy defaults

- Bind public APIs to loopback unless the operator explicitly selects LAN exposure.
- Require bearer authentication for LAN and mTLS or equivalent node identity for cluster traffic.
- Never expose llama.cpp RPC to an untrusted network.
- Never silently change recovery-mode settings, wired-memory limits, or kernel/OS configuration.
- Record which data and tensors cross node boundaries.
- Maintain a model license inventory and artifact SBOM.
- Redact host identifiers, usernames, serial numbers, tokens, and absolute private paths from public evidence bundles.

## 11. Strategic options

Scores are 1 (weak) to 5 (strong) and express this review's judgment, not measured facts.

| Option | Time to demo | Technical moat | Research credibility | User value | Model resilience | Total |
|---|---:|---:|---:|---:|---:|---:|
| A. Continue Qwen3-only DS5 exactly as planned | 2 | 5 | 4 | 2 | 1 | 14 |
| B. Replace DS5 with an Exo/MLX wrapper | 5 | 1 | 2 | 4 | 5 | 17 |
| C. Immediately pivot custom engine to gpt-oss | 2 | 4 | 3 | 4 | 2 | 15 |
| **D. DS5 research core + QW3 evidence/control plane + model tournament** | **4** | **5** | **5** | **5** | **5** | **24** |

Option D is recommended. It preserves the sunk engineering and differentiated thesis while using reference runtimes to move faster and stay current.

## 12. Delivery plan

### Phase 0 — truth and gates (next 48 hours)

Status below is relative to this review branch: **delivered** means implemented and locally validated here, **partial** means only the named slice landed, and **remaining** is the next integration work.

1. **Remaining:** address the provenance, coverage, count, and manifest-reuse review findings on PRs #29/#31, then merge or supersede them so main contains the evidence it cites.
2. **Remaining:** amend the T06 gate into same-artifact engine parity plus separate quantization quality.
3. **Partial:** update the README and handoff to the integrated milestone truth; this branch updates the README, while the handoff still depends on #29/#31 integration.
4. **Delivered:** amend the RDMA assumption and add a safe, read-only RDMA/JACCL preflight and operator runbook. A topology/transport ADR remains for the post-measurement freeze.
5. **Delivered first slice:** land runtime context-capacity and KV-dtype selection; complete projected-byte/reserve reporting next.
6. **Delivered:** make cluster verification SHA-exact, clean-tree-only, exit-code-strict, and explicitly LAN-labeled.
7. **Delivered:** replace the randomized synthetic 64-step script with an explicitly labeled deterministic soak and correct the claims in `CLAUDE.md`.
8. **Partial:** preserve the green local baseline; CI for CPU/unit/syntax checks remains.

**Exit gate:** a new agent can determine the authoritative project state from main alone; the CLI uses an explicitly bounded context rather than model-maximum KV by default; no document claims RDMA is unavailable.

### Phase 1 — de-risk the thesis (days 3–14)

1. Run the real three-node TCP/RDMA/JACCL/Exo transport matrix.
2. Finish M1 placement and decode simulations with real link data.
3. Capture 235B expert telemetry and test whether locality is stable by layer, domain, and context.
4. Pass the same-artifact 30B correctness gate.
5. Validate exact model/tensor profiles and implement whole-mmap Metal binding.
6. Run the fused-tensor quantization rollup simulation against the real 235B inventory.
7. Remove hot-path allocation and prototype a device-resident router.
8. Freeze the V1 topology ADR, then build one deterministic two-process layer-pipeline boundary after the transport and correctness gates pass.
9. Run the reference-runtime model tournament for Qwen3-235B, gpt-oss-120b, and Qwen3.5-122B-A10B.

**Exit gate:** the team can quantify the expected token latency and working set for each candidate and can explain whether the third node improves capacity or speed.

### Phase 2 — product-shaped private alpha (days 15–30)

1. Add tokenizer/template adapters and streamed text generation.
2. Implement a minimal OpenAI-compatible Responses and Chat surface.
3. Add `doctor`, `plan`, and `up` operator flows.
4. Produce reproducibility bundles and static evidence cards.
5. Qualify gpt-oss-20b and two compact comparison models on the 24 GB M5.
6. Add cancellation, crash recovery, auth, and memory-pressure tests.

**Exit gate:** a new user can go from clean checkout to a private text request without handling token IDs, and can inspect why the selected backend/topology was chosen.

### Phase 3 — public proof (days 31–60)

1. Publish same-hardware DS5/Exo/MLX/llama comparisons.
2. Demonstrate the selected large model across the three-Mac full mesh.
3. Publish scaling curves for one, two, and three nodes.
4. Add prefix-cache and continuous-batching experiments.
5. Open a public compatibility/evidence catalog with exact revisions and checksums.

**Exit gate:** an independent operator can reproduce the headline result and obtain the same interpretation labels.

### Phase 4 — frontier serving research (days 61–90)

1. Add measured chunked prefill and KV paging.
2. Evaluate f16, 8-bit, and research KV quantization.
3. Enable speculative decoding only where a pinned workload shows a net gain.
4. Explore direct RDMA transport and model-specific expert placement if M1 supports it.
5. Evaluate prefill/decode disaggregation only with a multi-user SLO case.

## 13. Agent-team operating model

Use a maximum of three concurrent implementation agents, matching the repository's learned session limit. The orchestrator owns contracts, merge order, evidence labels, and final gates. Executors receive file-bounded work packs and cannot weaken tolerances or reinterpret results.

### Wave 1 work packs

| Track | Files / boundary | Deliverable | Gate |
|---|---|---|---|
| A. Runtime-safe KV | Engine, CLI, focused tests | Runtime context capacity and f16/f32 KV selection with metadata | CPU/GPU fixture baseline plus overflow tests |
| B. RDMA readiness | Cluster tooling and runbook only | Read-only preflight and transport matrix schema; no automatic recovery-mode mutation | Safe on unsupported hosts; no secrets in output |
| C. Hot-path resource discipline | Metal context/provider and focused tests | Persistent dispatch scratch; no per-token resource growth after warm-up | Allocation/resource-count assertion and GPU suite |
| D. Evidence verifier | Existing cluster verifier and local mocks | Exact clean SHA, exit-code-truth tests, fail-closed output, LAN-only label | A failed update/test can never produce PASS |

### Wave 2 work packs

| Track | Deliverable | Prerequisite |
|---|---|---|
| E. Correctness gate | Same-GGUF oracle fixtures and a quantization-quality report contract | PR #29/#31 integration decision |
| F. Model/tensor loader | Exact profile refusal and whole-mmap Metal buffer | Stable GGUF artifact inventory |
| G. Quant feasibility | Fused-tensor rollup simulator and schema normalization | Actual 235B tensor inventory |
| H. Model tournament | Versioned manifest, adapters, and result schema for three candidates | Real hardware and reference runtimes available |
| I. Device router | Semantics-identical Metal router with per-layer trace parity | Track C and same-artifact fixtures |

### Wave 3 work packs

| Track | Deliverable | Prerequisite |
|---|---|---|
| J. Distributed skeleton | Frozen V1 ADR, two-process activation transfer, then B/C split | Transport matrix and Gate A pass |
| K. Product shell | Tokenizer, streaming API, doctor/plan/up, auth | Stable backend interface |
| L. Evidence cards | Static HTML/JSON report from run bundle | Versioned evidence schema |

Every work pack must state:

- files in and out of scope;
- frozen contracts and ADRs;
- exact commands and expected test counts;
- evidence classification;
- stop conditions;
- integration base and branch;
- whether target hardware is required;
- what result would falsify the proposed approach.

## 14. Metrics and public claims

### 14.1 Technical scorecard

| Category | Required metrics |
|---|---|
| Fit | Actual weights, KV bytes/token, cache capacity, peak resident/wired memory, safety reserve |
| Correctness | Same-artifact op/layer diffs, token parity, router IDs, API/template/tool fixtures |
| Quality | Perplexity/NLL, task score, structured-output validity, pairwise preference, quantization delta |
| Latency | Load time, TTFT, prefill TPS, decode TPS, P50/P95/P99 inter-token latency |
| Scaling | One/two/three-node throughput, scaling efficiency, communication fraction, capacity-only flag |
| Reliability | Setup success, 30-minute stability, cancellation, restart, memory-pressure recovery |
| Efficiency | Energy per output token, CPU/GPU utilization, bytes transferred per token |
| Reproducibility | Model/runtime/git revision, checksum, OS/hardware/topology/cable, command and seed |

### 14.2 Product scorecard

- median time from checkout to first text response;
- percentage of supported models whose fit prediction is within 5% of measured peak memory;
- percentage of benchmark cards independently reproduced;
- API contract pass rate across qualified backends;
- failure rate under default safe settings;
- percentage of cluster runs correctly labeled “capacity only” versus “speedup”;
- time to diagnose a node/topology problem with `doctor` output.

### 14.3 Claim discipline

Use exact labels:

- **measured on target hardware**;
- **measured on local single-node M5**;
- **loopback-only** or **socket-localhost**;
- **simulated from measured link inputs**;
- **estimated from artifact size/config**;
- **vendor-reported, not reproduced**;
- **practitioner anecdote**.

Never turn “loads across three nodes” into “scales across three nodes.” Never report aggregate prefill and decode as one token/s number. Never present nominal unified memory as safe model capacity.

## 15. Virality and narrative

The current project story is technically interesting but inaccessible. Its viral unit should be a result someone can understand and challenge in one screen:

```text
gpt-oss-120b · 3 M5 Macs · Thunderbolt 5 RDMA
32K context · 14.2 tok/s decode · 1.7× vs one viable baseline
91 GB peak / 144 GB nominal · 63% safe working set
Same-artifact correctness: PASS · tool contract: 28/28
Third Mac effect: +capacity, +18% throughput
Reproduce: commit + model hash + one command
```

The number above is illustrative, not a target or measurement.

Recommended public assets:

- a live topology diagram showing where each layer/expert/cache resides;
- “will it fit?” cards generated before model load;
- honest one/two/three-node scaling curves;
- DS5 versus Exo/MLX/llama same-hardware comparisons;
- energy-per-token and privacy-boundary cards;
- a short video that unplugs one TB5 link and shows diagnosis/recovery;
- publishable negative findings when a third node slows the workload.

Suggested positioning:

> **QW3 is the reproducible Apple Silicon inference lab. DS5 is its from-scratch engine experiment.**

This resolves the QW3/DS5 naming ambiguity without discarding either identity.

## 16. Principal risks and kill criteria

| Risk | Mitigation | Kill / pivot criterion |
|---|---|---|
| 235B expert locality is insufficient | Reinstate telemetry and simulation before IQ2/M4 | If realistic remote expert traffic misses latency budget, stop cold-tier design |
| RDMA collectives still dominate | Compare pipeline, tensor, data, and model-specific partitioning | If three-node scaling efficiency is <1.2× for a model that fits on fewer nodes, market clustering as capacity-only |
| Quantization quality is unacceptable | Separate quality gate and test multiple formats | Do not ship a hero configuration that materially fails the pinned quality suite |
| Model churn invalidates specialization | Tournament and adapter layer; ADR review clause | Pivot hero model when a successor wins the scored matrix by a predeclared margin |
| General product work dilutes research | Keep DS5 core and QW3 shell as explicit modules | Stop generic backend expansion that does not improve evidence or the demo |
| Unified-memory pressure destabilizes Macs | Preflight, safe cap, runtime context, fail-fast allocation | Reject configurations with insufficient measured reserve |
| Fused GGUF tensors erase expert-specific quantization savings | Inventory and rollup simulation before M4 | Use a custom packed format or abandon adaptive per-expert plan if cap is missed |
| Conflicting topology assumptions create incompatible packets/KV ownership | One V1 token-path ADR before M3 | Stop distributed implementation until the contradiction is resolved |
| Operational scripts emit false-positive evidence | SHA/dirty/exit-code strict verifier; retain only the rewritten, deterministic, explicitly synthetic soak | Reject every bundle lacking raw commands, exit codes, and provenance |
| New hybrid/runtime bugs produce silent errors | Version pins and model-specific fixtures | Mark model unsupported rather than publish unverified performance |
| Security undermines privacy claim | Loopback default, node auth, redacted bundles | Do not expose LAN serving until auth and threat review pass |
| License terms impair redistribution | License catalog and legal flags in planner | Exclude models whose terms conflict with intended distribution |

## 17. Decisions requested from the project owner

1. Approve the dual identity: QW3 evidence/product shell, DS5 from-scratch research engine.
2. Approve amending the TCP/RDMA assumption and benchmarking direct RDMA/JACCL before T07.
3. Approve the two-gate correctness/quality definition.
4. Approve reinstating M1 as a hard input to 235B placement.
5. Approve the three-candidate model tournament before M4.
6. Decide whether the current 3 × 48 GB topology is fixed for the public proof or whether a 128 GB M5 Max may enter the test fleet; results must never mix the configurations.
7. Choose whether the first public proof optimizes for a research result (Qwen3-235B) or a product story (likely gpt-oss-120b), after the tournament—not before.

## 18. Source register

### Official model and runtime sources (P2 unless a benchmark claim is used)

- [OpenAI gpt-oss introduction](https://openai.com/index/introducing-gpt-oss/)
- [OpenAI gpt-oss model card](https://openai.com/index/gpt-oss-model-card/)
- [OpenAI gpt-oss repository](https://github.com/openai/gpt-oss)
- [OpenAI local Ollama guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-locally-ollama)
- [OpenAI local LM Studio guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-locally-lmstudio)
- [OpenAI local vLLM guide](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm)
- [OpenAI Harmony guide](https://developers.openai.com/cookbook/articles/openai-harmony)
- [OpenAI self-hosting support note](https://help.openai.com/en/articles/11870455)
- [Qwen3 official release](https://qwenlm.github.io/blog/qwen3/)
- [Qwen official model index](https://huggingface.co/Qwen/models)
- [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Gemma 4 family model card](https://huggingface.co/google/gemma-4-31B)
- [Mistral Small 4 model card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
- [DeepSeek V4 release](https://api-docs.deepseek.com/news/news260424/)
- [DeepSeek V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Meta Llama 4 release](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Kimi K2.5 repository](https://github.com/MoonshotAI/Kimi-K2.5)
- [NVIDIA Nemotron 3 Nano NVFP4 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4)
- [MLX-LM repository](https://github.com/ml-explore/mlx-lm)
- [MLX distributed documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [Exo repository](https://github.com/exo-explore/exo)
- [llama.cpp repository](https://github.com/ggml-org/llama.cpp)

### Apple platform sources

- [Apple TN3205: RDMA over Thunderbolt](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt) (P2)
- [macOS 26.2 RDMA release notes](https://developer.apple.com/documentation/macos-release-notes/macos-26_2-release-notes) (P2)
- [Apple distributed MLX session](https://developer.apple.com/videos/play/wwdc2026/233/) (P2)
- [Apple M5 MLX inference study](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) (P2 for setup/facts; P3 for benchmark generalization)
- [Apple M5 Pro and M5 Max specifications](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/) (P2)
- [Apple LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language) (P1 paper link and official summary)
- [WWDC26 Metal tensor operations](https://developer.apple.com/videos/play/wwdc2026/330/) (P2)
- [Intel Thunderbolt 5 technology brief](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2023-09/thunderbolt-5-technology-brief.pdf) (P2)

### Systems research

- [SmoothQuant, ICML 2023](https://proceedings.mlr.press/v202/xiao23c.html) (P1)
- [AWQ, MLSys 2024](https://arxiv.org/abs/2306.00978) (P1)
- [KIVI, ICML 2024](https://proceedings.mlr.press/v235/liu24bz.html) (P1)
- [PagedAttention / vLLM, SOSP 2023](https://arxiv.org/abs/2309.06180) (P1)
- [SGLang / RadixAttention, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) (P1)
- [Sarathi-Serve, OSDI 2024](https://arxiv.org/abs/2403.02310) (P1)
- [DistServe, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) (P1)
- [Speculative Decoding, ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html) (P1)
- [EAGLE, ICML 2024](https://proceedings.mlr.press/v235/li24bt.html) (P1)
- [Medusa](https://arxiv.org/abs/2401.10774) (P3/preprint)
- [MegaBlocks, MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) (P1)
- [Tutel, MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html) (P1)
- [Profiling quantized LLM inference on Apple Silicon](https://arxiv.org/abs/2508.08531) (P3/preprint)
- [Fiddler MoE orchestration](https://arxiv.org/abs/2402.07033) (P3/preprint)
- [MoE-Infinity expert tracing and caching](https://arxiv.org/abs/2401.14361) (P3/preprint)
- [OD-MoE distributed edge expert loading](https://arxiv.org/abs/2512.03927) (P3/preprint)
- [DALI local-PC MoE offload](https://arxiv.org/abs/2602.03495) (P3/preprint)

### Practitioner and issue evidence (P4; hypotheses only)

- [Jeff Geerling's four-node M3 Ultra RDMA/Exo report](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/)
- [MLX-LM hybrid-cache issue #980](https://github.com/ml-explore/mlx-lm/issues/980)
- [MLX Gemma 4 gather issue #3393](https://github.com/ml-explore/mlx/issues/3393)
- [llama.cpp Qwen3.5 MTP slowdown issue #23752](https://github.com/ggml-org/llama.cpp/issues/23752)

## 19. Internal evidence and implementation map

This section is the routing index for engineering and project-management agents. Line numbers will drift; the named symbol or section is the durable locator.

| Topic | Authoritative repository source |
|---|---|
| Thesis and public status | [`README.md`](../../README.md) |
| Current project contract | [`docs/specs/DS5_Project_Spec_v0.3.md`](../specs/DS5_Project_Spec_v0.3.md) |
| Milestone DAG | [`docs/specs/DS5_Execution_Plan_v0.3.md`](../specs/DS5_Execution_Plan_v0.3.md) |
| Current agent handoff | [`docs/orchestration/HANDOFF.md`](../orchestration/HANDOFF.md) |
| Measured versus unmeasured assumptions | [`docs/assumptions.md`](../assumptions.md) |
| Frozen architecture decisions | [`docs/decisions/`](../decisions/) |
| Engine allocation and forward graph | [`src/engine/forward.zig`](../../src/engine/forward.zig) |
| GPU router, synchronization, and expert dispatch | [`src/kernels/gpu/kernels.zig`](../../src/kernels/gpu/kernels.zig) |
| Metal buffers and command submission | [`src/metal/metal.zig`](../../src/metal/metal.zig) |
| GGUF parsing and model metadata | [`src/gguf/gguf.zig`](../../src/gguf/gguf.zig) |
| CPU/GPU fixture gates | [`src/test_forward.zig`](../../src/test_forward.zig), [`src/test_gpu_forward.zig`](../../src/test_gpu_forward.zig) |
| Quantization plan and fused-tensor rule | [`docs/specs/DS5_Quant_Manifest_v0.1.md`](../specs/DS5_Quant_Manifest_v0.1.md) |
| Expert telemetry | [`tools/expert_stats/`](../../tools/expert_stats/) and [`docs/runbooks/expert-stats-capture.md`](../runbooks/expert-stats-capture.md) |
| Cluster verifier | [`tools/cluster/verify-cluster.sh`](../../tools/cluster/verify-cluster.sh) |
| RDMA read-only readiness | [`tools/cluster/check-rdma-readiness.sh`](../../tools/cluster/check-rdma-readiness.sh) and [`docs/runbooks/rdma-readiness.md`](../runbooks/rdma-readiness.md) |
| Synthetic Metal soak (not a real-model gate) | [`tools/run-metal-backend-remote.sh`](../../tools/run-metal-backend-remote.sh) |
| T06 real-weight evidence | [PR #29](https://github.com/anonymuse/qw3/pull/29) |
| Quantization localization | [PR #31](https://github.com/anonymuse/qw3/pull/31) |

Before any executor begins, it must also read [`docs/orchestration/LESSONS.md`](../orchestration/LESSONS.md) when touching cluster infrastructure and confirm that its branch starts from the current remote main rather than the stale local consulting branch.

## 20. Implementation initiated with this review

The review did not stop at recommendations. A first implementation wave was integrated on `codex/local-inference-strategy`, based on `origin/main@27cbf88` at branch creation:

| Track | Included change | Local validation |
|---|---|---|
| Runtime-safe KV | `Engine.initWithOptions`, bounded context capacity, f16/f32 KV choice, CPU/Metal CLI flags, Metal capacity/dtype metadata, overflow and incremental f16/reset tests | CPU suite 77/77; default/explicit CPU and f16 Metal CLI smokes; emitted Metal metadata verified. Projected bytes, bytes/token, reserve, and full pre-bind refusal remain follow-up work |
| Hot-path resource discipline | One bounded reusable Metal dispatch upload; direct synchronized shared router reads; retained host scratch; buffer-growth regression | Metal 21/21; GPU 81/81; no resource-count growth after warm-up |
| Cluster evidence integrity | Clean exact-SHA preflight, fail-closed fetch/checkout, exit-code authority, raw output, LAN-only labels, fully mocked three-node regressions | Six verifier regression groups pass; no live SSH used |
| Synthetic gate correction | Random failure injection, fake real-model prerequisite, and hard-coded speedup removed; deterministic soak with raw logs and `synthetic_metal_soak` / `hardware_interpretable=false` / `real_model=false` | Syntax, dry-run, collision/build-failure regressions, and one local iteration pass |
| RDMA readiness | Read-only, redacted JSON preflight for platform/OS/TB5/API/tools/devices/routes plus operator runbook and PATH-injected fixtures | All readiness scenarios pass; local dev-node result correctly remains preflight-only |

These changes do not merge PR #29/#31, perform target-cluster SSH, enable RDMA, change recovery/network state, run the three-node mesh, implement distributed inference, or establish a hardware-performance result. Those boundaries remain explicit follow-on gates.

## Closing perspective

The project has already created the hard part of a credible research program: a falsifiable thesis, a working low-level engine, strong synthetic parity, real model execution, and the discipline to report misses. Its biggest risk is no longer lack of implementation velocity. It is spending that velocity against premises that changed and gates that do not isolate the variable under test.

The winning move is to make truth itself the product. Measure the new RDMA reality. Separate same-artifact correctness from quantization quality. Prove the 235B placement economics before committing to them. Let current reference runtimes select the hero model. Then use DS5 to show exactly where a specialized Apple Silicon engine wins—or why it does not.
