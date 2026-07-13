# Review Response: Gemini Architecture Review (2026-07-10)

**Status:** Dispositioned
**Input:** External architecture review ("Gemini-arch-review 20260710"), received 2026-07-10
**Output:** `docs/specs/DS5_Project_Spec_v0.3.md` (adopted items folded in)
**Governing decisions unchanged:** ADR-001..005

## Summary judgment

The review contains three kinds of content: (a) background on antirez's "ds4" engine
that we cannot verify from its citations and do not depend on, (b) factual errors about
Qwen3-235B-A22B and about this project's architecture, and (c) a small set of sound
engineering principles that DS5 either already practices or should state explicitly.

No locked decision changes. Four items are adopted as explicit spec language; the rest
are rejected or remain deferred with reasons below.

## Factual corrections

| Review claim | Reality |
|---|---|
| Project is "qw4" | Project is DS5 (repo `qw3`) |
| Qwen3: 48 layers, 512 experts, 1M context | Qwen3-235B-A22B: 94 layers, 128 experts, top-8 active, 262K native context, 4 KV heads (GQA) |
| System is a "shared-nothing logits ensemble" | Category error — see R1 |
| 17.89× speculative-decoding speedup | Unsourced; no citation resolves to this number |
| ds4/DwarfStar background (DeepSeek v4 Flash 284B, RDMA Mac clusters) | Unverifiable from the given links; treated as color, not evidence. DS5 takes no dependency on it |

## Dispositions

### Rejected

**R1 — Shared-nothing logits ensemble / PackLLM fusion / STM token selection.**
The review's central frame assumes multiple nodes each running an *independent model*
whose output distributions are ensembled. DS5's premise is the opposite: no single
48GB node can hold Qwen3-235B, so there is nothing to ensemble. DS5 partitions **one
model** (experts and layers across B/C) and preserves exact top-8 routing semantics
(binding constraint, v0.2/v0.3). Logit fusion of partial models is not meaningful.
Everything downstream of this frame (asynchronous perplexity re-weighting, PRM
filter-gating interrupts) is rejected with it.

**R2 — Bypass XNU sockets via IOKit kernel driver / raw TB5 DMA.**
Contradicts the deliberate raw-libc I/O layer (`src/shared/sys.zig`) and inverts the
project's measure-first rule. M0 already produced real link numbers over the socket
path; M1 decode-sim determines whether transport RTT is even the binding constraint
before any transport heroics. Kernel extensions are effectively unavailable on modern
macOS, and DriverKit does not expose arbitrary TB5 DMA to userspace. A bounded
contingency ladder is now documented in the spec (§8): libc TCP (current, measured) →
`Network.framework` custom framing (only if decode-sim shows an RTT-bound ceiling below
target) → never a kernel driver.

**R3 — 1M-token context / ring buffer sized for 1M upfront.**
Out of scope. First performance context is 8K–32K; 64K stretch; 128K+ is research-gated
(v0.2 §5, v0.3 constraint). The *mechanism* (ring-buffered KV pages, pointer-advance
pruning) is adopted as A2 below at DS5's actual context budgets.

### Deferred (unchanged from v0.3)

**D1 — Speculative decoding (tree or linear, drafter on Node A).**
Already explicitly deferred (v0.3 change #5). Remains a post-M4 lever, contingent on
M4 landing and on measured verify-batch scaling on M5 Max — not on the review's
unsourced 17.89× figure. Node A's spare capacity is real; the sequencing decision
stands: correctness gates first.

**D2 — Agentic harness / PRM-style output gating.**
v0.2 Phase 5 remains deferred. Any future output-quality gating happens in the harness,
not as "hardware interrupts" in the decode loop.

### Adopted (folded into Project Spec v0.3, §7)

**A1 — Boot-time fixed memory budgets.** All steady-state decode allocations are made
at load from fixed per-node budgets (the existing 33.6GB static / 14.4GB reserve rule);
the decode hot path performs zero heap allocation. Was implicit in the 70/30 rule; now
a stated runtime principle.

**A2 — Ring-buffered KV pages.** KV cache lives in preallocated layer-owned pages;
context pruning advances page pointers rather than freeing memory. Consistent with the
existing KV page-table design; now stated.

**A3 — Sparse wire payloads.** Nothing crosses the mesh at full width when a top-k
projection suffices: routed-expert activations only (never all-expert), top-8 IDs +
gate weights, and top-K logits (K ≤ 64) if the sampling owner is ever remote from the
final layer. Decode-sim (M1) quantifies payload sizes; the one-packet-per-destination
rule (M3) is unchanged.

**A4 — Comptime specialization with load-time verification.** Qwen3-235B constants
(94 layers, 128 experts, top-8, hidden 4096, GQA 64/4/128) are baked at comptime per
ADR-005 contracts — and the GGUF loader refuses any file whose metadata disagrees with
the compiled constants. The review itself carried wrong constants; that failure mode
(plausible-but-wrong external specs) is exactly what the check guards against.

## Note on the ds4 lineage

If antirez's ds4 exists as described, it is prior art worth citing in the final
findings write-up — the shared-nothing framing differs from DS5's partitioned-model
design, and the contrast is clarifying. Verify the repo firsthand before citing;
nothing in DS5's plan depends on it either way.
