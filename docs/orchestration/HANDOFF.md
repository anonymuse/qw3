# DS5 Orchestration Handoff (2026-07-13)

**Status:** Active orchestration document  
**Updated:** 2026-07-13 ~14:30  
**Executor:** planning-architecture-review-b561d6  
**Governing specs:** DS5_Execution_Plan_v0.3.md, DS5_Project_Spec_v0.3.md, ADR-001..006  

---

## Executive Summary

Main branch (7ca4dbb) is green: 74/74 tests pass. M2a CPU forward (T04) merged; M2b GPU forward (T05) actively running under Sonnet executor (agentId: a43664913d9aca929). Critical blocking dependencies for T06/T07 are the 30B GGUF model (~32GB, placed at `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/`) and completion of T05. **Next executable step after T05:** T06 (real-weights correctness gate) when 30B GGUF lands.

---

## Execution State (as of 2026-07-13 ~14:30)

### Completed Tasks

| Task | Milestone | Status | Merged | Notes |
|---|---|---|---|---|
| T01 | M2 GGUF parser | ✅ DONE | 2026-07-12 | Zig GGUF loader + oracle fixture format (DS5T) |
| T02 | M2 attention kernels | ✅ DONE | 2026-07-12 | GQA + RoPE + (per ADR-005 amendment) f16 KV load-to-f32 compute |
| T03 | M1 viability model | ⏸️ DEFERRED | — | Owner deferred 2026-07-12; M1 data capture (routing telemetry) owns this |
| T04 | M2a CPU forward | ✅ DONE | 2026-07-13 13:23 | Engine + trace hook; 74/74 tests green |

**Merged commit:** 7ca4dbb  
**Test suite:** 74/74 pass (CPU forward verified against oracle fixtures)  
**Code freeze:** Interface frozen per ADR-005 (`src/shared/contracts.zig`)

### Running Executors

| Task | Milestone | Executor | Agent ID | Status | Deadline |
|---|---|---|---|---|---|
| T05 | M2b GPU forward | Sonnet | a43664913d9aca929 | RUNNING ~14:00 | Unblock T06 after 1–2 days or escalate |

**T05 scope:** Metal kernels for Q8_0 dequant+matmul, FP32-accumulate, f16 KV load (ADR-005 amendment), GPU fused expert MLP  
**T05 gates:** GPU forward output == CPU forward output, deterministic under oracle fixtures  
**T05 blockers:** None (CPU forward provides oracle; kernels ship with tests)  
**T05 expected output:** Green GPU tests, merged into integration branch

### Pending (blocked on T05 + 30B GGUF)

| Task | Milestone | Scope | Unblock condition |
|---|---|---|---|
| T06 | M2 real-weights | Qwen3-30B-A3B forward on real weights | T05 merged + 30B GGUF at `~/ds5-models/` |
| T07 | M3 distributed | 30B-A3B split across B/C over M0 transport | T06 gate + distributed test harness |

---

## Execution DAG and Dependencies

```
M0 (mesh reality)
  ├─ Link benchmarks + metadata
  └─ Model downloads (background task, user-triggered)
        ├─ 30B GGUF (~32GB) → T06 unblock
        └─ 235B GGUF (~85GB, no binary link) → M1 telemetry capture

M1 (viability model) [T03 deferred; M1 data capture = routing telemetry]
  ├─ Input: 235B router-calibration corpus + per-layer expert-usage JSON
  ├─ Output: docs/findings/f001 (projected tok/s ceiling decomposition)
  └─ Gate: go/no-go vs >12 tok/s target

M2 (single-node engine core)
  ├─ T01 ✅ GGUF parser
  ├─ T02 ✅ Attention kernels (GQA + RoPE)
  ├─ T04 ✅ CPU forward (engine + all kernels)
  ├─ T05 🔄 GPU forward (Metal Q8_0/matmul/MLP)
  └─ T06 ⏳ Real-weights gate (30B-A3B on real weights, all paths GPU)
       Gate: 30B output == oracle 30B (deterministic)

M3 (distributed correctness) [T07]
  ├─ Input: T06 + distributed transport harness
  ├─ Scope: 30B-A3B across B/C with packets + checksums
  └─ Gate: Distributed output == single-node output

M4 (235B placement + runtime) [stretch after M3]
  ├─ Placement/quant manifests
  ├─ I-quant dequant kernels
  ├─ Tiered expert residency + promotion
  └─ 8K/32K benchmarks vs >12 tok/s target

M5 (findings) [continuous]
  └─ docs/findings/ write-ups, README
```

---

## Blocking Dependencies for Next Phase

### 1. T05 Completion (in progress)

**What:** Metal GPU kernels (Q8_0 dequant, matmul, FP32-accumulate, fused MLP) match CPU oracle  
**Owner:** T05 executor (Sonnet, agentId a43664913d9aca929)  
**ETA:** 1–2 days from spawn (~2026-07-14/15)  
**Merge gate:** 74/74 GPU tests pass, GPU outputs bit-identical to CPU under deterministic seed  
**Action if blocked:** Escalate to project owner; check for Metal/MSL or fixture mismatch issues

### 2. 30B GGUF Arrival

**What:** Qwen3-30B-A3B-Instruct-2507 Q8_0 GGUF (~32GB)  
**Expected location:** `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/`  
**Download tool:** `./tools/download_models.sh` (user-triggered, runs on a worker node with >150GB free disk)  
**Status:** ⏳ PENDING (user responsibility per M0 runbook)  
**Critical for:** T06 unblock (real-weights correctness gate)  
**Action:** Unblock T06 executor immediately upon arrival

---

## Next Executor Checklist (T06 = Real-Weights Gate)

When 30B GGUF arrives and T05 merges:

1. **Verify T05 merged:** Check `git log main | head` contains T05 commit  
2. **Verify 30B GGUF:** Confirm `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/` exists and is readable  
3. **Run T06 scope:**  
   - Load 30B-A3B from real weights  
   - Run forward pass (all GPU paths, Q8_0 dequant path, MLP path, router path)  
   - Verify output == oracle 30B fixture (per `tests/fixtures/synthetic/model.gguf`)  
   - Verify determinism under `--seed` flag  
4. **Merge criteria:**  
   - New test suite `zig build test` with real-weights harness passes 100%  
   - No regression in CPU forward tests  
   - CI (if wired) is green  
5. **Next handoff:** After T06 merges, unblock T07 (M3 distributed) with distributed transport harness  

---

## Decisions Locked (No Re-litigation)

| ADR | Scope | Decided | Reference |
|---|---|---|---|
| ADR-001 | Model selection (Qwen3-235B-A22B + 30B-A3B bring-up) | 2026-07-08 | Binding |
| ADR-002 | Kernel strategy (zero linked ML library, oracle fixtures, no route-through-A) | 2026-07-08 | Binding |
| ADR-003 | Bring-up model (30B-A3B only, dense baseline cut) | 2026-07-08 | Binding |
| ADR-004 | Auxiliary hardware (no Mac minis, RTX box, or foreign nodes) | 2026-07-08 | Binding |
| ADR-005 | Interface freeze (contracts.zig, dtype set, kernel API, GGUF API, fixture format) | 2026-07-12 | Binding; amended 2026-07-13 for KV f16 |
| ADR-006 | MPP scope (matmul2d in scope for prefill/batch GEMM only) | 2026-07-13 | Binding |

**Amendment to ADR-005 (2026-07-13 13:30):** KV cache dtype frozen to **f16** (rationale: M3 inter-node decode bandwidth at 32K context). Attention loads f16 into f32 registers for computation. Existing f32 fixtures remain; tests adapt as T05 updates kernel reads.

---

## Known Issues & Landmines

| Issue | Scope | Mitigation |
|---|---|---|
| `zig build test` cosmetic failure line | Zig 0.16 wart; tests write to stderr | Build Summary + exit code 0 are ground truth; ignore "failed command:" |
| Worktree branch trap | Easy to commit on wrong branch | Always `git branch --show-current` before committing |
| Metal link flags required | Build issue | `-lobjc -framework Metal -framework Foundation -framework CoreGraphics` must be in build.zig |
| A-09: dispatch overhead unbounded | Performance risk | Microbench per-layer dispatch before M2 kernel design freeze; target <380 µs/layer measured |
| Fixture f32 vs f16 mismatch | T05 concern | Tests use f32 fixtures; T05 kernels read f16 KV; tolerance rule must adapt or new f16 fixtures required |

---

## Assumptions Still Unmeasured (Assumptions Ledger)

| Assumption | Measurement | Trigger |
|---|---|---|
| A-01: Node bandwidth (A:307GB/s, B/C:614GB/s) | M0 mesh run | User responsibility |
| A-02: TB5 link latency & jitter | M0 bench link (loopback done, mesh pending) | User responsibility |
| A-04: Expert routing skew enables tiering | M1 telemetry capture (235B corpus) | T03 deferred; blocking M1 findings |
| A-06: 33.6GB/node cap leaves runtime headroom | Real Metal heap behavior at 32K context | Measure during T06/T07 |
| A-09: Per-layer dispatch overhead | Microbench on real cluster node | Before M2 kernel freeze (do before T06) |

---

## Recommended Reading Order

For next executor:

1. **This file** (you are here) — handoff summary, execution state, DAG  
2. `docs/specs/DS5_Execution_Plan_v0.3.md` — milestone definitions and gates  
3. `docs/specs/DS5_Project_Spec_v0.3.md` — architecture, constraints, goals  
4. `docs/decisions/ADR-001..006.md` — locked trade-offs and rationales  
5. `docs/reviews/2026-07-12_airplane_arch_reviews_response.md` — what we adopted and rejected  
6. `docs/work-packs/2026-07-12-review-incorporation/*.md` — WP-1..3 deliverables  
7. `docs/backlog/DS5_Phase2_Optimization_Backlog.md` — deferred optimizations and when to revisit  

---

## Integration Playbook

When T05 completes and 30B GGUF arrives:

```sh
# Step 1: Verify T05 merged
git log main --oneline | head -3
# Expect: T05 commit visible

# Step 2: Verify 30B GGUF
ls -lh ~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/
# Expect: model.gguf ~32GB + metadata

# Step 3: Spawn T06 executor (real-weights correctness gate)
# - Load 30B from ~/ds5-models/
# - Forward pass GPU path vs oracle fixture
# - Determinism check
# - Merge when tests green

# Step 4: After T06 merges, spawn T07 (M3 distributed)
# - Transport + packet contracts
# - 30B across B/C
# - Checksum validation
```

---

## Roll-Forward Plan (If T05 Stalls)

If T05 is blocked >48 hours after spawn:

1. Check Metal/MSL compiler errors in T05 agent logs  
2. Escalate to project owner with:
   - T05 working tree state (`git status`)
   - Last successful test run output  
   - Any Zig/Metal/macOS toolchain issues  
3. Owner decision: pivot to simpler GPU path, or escalate toolchain issue  

If 30B GGUF doesn't arrive within 2 days of T05 merge:

1. Check `~/ds5-models/` for partial download or errors  
2. Verify download tool: `./tools/download_models.sh` ran without errors  
3. Owner escalation: check disk space, network, or mirror availability  

---

## Success Criteria for This Handoff

✅ **This phase (planning-architecture-review) is complete when:**

1. This handoff document is committed to the integration branch  
2. T05 executor can reference this document as a single source of truth  
3. No ambiguity about next steps (T06 scope, gates, blockers)  
4. ADRs 001–006 are all filed and current  
5. Integration branch is ready for T06/T07 executors  

---

**Co-authored by planning-architecture-review-b561d6 (2026-07-13)**  
**Next handoff:** T06 real-weights gate executor
