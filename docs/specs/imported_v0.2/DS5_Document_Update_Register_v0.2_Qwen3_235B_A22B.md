# DS5 Document Update Register v0.2: Qwen3-235B-A22B

**Document type:** Documentation control register  
**Status:** Planning baseline update  
**Date:** 2026-07-08  
**Model decision:** Qwen3-235B-A22B is the DS5 target model

---

## 1. Required documentation updates

| Document | Type | Current issue | Required update | Priority |
|---|---|---|---|---:|
| `DS5_Composite_Architecture_Planning_Model.md` | System architecture | Uses “Qwen3-235B-A22B-class” wording but should now reflect an exact locked model | Update status, decision section, model section, execution plan assumptions, and document references | P0 |
| `JW4_project_brief.md` | Project brief | Describes Gemma, 2-node, learning-engine scope | Supersede or split into separate precursor; do not leave as active DS5 target spec | P0 |
| `ChatGPT DS5 Analysis.md` | Source analysis | Contains the selected model recommendation and math | Archive as source analysis; link from ADR-001 and architecture baseline | P1 |
| `Deepseek DS5 Analysis.md` | Source analysis | Recommends DeepSeek-V3/R1 final target, conflicting with ADR-001 | Archive as alternative study; retain PDD/hot-expert ideas only | P1 |
| `Gemini DS5 Analysis.md` | Source analysis | Recommends Mixtral/static expert split, conflicting with ADR-001 | Archive as alternative study; retain useful diagram/implementation prompts only | P1 |
| `Z.AI DS5 Analysis.md` | Source analysis | Recommends Mixtral but flags per-token streaming risk | Archive as alternative study; retain strict streaming warning | P1 |

---

## 2. New required documents

| New document | Purpose | Owner | Status |
|---|---|---|---|
| `ADR_001_Model_Selection_Qwen3_235B_A22B.md` | Locks model target | Architecture | Generated |
| `DS5_Project_Spec_v0.2_Qwen3_235B_A22B.md` | Updates project scope and acceptance criteria | Project | Generated |
| `DS5_System_Architecture_v0.2_Qwen3_235B_A22B.md` | Updates architecture diagrams and views | Architecture | Generated |
| `DS5_Model_Runtime_Placement_Spec_v0.2_Qwen3_235B_A22B.md` | Defines placement, quant, KV, transport rules | Runtime | Generated |
| `DS5_Benchmark_and_Acceptance_Spec_v0.2_Qwen3_235B_A22B.md` | Defines benchmarks and phase gates | Benchmarking | Generated |
| `DS5_Risk_Register_v0.2_Qwen3_235B_A22B.md` | Tracks risks | Project/architecture | Generated |
| `DS5_Execution_Plan_Input_v0.2_Qwen3_235B_A22B.md` | Seeds execution plan | Planning | Generated |
| `DS5_README_Qwen3_Document_Update_Pack.md` | Index for generated pack | Documentation | Generated |

---

## 3. Specific edits for existing architecture baseline

Apply these changes to `DS5_Composite_Architecture_Planning_Model.md`:

| Section | Edit |
|---|---|
| Header | Change status to `Approved planning baseline pending Phase 0 measurements` |
| Executive architecture recommendation | Replace “Qwen3-235B-A22B-class” with `Qwen3-235B-A22B-Instruct-2507` as primary target |
| Core decisions | Add `Decision locked by ADR-001` |
| Source synthesis | Mark DeepSeek, Gemini, and Z.AI proposals as archived alternatives, not active target candidates |
| Architecture scope | Add `Qwen3-235B-A22B-Thinking-2507` as deferred optional variant |
| Model dimensions | Add exact model metadata and source revision/hash fields |
| Runtime view | Change local router mirrors from optimization idea to required performance-mode design after correctness mode |
| Storage view | Re-state that NVMe is never the steady-state active-weight path |
| Execution plan | Align phases with `DS5_Execution_Plan_Input_v0.2_Qwen3_235B_A22B.md` |
| Risks | Import risk IDs from the v0.2 risk register |

---

## 4. Specific edits for JW4 brief

Apply one of the following mutually exclusive updates to `JW4_project_brief.md`.

### Option A: Archive JW4 as precursor

Add to the top of the file:

```markdown
> **Status:** Archived precursor brief. This document describes the earlier Gemma/JW4 learning-engine scope. It is not the active DS5 target after ADR-001 selected Qwen3-235B-A22B.
```

### Option B: Split JW4 as a separate project

Add to the top of the file:

```markdown
> **Status:** Separate project track. JW4 remains a Gemma-focused learning engine and is not part of the DS5 Qwen3-235B-A22B execution plan unless explicitly scheduled as a precursor.
```

Recommended action: **Option A** unless there is dedicated capacity to run both tracks.

---

## 5. Documentation acceptance checklist

| Check | Required state |
|---|---|
| Model target appears in every active spec | `Qwen3-235B-A22B-Instruct-2507` |
| Optional thinking variant clearly deferred | Yes |
| DeepSeek/Kimi full checkpoints rejected as final target | Yes |
| Mixtral marked intermediate/comparator only | Yes |
| Dense 32B-70B marked bring-up/fallback only | Yes |
| JW4/Gemma no longer active DS5 target | Yes |
| Per-node static cap stated | 33.6GB |
| NVMe role stated | Promotion/backing/prefetch only |
| Top-k routing rule stated | Preserve exact model routing |
| Benchmarks tied to phase gates | Yes |
| Risk register linked | Yes |
