# DS5 Qwen3 Documentation Update Pack

**Document type:** Documentation pack index  
**Status:** Generated update pack  
**Date:** 2026-07-08  
**Decision locked:** DS5 will use **Qwen3-235B-A22B** as the project target model.

---

## 1. Confirmation

The DS5 project model decision is now locked:

> **Primary project model:** `Qwen3-235B-A22B-Instruct-2507`  
> **Model class:** Sparse Mixture-of-Experts, 235B total parameters, 22B activated parameters  
> **Optional later variant:** `Qwen3-235B-A22B-Thinking-2507`, only after the non-thinking instruct path is stable

This decision supersedes competing final-target proposals based on dense 70B, Mixtral 8x22B as the final model, DeepSeek-V3/R1 full checkpoints, Kimi-K2, and the earlier JW4/Gemma runtime scope. Dense 32B-70B and smaller Qwen MoE models remain valid bring-up and validation targets; they are not the final project target.

---

## 2. Generated Markdown files

| File | Purpose | Update type |
|---|---|---|
| `ADR_001_Model_Selection_Qwen3_235B_A22B.md` | Locks the model decision | New architecture decision record |
| `DS5_Project_Spec_v0.2_Qwen3_235B_A22B.md` | Recasts the project spec around Qwen3-235B-A22B | Project specification update |
| `DS5_System_Architecture_v0.2_Qwen3_235B_A22B.md` | Defines updated system architecture, diagrams, and planning dimensions | System architecture update |
| `DS5_Model_Runtime_Placement_Spec_v0.2_Qwen3_235B_A22B.md` | Defines runtime, node placement, quantization, KV, and storage rules | Runtime/system design spec |
| `DS5_Benchmark_and_Acceptance_Spec_v0.2_Qwen3_235B_A22B.md` | Defines phase gates, benchmark classes, and acceptance criteria | Verification spec |
| `DS5_Risk_Register_v0.2_Qwen3_235B_A22B.md` | Captures major technical, scope, and execution risks | Risk register |
| `DS5_Execution_Plan_Input_v0.2_Qwen3_235B_A22B.md` | Converts the decision into epics, workstreams, and dependencies | Planning-mode execution input |
| `DS5_Document_Update_Register_v0.2_Qwen3_235B_A22B.md` | Lists all existing and new docs that should be updated | Documentation control register |

---

## 3. Existing documents that should be updated

| Existing document | Required action | Notes |
|---|---|---|
| `DS5_Composite_Architecture_Planning_Model.md` | Update status to approved planning baseline; replace “Qwen3 class” language with exact model lock | Should become the top-level architecture baseline after review |
| `JW4_project_brief.md` | Either supersede or split into a precursor project | It currently describes a Gemma-focused, 2-node learning engine; Qwen3-235B-A22B requires a 3-node MoE runtime scope |
| `ChatGPT DS5 Analysis.md` | Archive as source analysis and link to ADR-001 | Keep as basis for model-selection math |
| `Deepseek DS5 Analysis.md` | Archive as alternative study | Preserve PDD/expert-locality concepts, but do not retain DeepSeek-V3/R1 as final target |
| `Gemini DS5 Analysis.md` | Archive as alternative study | Preserve useful diagram concepts; remove static Mixtral final-target assumption from active specs |
| `Z.AI DS5 Analysis.md` | Archive as alternative study | Preserve warning that strict per-token expert streaming is not viable |

---

## 4. Governance rule

After this update, any execution-plan item that assumes a different final model must be explicitly marked as one of:

- **bring-up target**;
- **benchmark comparator**;
- **fallback runtime**;
- **archived alternative**.

No project task should treat another model as the DS5 final target unless ADR-001 is superseded.
