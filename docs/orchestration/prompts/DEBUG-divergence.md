# DEBUG template — numeric divergence triage

**Model:** Haiku per-layer; Sonnet if two rounds fail. Use when an end-to-end
run diverges from oracle but individual kernels pass their fixtures. The
orchestrator fills the <angle-bracket> slots and spawns one agent per suspect.

---

You are debugging a numeric divergence in the DS5 engine (Zig + Metal).
Read-only context: `src/shared/contracts.zig`, ADR-005 §6 (forward recipe),
`docs/orchestration/HANDOFF.md` §5 (landmines).

**Symptom:** <e.g. prompt p2, greedy token diverges at step 7; trace hook
reports first divergence layer 3, op expert_mlp, max_abs_diff 4e-2>.

**Your scope:** layer <N>, op <op>, backend <cpu|metal>. Do not investigate
anything else. Do not edit contracts, fixtures, or tolerances.

**Procedure:**
1. Reproduce: `<exact command>`. Confirm the reported first-divergence point
   with the trace hook (`<flag>`); if it reproduces at a DIFFERENT point,
   report that immediately and stop (suspect nondeterminism: threadgroup
   atomics ordering, uninitialized buffer — see HANDOFF §5).
2. Isolate: dump the op's actual inputs at the divergence point (trace hook
   `--dump-dir`), run the op standalone through BOTH the CPU provider and
   (if backend=metal) the shader path on those inputs. Three-way compare:
   fixture-oracle semantics vs CPU vs Metal.
   - CPU wrong too → wiring bug upstream: verify buffer offsets/aliasing/
     position bookkeeping between the previous op's output and this input
     (most common: reused scratch buffer not re-uploaded, `pos` off by one,
     expert-bank byte offset).
   - Only Metal wrong → shader bug: check barriers, `precise::` math, params
     struct layout (`@sizeOf` Zig vs MSL comment), grid over-provisioning
     writes, f16→f32 promotion order in dequant.
3. Fix the smallest thing; rerun: the op standalone, then the layer trace,
   then the full prompt set, then `zig build test` + `test-metal` (+
   `test-gpu` if present).
4. Report: root cause in one sentence, the diff, evidence (before/after max
   diffs), and whether other call sites share the pattern.

**Hard rule:** if your conclusion is "the kernel/fixture/tolerance is wrong",
you do not change it — write the evidence and return it to the orchestrator.
Kernels that pass fixtures are innocent until trace-proven guilty.
