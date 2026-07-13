# Work Pack: Review Incorporation (2026-07-12)

**Origin:** Adopted items A5–A8 from `docs/reviews/2026-07-12_airplane_arch_reviews_response.md`
**Spec:** `docs/specs/DS5_Project_Spec_v0.3.md` (amended)
**Agent sizing:** Each pack is self-contained, doc/tooling-only, and executable by a
low-cost agent without cluster access. Packs own disjoint files — no cross-pack edits.

## Rules for all packs

1. Read the spec (`docs/specs/DS5_Project_Spec_v0.3.md`) and ADR-002/ADR-005 before
   writing anything. Do not change any ADR, gate, or milestone.
2. No runtime code changes. `src/` is out of bounds for these packs (kernel work is
   M2, gated on the interface freeze).
3. Deliverables must be runnable/checkable on the dev laptop (M5 Air, 24GB) where
   applicable; anything requiring the cluster or the 235B download ships as a runbook
   the operator executes later.
4. Python tooling uses the existing `.venv/` (torch 2.13, transformers 5.13, py 3.14).
5. Every claim sourced from the external reviews must be independently verified or
   marked unverified — the reviews have a track record of wrong constants.

## Packs

| Pack | Deliverable | Feeds |
|---|---|---|
| WP-1 `wp1-expert-stats-tooling.md` | `expert_stats.json` schema + capture tooling/runbook (routing frequency + imatrix/KLD sensitivity) | M1 placement simulator, f001 |
| WP-2 `wp2-quant-manifest-two-axis.md` | Placement/quant manifest format doc with per-expert precision column + assignment policy + loader refusal rules | M4 manifests |
| WP-3 `wp3-metal4-kernel-spike.md` | Research memo: Metal 4 / M5-gen GPU features relevant to DS5 kernels (bf16, tensor intrinsics, sub-byte types) | M2 kernel design |
