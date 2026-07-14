# T09 — Ship: f001 final, README, runbooks, PRs

**Model:** Haiku. **Branch:** `t09-ship` off `integration`. Runs in the last
two days REGARDLESS of how far T05–T08 got. Negative results get written up,
not hidden.

## Deliverables

1. **`docs/findings/f001-viability.md` final:** replace every placeholder
   marked in the T03 draft with measured values from `bench/results/` (mesh
   links, decode-sim on real numbers, telemetry skew, any real decode
   attempts). Every number cites its JSON file. Numbers still unmeasured stay
   explicitly labeled unmeasured — check `docs/assumptions.md` consistency
   and update its Status column for anything measured this week.
2. **README.md rewrite:** what DS5 is (from-scratch Zig+Metal distributed
   MoE engine, 3-node Apple Silicon), architecture sketch (transport /
   engine / kernels / fixtures layering), current honest status table
   (milestone gates passed/failed/pending, with links to findings), build +
   test commands (test, test-metal, test-gpu), fixture-first development
   explanation, and a **Limitations** section written plainly (what is
   unproven, what is placeholder, what broke).
3. **Runbook check:** `docs/runbook.md` + `docs/runbook-m3.md` reproduce from
   a clean clone (actually execute the loopback paths).
4. **PR hygiene:** every feature branch merged or explicitly parked with a
   one-line status in HANDOFF.md's scoreboard; open a final PR
   `integration` → `main` summarizing the milestone status table; CI-esque
   final run of all test suites pasted into the PR body.

## Definition of done

A newcomer can clone, build, run tests, run the loopback demos, and read
f001 + README to know exactly what is real. No claim without a citation to a
result file or test. HANDOFF.md scoreboard updated to final state.

## Forbidden

Rounding up. "Should work", "nearly", and uncited numbers are all bugs in
this deliverable.
