# Task: Run real 3-node mesh benchmark on Node C (max-2 M5 Max)

Execute the production mesh benchmark on the 3-node cluster as the blocking item for T07. This is one of three parallel node runs.

## Current context
- Blocker for T07: real 3-node `ds5 bench link` has never been run (only loopback smoke tests exist in bench/results/)
- Node C is currently idle (no T06 work assigned)
- All three nodes can run in parallel
- This prompt covers Node C only; Node A and Node B have parallel prompts

## Steps
1. SSH to Node C (max-2): `ssh jesse@max-2.local`
2. Navigate to qw3 repo (confirm main branch, PR #21 merged; may need to pull if stale)
3. Build if needed: `zig build -Doptimize=ReleaseFast`
4. Run: `./zig-out/bin/ds5 bench link --cluster manifests/cluster/lab.zon --self c --label mesh-run1`
5. Benchmark will connect to Nodes A and B over SSH mesh
6. Results written to bench/results/
7. Report: runtime, memory peak, any connection/timeout issues

## Acceptance
Benchmark completes, results logged with --label mesh-run1 tag, no SSH or serialization errors.
