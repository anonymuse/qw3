# Task: Run real 3-node mesh benchmark on Node B (max-1 M5 Max)

Execute the production mesh benchmark on the 3-node cluster as the blocking item for T07. This is one of three parallel node runs.

## Current context
- Blocker for T07: real 3-node `ds5 bench link` has never been run (only loopback smoke tests exist in bench/results/)
- Node B is also running T06 work (HF checkpoint download + fixture gen); schedule this for after T06 main work completes or run in background if resources allow
- All three nodes can run in parallel
- This prompt covers Node B only; Node A and Node C have parallel prompts

## Steps
1. SSH to Node B (max-1): `ssh jesse@max-1.local`
2. Navigate to repo at ~/Code/qw3 (confirm main branch, PR #21 merged)
3. Build if needed: `zig build -Doptimize=ReleaseFast`
4. Run: `./zig-out/bin/ds5 bench link --cluster manifests/cluster/lab.zon --self b --label mesh-run1`
5. Benchmark will connect to Nodes A and C over SSH mesh
6. Results written to bench/results/
7. Report: runtime, memory peak, any connection/timeout issues

## Acceptance
Benchmark completes, results logged with --label mesh-run1 tag, no SSH or serialization errors. Can run in parallel with T06 fixture work if background resources available.
