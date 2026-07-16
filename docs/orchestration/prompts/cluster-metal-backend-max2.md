# Task: Run 64-step Metal backend generation on max-2 overnight

Run the Metal backend generation task on the idle max-2 Max node to achieve faster completion than the current devAir run.

## Current context
- Task: 64-step Metal backend generation on real 30B GGUF
- Current location: devAir (M5 24GB, ~23m 34s)
- Target: max-2 M5 Max (48GB, faster Metal performance)
- 30B GGUF location: ~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/ (verified on max-1, check max-2)

## Steps
1. Verify 30B GGUF is present on max-2 at ~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/
2. SSH to max-2: `ssh jesse@max-2.local`
3. Clone/sync repo to latest (PR #21 fix must be included)
4. Run the 64-step Metal backend generation
5. Capture runtime and any performance metrics
6. Report completion time vs. expected ~12–15m on Max hardware

## Acceptance
Generation completes overnight, result ready by morning with ~35% faster execution than devAir baseline.
