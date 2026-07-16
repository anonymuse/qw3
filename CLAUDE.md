# DS5 Claude Code Integration

This document covers Claude Code CLI integration for the DS5 Qwen3 distributed inference project.

## Remote Execution: Metal Backend Generation on max-2

The Metal backend generation (64-step compilation + optimization pass) is configured for remote execution on the max-2 cluster node via Claude Code CLI.

### Quick Start

From any Claude Code session on your dev machine:

```bash
# Option 1: Run locally (devAir M5 24GB, ~23-24m)
claude run metal-backend-generation

# Option 2: Run on max-2 via SSH (M5 Max 48GB, ~12-15m expected)
claude run metal-backend-generation-max-2

# Option 3: Manual SSH (advanced)
ssh -4 jesse@max-2.local 'cd ~/Code/qw3 && ./tools/run-metal-backend-remote.sh'
```

### Prerequisites

**On max-2:**
- ✓ Repo cloned to `~/Code/qw3` (keep in sync: `git pull` to latest main)
- ✓ 30B GGUF present at `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/` (see [setup](#setup-max-2-first-time) below)
- ✓ Claude CLI installed (`/opt/homebrew/bin/claude`)

**On dev machine (Node D):**
- ✓ SSH keypair at `~/.ssh/id_ed25519` (auto-generated if missing)
- ✓ max-2 has your public key in `~/.ssh/authorized_keys`
- ✓ `ssh -4 jesse@max-2.local` works passwordless

### Setup: max-2 (First Time)

#### 1. Copy 30B GGUF from dev machine

```bash
# On dev machine (Node D):
rsync -av ~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/ \
  jesse@max-2.local:~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/

# Verify on max-2:
ssh jesse@max-2.local 'du -sh ~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/'
# Should show: 32G (approx)
```

#### 2. Verify repo sync and Zig build

```bash
ssh jesse@max-2.local <<'EOF'
cd ~/Code/qw3
git log -1 --oneline      # Should match your local HEAD
zig build                 # Verify build succeeds
echo "✓ Setup complete"
EOF
```

### Execution: Running the Generation

#### Via Claude Code CLI (Recommended)

Use the `metal-backend-generation-max-2` launcher from any session on your dev machine:

```
/claude run metal-backend-generation-max-2
```

This:
1. SSHes to `jesse@max-2.local`
2. Runs `./tools/run-metal-backend-remote.sh` in `~/Code/qw3`
3. Streams output back to your terminal
4. Saves metrics to `~/Code/qw3/bench/results/metal-generation-TIMESTAMP/`

#### Direct Script Execution (for testing on max-2)

If running the script directly on max-2 via a separate SSH session:

```bash
ssh jesse@max-2.local
cd ~/Code/qw3
./tools/run-metal-backend-remote.sh --output-dir ./bench/results/manual-run
```

### Interpreting Results

The script outputs a report (`metal-generation-report.txt`) with:

- **Duration**: Total wall-clock time for 64 steps
- **Speedup**: Ratio vs. devAir baseline (~1400s)
- **Node info**: Hostname, CPU count, memory, architecture

Example output:
```
Metal Generation Results:
  Total Steps: 64
  Failed Steps: 0
  Duration: 847s (14m 7s)
  Rate: 0.07 steps/sec

Performance vs Baseline:
  Baseline (devAir M5 24GB): ~1400s (~23m 34s)
  This run (max-2): 847s (14m 7s)
  Speedup: 1.7x
```

**Acceptance**: Generation completes overnight with ~35% faster execution than devAir baseline (target: ≤900s on max-2).

### Troubleshooting

#### SSH connection fails
```bash
# Check max-2 availability:
ping -c 1 max-2.local
ssh -4 jesse@max-2.local "echo OK"

# If SSH fails:
# 1. Verify ~/.ssh/id_ed25519 exists (if not, Claude Code can help generate)
# 2. Check max-2 authorized_keys has your public key:
ssh jesse@max-2.local 'cat ~/.ssh/authorized_keys | grep "ds5-dev"'
```

#### GGUF not found on max-2
```bash
# Copy from dev machine:
rsync -av ~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/ \
  jesse@max-2.local:~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/

# Verify:
ssh jesse@max-2.local 'ls -lah ~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/' | head -5
```

#### Build fails on max-2
```bash
# Ensure repo is up-to-date:
ssh jesse@max-2.local 'cd ~/Code/qw3 && git pull && zig build'

# Check Zig version (must be 0.16.0):
ssh jesse@max-2.local 'zig version'
```

#### Script times out or hangs
- Metal backend generation should complete in 12–20m on max-2
- If >30m, check `zig build test-metal` performance:
  ```bash
  ssh jesse@max-2.local 'cd ~/Code/qw3 && time zig build test-metal'
  ```

### Notes

- **Node D** (dev machine) is intentionally outside the 3-node cluster mesh (A/B/C). max-2 SSH access is the primary dev→cluster gateway.
- The repo at `~/Code/qw3` on max-2 must track main (PR #21 fix required). Sync before each run: `git pull`.
- GGUF is 32GB; rsync over LAN typically takes 5–10 min on first setup.
- The 64-step process is a full Metal shader backend gate (not loopback). It exercises real GGUF weights, quantization, and Metal dispatch.

### References

- T06 (real-weights gate): `docs/orchestration/HANDOFF.md` § "T06 M2c real-weights gate"
- Metal backend: `src/kernels/metal/` (Metal shaders: `kernels_a.metal`, `kernels_b.metal`)
- Build system: `build.zig`
- Cluster layout: `docs/runbook.md`, `manifests/cluster/lab.zon`
