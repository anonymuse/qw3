# DS5 cluster topology

Recorded by the orchestrator agent after `setup-ssh-mesh.sh` and
`verify-cluster.sh` both passed clean, 2026-07-15.

| Node | Machine | Role | Hostname | LAN IP (observed) | User | Zig | SSH pubkey comment |
|---|---|---|---|---|---|---|---|
| A | M5 Pro | `primary` (orchestrator) | `pro-1.local` | 192.168.1.95 | jesse | 0.16.0 | `ds5-pro-1` |
| B | M5 Max | `worker` + download | `max-1.local` | 192.168.1.98 | jesse | 0.16.0 | `ds5-max-1` |
| C | M5 Max | `worker` | `max-2.local` | 192.168.1.96 | jesse | 0.16.0 | `ds5-max-2` |

## Notes / gotchas hit during setup

- **Address nodes by `.local` hostname, not IP.** Node B's LAN IP drifted
  (`192.168.1.99` → `192.168.1.98`) between its bootstrap run and mesh setup —
  ordinary DHCP lease churn. `setup-ssh-mesh.sh` and `verify-cluster.sh` both
  use `.local` mDNS names exclusively for this reason. If a node's IP moves
  again, mDNS resolves it correctly with no script changes needed; only
  update the "observed" IP column above (informational only).
- **Force IPv4 (`-4`) for SSH.** On this LAN, `.local` mDNS resolution
  surfaces IPv6 link-local/ULA addresses ahead of the routable IPv4 one, so
  `ssh node.local` and plain `ping node.local` both fail with a misleading
  "No route to host". Fix: `ssh -4 ...`, and for `ping`, resolve to an IPv4
  address explicitly first (`dscacheutil -q host -a name <host>`) rather than
  pinging the `.local` name directly. Both scripts do this.
- **`~/.zshenv` needed the Homebrew `shellenv` line.** `bootstrap.sh` only
  adds `eval "$(/opt/homebrew/bin/brew shellenv)"` to `~/.zprofile`, which zsh
  only sources for login shells. Non-interactive SSH commands (`ssh host
  'zig version'`) use neither a login nor an interactive shell, so they only
  source `~/.zshenv` — without the fix, `zig`/`hf`/`claude` all report
  "command not found" over SSH despite being correctly installed (confirmed
  via absolute paths under `/opt/homebrew/bin` and Homebrew's Cellar). Added
  the same `eval` line to `~/.zshenv` on all three nodes; idempotent (checked
  with `grep -q` first).
- **Zig build test summary is on stderr.** `zig build test` writes its
  "Build Summary: ... tests passed" line to stderr, not stdout — a script
  redirecting stderr to `/dev/null` while grepping stdout for pass/fail will
  always see a mismatch even on a clean pass. `verify-cluster.sh` captures
  both streams merged (`2>&1`) before checking. Separately, Zig 0.16 also
  prints a cosmetic red `failed command:` line when a passing test writes to
  stderr (see `docs/orchestration/HANDOFF.md` landmine #1) — exit code and
  the Build Summary line are the ground truth, not that line.

## Coordination primitive used

**Pattern A: SSH from the primary node.** No Claude Code Remote Control /
peer-messaging primitive for driving separate Claude Code sessions on B and C
was available in the orchestrator's tool environment, so per the fallback
rule in `NEW-CHAT-PROMPT.md` this session drove B and C entirely over SSH
from Node A. Initial trust was seeded by a one-time manual step (the human
appended Node A's pubkey to `authorized_keys` on B and C directly); from
there, `setup-ssh-mesh.sh` distributed the remaining key pairs and verified
all 6 directions (B↔C directions verified by proxying one hop through their
own SSH access, e.g. `ssh B "ssh C true"`, since the orchestrator only has a
shell on A).

## Model download (T06 gate)

`~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf`
on Node B (max-1): 32,483,932,576 bytes, `hf download` reported
`✓ Downloaded`, GGUF magic header verified (`GGUF`, version 3). This
unblocks T06 for the remote DS5 orchestrator.
