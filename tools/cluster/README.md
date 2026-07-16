# DS5 3-Node Cluster — Operator Runbook

Brings up 1× M5 Pro + 2× M5 Max as an agent-coordinated cluster for DS5. The goal
is **maximum agentic automation**: humans do only the irreducible per-machine seams
(sudo, GUI toggles, Claude Code login), then AI agents take over.

## Node roles

| Node | Machine | Role | Notes |
|---|---|---|---|
| A | M5 Pro | `primary` (orchestrator) | Runs the coordinating Claude Code session |
| B | M5 Max | `worker` + `--download` | Stores the GGUF models (needs >150 GB free) |
| C | M5 Max | `worker` | Second compute node |

> The **download node is an M5 Max**, never the dev MacBook Air (24 GB) — the model
> set needs >150 GB of disk.

## What YOU provide (per the plan)

Three MacBooks, on the internet, logged in. Plus these irreducible seams:

1. **An Administrator account** on each — not just a password. The login user must
   be an admin (Homebrew and the sudo/SSH steps fail for a Standard user). Check
   with `id -Gn | grep -qw admin`; if missing, from an existing admin run
   `sudo dseditgroup -o edit -a <user> -t user admin`, then log out/in. On a fresh
   Mac the first Setup Assistant account is an admin by default. `bootstrap.sh`
   pre-checks this and stops early with instructions if it's not met.
2. **Same LAN** — all three on the same network (wired Ethernet preferred; same
   Wi-Fi is fine). Confirm the macOS firewall isn't blocking SSH
   (System Settings → Network → Firewall — off, or allow Remote Login).
3. **Claude Code login** — one interactive browser OAuth per machine (agents can't
   do this for you).
4. Optional: an HF account. The Qwen3 GGUFs are public/ungated, so **no token is
   required**; a token only raises rate limits.

Everything else is scripted or agent-driven.

## Sequence

**Notation:** **[AUTOMATED]** = the script does it. **[MANUAL]** = you do it by hand.

### Phase 0 — per machine (human, ~5 min each)

1. **[MANUAL]** Fetch `bootstrap.sh` onto the Mac. **Do not `git clone` yet** — git isn't on a
   fresh Mac (it ships with the Command Line Tools that bootstrap installs). But
   `curl` *is* built in, and the repo is public, so:
   ```sh
   curl -fsSL https://raw.githubusercontent.com/anonymuse/qw3/main/tools/cluster/bootstrap.sh -o bootstrap.sh
   ```
   (Or AirDrop it from another Mac.)
2. **[MANUAL]** Give each machine a clear name so nodes are distinguishable, e.g.:
   ```sh
   sudo scutil --set LocalHostName ds5-pro     # or ds5-max-1 / ds5-max-2
   ```
3. **[MANUAL]** Run the bootstrap for its role — **as yourself, never with `sudo`.** The script
   prompts for your password once (to prime sudo); running the whole thing under `sudo` breaks Homebrew, which refuses to install as root.
   ```sh
   caffeinate -s bash bootstrap.sh --role primary               # Node A
   caffeinate -s bash bootstrap.sh --role worker --download     # Node B
   caffeinate -s bash bootstrap.sh --role worker                # Node C
   ```
   To upload NODE FACTS to a public paste URL, add `--share`:
   ```sh
   caffeinate -s bash bootstrap.sh --role primary --share
   ```
   **[AUTOMATED]** The script does the following: clones the full repo to `~/Code/qw3`, installs Xcode CLT, Homebrew, git, node, huggingface CLI, and Zig; generates an ed25519 SSH key; enables Remote Login/SSH; and creates `~/ds5-models` on the download node. `caffeinate -s` keeps the Mac awake through the long installs.
4. **[MANUAL]** When it finishes, run `claude` once and complete the browser login.
5. **[MANUAL]** Copy the printed **NODE FACTS** block. The facts are also saved to
   `~/Code/ds5-cluster/node-facts-<hostname>.txt` — you can gather all three facts by:
   - Copying the printed block from each machine's terminal, or
   - Using **AirDrop** to transfer the `node-facts-<hostname>.txt` files to Node A.
   
   If passing `--share`, the script also uploads them to a public paste URL (note: the URL is PUBLIC and contains the SSH public key + LAN IP, but no secrets). You'll paste all three into the orchestrator chat in Phase 1.

If the script warns that **Zig isn't 0.16** or that **SSH couldn't be toggled from
the CLI**, that's expected on some setups:
- **[MANUAL]** If SSH can't be enabled from CLI, toggle it via the GUI: System Settings → General → Sharing → Remote Login.
- **[AUTOMATED]** The orchestrator agent will handle Zig version fixes automatically.

### Phase 1 — hand off to the agents

**Operator startup (on Node A, the M5 Pro):**

1. **[AUTOMATED]** Bootstrap already created `~/Code` and cloned the repo to `~/Code/qw3`.

2. **[MANUAL]** Navigate to the repo:
   ```sh
   cd ~/Code/qw3
   ```

3. **[MANUAL]** Run `claude` to launch Claude Code:
   ```sh
   claude
   ```
   On first launch, **accept the "trust this folder" prompt**. Launching `claude` from the repo root (`~/Code/qw3`) scopes Claude Code's trusted workspace to just the qw3 repo — do not run it from your home directory, as that would make the trusted domain broader than necessary.

4. **[MANUAL]** In the new Claude Code chat, paste the full contents of [`tools/cluster/NEW-CHAT-PROMPT.md`](NEW-CHAT-PROMPT.md), followed by the three NODE FACTS blocks you collected from each machine in Phase 0.

**Orchestrator automation (from Phase 1 onward):**

From there the orchestrator agent will:

1. Establish the coordination primitive (Remote Control peer messaging if
   available; otherwise SSH-from-primary — either works).
2. **[AUTOMATED]** Build the SSH mesh: distribute each node's public key to the other two using `ssh-copy-id` (already in NODE FACTS — no manual key transfer needed), verify passwordless SSH in all directions, and capture the process as `tools/cluster/setup-ssh-mesh.sh`.
3. Verify/repair the Zig 0.16 toolchain on each node.
4. Kick off the model download on Node B with resumability + checksum.
5. Clone qw3, build, and run `zig build test` on all three; confirm inter-node
   reachability.
6. Author the reusable scripts (`setup-ssh-mesh.sh`, `verify-cluster.sh`, etc.)
   and a topology doc as committed deliverables.

### Phase 2 — GitHub Authentication Setup

After the cluster is fully operational, set up secure GitHub authentication on all nodes:

1. **Review the setup guide**: Read [`tools/cluster/GITHUB-AUTH.md`](GITHUB-AUTH.md)
   for architecture, security considerations, and troubleshooting.

2. **Run the automated setup** on Node A (primary):
   ```sh
   bash ~/Code/qw3/tools/cluster/setup-github-auth.sh
   ```
   The script will:
   - Gather SSH public keys from all three nodes
   - Display them for you to add to GitHub (Deploy Keys)
   - Convert all git remotes from HTTPS to SSH
   - Set per-node git user identity
   - Verify GitHub connectivity

3. **Add keys to GitHub**:
   - Navigate to your repository settings → Deploy keys
   - For each node (A, B, C), add its public key with write access enabled
   - See [`GITHUB-AUTH.md`](GITHUB-AUTH.md) for detailed instructions

4. **Verify the setup**:
   ```sh
   bash ~/Code/qw3/tools/cluster/setup-github-auth.sh --verify-only
   ```

5. **Test a push from Node A**:
   ```sh
   cd ~/Code/qw3
   git config user.name "DS5 Cluster Test"
   git config user.email "ds5-test@cluster.local"
   echo "# Cluster setup complete" > CLUSTER-READY.md
   git add CLUSTER-READY.md
   git commit -m "Setup: cluster ready for DS5 tasks"
   git push origin main
   ```

**Security approach**:
- SSH key-based authentication (no PATs or passwords stored)
- Each node's ed25519 key (generated during bootstrap) is used for GitHub pushes
- Private keys never leave the nodes they're generated on
- Public keys are added as Deploy Keys (limited to this repo, write access controlled per-node)

## Directory structure

- `~/Code/qw3/tools/cluster/` — bootstrap and operational scripts
- `~/Code/ds5-cluster/` — generated node-facts files and agent-produced deliverables

## Deliverables produced

**Phase 0 & 1** (Agent-generated):
- `tools/cluster/bootstrap.sh` — per-node bootstrap.
- `tools/cluster/setup-ssh-mesh.sh` — key distribution (SSH mesh).
- `tools/cluster/verify-cluster.sh` — build + reachability verification.
- `tools/cluster/topology.md` — node map, addresses, Zig versions.
- `~/Code/ds5-cluster/node-facts-<hostname>.txt` — per-node facts (saved by bootstrap).
- Agent transcripts as evidence of the automated run.

**Phase 2** (GitHub authentication):
- `tools/cluster/GITHUB-AUTH.md` — comprehensive setup guide and troubleshooting.
- `tools/cluster/setup-github-auth.sh` — automated GitHub SSH auth setup.
- `~/Code/ds5-cluster/github-auth-setup.log` — setup script logs.
- `~/Code/ds5-cluster/github-pubkeys/` — public keys for GitHub Deploy Keys (saved by script).

## Why this unblocks the project

The model download on Node B lands
`~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/` (Q8_0), which unblocks **T06**
(real-weights gate) and, in turn, **T07** (M3 distributed). See
`docs/orchestration/HANDOFF.md` for the full task DAG.
