# New-chat orchestration prompt

Paste everything between the lines into a fresh Claude Code chat **on the M5 Pro
(primary node)**, then paste the three NODE FACTS blocks from `bootstrap.sh`
underneath it.

---

You are the **cluster-setup orchestrator** for the DS5 project. Set up a 3-node
Apple Silicon MacBook cluster using as much agentic automation as possible, and
produce reusable scripts + docs as committed deliverables.

**Nodes** (facts pasted below this prompt — hostname, LAN IP, user, ssh_pubkey each):
- Node A — M5 Pro — `primary` (this machine, where you're running)
- Node B — M5 Max — `worker` + download node (stores the GGUF models)
- Node C — M5 Max — `worker`

**Already done by the human** (via `tools/cluster/bootstrap.sh`): Xcode CLT,
Homebrew, git, node, huggingface CLI, Zig (may need version fix), Claude Code
installed + logged in, Remote Login/SSH enabled, an ed25519 key generated on each,
`~/ds5-models` created on Node B.

**Your job, in order:**

1. **Coordination primitive.** First establish how you'll drive the other two
   nodes. Prefer Claude Code Remote Control + peer messaging if it's available and
   the nodes are reachable as peers. If that's fiddly, fall back to plain SSH from
   this primary node into B and C — that is a fully acceptable Pattern A fallback,
   don't burn time forcing peer agents. State which one you're using and why.

2. **SSH mesh.** Build passwordless SSH in all directions. Each node's SSH **public** key is already in the pasted NODE FACTS blocks — nothing more to copy/paste. Your job is to write them into `authorized_keys` on the worker nodes.
   - Use `ssh-copy-id -i ~/.ssh/id_ed25519.pub <user>@<node>.local` to push keys from the primary to each worker.
   - If `ssh-copy-id` is absent, use the portable fallback: `cat ~/.ssh/id_ed25519.pub | ssh <user>@<node> 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'`.
   - You'll need each worker's login password once when prompted — **password auth works immediately once Remote Login is on**, so you don't need the keys to bootstrap the keys.
   - After installation, verify passwordless SSH works in all 6 directions with `ssh <node> true`.
   - Write the mesh setup up as an idempotent `tools/cluster/setup-ssh-mesh.sh`.
   
   **Advanced/optional:** If the worker nodes are running peered Claude Code sessions (Remote Control, same account), you can skip `ssh-copy-id` and instead `SendMessage` each worker its `authorized_keys` lines to install locally — fully hands-off, no passwords required. Use this only if peers are available.

3. **Toolchain repair.** On each node, confirm `zig version` is 0.16.x. If a node
   has the wrong Zig (brew often ships a different stable), download the matching
   0.16 build from ziglang.org, install to /opt/zig, and put it ahead of brew on
   PATH. Confirm `git`, `hf`/`huggingface-cli`, and `claude` are all present.

4. **Model download** (Node B only). Run, with resumability:
   ```sh
   hf download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
       --include "*Q8_0*" \
       --local-dir ~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf
   ```
   The model is public/ungated (no token needed). This is ~32 GB — run it under
   `caffeinate -s` or nohup so a sleep doesn't kill it, and poll for completion.
   Verify the downloaded files exist and are non-truncated (sha256 or size check).

5. **Cluster verification.** On all three nodes: the qw3 repo is already cloned to
   `~/Code/qw3` by bootstrap.sh — just run `zig build test` there and confirm it passes.
   If `~/Code/qw3` is somehow absent, fall back to cloning manually. Confirm each node can
   reach the others on the LAN. Write this up as an idempotent `tools/cluster/verify-cluster.sh`.

6. **Deliverables.** Commit to the qw3 repo under `tools/cluster/`:
   `setup-ssh-mesh.sh`, `verify-cluster.sh`, and a `topology.md` recording each
   node's role, hostname, LAN IP, and Zig version. Keep every script idempotent
   and re-runnable.

**Ground rules:**
- Favor idempotent scripts over one-off manual steps — the scripts ARE the
  deliverable; a transcript of manual commands is not reproducible.
- Ask the human only for the irreducible seams (an admin password if a remote
  sudo prompt blocks you, a GUI toggle if SSH won't enable from CLI, the qw3 clone
  credentials).
- Do NOT touch the qw3 model runtime or orchestration branches — you are setting up
  infrastructure, not doing DS5 tasks. When the model download completes, report
  that clearly: the remote orchestrator session is waiting on it to start task T06.

**Context:** This unblocks DS5 T06 (real-weights gate), which gates T07 (M3
distributed). Full task DAG in `docs/orchestration/HANDOFF.md`. The download landing
in `~/ds5-models/qwen3-30b-a3b-instruct-2507-gguf/` is the signal the rest of the
pipeline is waiting for.

Start by reading the pasted NODE FACTS, then tell me your coordination-primitive
choice and your step-by-step plan before executing.

---
