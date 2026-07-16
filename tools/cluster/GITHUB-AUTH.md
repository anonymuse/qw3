# GitHub SSH Authentication for DS5 Cluster

Sets up secure, SSH key-based push access to the GitHub repository on all three
compute nodes (A, B, C), enabling CI/CD and automated commits from the cluster
without storing passwords or personal access tokens.

## Architecture

Each node generates its own ed25519 SSH key dedicated to GitHub authentication,
separate from the cluster-mesh SSH key. This separation keeps the two trust
domains independent:

- **Cluster-mesh key** (`~/.ssh/id_ed25519`, comment `ds5-<hostname>`): proves
  membership in the compute cluster; grants lateral SSH movement across A/B/C.
- **GitHub key** (`~/.ssh/id_ed25519_github`, comment `ds5-github-<hostname>`):
  proves identity to GitHub; grants write access to *this repo only*.

If either key is compromised, revoke one without disrupting the other.

## Prerequisites

1. **Cluster is fully operational**: SSH mesh (`setup-ssh-mesh.sh`) has been run
   and verified on A/B/C.
2. **Node D enrolled** (if using Pattern B): `enroll-dev-node.sh` has completed
   so you can run scripts from the dev laptop.
3. **GitHub repo exists**: The repository must already be on GitHub and you have
   admin access (needed to add Deploy Keys).

## Setup

### 1. Run the automated setup on Node A

From Node A (or Node D, if enrolled), run:

```sh
bash ~/Code/qw3/tools/cluster/setup-github-auth.sh
```

This script will:
- Generate a GitHub-specific SSH key on each node if one doesn't exist
- Gather all three public keys from the cluster nodes
- Display them for you to add to GitHub as Deploy Keys
- Convert the git remote from HTTPS to SSH on all three nodes
- Set per-node git user identity (name/email) for commits
- Verify SSH connectivity to GitHub

**Expected output:**
```
==> Gathering public keys from all 3 nodes…
[ok] Node A (pro-1): ssh-ed25519 AAAA... ds5-github-pro-1
[ok] Node B (max-1): ssh-ed25519 AAAA... ds5-github-max-1
[ok] Node C (max-2): ssh-ed25519 AAAA... ds5-github-max-2

==> GitHub Deploy Key Setup
Visit: https://github.com/anonymuse/qw3/settings/keys/new
For each key below, paste it with "Write access" enabled:
…
```

### 2. Add Deploy Keys to GitHub

The script displays each node's public key. For each:

1. Navigate to your repository's **Settings → Deploy keys**
2. Click "Add deploy key"
3. **Title**: `ds5-<hostname>` (e.g., `ds5-pro-1`)
4. **Key**: Paste the full public key from the script output
5. **Allow write access**: ✓ (required for pushing)
6. Click "Add key"

Repeat for all three nodes.

### 3. Verify the setup

Run the verification step:

```sh
bash ~/Code/qw3/tools/cluster/setup-github-auth.sh --verify-only
```

This re-checks everything without repeating the setup:
- SSH key existence on each node
- Git remote is SSH (not HTTPS)
- SSH connectivity to GitHub from each node

### 4. Test a push from Node A

Once Deploy Keys are registered:

```sh
ssh -4 jesse@pro-1.local
cd ~/Code/qw3
git config user.name "DS5 Cluster"
git config user.email "ds5@cluster.local"
echo "# Cluster GitHub auth verified" > GITHUB-VERIFIED.md
git add GITHUB-VERIFIED.md
git commit -m "test: verify GitHub SSH auth from cluster"
git push origin main
```

If this succeeds, GitHub auth is working end to end.

## Security Considerations

### Key Separation

Keeping GitHub keys separate from cluster-mesh keys limits the blast radius if
either is compromised:

- A stolen cluster-mesh key (`~/.ssh/id_ed25519`) grants access to lateral SSH
  movement within A/B/C, but NOT to GitHub.
- A stolen GitHub key (`~/.ssh/id_ed25519_github`) grants access to push to
  *this GitHub repo only*, but NOT to lateral SSH within the cluster.

If the compromise scope is ever unclear, revoking one key doesn't require
re-keying the entire cluster.

### SSH config

Each node has an `.ssh/config` entry for `Host github.com` that explicitly
specifies `IdentityFile ~/.ssh/id_ed25519_github`, preventing accidental use of
other keys (e.g. the cluster-mesh key, or any personal keys on the node).

### Deploy Keys vs. SSH per-user

Deploy Keys are repository-specific (no access to other repos or org-level
resources) and time-limited (you can rotate/revoke per-node at any time without
rekeying the entire cluster). They also provide a clear audit trail: one key
per node, owned by a machine, not a person — ideal for CI/CD automation.

## Troubleshooting

### "SSH key not found on node X"

The script failed to generate a key on node X. Likely causes:

1. **SSH from Node A to X failed**: Check mDNS/LAN connectivity.
   ```bash
   ssh -4 -o BatchMode=yes jesse@max-1.local "id"
   ```

2. **`~/.ssh` directory doesn't exist or is unreadable**: The script creates it
   if missing, but if it exists with wrong permissions, it won't overwrite it.
   ```bash
   ssh jesse@max-1.local "chmod 700 ~/.ssh && ls -la ~/.ssh/id_ed25519_github"
   ```

3. **ssh-keygen hung or failed**: Re-run the setup script; it's idempotent.

### "Git remote is still HTTPS"

Run the setup script again. The script converts remotes using `git remote set-url`
and verifies the change, but if a node was unreachable during the first run, the
remote might not have been updated.

### "SSH to github.com failed"

1. **Verify the SSH key exists on the node:**
   ```bash
   ssh jesse@pro-1.local "cat ~/.ssh/id_ed25519_github.pub"
   ```

2. **Check that the public key is registered as a Deploy Key:**
   - Navigate to https://github.com/anonymuse/qw3/settings/keys
   - Confirm each node's key appears in the list

3. **Test SSH connectivity manually:**
   ```bash
   ssh -i ~/.ssh/id_ed25519_github -o UserKnownHostsFile=/dev/null git@github.com
   # Should print: "Hi anonymuse! You've successfully authenticated…"
   ```

4. **If the key is registered but SSH still fails:** GitHub may cache the old
   connection state. Wait 30 seconds and try again, or delete `~/.ssh/known_hosts`
   and retry.

### Git config (user.name / user.email) not persisting

The setup script sets local git config on each node (in the `~/Code/qw3`
repository only, not globally). These settings are stored in
`.git/config` within the repo and persist across sessions. Verify:

```bash
ssh jesse@pro-1.local "cd ~/Code/qw3 && git config user.name && git config user.email"
```

If empty, the script either didn't set them, or they were overwritten. Re-run
`setup-github-auth.sh --verify-only` to check, or manually set them:

```bash
ssh jesse@pro-1.local "cd ~/Code/qw3 && git config user.name 'DS5 Cluster' && git config user.email 'ds5@cluster.local'"
```

### "github.com not in known_hosts"

On first SSH connection to GitHub, the client asks whether to accept the host
key. The setup script uses `-o StrictHostKeyChecking=accept-new`, which accepts
the key automatically. If you see a prompt about `github.com`, either:

1. Re-run the setup script (it will accept the key and cache it).
2. Or manually SSH to github.com on the node to prime `known_hosts`:
   ```bash
   ssh -i ~/.ssh/id_ed25519_github git@github.com
   # Type 'yes' when prompted, then exit
   ```

## References

- `tools/cluster/setup-github-auth.sh` — automated setup script
- `tools/cluster/README.md` — Phase 2 overview
- `tools/cluster/topology.md` — cluster node addresses and access model
- GitHub documentation: [Deploy keys](https://docs.github.com/en/developers/overview/managing-deploy-keys#deploy-keys)
