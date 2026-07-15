#!/usr/bin/env bash
# DS5 cluster — distribute SSH pubkeys so all 3 nodes have passwordless SSH
# in every direction, then verify all 6.
#
# Run from the PRIMARY node (Node A / pro-1) after the one-time manual seed
# step (Node A's pubkey appended to authorized_keys on B and C — see
# tools/cluster/README.md / NEW-CHAT-PROMPT.md). That seed is what gives this
# script SSH access to B and C in the first place; everything from here on is
# scripted and idempotent — safe to re-run any time (e.g. after a node is
# reimaged, or to pick up a new node).
#
# Node hostnames are addressed via mDNS (.local) with -4 forced: Bonjour on
# this LAN prefers IPv6 link-local addresses that aren't actually routable
# from this host, which surfaces as a misleading "No route to host". DHCP
# leases have also been observed to drift (max-1 moved 192.168.1.99 ->
# .98 mid-setup) so hardcoded IPs are avoided entirely in favor of hostnames.

set -uo pipefail

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }

SSH="ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"

# name:user@host:pubkey
A_USER="jesse"; A_HOST="pro-1.local"
A_PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIO6NeytTDNf3nbD/SqeK/Nz+CQMEvA7Heq3B2OgGMa14 ds5-pro-1"

B_USER="jesse"; B_HOST="max-1.local"
B_PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMO/hSi6p32ydys86vSvCjTSWvsXu8GZYfOdKGAJUQQt ds5-max-1"

C_USER="jesse"; C_HOST="max-2.local"
C_PUBKEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL9wYGtpxMTM2zaN3jiUBslJpImocue3LWeAfLSA7f3J ds5-max-2"

# ensure_key_local <pubkey> — idempotently add a pubkey to this machine's own authorized_keys
ensure_key_local() {
  local pubkey="$1"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  touch "$HOME/.ssh/authorized_keys" && chmod 600 "$HOME/.ssh/authorized_keys"
  grep -qxF "$pubkey" "$HOME/.ssh/authorized_keys" || echo "$pubkey" >> "$HOME/.ssh/authorized_keys"
}

# ensure_key_remote <user@host> <pubkey> — idempotently add a pubkey to a remote authorized_keys over SSH
ensure_key_remote() {
  local dest="$1" pubkey="$2"
  $SSH "$dest" "
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    grep -qxF '$pubkey' ~/.ssh/authorized_keys || echo '$pubkey' >> ~/.ssh/authorized_keys
  "
}

say "Seeding local (A) authorized_keys with B's and C's pubkeys (for B->A, C->A)"
ensure_key_local "$B_PUBKEY"
ensure_key_local "$C_PUBKEY"
ok "A's authorized_keys up to date"

say "Distributing C's pubkey to B (for C->B), via A->B SSH"
if ensure_key_remote "$B_USER@$B_HOST" "$C_PUBKEY"; then
  ok "B's authorized_keys up to date"
else
  warn "Could not reach $B_HOST — is it awake and on the LAN? Re-run this script once it's up."
fi

say "Distributing B's pubkey to C (for B->C), via A->C SSH"
if ensure_key_remote "$C_USER@$C_HOST" "$B_PUBKEY"; then
  ok "C's authorized_keys up to date"
else
  warn "Could not reach $C_HOST — is it awake and on the LAN? Re-run this script once it's up."
fi

# verify <label> <ssh-args...> — run `true` over ssh, report pass/fail
verify() {
  local label="$1"; shift
  if $SSH "$@" true 2>/dev/null; then
    ok "$label"
    return 0
  else
    warn "$label FAILED"
    return 1
  fi
}

say "Verifying all 6 SSH directions"
FAILS=0
verify "A -> B" "$B_USER@$B_HOST"                                     || FAILS=$((FAILS+1))
verify "A -> C" "$C_USER@$C_HOST"                                     || FAILS=$((FAILS+1))
# B->C, C->B, B->A, C->A are proxied one hop through A, since this script
# only has a shell on A — the proxied `ssh` runs *on* B or C using their own
# keys against the target's authorized_keys we just distributed.
$SSH "$B_USER@$B_HOST" "ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 $C_USER@$C_HOST true" 2>/dev/null \
  && ok "B -> C" || { warn "B -> C FAILED"; FAILS=$((FAILS+1)); }
$SSH "$C_USER@$C_HOST" "ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 $B_USER@$B_HOST true" 2>/dev/null \
  && ok "C -> B" || { warn "C -> B FAILED"; FAILS=$((FAILS+1)); }
$SSH "$B_USER@$B_HOST" "ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 $A_USER@$A_HOST true" 2>/dev/null \
  && ok "B -> A" || { warn "B -> A FAILED"; FAILS=$((FAILS+1)); }
$SSH "$C_USER@$C_HOST" "ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 $A_USER@$A_HOST true" 2>/dev/null \
  && ok "C -> A" || { warn "C -> A FAILED"; FAILS=$((FAILS+1)); }

if [ "$FAILS" -eq 0 ]; then
  say "SSH mesh complete: all 6 directions verified passwordless."
else
  warn "$FAILS direction(s) failed. Re-run this script after fixing reachability; it's idempotent."
  exit 1
fi
