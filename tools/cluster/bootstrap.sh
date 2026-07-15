#!/usr/bin/env bash
# DS5 cluster node bootstrap — brand-new Apple Silicon Mac → agent-ready.
#
# Run ONCE per machine, in Terminal, as the logged-in admin user:
#   caffeinate -s bash bootstrap.sh --role primary      # the M5 Pro (orchestrator)
#   caffeinate -s bash bootstrap.sh --role worker        # each M5 Max
#   caffeinate -s bash bootstrap.sh --role worker --download   # the M5 Max that stores the model
#
# Idempotent: safe to re-run. It installs toolchain + Claude Code, enables SSH,
# generates an SSH key, and prints the node facts the orchestrator needs.
#
# It intentionally STOPS before authenticating Claude Code and before exchanging
# SSH keys across machines — those are the human/coordination seams (see README.md).

set -uo pipefail

ROLE="worker"
IS_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --role) : ;;                       # value handled below
    primary|worker) ROLE="$arg" ;;
    --download) IS_DOWNLOAD=1 ;;
  esac
done
# allow "--role primary" form
prev=""
for arg in "$@"; do
  [ "$prev" = "--role" ] && ROLE="$arg"
  prev="$arg"
done

ZIG_WANT="0.16"
MIN_DISK_GB=150
say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }

say "DS5 bootstrap — role=$ROLE download-node=$IS_DOWNLOAD"

# 1. Xcode Command Line Tools ------------------------------------------------
if ! xcode-select -p >/dev/null 2>&1; then
  say "Installing Xcode Command Line Tools (headless)…"
  touch /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
  PROD=$(softwareupdate -l 2>/dev/null | grep -Eo 'Label: Command Line Tools.*' | sed 's/^Label: //' | tail -1)
  if [ -n "$PROD" ]; then
    softwareupdate -i "$PROD" --verbose || warn "headless CLT install failed"
  fi
  rm -f /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
  if ! xcode-select -p >/dev/null 2>&1; then
    warn "CLT still missing. Run 'xcode-select --install' and click through the GUI dialog, then re-run this script."
    exit 1
  fi
fi
ok "Xcode Command Line Tools present"

# 2. Homebrew ----------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  say "Installing Homebrew…"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    || { warn "Homebrew install failed"; exit 1; }
fi
eval "$(/opt/homebrew/bin/brew shellenv)"
grep -q 'brew shellenv' ~/.zprofile 2>/dev/null || \
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
ok "Homebrew present: $(brew --version | head -1)"

# 3. Toolchain: git, zig, node (for Claude Code), huggingface CLI ------------
say "Installing git, node, huggingface-cli…"
brew install git node >/dev/null 2>&1 || warn "brew install git/node had issues"
brew install huggingface-cli >/dev/null 2>&1 || warn "brew huggingface-cli unavailable; will try pip"
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  python3 -m pip install --user -U huggingface_hub >/dev/null 2>&1 || warn "pip huggingface_hub failed"
fi

# Zig — the project pins 0.16, which may be a nightly/master build that brew's
# stable formula does NOT provide. Install, then verify hard.
say "Installing Zig (want $ZIG_WANT)…"
brew install zig >/dev/null 2>&1 || warn "brew install zig had issues"
ZIG_HAVE="$(zig version 2>/dev/null || echo none)"
case "$ZIG_HAVE" in
  ${ZIG_WANT}*) ok "Zig $ZIG_HAVE" ;;
  *) warn "Zig is '$ZIG_HAVE' but the project needs $ZIG_WANT.x.
        Download the matching build from https://ziglang.org/download/ (or the
        0.16 nightly), unpack to /opt/zig, and add it to PATH ahead of brew's.
        The orchestrator agent can also do this step for you." ;;
esac

# 4. Claude Code -------------------------------------------------------------
if ! command -v claude >/dev/null 2>&1; then
  say "Installing Claude Code (npm global)…"
  npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 \
    || warn "npm install of Claude Code failed — see https://claude.com/claude-code for the native installer"
fi
command -v claude >/dev/null 2>&1 && ok "Claude Code present: $(claude --version 2>/dev/null || echo installed)"

# 5. Enable Remote Login (SSH) ----------------------------------------------
say "Enabling Remote Login (SSH)…"
if sudo systemsetup -setremotelogin on 2>/dev/null; then
  ok "SSH enabled"
else
  warn "Could not toggle SSH from CLI (needs Full Disk Access for Terminal).
        Enable manually: System Settings → General → Sharing → Remote Login = ON."
fi

# 6. SSH key for the node mesh ----------------------------------------------
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
  say "Generating ed25519 SSH key…"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  ssh-keygen -t ed25519 -N "" -C "ds5-$(scutil --get LocalHostName 2>/dev/null || hostname)" -f "$HOME/.ssh/id_ed25519" >/dev/null
fi
ok "SSH public key ready"

# 7. Download-node disk check ------------------------------------------------
if [ "$IS_DOWNLOAD" = "1" ]; then
  AVAIL_GB=$(df -g "$HOME" | awk 'NR==2{print $4}')
  if [ "${AVAIL_GB:-0}" -lt "$MIN_DISK_GB" ]; then
    warn "Download node has ${AVAIL_GB}GB free; need >=${MIN_DISK_GB}GB for the model set."
  else
    ok "Download node free space: ${AVAIL_GB}GB"
  fi
  mkdir -p "$HOME/ds5-models"
fi

# 8. Node facts for the orchestrator ----------------------------------------
LOCALHOST="$(scutil --get LocalHostName 2>/dev/null || hostname)"
LANIP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo unknown)"
say "NODE FACTS — paste these to the orchestrator session"
cat <<EOF
  role:        $ROLE
  download:    $([ "$IS_DOWNLOAD" = 1 ] && echo yes || echo no)
  hostname:    ${LOCALHOST}.local
  lan_ip:      $LANIP
  user:        $USER
  zig:         $ZIG_HAVE
  claude:      $(command -v claude >/dev/null 2>&1 && echo installed || echo MISSING)
  ssh_pubkey:  $(cat "$HOME/.ssh/id_ed25519.pub")
EOF

say "NEXT (human): 1) run 'claude' once and finish the browser login.
              2) send the NODE FACTS block above to the orchestrator chat.
See tools/cluster/README.md for the full sequence."
