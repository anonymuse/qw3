#!/usr/bin/env bash
# DS5 model artifact downloads. Run this on a CLUSTER NODE with disk to spare
# (>150 GB free), not the dev laptop. Downloads are resumable.
#
# Requires the Hugging Face CLI:  pip install -U huggingface_hub
#
# TODO(pin): after the first successful download, record the exact repo
# revision hashes here and in docs/assumptions.md (risk R-012: pin artifacts).

set -euo pipefail

MODELS_DIR="${DS5_MODELS_DIR:-$HOME/ds5-models}"
mkdir -p "$MODELS_DIR"

HF=$(command -v hf || command -v huggingface-cli || true)
if [ -z "$HF" ]; then
    echo "error: hf CLI not found. Install with: pip install -U huggingface_hub" >&2
    exit 1
fi

echo "==> Downloading to $MODELS_DIR"

# 1. Bring-up model + oracle source (~32 GB): Qwen3-30B-A3B-Instruct-2507, Q8_0.
#    Used for M2 single-node correctness and golden-fixture generation.
"$HF" download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF \
    --include "*Q8_0*" \
    --local-dir "$MODELS_DIR/qwen3-30b-a3b-instruct-2507-gguf"

# 2. Final target, low-bit (~85 GB): Qwen3-235B-A22B-Instruct-2507, Q2-class.
#    M1 router-telemetry capture ONLY (slow mmap run via llama.cpp on one M5 Max).
#    Not a quality-bearing artifact.
"$HF" download unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF \
    --include "*UD-Q2_K_XL*" \
    --local-dir "$MODELS_DIR/qwen3-235b-a22b-instruct-2507-gguf"

echo "==> Done. Contents:"
du -sh "$MODELS_DIR"/*
