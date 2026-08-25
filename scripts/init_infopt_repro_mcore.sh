#!/usr/bin/env bash
# Bump nested Megatron-LM to main (includes NVIDIA/Megatron-LM#6521 te_native).
# Run from the RL repo root after `git submodule update --init --recursive`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MCORE_DIR="${ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

if ! git -C "${MCORE_DIR}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "Missing ${MCORE_DIR}; run: git submodule update --init --recursive" >&2
  exit 1
fi

git -C "${MCORE_DIR}" fetch origin main
git -C "${MCORE_DIR}" checkout origin/main

echo "Megatron-LM now at $(git -C "${MCORE_DIR}" rev-parse --short HEAD)"
rg -n "_BATCH_INVARIANT_BACKENDS" "${MCORE_DIR}/megatron/core/transformer/custom_layers/batch_invariant_kernels.py" | head -1
