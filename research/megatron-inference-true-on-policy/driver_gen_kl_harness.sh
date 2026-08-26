#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Container driver for gen_kl_harness — invoked by ray.sub via driver_command.sh.
# Do not embed bash control flow in the sbatch COMMAND one-liner; set env vars
# from run_gen_kl_harness.sh and call this script instead.

set -euo pipefail

: "${NRL_HARNESS_CONFIG:?NRL_HARNESS_CONFIG is required}"
NRL_HARNESS_NUM_BATCHES="${NRL_HARNESS_NUM_BATCHES:-1}"
: "${NRL_HARNESS_JSONL:?NRL_HARNESS_JSONL is required}"
: "${NRL_HARNESS_LOG_DIR:?NRL_HARNESS_LOG_DIR is required}"

RL_ROOT="${NRL_RL_ROOT:-/opt/nemo-rl}"
cd "${RL_ROOT}"

_detect_tms_cuda_major() {
    if [[ -n "${TMS_CUDA_MAJOR:-}" ]]; then
        return 0
    fi
    if command -v nvcc >/dev/null 2>&1; then
        TMS_CUDA_MAJOR="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' | head -1)"
        export TMS_CUDA_MAJOR
        echo "[driver] TMS_CUDA_MAJOR=${TMS_CUDA_MAJOR}" >&2
    else
        echo "WARNING: TMS_CUDA_MAJOR unset and nvcc not found; mcore uv sync may fail" >&2
    fi
}

_detect_tms_cuda_major

CONFIG_PATH="${RL_ROOT}/${NRL_HARNESS_CONFIG}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: harness config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ ! -f "${RL_ROOT}/examples/gen_kl_harness.py" ]]; then
    echo "ERROR: examples/gen_kl_harness.py not found under ${RL_ROOT}" >&2
    exit 1
fi

export FLA_TILELANG=0
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ZERO_KL_MODEL_PREFIX="${ZERO_KL_MODEL_PREFIX:-}"

_patch_mcore_venv_fla() {
    local patch_py="${RL_ROOT}/research/megatron-inference-true-on-policy/patch_mcore_venv_fla.py"
    if [[ ! -f "${patch_py}" ]]; then
        echo "WARNING: ${patch_py} missing; skipping tilelang strip." >&2
        return 0
    fi
    local venv_py="${NEMO_RL_VENV_DIR:?NEMO_RL_VENV_DIR is required}/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python"
    if [[ "${NRL_FORCE_REBUILD_VENVS:-false}" == "true" ]] || [[ ! -f "${venv_py}" ]]; then
        echo "[driver] building mcore worker venv (TMS_CUDA_MAJOR=${TMS_CUDA_MAJOR:-unset})" >&2
        uv run --extra mcore python "${patch_py}"
    fi
    uv run --no-sync --extra mcore python "${patch_py}" --post-sync
}

_patch_mcore_venv_fla

read -r -a extra_args <<< "${NRL_HARNESS_EXTRA_ARGS:-}"

exec uv run --no-sync --extra mcore examples/gen_kl_harness.py \
    --config "${NRL_HARNESS_CONFIG}" \
    --num-batches "${NRL_HARNESS_NUM_BATCHES}" \
    --jsonl "${NRL_HARNESS_JSONL}" \
    "logger.log_dir=${NRL_HARNESS_LOG_DIR}" \
    "${extra_args[@]}"
