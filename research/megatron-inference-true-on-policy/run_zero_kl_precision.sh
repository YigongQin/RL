#!/bin/bash
#SBATCH --job-name=zero-kl-precision
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/zero-kl-precision-%j-%x.out
#SBATCH --error=logs/zero-kl-precision-%j-%x.err

# Official entrypoint gate: examples/run_grpo.py + recipe. Read gen_kl_error at step 0.
#
# Same infopt recipes/cluster layout as run_gen_kl_harness.sh; entrypoint is
# examples/run_grpo.py (step-0 gen_kl_error matches the harness path, then one
# train step if MAX_STEPS>=1). Harness skips the train loop only.
#
# MODEL (colocated zero_train_gen_mismatch — yigongq/minf-onpolicy recipes):
#   qwen1.5b|qwen30ba3b|nanov3
#   PRECISION=bf16|mxfp8  ZERO_TRAIN_GEN_MISMATCH=false
#   ISL=10240  policy.max_total_sequence_length (qwen30ba3b default 10240)
#
# MODEL (inference-optimized / #3531 stack — dedicated infer nodes, BI + te_native):
#   infopt-debug-1n   1 node, Qwen1.5B, 4 train + 4 inference GPUs
#   infopt-debug-2n   2 nodes, Qwen30B MoE (submit with -N 2)
#   infopt-shared-1n  1 node shared-GPU path for 30B (sets NRL_MINF_SHARED_CLUSTER=1)
#   infopt            8 nodes, production det_ep8tp1 recipe (submit with -N 8)
#   MAX_STEPS=1       cheap step-0 gen_kl gate (default remains 2000 for long runs)
#   ISL=512           optional; overrides policy.max_total_sequence_length
#                     (recipe defaults: 1.5B debug=512, 30B debug=2048)
#
# NRL_FORCE_REBUILD_VENVS=true  rebuilds mcore worker venv via patch_mcore_venv_fla.py
# JOB_VENV_SUBDIR=...           override venv path
#                               (infopt default: infopt-mcore; colocated: job-${RUN_TAG})

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
[[ -f "${SCRIPT_DIR}/.env" ]] && set -a && source "${SCRIPT_DIR}/.env" && set +a

: "${RL_DIR:?Set RL_DIR in .env}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in .env}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY in .env}"
: "${WANDB_PROJECT:?Set WANDB_PROJECT in .env}"

MODEL="${MODEL:-}"
PRECISION="${PRECISION:-bf16}"
MAX_STEPS="${MAX_STEPS:-2000}"
ZERO_TRAIN_GEN_MISMATCH="${ZERO_TRAIN_GEN_MISMATCH:-true}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
# Prefer Slurm allocation, else explicit NUM_NODES, else per-MODEL default below.
USER_NUM_NODES="${NUM_NODES:-}"
DEFAULT_NODES=1

# STACK=colocated applies PRECISION / zero_train_gen_mismatch overrides.
# STACK=infopt leaves those to the recipe (batch_invariant + inference_optimized).
STACK="colocated"
EXTRA_FLAGS=()
case "${MODEL}" in
    qwen1.5b|qwen-1.5b)
        RUN_PREFIX="qwen1.5b"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-qwen1.5b-megatron-zero-train-gen-kl.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-250}"
        ;;
    qwen30ba3b|qwen-30ba3b)
        RUN_PREFIX="qwen30ba3b"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-10}"
        ISL="${ISL:-10240}"
        ;;
    nanov3|nano|nanov3-30ba3b)
        RUN_PREFIX="nanov3"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-nanov3-30ba3b-megatron-zero-train-gen-kl.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-10}"
        ;;
    infopt-debug-1n|debug-1n)
        STACK="infopt"
        RUN_PREFIX="infopt-debug-1n"
        GRPO_CONFIG="examples/configs/recipes/grpo_math_qwen15b_megatron_det_infopt_debug_1n8g.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-10}"
        DEFAULT_NODES=1
        ;;
    infopt-debug-2n|debug-2n)
        STACK="infopt"
        RUN_PREFIX="infopt-debug-2n"
        GRPO_CONFIG="examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_infopt_debug_2n8g.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-10}"
        DEFAULT_NODES=2
        ;;
    infopt-shared-1n)
        STACK="infopt"
        RUN_PREFIX="infopt-shared-1n"
        GRPO_CONFIG="examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_infopt_debug_2n8g.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-10}"
        DEFAULT_NODES=1
        export NRL_MINF_SHARED_CLUSTER="${NRL_MINF_SHARED_CLUSTER:-1}"
        EXTRA_FLAGS+=(
            "grpo.num_prompts_per_step=2"
            "grpo.num_generations_per_prompt=2"
            "policy.train_global_batch_size=4"
        )
        ;;
    infopt|infopt-qwen30ba3b)
        STACK="infopt"
        RUN_PREFIX="infopt-qwen30ba3b"
        GRPO_CONFIG="examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_ep8tp1.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-10}"
        DEFAULT_NODES=8
        ;;
    *)
        echo "ERROR: MODEL is required." >&2
        echo "  colocated: qwen1.5b, qwen30ba3b, nanov3" >&2
        echo "  infopt:    infopt-debug-1n, infopt-debug-2n, infopt-shared-1n, infopt" >&2
        exit 1
        ;;
esac

NUM_NODES="${SLURM_NNODES:-${USER_NUM_NODES:-${DEFAULT_NODES}}}"

if [[ ! -f "${RL_DIR}/${GRPO_CONFIG}" ]]; then
    echo "ERROR: config missing: ${RL_DIR}/${GRPO_CONFIG}" >&2
    if [[ "${STACK}" == "colocated" ]]; then
        echo "HINT: colocated recipes live on yigongq/minf-onpolicy; use MODEL=infopt-debug-* here." >&2
    fi
    exit 1
fi

if [[ "${STACK}" == "colocated" ]]; then
    [[ "${PRECISION}" == "mxfp8" ]] && EXTRA_FLAGS+=(
        "policy.megatron_cfg.fp8_cfg.enabled=true"
        "policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8"
    )
    [[ "${ZERO_TRAIN_GEN_MISMATCH}" == "false" ]] && EXTRA_FLAGS+=(
        "policy.megatron_cfg.zero_train_gen_mismatch=false"
    )
fi

# ISL → policy.max_total_sequence_length (colocated qwen30ba3b defaults above;
# set ISL for any infopt qwen run to shorten seqs for fast debugging).
if [[ -n "${ISL:-}" ]]; then
    EXTRA_FLAGS+=("policy.max_total_sequence_length=${ISL}")
fi

RUN_TAG="${EXP_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
if [[ "${STACK}" == "infopt" ]]; then
    WANDB_RUN_NAME="${RUN_PREFIX}-zero-kl-${RUN_TAG}"
else
    WANDB_RUN_NAME="${RUN_PREFIX}-zero-kl-${PRECISION}-${RUN_TAG}"
fi
CKPT_DIR="${CKPT_DIR:-${RL_DIR}/results/${WANDB_RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RL_DIR}/logs/${WANDB_RUN_NAME}}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" logs

# Infopt reuses the stable harness venv by default (same as run_gen_kl_harness.sh).
# Colocated keeps per-job isolation so concurrent runs do not clobber each other.
if [[ "${STACK}" == "infopt" ]]; then
    JOB_VENV_SUBDIR="${JOB_VENV_SUBDIR:-infopt-mcore}"
else
    JOB_VENV_SUBDIR="${JOB_VENV_SUBDIR:-job-${RUN_TAG}}"
fi
mkdir -p "${RL_DIR}/venvs/${JOB_VENV_SUBDIR}"

if [[ -n "${NRL_RAY_VENVS_MOUNT_HOST:-}" ]]; then
    mkdir -p "${NRL_RAY_VENVS_MOUNT_HOST}/${JOB_VENV_SUBDIR}"
    NEMO_RL_VENV_CONTAINER="/opt/ray_venvs/${JOB_VENV_SUBDIR}"
    NRL_RAY_VENVS_MOUNT_SUFFIX=",${NRL_RAY_VENVS_MOUNT_HOST}:/opt/ray_venvs"
else
    NEMO_RL_VENV_CONTAINER="/opt/nemo-rl/venvs/${JOB_VENV_SUBDIR}"
    NRL_RAY_VENVS_MOUNT_SUFFIX=""
fi

GRPO_ARGS=(
    --config "${GRPO_CONFIG}"
    "grpo.max_num_steps=${MAX_STEPS}"
    "cluster.num_nodes=${NUM_NODES}"
    "cluster.gpus_per_node=${GPUS_PER_NODE}"
    "checkpointing.enabled=true"
    "checkpointing.checkpoint_dir=${CKPT_DIR}"
    "checkpointing.save_period=${SAVE_PERIOD}"
    "checkpointing.keep_top_k=${KEEP_TOP_K:-3}"
    "logger.log_dir=${LOG_DIR}"
    "logger.wandb_enabled=true"
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${WANDB_RUN_NAME}"
    "${EXTRA_FLAGS[@]}"
    "$@"
)

UV_RUN=(uv run --no-sync --extra mcore)

cd "${RL_DIR}"
export TORCH_CUDA_ARCH_LIST='9.0 10.0'
export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${RL_DIR}:/opt/nemo-rl${NRL_RAY_VENVS_MOUNT_SUFFIX}"
[[ -d /scratch ]] && export MOUNTS="${MOUNTS},/scratch:/scratch"
export GPUS_PER_NODE
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_CONTAINER}"

# Ray _env_builder runs uv sync inside worker processes; torch-memory-saver needs this.
export SETUP_COMMAND='if [[ -z "${TMS_CUDA_MAJOR:-}" ]] && command -v nvcc >/dev/null 2>&1; then export TMS_CUDA_MAJOR="$(nvcc --version | sed -n '"'"'s/.*release \([0-9][0-9]*\).*/\1/p'"'"' | head -1)"; fi'

CACHE_EXPORT=""
if [[ "${NRL_USE_WARM_UV_CACHE:-false}" == "true" ]]; then
    mkdir -p "${NRL_WARM_UV_CACHE_DIR:-${RL_DIR}/uv_cache}"
    CACHE_EXPORT="export UV_CACHE_DIR=${NRL_WARM_UV_CACHE_DIR:-${RL_DIR}/uv_cache} && "
fi

# Megatron-Bridge imports hybrid GatedDeltaNet → fla at worker import time for all
# models. fla probes tilelang on import; Py3.13 + tvm-ffi>=0.1.12 crashes unless
# tilelang is removed after each uv sync. Unset NRL_FORCE_REBUILD_VENVS after the
# initial patch so Ray _env_builder reuses the patched venv instead of rebuilding.
MCORE_FLA_SETUP="export FLA_TILELANG=0 FLA_DISABLE_BACKEND_DISPATCH=1 ZERO_KL_MODEL_PREFIX=${RUN_PREFIX} && \
cd /opt/nemo-rl && \
uv run --extra mcore python research/megatron-inference-true-on-policy/patch_mcore_venv_fla.py && \
unset NRL_FORCE_REBUILD_VENVS && \
uv run --no-sync --extra mcore python research/megatron-inference-true-on-policy/patch_mcore_venv_fla.py --post-sync && "

# Non-colocated infopt recipes use refit_backend=nvshmem; Megatron's copy
# service hard-requires NVSHMEM_MAX_CTAS=2 (hangs/fails otherwise).
NVSHMEM_EXPORT=""
if [[ "${STACK}" == "infopt" ]]; then
    NVSHMEM_EXPORT="export NVSHMEM_MAX_CTAS=2 && "
fi

export COMMAND="${CACHE_EXPORT}\
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_CONTAINER} && \
export NRL_MCORE_STRIP_TILELANG=1 && \
export NRL_MINF_SHARED_CLUSTER=${NRL_MINF_SHARED_CLUSTER:-0} && \
${NVSHMEM_EXPORT}\
export PYTHONUNBUFFERED=1 && \
export UV_HTTP_TIMEOUT=900 && \
export HF_HOME=${HF_HOME} && \
export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
export HF_TOKEN=${HF_TOKEN} && \
export WANDB_API_KEY=${WANDB_API_KEY} && \
export WANDB_ENTITY=${WANDB_ENTITY} && \
export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
export FLA_TILELANG=0 FLA_DISABLE_BACKEND_DISPATCH=1 && \
${MCORE_FLA_SETUP}\
unset NRL_FORCE_REBUILD_VENVS && \
cd /opt/nemo-rl && \
${UV_RUN[*]} examples/run_grpo.py ${GRPO_ARGS[*]}"

echo "MODEL=${RUN_PREFIX} STACK=${STACK} CONFIG=${GRPO_CONFIG}"
[[ "${STACK}" == "colocated" ]] && echo "PRECISION=${PRECISION} ZERO_TRAIN_GEN_MISMATCH=${ZERO_TRAIN_GEN_MISMATCH}"
echo "VENV_DIR=${NEMO_RL_VENV_CONTAINER}"
[[ -n "${ISL:-}" ]] && echo "ISL=${ISL} (policy.max_total_sequence_length)"
echo "NODES=${NUM_NODES}x${GPUS_PER_NODE}  MAX_STEPS=${MAX_STEPS}  RUN=${WANDB_RUN_NAME}"
if [[ "${STACK}" == "infopt" && -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -lt "${DEFAULT_NODES}" ]]; then
    echo "WARN: ${RUN_PREFIX} wants ${DEFAULT_NODES} nodes but Slurm allocation is ${SLURM_NNODES}; resubmit with -N ${DEFAULT_NODES}." >&2
fi
echo "COMMAND: ${COMMAND}"
source ray.sub
