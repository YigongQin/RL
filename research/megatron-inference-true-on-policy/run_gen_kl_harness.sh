#!/bin/bash
#SBATCH --job-name=gen-kl-harness
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=logs/gen-kl-harness-%j-%x.out
#SBATCH --error=logs/gen-kl-harness-%j-%x.err

# Cheap gen_kl_error gate — one rollout + train-path get_logprobs, no GRPO training loop.
#
# MODEL=infopt-debug-1n|infopt-debug-2n|infopt|qwen1.5b|qwen30ba3b
#   infopt-debug-1n  (default) 1 node, Qwen1.5B, 4 train + 4 inference GPUs
#   infopt-debug-2n            2 nodes, Qwen30B MoE, 1 train + 1 inference node
#   infopt                     8 nodes, production #3531 repro
#   qwen1.5b|qwen30ba3b         colocated zero_train_gen_mismatch (yigongq/minf-onpolicy branch only)
#
# NUM_BATCHES=1
# JOB_VENV_SUBDIR=infopt-mcore   default stable mcore venv (reuse across jobs)
# NRL_FORCE_REBUILD_VENVS=true   wipe + rebuild mcore venv (needs nvcc / TMS_CUDA_MAJOR)
# NRL_MINF_SHARED_CLUSTER=1      optional 1-node shared-GPU path for 30B (B200 memory)

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
[[ -f "${SCRIPT_DIR}/.env" ]] && set -a && source "${SCRIPT_DIR}/.env" && set +a

: "${RL_DIR:?Set RL_DIR in .env}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE in .env}"
: "${HF_TOKEN:?Set HF_TOKEN in .env}"
: "${HF_HOME:?Set HF_HOME in .env}"

MODEL="${MODEL:-infopt-debug-1n}"
NUM_BATCHES="${NUM_BATCHES:-1}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
NUM_NODES="${SLURM_NNODES:-${NUM_NODES:-1}}"

EXTRA_FLAGS=()
case "${MODEL}" in
    infopt-debug-1n|debug-1n)
        RUN_PREFIX="infopt-debug-1n"
        HARNESS_CONFIG="examples/configs/recipes/grpo_math_qwen15b_megatron_det_infopt_debug_1n8g.yaml"
        NUM_NODES="${NUM_NODES:-1}"
        ;;
    infopt-debug-2n|debug-2n)
        RUN_PREFIX="infopt-debug-2n"
        HARNESS_CONFIG="examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_infopt_debug_2n8g.yaml"
        NUM_NODES="${NUM_NODES:-2}"
        ;;
    infopt-shared-1n)
        RUN_PREFIX="infopt-shared-1n"
        HARNESS_CONFIG="examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_infopt_debug_2n8g.yaml"
        NUM_NODES="${NUM_NODES:-1}"
        export NRL_MINF_SHARED_CLUSTER="${NRL_MINF_SHARED_CLUSTER:-1}"
        EXTRA_FLAGS+=(
            "grpo.num_prompts_per_step=2"
            "grpo.num_generations_per_prompt=2"
            "policy.train_global_batch_size=4"
        )
        ;;
    infopt|infopt-qwen30ba3b)
        RUN_PREFIX="infopt-qwen30ba3b"
        HARNESS_CONFIG="examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_ep8tp1.yaml"
        NUM_NODES="${NUM_NODES:-8}"
        ;;
    qwen1.5b|qwen-1.5b)
        RUN_PREFIX="qwen1.5b"
        HARNESS_CONFIG="examples/configs/recipes/llm/grpo-qwen1.5b-megatron-zero-train-gen-kl.yaml"
        NUM_NODES="${NUM_NODES:-1}"
        if [[ ! -f "${RL_DIR}/${HARNESS_CONFIG}" ]]; then
            echo "ERROR: ${HARNESS_CONFIG} is on yigongq/minf-onpolicy; use MODEL=infopt-debug-1n here." >&2
            exit 1
        fi
        ;;
    qwen30ba3b|qwen-30ba3b)
        RUN_PREFIX="qwen30ba3b"
        HARNESS_CONFIG="examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl.yaml"
        NUM_NODES="${NUM_NODES:-1}"
        if [[ ! -f "${RL_DIR}/${HARNESS_CONFIG}" ]]; then
            echo "ERROR: ${HARNESS_CONFIG} is on yigongq/minf-onpolicy; use MODEL=infopt-debug-2n here." >&2
            exit 1
        fi
        ISL="${ISL:-10240}"
        EXTRA_FLAGS+=("policy.max_total_sequence_length=${ISL}")
        ;;
    *)
        echo "ERROR: MODEL must be infopt-debug-1n, infopt-debug-2n, infopt-shared-1n, infopt, qwen1.5b, or qwen30ba3b." >&2
        exit 1
        ;;
esac

RUN_TAG="${EXP_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
JSONL="${JSONL:-${RL_DIR}/logs/gen_kl_harness_${RUN_PREFIX}_${RUN_TAG}.jsonl}"
LOG_DIR="${LOG_DIR:-${RL_DIR}/logs/gen_kl_harness_${RUN_PREFIX}_${RUN_TAG}}"
mkdir -p "${LOG_DIR}" logs

JOB_VENV_SUBDIR="${JOB_VENV_SUBDIR:-infopt-mcore}"
mkdir -p "${RL_DIR}/venvs/${JOB_VENV_SUBDIR}"

if [[ -n "${NRL_RAY_VENVS_MOUNT_HOST:-}" ]]; then
    mkdir -p "${NRL_RAY_VENVS_MOUNT_HOST}/${JOB_VENV_SUBDIR}"
    NEMO_RL_VENV_CONTAINER="/opt/ray_venvs/${JOB_VENV_SUBDIR}"
    NRL_RAY_VENVS_MOUNT_SUFFIX=",${NRL_RAY_VENVS_MOUNT_HOST}:/opt/ray_venvs"
else
    NEMO_RL_VENV_CONTAINER="/opt/nemo-rl/venvs/${JOB_VENV_SUBDIR}"
    NRL_RAY_VENVS_MOUNT_SUFFIX=""
fi

HARNESS_EXTRA=(
    "cluster.num_nodes=${NUM_NODES}"
    "cluster.gpus_per_node=${GPUS_PER_NODE}"
    "${EXTRA_FLAGS[@]}"
    "$@"
)

# Preflight on submit host (fail before allocating nodes).
if [[ ! -f "${RL_DIR}/${HARNESS_CONFIG}" ]]; then
    echo "ERROR: config missing: ${RL_DIR}/${HARNESS_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${RL_DIR}/examples/gen_kl_harness.py" ]]; then
    echo "ERROR: examples/gen_kl_harness.py missing (wrong branch?)" >&2
    exit 1
fi
DRIVER_SCRIPT="${SCRIPT_DIR}/driver_gen_kl_harness.sh"
if [[ ! -f "${DRIVER_SCRIPT}" ]]; then
    echo "ERROR: ${DRIVER_SCRIPT} missing" >&2
    exit 1
fi
chmod +x "${DRIVER_SCRIPT}"

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

# Pass harness settings via env; run a real script inside the container (no inline if/fi).
export NRL_HARNESS_CONFIG="${HARNESS_CONFIG}"
export NRL_HARNESS_NUM_BATCHES="${NUM_BATCHES}"
export NRL_HARNESS_JSONL="${JSONL}"
export NRL_HARNESS_LOG_DIR="${LOG_DIR}"
export NRL_HARNESS_EXTRA_ARGS="${HARNESS_EXTRA[*]}"
export ZERO_KL_MODEL_PREFIX="${RUN_PREFIX}"

# Non-colocated infopt recipes use refit_backend=nvshmem; Megatron requires CTAS=2.
NVSHMEM_EXPORT=""
case "${MODEL}" in
    infopt*|debug-*)
        NVSHMEM_EXPORT="export NVSHMEM_MAX_CTAS=2 && "
        ;;
esac

export COMMAND="${CACHE_EXPORT}\
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_CONTAINER} && \
export NRL_RL_ROOT=/opt/nemo-rl && \
export NRL_HARNESS_CONFIG=${NRL_HARNESS_CONFIG} && \
export NRL_HARNESS_NUM_BATCHES=${NRL_HARNESS_NUM_BATCHES} && \
export NRL_HARNESS_JSONL=${NRL_HARNESS_JSONL} && \
export NRL_HARNESS_LOG_DIR=${NRL_HARNESS_LOG_DIR} && \
export NRL_HARNESS_EXTRA_ARGS=\"${NRL_HARNESS_EXTRA_ARGS}\" && \
export NRL_MINF_SHARED_CLUSTER=${NRL_MINF_SHARED_CLUSTER:-0} && \
${NVSHMEM_EXPORT}\
export ZERO_KL_MODEL_PREFIX=${ZERO_KL_MODEL_PREFIX} && \
export PYTHONUNBUFFERED=1 && \
export UV_HTTP_TIMEOUT=900 && \
export HF_HOME=${HF_HOME} && \
export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
export HF_TOKEN=${HF_TOKEN} && \
export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
bash /opt/nemo-rl/research/megatron-inference-true-on-policy/driver_gen_kl_harness.sh"

echo "MODEL=${RUN_PREFIX} CONFIG=${HARNESS_CONFIG}"
echo "VENV_DIR=${NEMO_RL_VENV_CONTAINER}"
echo "NODES=${NUM_NODES}x${GPUS_PER_NODE}  NUM_BATCHES=${NUM_BATCHES}"
echo "JSONL=${JSONL}"
echo "COMMAND: ${COMMAND}"
source ray.sub
