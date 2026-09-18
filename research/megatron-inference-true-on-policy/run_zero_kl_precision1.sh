#!/bin/bash
#SBATCH --job-name=zero-kl-precision
#SBATCH --account=coreai_chef_posttrain
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/zero-kl-precision-%j-%x.out
#SBATCH --error=logs/zero-kl-precision-%j-%x.err

# Official entrypoint gate: examples/run_grpo.py + recipe. Read gen_kl_error at step 0.
#
# MODEL:
#   qwen1.5b|qwen1.5b-mxfp8
#       colocated TE + mxfp8 zero-KL
#   qwen3-1.7b-infopt
#       colocated inference_optimized zero-KL (Qwen3 dense; no QKV bias)
#   qwen30ba3b|qwen30ba3b-mxfp8
#       colocated TE + mxfp8 zero-KL (ISL default 10240)
#   qwen30ba3b-infopt
#       non-colocated inference_optimized zero-KL bf16/vllm MoE (submit with -N 2)
#   qwen30ba3b-mxfp8-te-infopt
#   qwen30ba3b-mxfp8-torch-infopt
#       non-colocated inference_optimized + TE native MXFP8 MoE (PR #6933; -N 2, GB200)
#   qwen30ba3b-mega-infopt
#       non-colocated inference_optimized + FlashInfer expert-parallel megakernel
#       (bf16; -N 2, GB200). Both sides run the megakernel: generation for the
#       expert compute, and the training forward via moe_mega_training_forward.
#       Needs the megakernel integration present in the Megatron-LM submodule;
#       it is developed there in place and is not on the pinned commit.
#   qwen30ba3b-mega-mxfp8-infopt
#       the same, with the megakernel at MXFP8 on both sides. The two forwards
#       stay bitwise identical, so zero-KL holds; the backward still comes from
#       the TE bf16 recompute, making the gradient a straight-through estimator.
#       Convergence against the bf16 arm is not established -- an experiment.
#
# Local Megatron-LM changes under ${RL_DIR} take effect via the /opt/nemo-rl mount
# (editable megatron-core install). Rebuild worker venv once after submodule edits.
#
# Optional overrides:
#   SMOKE=true               cheap single-step gen_kl gate: MAX_STEPS=1, ISL=1024,
#                            4 prompts x 2 generations, max_new_tokens=256, mcore
#                            max_tokens=ISL (the recipe's 16384 is sized for ISL 10240
#                            and makes the te backend chunk 64x per GEMM instead of 4x).
#                            Tune with SMOKE_PROMPTS / SMOKE_GENERATIONS / SMOKE_GBS /
#                            SMOKE_MAX_NEW_TOKENS / SMOKE_MAX_TOKENS; MAX_STEPS and ISL
#                            still win if set. Keep ISL well above the ~300-token
#                            dapomath prompt tail: a prompt that fills the window leaves
#                            no generation region and makes gen_kl_error meaningless.
#   MAX_STEPS=1              cheap step-0 gen_kl gate (default 2000)
#   ISL=512                  policy.max_total_sequence_length
#   ZERO_TRAIN_GEN_MISMATCH=false
#   EXP_TAG=my-chain         run identity for CKPT_DIR / LOG_DIR / wandb name
#                            (default: SLURM_JOB_ID). Share it across chained jobs
#                            so each leg resumes the previous one -- see (5) below.
#   CHECKPOINTING_ENABLED=true   default false (recipes also set checkpointing.enabled=false)
#   NRL_FORCE_REBUILD_VENVS=true   rm + serial pre-build worker venv on driver (not Ray)
#   NRL_DRIVER_UV_SYNC=false       skip driver uv sync (default: true; required when code != image)
#   NRL_USE_WARM_UV_CACHE=false    skip persistent uv cache (default: true)
#   NRL_WARM_UV_CACHE_DIR=...      default: ${RL_DIR}/uv_cache
#   JOB_VENV_SUBDIR=...      default: infopt-mcore (reuse existing; no rebuild unless forced)
#   NVTE_WITH_NCCL_EP=0              disable TE bundled NCCL EP build (default 0; TE 2.18 on GB200)
#   NVTE_CUDA_ARCHS=100              TE CUDA archs (default 100 -> sm_100a/103a; TE ignores
#                                    TORCH_CUDA_ARCH_LIST). Changing this invalidates every
#                                    cached TE object and forces a full rebuild.
#
# Worker venv is built once on the driver before run_grpo (avoids lustre races when Ray
# _env_builder runs uv sync on multiple nodes). Ray workers reuse it with
# NRL_FORCE_REBUILD_VENVS=false.
#
# --- Copy/paste (from this directory; source .env first: set -a && source .env && set +a) ---
#
# 1) After pyproject.toml / uv.lock TE bump (e.g. release_v2.18):
#    cd "${RL_DIR}" && uv lock
#    TE 2.18: NVTE_WITH_NCCL_EP=0 avoids bundled nccl_ep compile failure (NCCL_GIN_* undefined).
#
# 2) Rebuild shared MegatronPolicyWorker venv on 1 node (same venv infopt stack uses):
#    NRL_FORCE_REBUILD_VENVS=true MAX_STEPS=1 ISL=256 sbatch --export=ALL,MODEL=qwen30ba3b-mxfp8,NVTE_WITH_NCCL_EP=0 ./run_zero_kl_precision1.sh
#    Watch driver log for: worker venv OK 2.18...
#
# 3) Run mxfp8-te-infopt zero-KL (2 nodes: train + gen):
#    MAX_STEPS=2 sbatch -N 2 --export=ALL,MODEL=qwen30ba3b-mxfp8-te-infopt,NRL_FORCE_REBUILD_VENVS=false ./run_zero_kl_precision1.sh
#
# 4) Smoke the same bed (one step, small batch, generation region intact):
#    sbatch -N 2 --export=ALL,SMOKE=true,MODEL=qwen30ba3b-mxfp8-te-infopt,NRL_FORCE_REBUILD_VENVS=false ./run_zero_kl_precision1.sh
#
# 5) Chain N legs past the wall clock (one chain per model, run concurrently).
#    EXP_TAG is what makes a chain a chain: RUN_TAG defaults to SLURM_JOB_ID, so
#    without it every leg invents its own CKPT_DIR and restarts from step 0.
#    CHECKPOINTING_ENABLED=true is likewise required -- grpo resumes from the
#    latest checkpoint in checkpoint_dir, and with saving off there is none.
#    afterany (not afterok) so a leg that hits the time limit still hands off.
#    The tag needs no model name: RUN_PREFIX already prefixes it, so chain1 gives
#    qwen30ba3b-infopt-zero-kl-chain1 and ...-mega-infopt-zero-kl-chain1.
#
#    The leg has to be long enough to reach SAVE_PERIOD or the chain is a
#    treadmill: the first attempt ran 2h legs at ~325 s/step, reached step 19,
#    never hit the default save_period=35, and so every leg resumed from an empty
#    CKPT_DIR and redid steps 0-19. At that step time a 4h leg is ~40 steps after
#    ~15 min of startup, which clears 35 with little to spare -- if step time
#    grows as generations lengthen, raise --time before lowering SAVE_PERIOD.
#    for m in qwen30ba3b-infopt qwen30ba3b-mega-infopt; do
#        dep=""
#        for leg in $(seq 1 6); do
#            jid=$(sbatch --parsable ${dep:+--dependency=afterany:${dep}} -N 2 --time=04:00:00 \
#                --export=ALL,MODEL=${m},EXP_TAG=chain1,CHECKPOINTING_ENABLED=true,NRL_FORCE_REBUILD_VENVS=false \
#                ./run_zero_kl_precision1.sh)
#            echo "${m} leg ${leg}: ${jid}"
#            dep="${jid}"
#        done
#    done
#

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
SMOKE="${SMOKE:-false}"
# A zero-KL number only means something when every sample has a non-empty generation
# region, so ISL has to clear the prompt-length tail (dapomath prompts reached 297
# tokens in job 388851) -- but nothing above that buys accuracy. Measured per-token KL
# in that job was flat across the window (0.0049 over the first 256 generated tokens,
# ~0.0030 out at 1536+), so the error is per-token noise rather than drift through the
# KV cache and a short generation measures the same number as a long one. SMOKE
# therefore spends tokens only where they change the answer: a window with prompt
# headroom, and just enough decode steps to average over.
if [[ "${SMOKE}" == "true" ]]; then
    MAX_STEPS="${MAX_STEPS:-1}"
    ISL="${ISL:-1024}"
    SMOKE_PROMPTS="${SMOKE_PROMPTS:-4}"
    SMOKE_GENERATIONS="${SMOKE_GENERATIONS:-2}"
fi
MAX_STEPS="${MAX_STEPS:-2000}"
ZERO_TRAIN_GEN_MISMATCH="${ZERO_TRAIN_GEN_MISMATCH:-true}"
NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
NRL_DRIVER_UV_SYNC="${NRL_DRIVER_UV_SYNC:-true}"
NRL_USE_WARM_UV_CACHE="${NRL_USE_WARM_UV_CACHE:-true}"
NRL_WARM_UV_CACHE_DIR="${NRL_WARM_UV_CACHE_DIR:-${RL_DIR}/uv_cache}"
NVTE_WITH_NCCL_EP="${NVTE_WITH_NCCL_EP:-0}"
# TE ignores TORCH_CUDA_ARCH_LIST and reads NVTE_CUDA_ARCHS, defaulting to
# 75;80;89;90;100;120. Its arch-specific sources (the grouped activation and cast
# kernels) compile once per standard arch plus once per Blackwell target, so the
# default costs 7 nvcc passes on files that take ~20 min each. 100 expands to
# sm_100a (GB200) + sm_103a (GB300), which is all this stack runs on.
NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-100}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
# Prefer Slurm allocation, else explicit NUM_NODES, else per-MODEL default below.
USER_NUM_NODES="${NUM_NODES:-}"
DEFAULT_NODES=1

# STACK=colocated: shared-rank TE or colocated infopt recipes.
# STACK=infopt: non-colocated infopt (sets NVSHMEM_MAX_CTAS for refit).
STACK="colocated"
# Whether the recipe's refit_backend is nvshmem, which is what NVSHMEM_MAX_CTAS
# below is for. Per model, because the cap is not free for recipes that do not
# need it: a model whose kernels use NVSHMEM for anything else pays it too.
NVSHMEM_REFIT="true"
EXTRA_FLAGS=()
case "${MODEL}" in
    qwen1.5b|qwen-1.5b|qwen1.5b-mxfp8)
        RUN_PREFIX="qwen1.5b-mxfp8"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-qwen1.5b-megatron-zero-train-gen-kl-mxfp8.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-250}"
        ;;
    qwen3-1.7b-infopt|qwen3.1.7b-infopt|qwen-3-1.7b-infopt)
        RUN_PREFIX="qwen3-1.7b-infopt"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-qwen3-1.7b-megatron-zero-train-gen-kl-infopt-colocated.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-250}"
        ;;
    qwen30ba3b|qwen-30ba3b|qwen30ba3b-mxfp8)
        RUN_PREFIX="qwen30ba3b-mxfp8"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl-mxfp8.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-50}"
        ISL="${ISL:-10240}"
       # EXTRA_FLAGS+=(
       #     "grpo.async_grpo.recompute_kv_cache_after_weight_updates=true"
       # )
        ;;
    qwen30ba3b-infopt|qwen-30ba3b-infopt)
        STACK="infopt"
        RUN_PREFIX="qwen30ba3b-infopt"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl-infopt-noncolocated.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-35}"
        DEFAULT_NODES=2
        ISL="${ISL:-10240}"
        ;;
    qwen30ba3b-mega-infopt|qwen-30ba3b-mega-infopt)
        STACK="infopt"
        RUN_PREFIX="qwen30ba3b-mega-infopt"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl-mega-infopt-noncolocated.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-35}"
        DEFAULT_NODES=2
        ISL="${ISL:-10240}"
        # This recipe pins refit_backend: nccl, so it needs no NVSHMEM_MAX_CTAS
        # and must not inherit it -- see the NVSHMEM_EXPORT note below.
        NVSHMEM_REFIT="false"
        ;;
    qwen30ba3b-mega-mxfp8-infopt|qwen-30ba3b-mega-mxfp8-infopt)
        STACK="infopt"
        RUN_PREFIX="qwen30ba3b-mega-mxfp8-infopt"
        GRPO_CONFIG="examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl-mega-mxfp8-infopt-noncolocated.yaml"
        SAVE_PERIOD="${SAVE_PERIOD:-35}"
        DEFAULT_NODES=2
        ISL="${ISL:-10240}"
        NVSHMEM_REFIT="false"
        ;;
    *)
        echo "ERROR: MODEL is required." >&2
        echo "  colocated TE+mxfp8:  qwen1.5b, qwen30ba3b" >&2
        echo "  colocated infopt:    qwen3-1.7b-infopt" >&2
        echo "  non-colocated infopt: qwen30ba3b-infopt (-N 2)" >&2
        echo "  non-colocated TE mxfp8 infopt: qwen30ba3b-mxfp8-te-infopt (-N 2, GB200)" >&2
        echo "  non-colocated torch mxfp8 infopt (A/B): qwen30ba3b-mxfp8-torch-infopt (-N 2)" >&2
        echo "  non-colocated flashinfer megakernel: qwen30ba3b-mega-infopt (-N 2, GB200)" >&2
        echo "  ... the same at MXFP8 (straight-through gradient): qwen30ba3b-mega-mxfp8-infopt" >&2
        exit 1
        ;;
esac

NUM_NODES="${SLURM_NNODES:-${USER_NUM_NODES:-${DEFAULT_NODES}}}"

if [[ "${STACK}" == "infopt" && "${NUM_NODES}" -lt "${DEFAULT_NODES}" ]]; then
    echo "ERROR: ${RUN_PREFIX} needs ${DEFAULT_NODES} nodes (train + gen); got NUM_NODES=${NUM_NODES}." >&2
    echo "  sbatch -N ${DEFAULT_NODES} ... MODEL=${MODEL} $0" >&2
    exit 1
fi

if [[ ! -f "${RL_DIR}/${GRPO_CONFIG}" ]]; then
    echo "ERROR: config missing: ${RL_DIR}/${GRPO_CONFIG}" >&2
    exit 1
fi

[[ "${ZERO_TRAIN_GEN_MISMATCH}" == "false" ]] && EXTRA_FLAGS+=(
    "policy.megatron_cfg.zero_train_gen_mismatch=false"
)

# ISL → policy.max_total_sequence_length (qwen30ba3b-mxfp8 defaults to 10240 above).
if [[ -n "${ISL:-}" ]]; then
    EXTRA_FLAGS+=("policy.max_total_sequence_length=${ISL}")
fi

# The dapomath recipes pin max_new_tokens to a literal 8192 instead of tracking
# policy.max_total_sequence_length, so lowering ISL alone leaves generation asking for
# more than the window holds: every request gets clamped and short-ISL samples end up
# with a prompt longer than the whole SMOKEbudget and no generation region at all.
# The log-prob microbatch, in one place because two things read it: the pass
# itself and the megakernel's token cap, which is ISL x this. 4 is what the 30B
# configs set; SMOKE drops to 1 for the divisibility reason spelled out below.
if [[ "${SMOKE}" == "true" ]]; then
    LOGPROB_MBS="${LOGPROB_MBS:-${MEGA_LOGPROB_MBS:-1}}"
else
    LOGPROB_MBS="${LOGPROB_MBS:-${MEGA_LOGPROB_MBS:-4}}"
fi

if [[ "${SMOKE}" == "true" ]]; then
    EXTRA_FLAGS+=(
        # Decode steps dominate generation wall clock, and this is the cheap knob the
        # flat-KL measurement licenses: 256 tokens x 8 samples is ~2k scored tokens,
        # half of which come back bitwise equal, which still separates 0 from 3e-3.
        "policy.generation.max_new_tokens=${SMOKE_MAX_NEW_TOKENS:-256}"
        "grpo.num_prompts_per_step=${SMOKE_PROMPTS}"
        "grpo.num_generations_per_prompt=${SMOKE_GENERATIONS}"
        "policy.train_global_batch_size=${SMOKE_GBS:-$((SMOKE_PROMPTS * SMOKE_GENERATIONS))}"
        # The recipe's 16384-token engine budget is sized for ISL 10240. It is not just
        # memory: the te grouped-GEMM backend buys batch invariance by replaying each
        # MoE GEMM in ceil(max_tokens/256) fixed-M chunks of num_experts*256 rows, so a
        # budget far above the window multiplies padded GEMM work per layer.
        "policy.generation.mcore_generation_config.max_tokens=${SMOKE_MAX_TOKENS:-${ISL}}"
        # Every 30B arm here inherits logprob_batch_size=4 from
        # grpo_math_qwen30ba3b_megatron.yaml, and the log-prob microbatch has to
        # divide the per-rank sequence count. The smoke sizes above give 4x2=8
        # sequences over DP=4, so each rank gets 2, and the pass asserts
        # "Data dict size (2) is not a multiple of the provided microbatch size
        # (4)" -- which is where job 424479 died. Not specific to mega: it is
        # reachable by any arm whose smoke batch is smaller than 4 per rank.
        # 1 divides whatever the smoke sizing turns out to be, and the extra
        # microbatches cost nothing in a pass no one is timing.
        "policy.logprob_batch_size=${LOGPROB_MBS}"
    )
fi

# The megakernel's token cap is a hard reject rather than a hint, and FlashInfer
# allocates its symmetric workspace from it once per MoE layer, so it has to
# track ISL instead of sitting at the recipe's full-length value: a 1024-token
# pipeclean would otherwise pay for a 10240-token workspace in all 48 layers.
# The two sides are capped separately because they see different widths --
# generation sees whole prefills bounded by the engine's max_tokens, training
# sees the log-prob microbatch, which is the wider of the two.
if [[ "${RUN_PREFIX}" == qwen30ba3b-mega-*infopt ]]; then
    EXTRA_FLAGS+=(
        # Derived from the same LOGPROB_MBS the log-prob pass is given, so the
        # two cannot drift. The megakernel's cap is a hard reject, so a silent
        # disagreement between the cap and the actual microbatch is a crash.
        "policy.megatron_cfg.inference_mega_max_tokens_per_rank=$(( ${ISL:-10240} * LOGPROB_MBS ))"
    )
    # Only under SMOKE, where the block above shrank the engine budget this has
    # to stay paired with. Otherwise the recipe's own paired values still hold.
    if [[ "${SMOKE}" == "true" ]]; then
        EXTRA_FLAGS+=(
            "policy.generation.mcore_generation_config.inference_mega_max_tokens_per_rank=${SMOKE_MAX_TOKENS:-${ISL}}"
        )
    fi
fi

RUN_TAG="${EXP_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}}"
WANDB_RUN_NAME="${RUN_PREFIX}-zero-kl-${RUN_TAG}"
# A chain is one training run served by several jobs, so it should be one wandb
# run too: a fixed id plus resume=allow makes leg 2+ append to the curve the
# previous leg started instead of opening a new one at step 0. The id is derived
# from the run name rather than the job id for the same reason CKPT_DIR is --
# sharing EXP_TAG is what ties the legs together. wandb ids take only
# [A-Za-z0-9_-], and RUN_PREFIX carries dots for models like qwen3-1.7b.
WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_RUN_NAME//[^A-Za-z0-9_-]/-}}"
CKPT_DIR="${CKPT_DIR:-${RL_DIR}/results/${WANDB_RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${RL_DIR}/logs/${WANDB_RUN_NAME}}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" logs

# Shared Ray worker venv (MegatronPolicyWorker / mcore). Pre-built serially on the driver.
JOB_VENV_SUBDIR="${JOB_VENV_SUBDIR:-infopt-mcore}"
WORKER_VENV_FQN="nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"

if [[ -n "${NRL_RAY_VENVS_MOUNT_HOST:-}" ]]; then
    mkdir -p "${NRL_RAY_VENVS_MOUNT_HOST}/${JOB_VENV_SUBDIR}"
    NEMO_RL_VENV_CONTAINER="/opt/ray_venvs/${JOB_VENV_SUBDIR}"
    WORKER_VENV_HOST="${NRL_RAY_VENVS_MOUNT_HOST}/${JOB_VENV_SUBDIR}/${WORKER_VENV_FQN}"
    NRL_RAY_VENVS_MOUNT_SUFFIX=",${NRL_RAY_VENVS_MOUNT_HOST}:/opt/ray_venvs"
else
    NEMO_RL_VENV_CONTAINER="/opt/nemo-rl/venvs/${JOB_VENV_SUBDIR}"
    WORKER_VENV_HOST="${RL_DIR}/venvs/${JOB_VENV_SUBDIR}/${WORKER_VENV_FQN}"
    NRL_RAY_VENVS_MOUNT_SUFFIX=""
    mkdir -p "${RL_DIR}/venvs/${JOB_VENV_SUBDIR}"
fi

if [[ "${NRL_FORCE_REBUILD_VENVS}" == "true" ]]; then
    echo "NRL_FORCE_REBUILD_VENVS=true: removing worker venv at ${WORKER_VENV_HOST}"
    rm -rf "${WORKER_VENV_HOST}" || {
        echo "ERROR: failed to remove ${WORKER_VENV_HOST} (stale lustre handle?)." >&2
        echo "  Retry after the job ends, or rm from a compute node in the allocation." >&2
        exit 1
    }
fi

CHECKPOINTING_ENABLED="${CHECKPOINTING_ENABLED:-false}"

GRPO_ARGS=(
    --config "${GRPO_CONFIG}"
    "grpo.max_num_steps=${MAX_STEPS}"
    "cluster.num_nodes=${NUM_NODES}"
    "cluster.gpus_per_node=${GPUS_PER_NODE}"
    "checkpointing.enabled=${CHECKPOINTING_ENABLED}"
    "logger.log_dir=${LOG_DIR}"
    "logger.wandb_enabled=true"
    "logger.wandb.project=${WANDB_PROJECT}"
    "logger.wandb.name=${WANDB_RUN_NAME}"
    # WandbLogger splats logger.wandb straight into wandb.init(), so id/resume
    # reach it untouched -- but CLI overrides run under OmegaConf struct mode and
    # WandbConfig declares neither key, so plain assignment raises. ++ force-adds.
    "++logger.wandb.id=${WANDB_RUN_ID}"
    "++logger.wandb.resume=allow"
    "${EXTRA_FLAGS[@]}"
    "$@"
)
if [[ "${CHECKPOINTING_ENABLED}" == "true" ]]; then
    GRPO_ARGS+=(
        "checkpointing.checkpoint_dir=${CKPT_DIR}"
        "checkpointing.save_period=${SAVE_PERIOD}"
        "checkpointing.keep_top_k=${KEEP_TOP_K:-3}"
    )
fi

UV_RUN=(uv run --no-sync --extra mcore)

# Driver uses the container project venv (/opt/nemo_rl_venv), not NEMO_RL_VENV_DIR.
# uv run --no-sync skips installing new deps; sync the driver venv by default so
# imports like nemo.lens stay current when mounted code differs from the image.
DRIVER_UV_SYNC=""
if [[ "${NRL_DRIVER_UV_SYNC}" == "true" ]]; then
    DRIVER_UV_SYNC="uv sync --extra mcore && "
fi

cd "${RL_DIR}"
# Blackwell only, and the arch-conditional 'a' variants, which is what the
# FlashInfer kernels are written against. 10.3 covers GB300 (Blackwell Ultra);
# leaving it out is what killed job 423540, where the JIT-built RoPE kernel
# came back "no kernel image is available for execution on the device" on an
# sm_103 node. Hopper is dropped rather than carried: this bed is GB200/GB300,
# and every extra arch is another full compile of each JIT module, which the
# log reports as minutes apiece.
#
# Only started mattering once the mcore extra took flashinfer from a git rev,
# which meant dropping flashinfer-cubin/flashinfer-jit-cache -- the prebuilt
# binaries that used to cover sm_103 without compiling anything.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0a 10.3a}"
# FlashInfer does not read TORCH_CUDA_ARCH_LIST. Its JIT builds a
# CompilationContext of its own (jit/cpp_ext.py) whose only env input is
# FLASHINFER_CUDA_ARCH_LIST (compilation_context.py); with that unset it
# infers the arch from torch.cuda.get_device_capability(). Setting it above
# and stopping there is why job 424154 died the same way 423540 did.
#
# Stated rather than inferred because the inference is what failed, and
# because a wrong answer here is invisible: the build succeeds and the
# mismatch only surfaces as "no kernel image" at the first kernel launch.
# Unsuffixed values on purpose -- FlashInfer normalizes 10.x to the
# arch-conditional 10.xa itself, and rejects the suffixed spelling that
# TORCH_CUDA_ARCH_LIST wants.
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-10.0 10.3}"
# A persistent cache, shared across jobs, which is safe because FlashInfer
# keys the path on both its version and the arch set it built for
# (.../0.7.0/100a_103a/cached_ops) -- a GB200 build and a GB300 build cannot
# collide. Left in the container's HOME it was ephemeral instead, so every
# run paid the multi-minute build of every module, during which rank 0
# compiles while the other ranks wait on it.
#
# Point it at a fresh directory if a run ever dies mid-build and the cache
# looks poisoned; check_flashinfer_arch.py --purge clears it for this arch.
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${RL_DIR}/jit_cache/shared}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}"
export NVTE_WITH_NCCL_EP
export NVTE_CUDA_ARCHS
export CONTAINER="${CONTAINER_IMAGE}"
export MOUNTS="/lustre:/lustre,${RL_DIR}:/opt/nemo-rl${NRL_RAY_VENVS_MOUNT_SUFFIX}"
[[ -d /scratch ]] && export MOUNTS="${MOUNTS},/scratch:/scratch"
export GPUS_PER_NODE
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_CONTAINER}"

# Ray _env_builder runs uv sync inside worker processes; torch-memory-saver needs this.
export SETUP_COMMAND='if [[ -z "${TMS_CUDA_MAJOR:-}" ]] && command -v nvcc >/dev/null 2>&1; then export TMS_CUDA_MAJOR="$(nvcc --version | sed -n '"'"'s/.*release \([0-9][0-9]*\).*/\1/p'"'"' | head -1)"; fi'

WORKER_VENV_CONTAINER="${NEMO_RL_VENV_CONTAINER}/${WORKER_VENV_FQN}"
WORKER_VENV_PREBUILD="WORKER_VENV=${WORKER_VENV_CONTAINER} && \
if [[ -x \${WORKER_VENV}/bin/python ]] && ! \${WORKER_VENV}/bin/python -c 'import ray, megatron.bridge, transformer_engine' 2>/dev/null; then \
  echo 'Worker venv corrupt or incomplete; removing and rebuilding...' && rm -rf \${WORKER_VENV}; \
fi && \
if [[ ! -x \${WORKER_VENV}/bin/python ]]; then \
  echo \"Pre-building worker venv (single-process) at \${WORKER_VENV}...\" && \
  uv venv --allow-existing \${WORKER_VENV} && \
  UV_PROJECT_ENVIRONMENT=\${WORKER_VENV} uv sync --directory /opt/nemo-rl --extra mcore && \
  \${WORKER_VENV}/bin/python -c 'import ray, megatron.bridge, transformer_engine; print(\"worker venv OK\", transformer_engine.__version__)'; \
fi && \
export NRL_FORCE_REBUILD_VENVS=false"

CACHE_EXPORT=""
if [[ "${NRL_USE_WARM_UV_CACHE}" == "true" ]]; then
    mkdir -p "${NRL_WARM_UV_CACHE_DIR}"
    export UV_CACHE_DIR="${NRL_WARM_UV_CACHE_DIR}"
    CACHE_EXPORT="export UV_CACHE_DIR=${UV_CACHE_DIR} && "
fi

# Non-colocated infopt recipes use refit_backend=nvshmem; Megatron's copy
# service hard-requires NVSHMEM_MAX_CTAS=2 (hangs/fails otherwise).
#
# Skipped when the recipe refits over nccl instead, because the cap is global
# rather than scoped to the copy service: the FlashInfer megakernel carries its
# own expert-parallel traffic over NVSHMEM, so leaving this set would throttle
# the kernel this stack exists to measure.
NVSHMEM_EXPORT=""
if [[ "${STACK}" == "infopt" && "${NVSHMEM_REFIT}" == "true" ]]; then
    NVSHMEM_EXPORT="export NVSHMEM_MAX_CTAS=2 && "
fi

# Per-layer train/generation fingerprints, for locating which of the 48 layers
# separates first. Off unless MEGA_PARITY_DUMP=1: the dump adds a device-to-host
# copy per MoE forward, and the comparison it feeds is a debugging step rather
# than something a measurement run should carry.
#
# Defaults to rank 0 and the MoE output only. Raise MEGA_PARITY_DUMP_MAX_MB if
# the dump truncates before generation and the log-prob pass have both run --
# the writer stops at the cap rather than growing without bound.
PARITY_DUMP_EXPORT=""
if [[ "${MEGA_PARITY_DUMP:-0}" == "1" ]]; then
    MEGA_PARITY_DUMP_DIR="${MEGA_PARITY_DUMP_DIR:-${RL_DIR}/parity_dumps/${JOB_NAME:-run}-$$}"
    mkdir -p "${MEGA_PARITY_DUMP_DIR}"
    PARITY_DUMP_EXPORT="export MEGA_PARITY_DUMP_DIR=${MEGA_PARITY_DUMP_DIR} && \
export MEGA_PARITY_DUMP_RANKS=${MEGA_PARITY_DUMP_RANKS:-0} && \
export MEGA_PARITY_DUMP_MAX_MB=${MEGA_PARITY_DUMP_MAX_MB:-512} && "
    if [[ -n "${MEGA_PARITY_DUMP_LAYERS:-}" ]]; then
        PARITY_DUMP_EXPORT="${PARITY_DUMP_EXPORT}export MEGA_PARITY_DUMP_LAYERS=${MEGA_PARITY_DUMP_LAYERS} && "
    fi
    echo "-- parity dump enabled -> ${MEGA_PARITY_DUMP_DIR}"
fi

# Brackets the three phases inside a reshard swap -- buffer harmonization, plan
# execution, and the destination-only weight refresh -- so a refit that stalls
# says which phase it stalled in. Both sides print, so a phase one side entered
# and the other did not is visible directly. On by default for mega: it is a few
# lines per refit, and without it a stalled refit prints nothing at all.
REFIT_TRACE_EXPORT=""
if [[ "${MEGA_REFIT_TRACE:-1}" == "1" ]]; then
    REFIT_TRACE_EXPORT="export MEGA_REFIT_TRACE=1 && "
fi
# Escape hatch for the post-refit weight-refresh wait, in case scoping it to the
# refit's own stream is still not enough to let it drain. See refit.py.
if [[ "${MEGA_REFIT_SKIP_SYNC:-0}" == "1" ]]; then
    REFIT_TRACE_EXPORT="${REFIT_TRACE_EXPORT}export MEGA_REFIT_SKIP_SYNC=1 && "
    echo "-- refit weight-refresh sync disabled"
fi

export COMMAND="${CACHE_EXPORT}\
export NEMO_RL_VENV_DIR=${NEMO_RL_VENV_CONTAINER} && \
export NRL_MINF_SHARED_CLUSTER=${NRL_MINF_SHARED_CLUSTER:-0} && \
${NVSHMEM_EXPORT}${PARITY_DUMP_EXPORT}${REFIT_TRACE_EXPORT}\
export PYTHONUNBUFFERED=1 && \
export UV_HTTP_TIMEOUT=900 && \
export HF_HOME=${HF_HOME} && \
export TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' && \
export FLASHINFER_CUDA_ARCH_LIST='${FLASHINFER_CUDA_ARCH_LIST}' && \
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE} && \
export NVTE_WITH_NCCL_EP=${NVTE_WITH_NCCL_EP} && \
export NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS} && \
export HF_TOKEN=${HF_TOKEN} && \
export WANDB_API_KEY=${WANDB_API_KEY} && \
export WANDB_ENTITY=${WANDB_ENTITY} && \
export CUDA_DEVICE_MAX_CONNECTIONS=1 && \
cd /opt/nemo-rl && \
${DRIVER_UV_SYNC}\
${WORKER_VENV_PREBUILD} && \
${UV_RUN[*]} examples/run_grpo.py ${GRPO_ARGS[*]}"

echo "MODEL=${RUN_PREFIX} STACK=${STACK} CONFIG=${GRPO_CONFIG}"
echo "ZERO_TRAIN_GEN_MISMATCH=${ZERO_TRAIN_GEN_MISMATCH}"
echo "NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS} (host rm only; Ray sees false after pre-build)"
echo "NVTE_WITH_NCCL_EP=${NVTE_WITH_NCCL_EP} NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}"
echo "NRL_DRIVER_UV_SYNC=${NRL_DRIVER_UV_SYNC}"
if [[ "${NRL_USE_WARM_UV_CACHE}" == "true" ]]; then
    echo "NRL_USE_WARM_UV_CACHE=true UV_CACHE_DIR=${UV_CACHE_DIR}"
fi
echo "WORKER_VENV=${WORKER_VENV_HOST}"
echo "VENV_DIR=${NEMO_RL_VENV_CONTAINER}"
[[ -n "${ISL:-}" ]] && echo "ISL=${ISL} (policy.max_total_sequence_length)"
if [[ "${SMOKE}" == "true" ]]; then
    SMOKE_MAX_TOKENS_EFF="${SMOKE_MAX_TOKENS:-${ISL}}"
    echo "SMOKE=true prompts=${SMOKE_PROMPTS} generations=${SMOKE_GENERATIONS} gbs=${SMOKE_GBS:-$((SMOKE_PROMPTS * SMOKE_GENERATIONS))} max_new_tokens=${SMOKE_MAX_NEW_TOKENS:-256} max_tokens=${SMOKE_MAX_TOKENS_EFF} (te chunks=$(( (SMOKE_MAX_TOKENS_EFF + 255) / 256 )))"
fi
echo "NODES=${NUM_NODES}x${GPUS_PER_NODE}  MAX_STEPS=${MAX_STEPS}  RUN=${WANDB_RUN_NAME}"
# The config checks that rejected jobs 422181 and 423215 both run inside the
# policy worker, so a bad key costs a full allocation and ~8 minutes to find.
# Printed with the assembled overrides so a pre-submit check sees this exact
# config rather than the recipe's defaults.
echo "PREFLIGHT: ${RL_DIR}/venvs/${JOB_VENV_SUBDIR}/${WORKER_VENV_FQN}/bin/python ${SCRIPT_DIR}/preflight_zero_kl.py --config ${GRPO_CONFIG} ${EXTRA_FLAGS[*]}"
echo "COMMAND: ${COMMAND}"
# Off by default, so a launch is a launch. With it set the assembled config and
# the two commands above are printed and nothing is submitted, which is the
# cheap way to see what a new MODEL or override actually resolves to before it
# costs an allocation.
if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "DRY_RUN=true: not submitting. Run the PREFLIGHT line above first."
    exit 0
fi
source ray.sub


