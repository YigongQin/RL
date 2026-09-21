# Zero train/generation KL with Megatron inference

Setup and run commands for the mega-parity branch. What the work is and why
each piece is needed lives in `DESIGN-moe-flashinfer-mega-prototype.md` and in
`megatron/core/inference/moe/mega/QUANTIZED_BLOCKERS.md` inside the
Megatron-LM submodule; this file is only how to get a working tree and launch
something.

## Getting the tree

Three repos, all on forks, nested two deep:

```
NeMo RL                       yigongq/mega-parity
└── Megatron-Bridge           yigongq/inf-backend
    └── Megatron-LM           yigongq/inf-backend
```

Megatron-Bridge is forked for a structural reason rather than because it
changed much: Megatron-LM is a submodule *of Bridge*, and a nested submodule's
URL can only be set in the repo that declares it. NeMo RL cannot redirect it
from the parent, so reaching the Megatron-LM parity branch through a recursive
clone requires a Bridge fork regardless. Bridge itself carries one dependency
change.

```bash
git clone --recursive -b yigongq/mega-parity \
    https://github.com/YigongQin/RL.git
```

Verify before doing anything else — a partially-initialised submodule fails
much later, during a `uv sync`, as a confusing import error:

```bash
git -C RL submodule status --recursive | grep -i megatron
```

Both lines must be prefixed with a space. A `-` means uninitialised and a `+`
means the checkout does not match the recorded commit.

### If you already cloned before the URLs changed

Changed submodule URLs do not propagate into existing clones. `git pull` alone
leaves you fetching the old upstream and silently on the wrong commit:

```bash
cd RL
git fetch origin && git checkout yigongq/mega-parity && git pull
git submodule sync --recursive      # adopt the new URLs -- this is the step people miss
git submodule update --init --recursive
```

`shallow = true` was removed from the Bridge submodule deliberately. A shallow
submodule clone fetches only the branch tip, so the recorded commit becomes
unreachable as soon as that tip moves — which works for whoever pushed and
fails for everyone else.

## Environment

Everything runs inside the container. Copy `.env` (not in git; it holds
`HF_TOKEN` and `WANDB_API_KEY`) and set at least:

```
RL_DIR=/path/to/RL
CONTAINER_IMAGE=/path/to/image.sqsh
HF_HOME=...
WANDB_ENTITY=...  WANDB_PROJECT=...
```

Then, from this directory:

```bash
set -a && source .env && set +a
```

### Worker venv

The `MegatronPolicyWorker` venv is what actually imports Megatron-LM, so
submodule edits are invisible until it is rebuilt. Run this **as root inside
the container** — the venv's interpreter is a symlink into `/root/.local`, and
as a normal user `uv` reports it as a permission error on a path that appears
to exist:

```bash
UV_PROJECT_ENVIRONMENT=$WORKER_VENV uv sync --directory "$RL_DIR" --extra mcore
```

After a `pyproject.toml` change, relock first:

```bash
cd "$RL_DIR" && uv lock
```

FlashInfer comes from a git rev, not a release: `flashinfer.moe_ep` — the
expert-parallel megakernels — is in no published wheel. `uv lock` therefore
takes minutes and will fail on any exact `flashinfer-python` pin reintroduced
upstream.

## Unit tests

Run from inside the Megatron-LM submodule, in the container. `FI_OVERLAY=0`
selects the venv's FlashInfer rather than a local source overlay; use it unless
you are deliberately testing an uncommitted FlashInfer.

```bash
cd $RL_DIR/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM

# mega MoE train/generation parity: forward parity + a fwd/bwd loop, ~3 min
FI_OVERLAY=0 ./scripts/local/run_mega_training_tests.sh

# one phase at a time: weights (CPU-only, no Blackwell needed), parity, gen,
# block (whole transformer layer), update (the RL step-2 repro), drift
# (wgrad against the mega forward), all
FI_OVERLAY=0 PHASES=gen     ./scripts/local/run_mega_training_tests.sh
FI_OVERLAY=0 PHASES=weights ./scripts/local/run_mega_training_tests.sh

# bf16 and mxfp8 are both exercised by default; narrow or widen with
FI_OVERLAY=0 ./scripts/local/run_mega_training_tests.sh --precision mxfp8
FI_OVERLAY=0 ./scripts/local/run_mega_training_tests.sh --precision all  # adds nvfp4

# Mamba train/generation parity (single rank, seconds)
./scripts/local/run_mamba_parity_tests.sh
```

Logs land in `logs/` inside the submodule. Each suite prints its findings as
`[mega-metric]` / `[mamba-parity]` lines on success as well as failure — the
numbers are the output, not just the pass.

## Config preflight

Recipe mistakes otherwise surface forty minutes into a multi-node job. This
resolves and validates the config locally, the same way the workers will:

```bash
python preflight_zero_kl.py --config ../../examples/configs/recipes/llm/\
grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl-mega-infopt-noncolocated.yaml
```

`check_flashinfer_arch.py` is separate and narrower: it JIT-builds one
FlashInfer kernel and runs it, to confirm `FLASHINFER_CUDA_ARCH_LIST` covers
the device. FlashInfer ignores `TORCH_CUDA_ARCH_LIST`, which is how a GB300
run reaches a "no kernel image is available" failure with an arch list that
looks correct.

## Launching

`run_zero_kl_precision1.sh` is the SLURM entrypoint; `MODEL` selects the arm
and its header documents every one. The KL to read is `gen_kl_error` at step 0.

```bash
# one-step gate: is the bed alive and is step-0 KL zero
sbatch -N 2 --export=ALL,SMOKE=true,MODEL=qwen30ba3b-mega-infopt,\
NRL_FORCE_REBUILD_VENVS=false ./run_zero_kl_precision1.sh

# short real run
MAX_STEPS=2 sbatch -N 2 --export=ALL,MODEL=qwen30ba3b-mega-infopt,\
NRL_FORCE_REBUILD_VENVS=false ./run_zero_kl_precision1.sh

# megakernel at MXFP8 on both sides (bitwise-equal forwards, straight-through backward)
MAX_STEPS=2 sbatch -N 2 --export=ALL,MODEL=qwen30ba3b-mega-mxfp8-infopt,\
NRL_FORCE_REBUILD_VENVS=false ./run_zero_kl_precision1.sh
```

Set `NRL_FORCE_REBUILD_VENVS=true` on the first job after a dependency change,
and back to `false` afterwards — a rebuild on every job wastes roughly fifteen
minutes each time. Use `-N 2` for any `*-infopt` arm: those are non-colocated
and want a train node plus a dedicated generation node.

For runs longer than the wall clock, the script's header has a chaining recipe.
Two things make a chain a chain rather than a treadmill: `EXP_TAG`, without
which every leg invents its own checkpoint directory and restarts from step 0,
and `CHECKPOINTING_ENABLED=true`, without which there is nothing to resume from.
