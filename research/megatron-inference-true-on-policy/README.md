# Megatron-Inference True On-Policy Study

Colocated Megatron inference GRPO with `zero_train_gen_mismatch` (`gen_kl_error` → 0).

## Setup

```bash
cp research/megatron-inference-true-on-policy/.env.template research/megatron-inference-true-on-policy/.env
# Edit .env: RL_DIR, CONTAINER_IMAGE, HF_TOKEN, HF_HOME, WANDB_API_KEY, WANDB_PROJECT
```

## Run

**Full GRPO** (training loop; recipe knobs live in yaml):

```bash
cd research/megatron-inference-true-on-policy

sbatch --export=MODEL=qwen1.5b,PRECISION=bf16  run_zero_kl_precision.sh
sbatch --export=MODEL=qwen30ba3b,PRECISION=bf16  run_zero_kl_precision.sh
sbatch --export=MODEL=qwen30ba3b,PRECISION=mxfp8 run_zero_kl_precision.sh
sbatch --export=MODEL=nanov3,PRECISION=bf16  run_zero_kl_precision.sh
sbatch --export=MODEL=nanov3,PRECISION=mxfp8 run_zero_kl_precision.sh
sbatch --export=MODEL=qwen30ba3b,ZERO_TRAIN_GEN_MISMATCH=false run_zero_kl_precision.sh
```

**Cheap gen_kl gate** (one rollout + score, no optimizer step):

```bash
sbatch --export=MODEL=infopt,NUM_BATCHES=1 run_gen_kl_harness.sh
sbatch --nodes=1 --export=MODEL=qwen1.5b run_gen_kl_harness.sh
```

Recipes:
- `examples/configs/recipes/llm/grpo-qwen1.5b-megatron-zero-train-gen-kl.yaml`
- `examples/configs/recipes/llm/grpo-dapomath17k-qwen-30ba3b-megatron-zero-train-gen-kl.yaml`
  (`dapo_long_cot: true` for long CoT DAPO grading)
- `examples/configs/recipes/llm/grpo-nanov3-30ba3b-megatron-zero-train-gen-kl.yaml`
