# Inference-optimized zero-KL reproduction branch

Branch: `yigongq/infopt-zero-kl-repro`

## Stack

| Layer | Source |
|-------|--------|
| Megatron-LM | `main` ≥ #6521 (`te_native` batch-invariant backend) |
| Megatron-Bridge | latest `main` |
| NeMo-RL | #3531 minimal port + config-driven batch-invariant |

This is **not** the colocated `zero_train_gen_mismatch` / `transformer_engine` path.
It uses **dedicated-node** `inference_optimized` generation with vLLM fused MoE.

## Setup

```bash
git checkout yigongq/infopt-zero-kl-repro
git submodule update --init --recursive
./scripts/init_infopt_repro_mcore.sh   # Megatron-LM main ≥ #6521
uv sync --extra mcore
```

## Debug setups (start here)

| MODEL | Nodes | What it tests |
|-------|-------|---------------|
| `infopt-debug-1n` (default) | **1** | Qwen1.5B, non-colocated infopt + batch-invariant, 4 train + 4 inference GPUs |
| `infopt-debug-2n` | **2** | Qwen30B MoE EP8, 1 train + 1 inference node, tiny batches |
| `infopt-shared-1n` | **1** | Qwen30B on shared GPUs (`NRL_MINF_SHARED_CLUSTER=1`, needs ~180GB/GPU) |
| `infopt` | **8** | Production #3531 repro |

```bash
cd research/megatron-inference-true-on-policy

# recommended first gate (1 node)
sbatch run_gen_kl_harness.sh

# MoE infopt on 2 nodes
sbatch --nodes=2 --export=MODEL=infopt-debug-2n run_gen_kl_harness.sh

# production scale
sbatch --nodes=8 --export=MODEL=infopt run_gen_kl_harness.sh
```

Recipes:
- `examples/configs/recipes/grpo_math_qwen15b_megatron_det_infopt_debug_1n8g.yaml`
- `examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_infopt_debug_2n8g.yaml`

## Run (cheap gate)

From repo root (local):

```bash
python examples/gen_kl_harness.py \
  --config examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_ep8tp1.yaml \
  --num-batches 1
```

On cluster (same venv patch + Ray launcher as colocated study):

```bash
cd research/megatron-inference-true-on-policy
sbatch --export=MODEL=infopt,NUM_BATCHES=1 run_gen_kl_harness.sh
# colocated cheap gate (1 node):
sbatch --nodes=1 --export=MODEL=qwen1.5b run_gen_kl_harness.sh
```

Check the jsonl output for `"gen_kl_error": 0.0` (path printed at job start).

## Key yaml knobs

```yaml
policy:
  megatron_cfg:
    batch_invariant_mode: true
    batch_invariant_backend: te_native
    attention_backend: flash
    flash_attention_version: 4          # wired in setup.py → model_cfg
    moe_permute_fusion: false           # required for det MoE combine (also forced in setup.py)
    moe_pad_experts_for_cuda_graph_inference: false
    env_vars:
      NRL_FUSED_TRAIN_LOGPROB: "1"
      NRL_REFIT_PARAM_SYNC: "1"
  generation:
    colocated:
      enabled: false
    mcore_generation_config:
      transformer_impl: inference_optimized
      inference_grouped_gemm_backend: vllm
      inference_moe_token_dispatcher_type: nvls
      logprobs_mode: raw_logprobs
```

NeMo-RL `setup.py` also:
- calls `enable_batch_invariant_mode(backend=te_native)` on every worker (TE GEMM via workspace starvation)
- forces unfused MoE permute + no expert padding when `batch_invariant_mode: true`
- defaults `flash_attention_version: 4` if omitted under batch-invariant

## What we did **not** port from #3531

- utkarsh Megatron-LM fork (replaced by upstream #6521)
- `NRL_BATCH_INVARIANT=1` env gate (replaced by yaml `batch_invariant_mode`)
- colocated-only `core_patches` stack (infopt reuses `apply_te_bik_attention_assert_skip_patch` on TE 2.15 until container bumps TE)
- Debug probes (`NRL_PROBE_*`, layer dumps) — left env-gated, default off
