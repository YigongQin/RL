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

## Run (cheap gate)

```bash
python examples/gen_kl_harness.py \
  --config examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_ep8tp1.yaml \
  --num-batches 1
```

Check `gen_kl_harness_results.jsonl` for `"gen_kl_error": 0.0`.

## Key yaml knobs

```yaml
policy:
  megatron_cfg:
    batch_invariant_mode: true
    batch_invariant_backend: te_native
    attention_backend: flash
    flash_attention_version: 4
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

## What we did **not** port from #3531

- utkarsh Megatron-LM fork (replaced by upstream #6521)
- `NRL_BATCH_INVARIANT=1` env gate (replaced by yaml `batch_invariant_mode`)
- Debug probes (`NRL_PROBE_*`, layer dumps) — left env-gated, default off
