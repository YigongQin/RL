# Bitwise-deterministic true on-policy GRPO on the inference-optimized Megatron engine (EP8/TP1)

Generation logprobs bitwise-identical to the training/scoring forward:
TensorBoard `train/gen_kl_error == 0.000000e+00` over full-lr GRPO
(18/18 steps, prod bed seq4096/GBS512, 5,217 gen tok/s/GPU = 1.02x tax vs
matched non-deterministic; small bed 20/20 at 1,368 tok/s/GPU).

## Megatron-LM dependency

This reproduction branch pins Megatron-LM **main** at or after
[NVIDIA/Megatron-LM#6521](https://github.com/NVIDIA/Megatron-LM/pull/6521)
(`batch_invariant_backend=te_native`, batch-invariant vLLM fused MoE).

Megatron-Bridge is bumped to latest main; nested Megatron-LM is bumped to
latest main (includes #6521).

The original [NeMo-RL PR #3531](https://github.com/NVIDIA-NeMo/RL/pull/3531)
used a companion Megatron-LM fork; this branch replaces that fork with upstream
#6521 and wires batch-invariant mode through yaml instead of `NRL_BATCH_INVARIANT=1`.

## What this branch adds (NeMo-RL side, ported from #3531)

- fail-fast validation for colocated-vs-dedicated mcore generation config
- non-colocated (dedicated-node) megatron generation support
- deterministic scoring path: fused logprob (`float() -> log_softmax -> gather`) via `NRL_FUSED_TRAIN_LOGPROB=1`
- refit param-gather sync barrier via `NRL_REFIT_PARAM_SYNC=1`
- batch-invariant activation via `megatron_cfg.batch_invariant_mode` + `batch_invariant_backend: te_native`
- inference-optimized GPT layer spec hook in `setup.py` (Bridge does not assign it for GPT yet)
- certified recipe: `examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_ep8tp1.yaml`
- determinism gate: `examples/gen_kl_harness.py`

## Quick repro

```bash
# After uv sync with mcore extra on 8 nodes (train + dedicated inference):
python examples/gen_kl_harness.py \
  --config examples/configs/recipes/grpo_math_qwen30ba3b_megatron_det_ep8tp1.yaml
```

Expect `gen_kl_error: 0.0` in the harness JSONL output.

## Scope

EP=8, TP=1 inference layout, decode CUDA graphs. Router replay OFF.
