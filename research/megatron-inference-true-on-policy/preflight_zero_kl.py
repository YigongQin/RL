#!/usr/bin/env python3
"""Run a recipe's config validation locally, without SLURM, Ray, or GPUs.

Both zero-KL bring-up failures so far were config errors that took eight
minutes and two nodes to surface, because the checks that catch them run inside
the policy worker, which is the last thing to start. Nothing about either check
needs a GPU or a cluster, so this runs the same two of them here:

  1. enable_zero_train_gen_kl  -- the zero-KL resolver and its validation
     (package and Megatron-Core commit gates, backend compatibility, recompute,
     the mega token caps). Caught job 422181's commit-gate failure.
  2. validate_and_set_config + ConfigContainer.validate -- builds the real
     Megatron-Bridge config and runs TransformerConfig.__post_init__. Caught
     job 423215's transformer_impl failure.
  3. the same two, against the config a dedicated generation model builds from
     (merged_inference_megatron_cfg). Non-colocated generation runs its own
     policy whose megatron_cfg has mcore_generation_config layered on top, so
     checks 1 and 2 passing for training says nothing about it. Caught job
     430144's cuda_graph_impl failure.

Config loading mirrors examples/run_grpo.py exactly -- same loader, same hydra
override parsing, same resolution -- so what is checked is what would run.
Pass the config and overrides exactly as they would be passed to run_grpo:

    ./preflight_zero_kl.py --config examples/configs/recipes/llm/<recipe>.yaml \
        policy.max_total_sequence_length=1024 ...

Run it before sbatch, from inside the container, using the worker venv's
python. run_zero_kl_precision1.sh prints the matching invocation for whatever
MODEL and ISL you gave it, so the overrides checked are the ones it would send.

Step 2 needs the model's HF config, from the cache or the hub. Where it cannot
be reached, that step is reported as skipped rather than passed: step 1 alone
would not have caught the transformer_impl error.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# Worker processes reach these through the installed package; a bare script run
# from this directory needs the repo root on the path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_config(config_path: str, overrides: list[str]) -> dict:
    """Load and resolve a recipe the way run_grpo.main does."""
    from omegaconf import OmegaConf

    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    return OmegaConf.to_container(config, resolve=True)


def _check_zero_kl(policy_cfg: dict) -> tuple[bool, str | None]:
    """Resolve and validate the zero-KL knobs, as the worker does at startup."""
    from nemo_rl.models.megatron.setup import enable_zero_train_gen_kl

    if not policy_cfg.get("megatron_cfg", {}).get("zero_train_gen_mismatch"):
        return True, "not a zero_train_gen_mismatch recipe; nothing to check"
    # apply_kernels touches CUDA, which is the one thing this script must not
    # do. The resolver and its validation are unaffected by the distinction.
    enable_zero_train_gen_kl(policy_cfg, apply_kernels=False)
    return True, None


def _check_megatron_config(policy_cfg: dict) -> tuple[bool, str | None]:
    """Build the Megatron-Bridge config, running TransformerConfig.__post_init__."""
    from nemo_rl.models.megatron.setup import validate_and_set_config

    hf_model_name = policy_cfg["model_name"]
    runtime = validate_and_set_config(
        policy_cfg,
        0,  # rank
        hf_model_name,
        None,  # pretrained_path
        None,  # weights_path
        None,  # optimizer_path
        skip_weight_load=True,
    )
    runtime.megatron_cfg.validate()
    return True, None


def _check_generation_config(policy_cfg: dict) -> tuple[bool, str | None]:
    """Validate the config the dedicated generation workers build from.

    Non-colocated Megatron generation stands up a second policy whose
    megatron_cfg is merged_inference_megatron_cfg: the training values with
    mcore_generation_config layered on top. Its workers then run the same
    zero-KL resolve and the same Megatron-Bridge build that the two checks
    above run for training -- but against that merge, where every knob the
    recipe set under mcore_generation_config has replaced its training value.

    Job 430144 died here and only here: the generation model graphs
    (cuda_graph_impl='local') while training does not, and a train-side gate
    read the merge's value as if it were the training one.
    """
    from copy import deepcopy

    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg
    from nemo_rl.models.megatron.setup import enable_zero_train_gen_kl

    generation = policy_cfg.get("generation") or {}
    if generation.get("backend") != "megatron":
        return True, "generation backend is not megatron; no second model"
    if (generation.get("colocated") or {}).get("enabled", True):
        return True, "colocated generation shares the training model"

    gen_policy_cfg = deepcopy(policy_cfg)
    gen_policy_cfg["megatron_cfg"] = merged_inference_megatron_cfg(gen_policy_cfg)
    gen_mcore = gen_policy_cfg["megatron_cfg"]
    if gen_mcore.get("zero_train_gen_mismatch"):
        enable_zero_train_gen_kl(gen_policy_cfg, apply_kernels=False)
    _check_megatron_config(gen_policy_cfg)
    return True, (
        f"cuda_graph_impl={gen_mcore.get('cuda_graph_impl', 'none')!r}, "
        f"moe_mega_training_forward={gen_mcore.get('moe_mega_training_forward', False)!r}"
    )


def _check_batch_shapes(config: dict) -> tuple[bool, str | None]:
    """Check the batch divisibility rules that otherwise fire deep into a run.

    Job 424479 spent eight minutes and two nodes generating a full rollout
    before the log-prob pass rejected it: the per-rank sequence count did not
    divide the log-prob microbatch. None of that arithmetic needs a GPU, a
    cluster, or a rollout -- it is fixed the moment the config resolves.

    The rules are transcribed from where they actually assert, so the error
    here names the same quantity the run would:

      grpo.py:5127                 rollout size divisible by DP
      batched_data_dict.py:1011    per-rank size divisible by the microbatch
      lm_policy.py:294             tmbs == logprob mbs, but only at TP >= 4

    Data-parallel size is derived rather than read, because nothing writes it
    down: it is whatever GPUs are left after generation takes its share,
    divided by the model-parallel factors.
    """
    grpo = config["grpo"]
    policy = config["policy"]
    cluster = config["cluster"]
    mcore = policy.get("megatron_cfg", {}) or {}

    prompts = grpo["num_prompts_per_step"]
    generations = grpo["num_generations_per_prompt"]
    rollout = prompts * generations

    gbs = policy["train_global_batch_size"]
    tmbs = policy["train_micro_batch_size"]
    logprob_mbs = policy.get("logprob_batch_size", tmbs)
    seq_len = policy["max_total_sequence_length"]

    tp = mcore.get("tensor_model_parallel_size", 1)
    pp = mcore.get("pipeline_model_parallel_size", 1)
    cp = mcore.get("context_parallel_size", 1)

    total_gpus = cluster["gpus_per_node"] * cluster["num_nodes"]
    colocated = policy.get("generation", {}).get("colocated", {}) or {}
    if colocated.get("enabled", True):
        train_gpus = total_gpus
    else:
        resources = colocated.get("resources", {}) or {}
        gen_gpus = resources.get("gpus_per_node", 0) * resources.get("num_nodes", 0)
        train_gpus = total_gpus - gen_gpus
    dp = train_gpus // (tp * pp * cp)

    print(
        f"     rollout={rollout} ({prompts}x{generations})  gbs={gbs}  tmbs={tmbs}  "
        f"logprob_mbs={logprob_mbs}  seq_len={seq_len}"
    )
    print(
        f"     gpus={total_gpus} train={train_gpus} gen={total_gpus - train_gpus}  "
        f"tp={tp} pp={pp} cp={cp} -> dp={dp}"
    )

    problems = []
    if dp < 1:
        problems.append(f"no GPUs left for training: {train_gpus} after generation")
        dp = 1
    if rollout % generations:
        problems.append(f"rollout {rollout} not divisible by num_generations {generations}")
    if rollout % dp:
        problems.append(f"rollout {rollout} not divisible by dp {dp} (grpo.py:5127)")
    if gbs > rollout:
        problems.append(f"train_global_batch_size {gbs} exceeds rollout {rollout}")
    if gbs % dp:
        problems.append(f"train_global_batch_size {gbs} not divisible by dp {dp}")
    else:
        per_rank = gbs // dp
        print(f"     per-rank sequences={per_rank}")
        if per_rank % tmbs:
            problems.append(f"per-rank {per_rank} not divisible by tmbs {tmbs}")
        if per_rank % logprob_mbs:
            problems.append(
                f"per-rank {per_rank} not divisible by logprob_batch_size "
                f"{logprob_mbs} (batched_data_dict.py:1011) -- this is job 424479"
            )
    if tp >= 4 and tmbs != logprob_mbs:
        problems.append(
            f"TP={tp} requires train_micro_batch_size ({tmbs}) == "
            f"logprob_batch_size ({logprob_mbs}) (lm_policy.py:294)"
        )

    # Mega's token cap is a hard reject, so it has to cover the widest forward
    # each side sees: the log-prob microbatch for training, the engine's own
    # window for generation.
    train_cap = mcore.get("inference_mega_max_tokens_per_rank")
    if train_cap is not None:
        needed = seq_len * logprob_mbs
        if train_cap < needed:
            problems.append(
                f"train inference_mega_max_tokens_per_rank {train_cap} < "
                f"seq_len x logprob_mbs = {needed}"
            )
        gen_cfg = policy.get("generation", {}).get("mcore_generation_config", {}) or {}
        gen_cap = gen_cfg.get("inference_mega_max_tokens_per_rank")
        gen_window = gen_cfg.get("max_tokens")
        if gen_cap is not None and gen_window is not None and gen_cap < gen_window:
            problems.append(
                f"generation inference_mega_max_tokens_per_rank {gen_cap} < "
                f"max_tokens {gen_window}"
            )
        print(f"     mega caps: train={train_cap} gen={gen_cap} (window {gen_window})")

    if problems:
        raise ValueError("; ".join(problems))
    return True, f"dp={dp}, per-rank={gbs // dp}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the recipe YAML")
    args, overrides = parser.parse_known_args()

    print(f"-- config:    {args.config}")
    if overrides:
        print(f"-- overrides: {' '.join(overrides)}")

    try:
        config = _load_config(args.config, overrides)
    except Exception:
        print("\nFAIL: config did not load or resolve\n")
        traceback.print_exc()
        return 1

    policy_cfg = config["policy"]
    failed = False

    # Batch shapes first: it is the cheapest check and the one whose failure
    # costs a whole rollout to discover.
    for label, check, optional, arg in (
        ("batch shapes", _check_batch_shapes, False, config),
        ("zero-KL resolve + validate", _check_zero_kl, False, policy_cfg),
        ("Megatron config + post_init", _check_megatron_config, True, policy_cfg),
        ("generation model config", _check_generation_config, True, policy_cfg),
    ):
        try:
            _, note = check(arg)
        except Exception as exc:
            # An optional check cannot distinguish a real config error from a
            # missing HF cache by exception type, so it reports the error and
            # says it did not run rather than claiming either outcome.
            if optional and not isinstance(exc, ValueError):
                print(f"-- SKIP  {label}: could not run ({type(exc).__name__}: {exc})")
                continue
            print(f"-- FAIL  {label}")
            print()
            traceback.print_exc()
            print()
            failed = True
            continue
        print(f"-- ok    {label}" + (f" ({note})" if note else ""))

    if failed:
        print("\npreflight FAILED -- fix the config before submitting\n")
        return 1
    print("\npreflight passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
