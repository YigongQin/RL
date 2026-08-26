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
#
# gen_kl_harness.py — cheap replica of the NeMo-RL gen_kl measurement path.
#
# Reuses setup() / run_multi_turn_rollout / get_logprobs / calculate_kl exactly as
# grpo_train does, without the training loop.
#
# Config access rules (must match examples/run_grpo.py + grpo_train):
#   master_config.grpo.*          -> GRPOConfig BaseModel (attributes)
#   master_config.loss_fn.*       -> ClippedPGLossConfig BaseModel (attributes)
#   master_config.policy[...]     -> PolicyConfig TypedDict (subscript)
#   master_config.logger[...]     -> logger TypedDict (subscript)
#   setup() returns 13 values     -> match run_grpo.py unpack

import argparse
import json
import os
import pprint
import time

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _preserve_router_replay_routed_experts,
    add_grpo_token_loss_masks_and_generation_logprobs,
    refit_policy_generation,
    setup,
)
from nemo_rl.algorithms.utils import calculate_kl, get_tokenizer, masked_mean
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    parser = argparse.ArgumentParser(description="gen_kl determinism harness")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="How many dataloader batches to measure (each = one gen+score+kl pass)",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=os.environ.get("NRL_GENKL_JSONL", "gen_kl_harness_results.jsonl"),
        help="Where to append per-batch results",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _report(tag, gen_lp, prev_lp, token_mask, sample_mask, kl_type):
    """Compute gen_kl as ClippedPGLossFn, plus diagnostics."""
    gen = gen_lp[:, 1:]
    prev = prev_lp[:, 1:]
    mask = token_mask[:, 1:] * sample_mask.unsqueeze(-1)
    global_valid_toks = mask.sum()

    gen_kl = calculate_kl(
        logprobs=gen,
        logprobs_reference=prev,
        kl_type=kl_type,
        input_clamp_value=None,
        output_clamp_value=None,
    )
    gen_kl_mean = masked_mean(
        gen_kl, mask, global_normalization_factor=global_valid_toks
    ).item()

    lp_abs = torch.abs(gen - prev)
    m = mask.bool()
    n_valid = int(m.sum().item())
    if n_valid == 0:
        return {
            "tag": tag,
            "gen_kl_error": gen_kl_mean,
            "n_valid_toks": 0,
        }
    masked_abs = lp_abs[m]
    max_abs = masked_abs.max().item()
    mean_abs = masked_abs.mean().item()
    n_exact = int((masked_abs == 0.0).sum().item())
    mult_prob_error = masked_mean(
        torch.exp(lp_abs * mask), mask, global_normalization_factor=global_valid_toks
    ).item()

    return {
        "tag": tag,
        "gen_kl_error": gen_kl_mean,
        "mult_prob_error": mult_prob_error,
        "max_abs_lp_err": max_abs,
        "mean_abs_lp_err": mean_abs,
        "n_valid_toks": n_valid,
        "n_exact_toks": n_exact,
        "frac_exact": n_exact / n_valid,
    }


def _victim_diags(train_data, prev_logprobs, repeated_batch):
    """Per-sequence mismatch localization (optional diagnostics)."""
    gen = train_data["generation_logprobs"][:, 1:]
    prev = prev_logprobs[:, 1:]
    m = (
        train_data["token_mask"][:, 1:] * train_data["sample_mask"].unsqueeze(-1)
    ).bool()
    diff = (gen != prev) & m
    bad_per_seq = diff.sum(dim=1)
    victims = (bad_per_seq > 0).nonzero(as_tuple=True)[0]
    trunc = repeated_batch.get("truncated", None)
    if trunc is not None and not torch.is_tensor(trunc):
        trunc = torch.tensor(trunc, dtype=torch.bool)
    n_trunc_total = int(trunc.sum().item()) if trunc is not None else -1
    vdiags = []
    for vi in victims.tolist():
        row = diff[vi].nonzero(as_tuple=True)[0]
        mrow = m[vi].nonzero(as_tuple=True)[0]
        gen_start = int(mrow.min().item())
        gen_end = int(mrow.max().item())
        vdiags.append(
            {
                "idx": vi,
                "n_bad": int(bad_per_seq[vi].item()),
                "n_valid": int(m[vi].sum().item()),
                "gen_span": [gen_start, gen_end],
                "first_bad_off": int(row.min().item()) - gen_start,
                "last_bad_off": int(row.max().item()) - gen_start,
                "input_len": int(train_data["input_lengths"][vi].item()),
                "truncated": bool(trunc[vi].item()) if trunc is not None else None,
            }
        )
    n_vt = sum(1 for v in vdiags if v["truncated"]) if trunc is not None else -1
    return vdiags, n_vt, n_trunc_total


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        raise SystemExit("--config is required (use the det bed yaml)")

    config = load_config(args.config)
    print(f"[HARNESS] Loaded configuration from: {args.config}", flush=True)
    if overrides:
        print(f"[HARNESS] Overrides: {overrides}", flush=True)
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("[HARNESS] Final config:")
    pprint.pprint(config)

    # logger remains dict-like (TypedDict); grpo/loss_fn are BaseModels.
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"[HARNESS] log dir: {config.logger['log_dir']}", flush=True)

    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, "generation config required"
    draft_cfg = config.policy.get("draft") or {}
    has_refit_draft_weights = bool(draft_cfg.get("enabled"))
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
    )

    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config.data, config.env
    )

    (
        policy,
        policy_generation,
        _nemo_gym,
        _cluster,
        dataloader,
        _val_dataloader,
        _loss_fn,
        _logger,
        checkpointer,
        _grpo_state,
        master_config,
        _teacher_worker_groups,
        _alias_to_group_alias,
    ) = setup(config, tokenizer, dataset, val_dataset, policy_factory=None)

    # GRPOConfig / ClippedPGLossConfig → attributes; PolicyConfig → subscript.
    kl_type = master_config.loss_fn.reference_policy_kl_type
    num_gen = master_config.grpo.num_generations_per_prompt
    max_rollout_turns = master_config.grpo.max_rollout_turns
    make_div_by = master_config.policy["make_sequence_length_divisible_by"]
    max_seq_len = master_config.policy["max_total_sequence_length"]
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]

    print(
        f"[HARNESS] kl_type={kl_type} num_gen={num_gen} "
        f"make_div_by={make_div_by} max_seq_len={max_seq_len}",
        flush=True,
    )

    need_refit = not (
        isinstance(policy_generation, MegatronGeneration) and colocated_inference
    )
    print(
        f"[HARNESS] colocated={colocated_inference} need_refit={need_refit}",
        flush=True,
    )
    refit_done = False
    results = []

    try:
        with checkpointer:
            n_done = 0
            for batch in dataloader:
                if n_done >= args.num_batches:
                    break
                t0 = time.time()
                repeated_batch = batch.repeat_interleave(num_gen)

                if need_refit and not refit_done:
                    refit_policy_generation(
                        policy, policy_generation, colocated_inference
                    )
                    refit_done = True
                else:
                    if colocated_inference:
                        policy.offload_after_refit()
                    policy_generation.prepare_for_generation()

                repeated_batch, _rollout_metrics = run_multi_turn_rollout(
                    policy_generation=policy_generation,
                    input_batch=repeated_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=max_seq_len,
                    max_rollout_turns=max_rollout_turns,
                    greedy=False,
                )
                policy_generation.finish_generation()
                t_gen = time.time() - t0

                add_grpo_token_loss_masks_and_generation_logprobs(
                    repeated_batch["message_log"]
                )
                flat_messages, input_lengths = batched_message_log_to_flat_message(
                    repeated_batch["message_log"],
                    pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    make_sequence_length_divisible_by=make_div_by,
                )
                train_data = BatchedDataDict(
                    {
                        "input_ids": flat_messages["token_ids"],
                        "input_lengths": input_lengths,
                        "generation_logprobs": flat_messages["generation_logprobs"],
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                )
                _preserve_router_replay_routed_experts(
                    train_data, flat_messages, master_config.policy
                )
                train_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
                train_data.to("cpu")

                policy.prepare_for_lp_inference()
                logprob_data = BatchedDataDict(
                    {
                        "input_ids": train_data["input_ids"],
                        "input_lengths": train_data["input_lengths"],
                        "token_mask": flat_messages["token_loss_mask"],
                        "sample_mask": repeated_batch["loss_multiplier"],
                    }
                )
                _preserve_router_replay_routed_experts(
                    logprob_data, flat_messages, master_config.policy
                )
                logprob_data.update(
                    flat_messages.get_multimodal_dict(as_tensors=False)
                )

                t1 = time.time()
                prev_logprobs = policy.get_logprobs(logprob_data)["logprobs"]
                t_score = time.time() - t1

                rec = _report(
                    f"batch{n_done}",
                    train_data["generation_logprobs"],
                    prev_logprobs,
                    train_data["token_mask"],
                    train_data["sample_mask"],
                    kl_type,
                )
                rec["t_gen_s"] = round(t_gen, 2)
                rec["t_score_s"] = round(t_score, 2)
                rec["batch_size"] = int(train_data["input_ids"].shape[0])
                rec["seq_len"] = int(train_data["input_ids"].shape[1])

                vdiags, n_vt, n_trunc_total = _victim_diags(
                    train_data, prev_logprobs, repeated_batch
                )
                rec["n_victims"] = len(vdiags)
                rec["n_victims_truncated"] = n_vt
                rec["n_truncated_total"] = n_trunc_total
                rec["victims"] = vdiags[:40]
                print(
                    f"[VICTIMS] {rec['tag']}: {len(vdiags)} victim seqs / "
                    f"{rec['batch_size']} (truncated victims: {n_vt}/{n_trunc_total})",
                    flush=True,
                )
                for v in vdiags[:20]:
                    glen = v["gen_span"][1] - v["gen_span"][0] + 1
                    print(
                        f"[VICTIMS]   idx={v['idx']:5d} bad={v['n_bad']:5d}/"
                        f"{v['n_valid']:5d} first_bad@{v['first_bad_off']}/{glen} "
                        f"last@{v['last_bad_off']} in_len={v['input_len']} "
                        f"trunc={v['truncated']}",
                        flush=True,
                    )
                results.append(rec)

                print(
                    f"\n[GENKL] {rec['tag']}: gen_kl_error={rec['gen_kl_error']:.3e}  "
                    f"max|Δlp|={rec.get('max_abs_lp_err', float('nan')):.3e}  "
                    f"exact={rec.get('n_exact_toks', 0)}/{rec.get('n_valid_toks', 0)} "
                    f"({rec.get('frac_exact', 0.0) * 100:.2f}%)  "
                    f"mult_prob_err={rec.get('mult_prob_error', float('nan')):.6f}  "
                    f"[gen {t_gen:.1f}s score {t_score:.1f}s]\n",
                    flush=True,
                )
                with open(args.jsonl, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                n_done += 1
    finally:
        try:
            if policy_generation is not None:
                policy_generation.finish_generation()
        except Exception as exc:  # noqa: BLE001 — best-effort shutdown
            print(f"[HARNESS] finish_generation during shutdown: {exc}", flush=True)

    if results:
        kls = [r["gen_kl_error"] for r in results]
        maxes = [r.get("max_abs_lp_err", 0.0) for r in results]
        print("\n" + "=" * 60, flush=True)
        print(f"[GENKL SUMMARY] batches={len(results)}", flush=True)
        print(
            f"[GENKL SUMMARY] gen_kl_error: min={min(kls):.3e} "
            f"max={max(kls):.3e} mean={sum(kls) / len(kls):.3e}",
            flush=True,
        )
        print(
            f"[GENKL SUMMARY] worst max|Δlp| over batches = {max(maxes):.3e}",
            flush=True,
        )
        all_exact = all(m == 0.0 for m in maxes)
        print(
            f"[GENKL SUMMARY] VERDICT: "
            f"{'BITWISE-EXACT ZERO' if all_exact else 'NON-ZERO RESIDUAL'}",
            flush=True,
        )
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
