# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed under an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models.megatron.zero_train_gen_mismatch import (
    ZeroTrainGenValidation,
    mega_backend_selected,
    resolve_zero_train_gen_mismatch,
    validate_zero_train_gen_mismatch,
)


def _zero_kl_config(**kwargs: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "precision": "bfloat16",
        "megatron_cfg": {
            "zero_train_gen_mismatch": True,
            "tensor_model_parallel_size": 1,
            "context_parallel_size": 1,
            "attention_backend": "flash",
            "flash_attention_version": 4,
            "batch_invariant_backend": "te_native",
            "batch_invariant_collective": "ordered",
        },
        "generation": {
            "backend": "megatron",
            "mcore_generation_config": {
                "transformer_impl": "inference_optimized",
            },
        },
    }
    if "megatron_cfg" in kwargs:
        base["megatron_cfg"] = {**base["megatron_cfg"], **kwargs.pop("megatron_cfg")}
    base.update(kwargs)
    return base


def test_collects_multiple_violations():
    config = _zero_kl_config(
        megatron_cfg={"multi_latent_attention": True, "use_mla": True},
    )
    config["generation"]["backend"] = "vllm"

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert len(result.violations) >= 2
    with pytest.raises(ValueError, match="failed validation"):
        result.raise_if_invalid("zero_train_gen_mismatch failed validation:")


def test_transformer_engine_gen_warns_not_fails():
    config = _zero_kl_config()
    config["generation"]["mcore_generation_config"]["transformer_impl"] = (
        "transformer_engine"
    )

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert result.violations == []
    assert any("inference_optimized" in w for w in result.warnings)


def test_flashinfer_mxfp8_rejected_for_zero_train_gen_mismatch():
    config = _zero_kl_config(
        megatron_cfg={
            "fp8_cfg": {
                "enabled": True,
                "fp8_recipe": "mxfp8",
            }
        }
    )
    config["generation"]["mcore_generation_config"][
        "inference_grouped_gemm_backend"
    ] = "flashinfer"

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )

    assert any("not bitwise identical" in violation for violation in result.violations)


def test_resolve_applies_batch_invariant_defaults():
    config = _zero_kl_config(
        megatron_cfg={"batch_invariant_mode": False, "moe_permute_fusion": True}
    )
    with pytest.warns(UserWarning, match="zero_train_gen_mismatch"):
        resolve_zero_train_gen_mismatch(config)

    assert config["megatron_cfg"]["batch_invariant_mode"] is True
    assert config["megatron_cfg"]["moe_permute_fusion"] is False


def _mega_config(**kwargs: Any) -> dict[str, Any]:
    """A zero-KL config whose generation selects the FlashInfer megakernel."""
    config = _zero_kl_config(**kwargs)
    config["generation"]["mcore_generation_config"].update(
        {
            "inference_grouped_gemm_backend": "flashinfer_mega",
            "inference_mega_max_tokens_per_rank": 10240,
        }
    )
    return config


def test_mega_backend_is_detected_from_generation_config():
    assert mega_backend_selected(_mega_config()) is True
    assert mega_backend_selected(_zero_kl_config()) is False


def test_resolve_forces_the_mega_training_forward():
    """The megakernel on the generation side implies it on the training side.

    Without this the training forward runs TransformerEngine against a
    megakernel generation, which is the ~6e-3 mismatch zero-KL exists to remove.
    """
    config = _mega_config()
    resolve_zero_train_gen_mismatch(config)

    mc = config["megatron_cfg"]
    assert mc["moe_mega_training_forward"] is True
    assert mc["activation_checkpointing"] is True
    assert mc["recompute_granularity"] == "selective"
    assert "moe" in mc["recompute_modules"]
    inference = config["generation"]["mcore_generation_config"]
    assert inference["inference_mega_precision"] == "bf16"


def test_resolve_appends_moe_to_existing_recompute_modules():
    """recompute_modules is additive: the recipe may already use it for memory."""
    config = _mega_config(megatron_cfg={"recompute_modules": ["core_attn"]})
    resolve_zero_train_gen_mismatch(config)

    assert config["megatron_cfg"]["recompute_modules"] == ["core_attn", "moe"]


def test_resolve_is_idempotent_on_recompute_modules():
    config = _mega_config(megatron_cfg={"recompute_modules": ["moe"]})
    resolve_zero_train_gen_mismatch(config)
    resolve_zero_train_gen_mismatch(config)

    assert config["megatron_cfg"]["recompute_modules"] == ["moe"]


def test_a_precision_without_a_weight_packer_is_rejected_not_rewritten():
    """nvfp4 is refused rather than quietly turned into bf16.

    It used to be forced, from when bf16 was the only precision that worked.
    Now that mxfp8 is a real choice, silently substituting would mean a recipe
    asking for one precision could run another and report success -- so the
    resolver propagates what was asked for and the validator refuses it.

    The refusal is about refit, not accuracy: FlashInfer preprocesses and
    snapshots nvfp4 weights, so a refit never reaches the kernel and generation
    keeps sampling from the previous step's policy.
    """
    config = _mega_config()
    config["generation"]["mcore_generation_config"]["inference_mega_precision"] = (
        "nvfp4"
    )
    resolve_zero_train_gen_mismatch(config)

    assert (
        config["generation"]["mcore_generation_config"]["inference_mega_precision"]
        == "nvfp4"
    )
    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert any("inference_mega_precision" in v for v in result.violations)


def test_resolve_propagates_mxfp8_to_the_training_side():
    """The precision has to match on both sides or the forwards are not bitwise.

    Only the generation value is written in a recipe; MCore validates the
    training forward against the training config's own precision, so a training
    side left at the bf16 default would run a different kernel from generation.
    """
    config = _mega_config()
    config["generation"]["mcore_generation_config"]["inference_mega_precision"] = (
        "mxfp8"
    )
    resolve_zero_train_gen_mismatch(config)

    mc = config["megatron_cfg"]
    assert mc["inference_mega_precision"] == "mxfp8"
    # The gradient still comes from the bf16 recompute, which MCore refuses to
    # pair with a quantized forward unless the recipe has accepted it.
    assert mc["moe_mega_training_straight_through"] is True


def test_bf16_does_not_get_the_straight_through_opt_in():
    """MCore rejects the flag at bf16, so it must not be set unconditionally."""
    config = _mega_config()
    resolve_zero_train_gen_mismatch(config)

    assert not config["megatron_cfg"].get("moe_mega_training_straight_through")


def test_mxfp8_without_the_straight_through_opt_in_is_rejected():
    """The opt-in is the recipe accepting a straight-through gradient."""
    config = _mega_config()
    config["generation"]["mcore_generation_config"]["inference_mega_precision"] = (
        "mxfp8"
    )
    resolve_zero_train_gen_mismatch(config)
    # Undo what the resolver added, standing in for a caller that validates a
    # hand-built config rather than a resolved one.
    config["megatron_cfg"]["moe_mega_training_straight_through"] = False

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert any(
        "moe_mega_training_straight_through" in v for v in result.violations
    ), result.violations


def test_a_precision_mismatch_between_the_two_sides_is_rejected():
    """The failure this guards is silent: parity breaks with no error."""
    config = _mega_config()
    config["generation"]["mcore_generation_config"]["inference_mega_precision"] = (
        "mxfp8"
    )
    resolve_zero_train_gen_mismatch(config)
    config["megatron_cfg"]["inference_mega_precision"] = "bf16"

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert any("does not match generation" in v for v in result.violations)


def test_resolve_leaves_non_mega_recipes_untouched():
    """The mega defaults must not leak into the backends that were already working."""
    config = _zero_kl_config()
    resolve_zero_train_gen_mismatch(config)

    mc = config["megatron_cfg"]
    assert "moe_mega_training_forward" not in mc
    assert "recompute_granularity" not in mc
    assert "recompute_modules" not in mc


def test_resolved_mega_config_passes_validation():
    """The resolver and the validator have to agree on what a mega recipe is.

    Asserted as a pair because they encode the same rules twice: the resolver
    sets the fields and the validator refuses configs without them, so a drift
    between the two would make every mega run fail on a config nothing can
    produce.
    """
    config = _mega_config()
    resolve_zero_train_gen_mismatch(config)

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert result.violations == []


def test_mega_validation_requires_the_training_forward():
    """An unresolved config, i.e. a caller that validated without resolving."""
    config = _mega_config()

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert any("moe_mega_training_forward" in v for v in result.violations)


def test_mega_validation_requires_moe_recompute():
    """The mega forward saves no intermediates, so the backward needs the recompute."""
    config = _mega_config(
        megatron_cfg={
            "moe_mega_training_forward": True,
            "recompute_granularity": "full",
        }
    )

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert any("recompute_granularity" in v for v in result.violations)


def test_mega_validation_rejects_local_cuda_graphs():
    """Local graphs capture the MoE layer whole, which disables that recompute."""
    config = _mega_config(
        megatron_cfg={
            "moe_mega_training_forward": True,
            "recompute_granularity": "selective",
            "recompute_modules": ["moe"],
            "cuda_graph_impl": "local",
        }
    )

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert any("cuda_graph_impl" in v for v in result.violations)


def _generation_side_config(**kwargs: Any) -> dict[str, Any]:
    """The policy config a dedicated (non-colocated) generation model gets.

    MegatronGeneration stands up its own Policy whose megatron_cfg is the
    training one with mcore_generation_config layered on top, and those workers
    then run the same resolve and validation as the training workers.
    """
    from nemo_rl.models.generation.megatron.config import (
        merged_inference_megatron_cfg,
    )

    config = _mega_config(**kwargs)
    config["generation"]["colocated"] = {"enabled": False}
    resolve_zero_train_gen_mismatch(config)
    return {**config, "megatron_cfg": merged_inference_megatron_cfg(config)}


def test_generation_merge_drops_the_training_forward():
    """It is a training knob, and generation reaches the kernel without it.

    Left inherited it also puts the generation model in a state MCore rejects,
    because it requires an MoE recompute that local CUDA graphs remove.
    """
    gen_config = _generation_side_config()

    assert gen_config["megatron_cfg"]["moe_mega_training_forward"] is False
    assert gen_config["megatron_cfg"]["is_inference_model"] is True


def test_generation_config_may_graph_while_training_does_not():
    """Job 430144: a correct setup rejected because the two sides were conflated.

    Training must stay eager for the MoE recompute the mega backward comes
    from, while generation has no backward and graphs for throughput. Both
    values live under the key `cuda_graph_impl`, and the generation workers
    validate the merge, so the train-side gate saw 'local' and refused.
    """
    gen_config = _generation_side_config()
    gen_config["generation"]["mcore_generation_config"]["cuda_graph_impl"] = "local"
    gen_config["megatron_cfg"]["cuda_graph_impl"] = "local"

    result = validate_zero_train_gen_mismatch(
        gen_config, check_packages=False, check_platform=False
    )
    assert result.violations == []


def test_resolve_leaves_the_training_knobs_off_the_generation_config():
    """Resolve runs again inside the generation worker, so it must agree too.

    Otherwise it puts back what the merge deliberately dropped and the
    validation that follows it in the same call fails on its own output.
    """
    gen_config = _generation_side_config()
    resolve_zero_train_gen_mismatch(gen_config)

    mc = gen_config["megatron_cfg"]
    assert mc["moe_mega_training_forward"] is False
    assert mc["activation_checkpointing"] is False
    # Still selects the megakernel: generation drives it through the backend,
    # which is why dropping the training forward costs it nothing.
    assert mc["inference_grouped_gemm_backend"] == "flashinfer_mega"
    assert mc["transformer_impl"] == "inference_optimized"


def test_mega_validation_requires_the_token_cap():
    """The cap is a hard workspace bound; the kernel rejects a wider forward."""
    config = _mega_config()
    resolve_zero_train_gen_mismatch(config)
    del config["generation"]["mcore_generation_config"][
        "inference_mega_max_tokens_per_rank"
    ]

    result = validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False
    )
    assert any(
        "inference_mega_max_tokens_per_rank" in v for v in result.violations
    )


@patch("nemo_rl.models.megatron.zero_train_gen_mismatch._validate_megatron_core_commit")
@patch("nemo_rl.models.megatron.zero_train_gen_mismatch._package_version")
def test_mega_is_held_to_the_same_commit_gate(mock_pkg_version, mock_mcore_commit):
    """Mega adds no commit gate of its own; moe_ep importability is its guard.

    A mega-specific SHA could only name a commit that is not upstream yet, so
    it would pass on the tree it was read from and fail on every other, which
    is worse than not checking.
    """
    from nemo_rl.models.megatron.zero_train_gen_mismatch import (
        MEGATRON_CORE_MIN_COMMIT_SHA,
    )

    mock_pkg_version.side_effect = lambda name: None
    config = _mega_config()
    resolve_zero_train_gen_mismatch(config)

    # Stubbed rather than imported for real: the package check imports
    # flashinfer.moe_ep, which is heavy and, against a mismatched cubin wheel,
    # raises something other than ImportError.
    stubs = {"flashinfer": MagicMock(), "flashinfer.moe_ep": MagicMock()}
    with patch.dict(sys.modules, stubs):
        validate_zero_train_gen_mismatch(
            config, check_packages=True, check_platform=False
        )
        mega_gates = [call.args[0] for call in mock_mcore_commit.call_args_list]

        mock_mcore_commit.reset_mock()
        validate_zero_train_gen_mismatch(
            _zero_kl_config(), check_packages=True, check_platform=False
        )
        plain_gates = [call.args[0] for call in mock_mcore_commit.call_args_list]

    assert mega_gates == [MEGATRON_CORE_MIN_COMMIT_SHA]
    assert plain_gates == [MEGATRON_CORE_MIN_COMMIT_SHA]


@patch("nemo_rl.models.megatron.zero_train_gen_mismatch._validate_megatron_core_commit")
@patch("nemo_rl.models.megatron.zero_train_gen_mismatch._package_version")
def test_package_check_requires_te(mock_pkg_version, mock_mcore_commit):
    mock_mcore_commit.side_effect = lambda _min, out: None
    mock_pkg_version.side_effect = lambda name: None

    config = _zero_kl_config()
    result = validate_zero_train_gen_mismatch(
        config, check_packages=True, check_platform=False
    )
    msgs = " ".join(result.violations)
    assert "transformer_engine" in msgs
    assert "flash_attn" in msgs


def test_megatron_commit_too_old():
    out = ZeroTrainGenValidation()

    with (
        patch(
            "nemo_rl.models.megatron.zero_train_gen_mismatch._installed_megatron_core_commit",
            return_value="abc123",
        ),
        patch(
            "nemo_rl.models.megatron.zero_train_gen_mismatch._megatron_core_source_root",
            return_value="/tmp/mcore",
        ),
        patch(
            "nemo_rl.models.megatron.zero_train_gen_mismatch._git_is_ancestor",
            return_value=False,
        ),
    ):
        from nemo_rl.models.megatron.zero_train_gen_mismatch import (
            _validate_megatron_core_commit,
            MEGATRON_CORE_MIN_COMMIT_SHA,
        )

        _validate_megatron_core_commit(MEGATRON_CORE_MIN_COMMIT_SHA, out)

    assert any("not at or after" in v for v in out.violations)
