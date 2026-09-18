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

from typing import Any
from unittest.mock import patch

import pytest

from nemo_rl.models.megatron.zero_train_gen_mismatch import (
    ZeroTrainGenValidation,
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
            MEGATRON_CORE_MIN_COMMIT_SHA,
            _validate_megatron_core_commit,
        )

        _validate_megatron_core_commit(MEGATRON_CORE_MIN_COMMIT_SHA, out)

    assert any("not at or after" in v for v in out.violations)


@pytest.mark.parametrize("is_minimum", [True, False])
def test_megatron_mainline_commit_accepted(is_minimum):
    from nemo_rl.models.megatron import zero_train_gen_mismatch as zero_kl

    minimum = zero_kl.MEGATRON_CORE_MIN_COMMIT_SHA
    head = minimum if is_minimum else "mainline-descendant"
    out = ZeroTrainGenValidation()
    with (
        patch.object(zero_kl, "_megatron_core_source_root", return_value="/tmp/mcore"),
        patch.object(zero_kl, "_installed_megatron_core_commit", return_value=head),
        patch.object(zero_kl, "_git_is_ancestor", return_value=True) as ancestry,
    ):
        zero_kl._validate_megatron_core_commit(minimum, out)

    assert out.violations == []
    if is_minimum:
        ancestry.assert_not_called()
    else:
        ancestry.assert_called_once_with(minimum, head, "/tmp/mcore")


@pytest.mark.parametrize("backend", ["torch", "vllm"])
def test_mxfp8_zero_mismatch_accepts_exact_parity_backends(backend):
    from nemo_rl.models.megatron.zero_train_gen_mismatch import (
        configure_zero_train_gen_mismatch,
    )

    config = _zero_kl_config(
        megatron_cfg={
            "fp8_cfg": {"enabled": True, "fp8_recipe": "mxfp8", "fp8_param": True},
            "model_overrides": {"moe_use_grouped_tensor": True},
        }
    )
    config["generation"]["mcore_generation_config"][
        "inference_grouped_gemm_backend"
    ] = backend

    configure_zero_train_gen_mismatch(config, apply_kernels=False)

    assert config["megatron_cfg"]["model_overrides"]["moe_use_grouped_tensor"] is True
    assert (
        config["generation"]["mcore_generation_config"]["logprobs_mode"]
        == "raw_logprobs"
    )
