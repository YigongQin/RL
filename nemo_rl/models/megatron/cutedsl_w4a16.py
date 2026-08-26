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

"""Pack BF16 Megatron expert weights into CuteDSL W4A16 NVFP4 snapshots."""

from __future__ import annotations

from typing import Any


def maybe_pack_cutedsl_w4a16_weights(model: Any, megatron_cfg: dict[str, Any]) -> None:
    """Re-quantize expert masters after an optimizer step or refit.

    No-op unless ``megatron_cfg['moe_cutedsl_w4a16']`` is true. Fail loud if the
    Megatron helper is missing so a stale submodule cannot silently score in BF16.
    """
    if not megatron_cfg.get("moe_cutedsl_w4a16"):
        return
    try:
        from megatron.core.transformer.moe.cutedsl_w4a16 import pack_model_expert_weights
        from megatron.core.utils import unwrap_model
    except ImportError as exc:
        raise RuntimeError(
            "moe_cutedsl_w4a16=True but megatron.core.transformer.moe.cutedsl_w4a16 "
            "is not importable. Use the Megatron-LM revision that adds this module."
        ) from exc
    pack_model_expert_weights(unwrap_model(model))
