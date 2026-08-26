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

from nemo_rl.models.megatron.cutedsl_w4a16 import maybe_pack_cutedsl_w4a16_weights


def test_maybe_pack_cutedsl_w4a16_weights_noop_when_disabled():
    maybe_pack_cutedsl_w4a16_weights(object(), {})
    maybe_pack_cutedsl_w4a16_weights(object(), {"moe_cutedsl_w4a16": False})
