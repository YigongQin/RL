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
# See the License for the License for the specific language governing permissions
# and limitations under the License.

"""Post-process the MegatronPolicyWorker venv for FLA/tilelang import safety.

Megatron-Bridge imports hybrid GatedDeltaNet → fla at worker import time for all
models (including Qwen). fla 0.5.1 probes tilelang on import; Py3.13 +
apache-tvm-ffi>=0.1.12 crashes in tvm_ffi registry (type __dict__ not writable).

Keep apache-tvm-ffi>=0.1.12 for FA4; uninstall tilelang after every uv sync.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from nemo_rl.utils.venvs import create_local_venv

_VENV = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
_PE = "uv run --extra mcore"
_FORCE = os.environ.get("NRL_FORCE_REBUILD_VENVS", "false").lower() == "true"
_MODEL = os.environ.get("ZERO_KL_MODEL_PREFIX", "")


def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _venv_python() -> str:
    venv_dir = os.path.normpath(os.environ.get("NEMO_RL_VENV_DIR", "venvs"))
    return os.path.join(venv_dir, _VENV, "bin", "python")


def _strip_tilelang(py: str) -> None:
    _run(["uv", "pip", "uninstall", "--python", py, "-y", "tilelang"], check=False)

    fla_env = {
        **os.environ,
        "FLA_TILELANG": "0",
        "FLA_DISABLE_BACKEND_DISPATCH": "1",
    }
    probe = subprocess.run(
        [py, "-c", "import fla"],
        env=fla_env,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        print(
            f"[mcore-fla] fla import failed after tilelang removal (model={_MODEL or 'unknown'}):\n"
            f"{probe.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-sync",
        action="store_true",
        help="Only strip tilelang from an existing mcore worker venv (no uv sync).",
    )
    args = parser.parse_args()

    py = _venv_python()
    if args.post_sync:
        if not os.path.isfile(py):
            print(
                f"[mcore-fla] worker venv not present yet ({py}); "
                "Ray workers will create it — skipping pre-flight tilelang strip.",
                flush=True,
            )
            return
    elif _FORCE or not os.path.isfile(py):
        # create_local_venv always runs `uv sync` (even without NRL_FORCE_REBUILD_VENVS).
        # Skip that when the Lustre worker interpreter already exists.
        py = create_local_venv(_PE, _VENV, force_rebuild=_FORCE)
    else:
        print(f"[mcore-fla] reusing existing worker venv (no uv sync): {py}", flush=True)

    _strip_tilelang(py)
    print(f"[mcore-fla] mcore venv ready (model={_MODEL or 'unknown'}): {py}", flush=True)


if __name__ == "__main__":
    main()
