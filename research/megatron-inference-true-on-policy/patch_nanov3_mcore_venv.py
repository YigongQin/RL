import os
import subprocess
import sys

from nemo_rl.utils.venvs import create_local_venv

_VENV = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
_pe = "uv run --extra mcore"
_force = os.environ.get("NRL_FORCE_REBUILD_VENVS", "false").lower() == "true"
_py = create_local_venv(_pe, _VENV, force_rebuild=_force)
subprocess.run(
    ["uv", "pip", "install", "--python", _py, "apache-tvm-ffi==0.1.11", "--force-reinstall"],
    check=False,
)
if subprocess.run([_py, "-c", "import tilelang"], capture_output=True).returncode != 0:
    subprocess.run(["uv", "pip", "uninstall", "--python", _py, "-y", "tilelang"], check=False)
    if subprocess.run([_py, "-c", "import fla"], capture_output=True).returncode != 0:
        print("[nanov3] fla import still failing after tilelang patch", file=sys.stderr)
        sys.exit(1)
print(f"[nanov3] mcore venv ready for FLA import: {_py}", flush=True)
