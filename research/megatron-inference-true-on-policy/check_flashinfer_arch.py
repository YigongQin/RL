#!/usr/bin/env python3
"""Reproduce the RoPE "no kernel image" failure in seconds, on one GPU.

Jobs 423540 and 424154 both died in the same place: FlashInfer JIT-builds its
rope module, the build succeeds, and the first kernel launch comes back "no
kernel image is available for execution on the device". Each round trip to find
that out cost two nodes and eight minutes, and the failure needs neither -- one
GPU and the one op are enough.

What it prints, in the order that matters when the answer is wrong:

  * the device's real compute capability;
  * what FlashInfer's CompilationContext decided to target, which is the number
    that has to match and the one nothing in the traceback shows;
  * the -gencode flags that go to nvcc;
  * where the JIT cache lives, since a module built on a GB200 node and reused
    on a GB300 one fails identically to building for the wrong arch, and the
    cache path keys on FlashInfer's version but not on arch.

Then it runs apply_rope_with_cos_sin_cache, which is the exact call that failed.

    srun --overlap --pty --jobid=<job> --ntasks=1 --gres=gpu:1 \
        --container-image=<image> --container-mounts=/lustre:/lustre \
        <venv>/bin/python check_flashinfer_arch.py

Run it the way the workers run: same container, same venv, same node type. A
GB200 node will pass while a GB300 one fails, which is the whole point.

  --purge   delete the cache for this FlashInfer version first, so the build
            is real rather than a cache hit. Use it whenever the arch list
            changed, since a cached module predates the change.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purge", action="store_true", help="drop the JIT cache first")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1

    major, minor = torch.cuda.get_device_capability()
    print(f"-- device:        {torch.cuda.get_device_name()}")
    print(f"-- capability:    sm_{major}{minor}")
    for var in ("FLASHINFER_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST", "FLASHINFER_WORKSPACE_BASE"):
        print(f"-- {var:26} {os.environ.get(var, '<unset>')}")

    import flashinfer
    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit import env as jit_env

    print(f"-- flashinfer:    {flashinfer.__version__} ({os.path.dirname(flashinfer.__file__)})")

    # The targets FlashInfer will build for. Read from a fresh context, which is
    # what the JIT itself constructs per build, so this cannot drift from it.
    context = CompilationContext()
    targets = sorted(f"sm_{maj}{min_}" for maj, min_ in context.TARGET_CUDA_ARCHS)
    print(f"-- jit targets:   {', '.join(targets) or '<none>'}")
    print(f"-- gencode:       {' '.join(context.get_nvcc_flags_list()[:len(targets)])}")

    cache = jit_env.FLASHINFER_JIT_DIR if hasattr(jit_env, "FLASHINFER_JIT_DIR") else None
    print(f"-- jit cache:     {cache or jit_env.FLASHINFER_CACHE_DIR}")

    # The arch the device needs, spelled the way FlashInfer spells it. Blackwell
    # targets are arch-conditional: an sm_100a binary does not run on sm_103.
    needed = f"sm_{major}{minor}a" if major >= 10 else f"sm_{major}{minor}"
    if needed not in targets:
        print(f"\nMISMATCH: device needs {needed}, JIT targets {targets or 'nothing'}")
        print("Set FLASHINFER_CUDA_ARCH_LIST (unsuffixed, e.g. '10.0 10.3') and rerun --purge.\n")

    if args.purge:
        for path in {str(cache or ""), str(jit_env.FLASHINFER_CACHE_DIR)}:
            if path and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                print(f"-- purged:        {path}")

    print("\n-- building rope and launching it (first build takes minutes)\n", flush=True)

    # Shapes are arbitrary; only the kernel's existence for this arch is at issue.
    tokens, heads, head_dim = 4, 8, 128
    dtype = torch.bfloat16
    query = torch.randn(tokens, heads * head_dim, device="cuda", dtype=dtype)
    key = torch.randn(tokens, heads * head_dim, device="cuda", dtype=dtype)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    cos_sin_cache = torch.randn(tokens, head_dim, device="cuda", dtype=torch.float32)

    try:
        flashinfer.rope.apply_rope_with_cos_sin_cache(
            positions=positions,
            query=query,
            key=key,
            head_size=head_dim,
            cos_sin_cache=cos_sin_cache,
            is_neox=True,
        )
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}\n")
        if "no kernel image" in str(exc):
            print(
                "This is the job's failure, reproduced. The build targeted an arch\n"
                f"this device does not run. Device needs {needed}; JIT targeted "
                f"{targets or 'nothing'}.\n"
            )
        return 1

    print(f"\nPASS: rope ran on {needed}. This failure mode is cleared.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
