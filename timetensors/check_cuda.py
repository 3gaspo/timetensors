"""Print CUDA diagnostics for Slurm jobs before launching experiments."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import torch


def _run_nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "nvidia-smi: not found on PATH"
    try:
        result = subprocess.run(
            [executable],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return f"nvidia-smi failed before execution: {exc}"
    output = "\n".join(part for part in [result.stdout, result.stderr] if part)
    return output.strip() or f"nvidia-smi exited with code {result.returncode}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--expected-torch", default=None)
    args = parser.parse_args(argv)

    print("===== CUDA preflight =====", flush=True)
    print(f"python={sys.version}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"torch.version.cuda={torch.version.cuda}", flush=True)
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ID",
    ):
        print(f"{name}={os.environ.get(name)}", flush=True)
    print(_run_nvidia_smi(), flush=True)
    version_ok = True
    if args.expected_torch is not None:
        version_ok = torch.__version__.startswith(args.expected_torch)
        print(f"expected_torch={args.expected_torch}", flush=True)
        print(f"torch_version_matches={version_ok}", flush=True)
        if not version_ok:
            print(
                "Torch version does not match the project pin; rebuild the environment.",
                flush=True,
            )
    try:
        available = torch.cuda.is_available()
        print(f"torch.cuda.is_available={available}", flush=True)
        print(f"torch.cuda.device_count={torch.cuda.device_count()}", flush=True)
        if available:
            current = torch.cuda.current_device()
            print(f"torch.cuda.current_device={current}", flush=True)
            print(f"torch.cuda.device_name={torch.cuda.get_device_name(current)}", flush=True)
    except Exception as exc:
        print(f"torch.cuda check failed: {exc}", flush=True)
        available = False
    print("===== End CUDA preflight =====", flush=True)
    if version_ok and (available or args.allow_cpu):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
