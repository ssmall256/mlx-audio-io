from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import traceback
from typing import Sequence

from ._native_loader import remediation_text


def _run_child_process() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "mlx_audio_io.doctor", "--child"],
        capture_output=True,
        text=True,
        check=False,
    )


def _child_main() -> int:
    from ._native_loader import load_native_module

    try:
        load_native_module()
    except Exception:
        traceback.print_exc()
        return 1

    print("Native import check: OK")
    return 0


def _print_child_output(proc: subprocess.CompletedProcess[str]) -> None:
    if proc.stdout.strip():
        print("Child stdout:")
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print("Child stderr:")
        print(proc.stderr.strip())


def _print_corruption_guidance() -> None:
    print("Likely cause: code-sign failure or binary corruption.")
    print("Reinstall the environment and do not copy `.venv` between machines.")
    print("Fix: rm -rf .venv && uv venv --python 3.11 && uv sync")


def _handle_child_result(proc: subprocess.CompletedProcess[str]) -> int:
    if proc.returncode == 0:
        print("Doctor check passed.")
        _print_child_output(proc)
        return 0

    if proc.returncode == 137 or proc.returncode < 0:
        if proc.returncode < 0:
            sig = signal.Signals(-proc.returncode).name
            print(f"Doctor check failed: child terminated by signal {sig} ({-proc.returncode}).")
        else:
            print("Doctor check failed: child exited with status 137.")

        _print_corruption_guidance()
        _print_child_output(proc)
        return 1

    print(f"Doctor check failed: child exited with status {proc.returncode}.")
    print(remediation_text())
    _print_child_output(proc)
    return 1


def _print_summary() -> None:
    from . import show_build_info

    info = show_build_info()
    print("mlx-audio-io doctor")
    for key in (
        "native_path",
        "build_os_name",
        "build_os_version",
        "build_deployment_target",
        "build_arch",
        "build_python_tag",
        "runtime_os_name",
        "runtime_os_version",
        "runtime_arch",
        "runtime_python_tag",
    ):
        print(f"{key}: {info.get(key)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mlx-audio-io native diagnostics")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        return _child_main()

    _print_summary()
    proc = _run_child_process()
    return _handle_child_result(proc)


if __name__ == "__main__":
    raise SystemExit(main())
