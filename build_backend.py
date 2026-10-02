"""Project build backend wrapper.

This wraps scikit-build-core so local macOS builds avoid selecting pyenv shim
scripts as the CMake executable.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _configure_cmake_executable() -> None:
    if platform.system() != "Darwin":
        return
    if os.environ.get("CMAKE_EXECUTABLE"):
        return

    for candidate in ("/opt/homebrew/bin/cmake", "/usr/local/bin/cmake"):
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            os.environ["CMAKE_EXECUTABLE"] = candidate
            return


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    # Imported lazily so this module stays importable without the build
    # toolchain present.
    from scikit_build_core import build as _backend

    _configure_cmake_executable()
    return getattr(_backend, name)(*args, **kwargs)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    return val not in {"0", "false", "no", "off", ""}


def _repair_macos_wheel(wheel_path: Path, wheel_directory: Path) -> str:
    # delocate vendors external dylibs into the wheel and rewrites load paths.
    with tempfile.TemporaryDirectory(prefix="mlx_audio_io_delocate_") as tmpdir:
        cmd = [
            sys.executable,
            "-m",
            "delocate.cmd.delocate_wheel",
            "--wheel-dir",
            tmpdir,
            str(wheel_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = "\n".join(
                part for part in (proc.stdout.strip(), proc.stderr.strip()) if part
            )
            raise RuntimeError(
                "Failed to repair macOS wheel with delocate.\n"
                "Set MLX_AUDIO_IO_REPAIR_WHEEL=0 to skip repair.\n"
                f"{detail}"
            )

        repaired = sorted(Path(tmpdir).glob("*.whl"))
        if not repaired:
            raise RuntimeError("delocate did not produce a repaired wheel artifact.")

        final_path = wheel_directory / repaired[0].name
        final_path.write_bytes(repaired[0].read_bytes())
        if repaired[0].name != wheel_path.name and wheel_path.exists():
            wheel_path.unlink()
        return repaired[0].name


def _repair_linux_wheel(wheel_path: Path, wheel_directory: Path) -> str:
    # Linux repair is optional because many local dev environments do not use
    # manylinux-compliant toolchains; CI release jobs can enable this explicitly.
    with tempfile.TemporaryDirectory(prefix="mlx_audio_io_auditwheel_") as tmpdir:
        cmd = [
            sys.executable,
            "-m",
            "auditwheel",
            "repair",
            "--wheel-dir",
            tmpdir,
            str(wheel_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = "\n".join(
                part for part in (proc.stdout.strip(), proc.stderr.strip()) if part
            )
            raise RuntimeError(
                "Failed to repair Linux wheel with auditwheel.\n"
                "Set MLX_AUDIO_IO_REPAIR_WHEEL=0 or MLX_AUDIO_IO_REPAIR_LINUX=0 to skip repair.\n"
                f"{detail}"
            )

        repaired = sorted(Path(tmpdir).glob("*.whl"))
        if not repaired:
            raise RuntimeError("auditwheel did not produce a repaired wheel artifact.")

        final_path = wheel_directory / repaired[0].name
        final_path.write_bytes(repaired[0].read_bytes())
        if repaired[0].name != wheel_path.name and wheel_path.exists():
            wheel_path.unlink()
        return repaired[0].name


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    wheel_name = _call("build_wheel", wheel_directory, config_settings, metadata_directory)
    if not _env_flag("MLX_AUDIO_IO_REPAIR_WHEEL", True):
        return wheel_name

    wheel_dir = Path(wheel_directory)
    wheel_path = wheel_dir / wheel_name
    if not wheel_path.exists():
        return wheel_name

    if platform.system() == "Darwin":
        return _repair_macos_wheel(wheel_path, wheel_dir)
    elif platform.system() == "Linux" and _env_flag("MLX_AUDIO_IO_REPAIR_LINUX", False):
        return _repair_linux_wheel(wheel_path, wheel_dir)
    return wheel_name


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _call("build_sdist", sdist_directory, config_settings)


def get_requires_for_build_wheel(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _call("get_requires_for_build_wheel", config_settings)


def get_requires_for_build_sdist(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _call("get_requires_for_build_sdist", config_settings)


def prepare_metadata_for_build_wheel(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _call("prepare_metadata_for_build_wheel", metadata_directory, config_settings)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _call("build_editable", wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_editable(
    config_settings: dict[str, Any] | None = None,
) -> list[str]:
    return _call("get_requires_for_build_editable", config_settings)


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _call("prepare_metadata_for_build_editable", metadata_directory, config_settings)
