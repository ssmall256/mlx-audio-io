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
from importlib.util import find_spec
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
    # Imported lazily so the pure-Python hooks below stay importable without
    # the build toolchain present -- otherwise the requirement-pairing logic,
    # which is the part most worth testing, can only be tested where a build
    # could already run.
    from scikit_build_core import build as _backend

    _configure_cmake_executable()
    return getattr(_backend, name)(*args, **kwargs)


# MLX statically links nanobind and registers `mlx::core::array` in a shared
# NB_DOMAIN type registry. An extension built with a different nanobind compiles
# and links cleanly, then fails on the first call with "Unable to convert
# function return value to a Python type" -- an error that never says nanobind.
# Pairs come from MLX's own CMakeLists FetchContent GIT_TAG.
_NANOBIND_FOR_MLX = {
    "0.31": "2.12.0",
    "0.32": "2.15.0",
}

_BUILD_MLX_ENV = "MLX_AUDIO_IO_BUILD_MLX"


def _mlx_minor(version: str) -> str:
    parts = version.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else version


def _installed_version(distribution: str) -> str | None:
    try:
        from importlib import metadata

        return metadata.version(distribution)
    except Exception:
        return None


def paired_build_requirements() -> list[str]:
    """Converge the build environment on one MLX and the nanobind it needs.

    pip resolves `mlx` and `nanobind` from `[build-system] requires`
    independently, in an isolated environment that is also resolved
    independently of the environment the wheel will be installed into. Two
    things go wrong there, and both produce a binary that imports fine and
    fails later:

      * a nanobind MLX was never built with -> TypeError on the first call
      * an MLX minor the user does not have at runtime -> dlopen symbol error

    A version range cannot express "must equal whatever MLX used", but this
    hook can: pip installs whatever it returns into that same build
    environment, so an exact pin here is the one place a backend can fix the
    combination it was handed. `MLX_AUDIO_IO_BUILD_MLX` crosses the isolation
    boundary, which installed metadata cannot, so a caller whose runtime MLX is
    not the newest allowed can still build against it.
    """
    extra: list[str] = []
    requested = os.environ.get(_BUILD_MLX_ENV, "").strip()
    if requested:
        extra.append(f"mlx=={requested}")
    target = requested or _installed_version("mlx")
    if target:
        nanobind = _NANOBIND_FOR_MLX.get(_mlx_minor(target))
        if nanobind:
            extra.append(f"nanobind=={nanobind}")
    return extra


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    return val not in {"0", "false", "no", "off", ""}


def _mlx_library_search_paths() -> list[str]:
    paths: list[str] = []
    spec = find_spec("mlx")
    if spec is None:
        return paths
    if spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            lib_dir = Path(location) / "lib"
            if lib_dir.is_dir():
                paths.append(str(lib_dir))
    elif spec.origin:
        lib_dir = Path(spec.origin).resolve().parent / "lib"
        if lib_dir.is_dir():
            paths.append(str(lib_dir))
    return paths


def _repair_macos_wheel(wheel_path: Path, wheel_directory: Path) -> str:
    # delocate vendors external dylibs into the wheel and rewrites load paths.
    with tempfile.TemporaryDirectory(prefix="mlx_audio_io_delocate_") as tmpdir:
        cmd = [
            sys.executable,
            "-m",
            "delocate.cmd.delocate_wheel",
            "--wheel-dir",
            tmpdir,
            "--exclude",
            "libmlx.dylib",
            str(wheel_path),
        ]
        env = os.environ.copy()
        search_paths = _mlx_library_search_paths()
        if search_paths:
            fallback = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            merged = search_paths + ([fallback] if fallback else [])
            env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(merged)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
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
    return list(_call("get_requires_for_build_wheel", config_settings)) + paired_build_requirements()


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
    return list(_call("get_requires_for_build_editable", config_settings)) + paired_build_requirements()


def prepare_metadata_for_build_editable(
    metadata_directory: str,
    config_settings: dict[str, Any] | None = None,
) -> str:
    return _call("prepare_metadata_for_build_editable", metadata_directory, config_settings)
