"""Project build backend wrapper.

This wraps scikit-build-core so local macOS builds avoid selecting pyenv shim
scripts as the CMake executable.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from scikit_build_core import build as _backend


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
    _configure_cmake_executable()
    return getattr(_backend, name)(*args, **kwargs)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _call("build_wheel", wheel_directory, config_settings, metadata_directory)


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
