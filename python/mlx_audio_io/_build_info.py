from __future__ import annotations

import importlib.metadata as importlib_metadata
import importlib.resources as importlib_resources
import importlib.util
import json
from pathlib import Path
from typing import Any


def _normalize_optional(value: Any) -> Any:
    if value in ("", "null", "None"):
        return None
    return value


def _native_package_dir() -> Path | None:
    """Directory holding the compiled extension.

    Editable installs resolve ``mlx_audio_io`` to the source tree, which only
    contains ``_build_info.json.in``. The generated JSON lives beside the
    compiled ``_core`` module in site-packages, so look there too -- otherwise
    build metadata reads as unknown and the MLX version gate silently no-ops in
    exactly the checkouts where it is most likely to matter.
    """
    try:
        spec = importlib.util.find_spec("mlx_audio_io._core")
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def _read_embedded_build_info() -> dict[str, Any]:
    text: str | None = None
    try:
        text = importlib_resources.files("mlx_audio_io").joinpath("_build_info.json").read_text(
            encoding="utf-8"
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        text = None

    if text is None:
        native_dir = _native_package_dir()
        if native_dir is not None:
            try:
                text = (native_dir / "_build_info.json").read_text(encoding="utf-8")
            except OSError:
                text = None

    if text is None:
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return {k: _normalize_optional(v) for k, v in data.items()}


def _extract_arch_from_platform_tag(platform_tag: str) -> str | None:
    lower = platform_tag.lower()
    known_arches = (
        "x86_64",
        "amd64",
        "aarch64",
        "arm64",
        "ppc64le",
        "s390x",
        "riscv64",
        "armv7l",
        "armv6l",
        "i686",
        "x86",
        "universal2",
    )

    for arch in known_arches:
        if lower == arch or lower.endswith(f"_{arch}") or lower.endswith(f"-{arch}"):
            return arch

    return None


def _parse_wheel_tag_fallback() -> dict[str, Any]:
    fallback: dict[str, Any] = {}

    try:
        dist = importlib_metadata.distribution("mlx-audio-io")
    except importlib_metadata.PackageNotFoundError:
        return fallback

    wheel_text = dist.read_text("WHEEL") or ""
    tags: list[str] = []
    for line in wheel_text.splitlines():
        if line.startswith("Tag:"):
            tags.append(line.split(":", 1)[1].strip())

    if not tags:
        return fallback

    primary = tags[0]
    parts = primary.split("-")
    if len(parts) < 3:
        return fallback

    interpreter, _abi, platform_tag = parts[0], parts[1], parts[2]
    fallback["python_tag"] = interpreter

    if platform_tag.startswith("macosx_"):
        segs = platform_tag.split("_")
        if len(segs) >= 4:
            fallback["build_os_name"] = "Darwin"
            fallback["build_os_version"] = f"{segs[1]}.{segs[2]}"
            fallback["deployment_target"] = f"{segs[1]}.{segs[2]}"
            fallback["arch"] = segs[3]
    else:
        lower = platform_tag.lower()
        if "linux" in lower:
            fallback["build_os_name"] = "Linux"
        if "win" in lower:
            fallback["build_os_name"] = "Windows"

        arch = _extract_arch_from_platform_tag(platform_tag)
        if arch:
            fallback["arch"] = arch

    return fallback


def _package_version() -> str | None:
    try:
        return importlib_metadata.version("mlx-audio-io")
    except importlib_metadata.PackageNotFoundError:
        return None


def load_build_info() -> dict[str, Any]:
    """Load build metadata embedded in the wheel, with safe fallbacks."""
    info: dict[str, Any] = {
        "build_os_name": None,
        "build_os_version": None,
        "deployment_target": None,
        "build_mlx_version": None,
        "compatible_mlx_versions": None,
        "arch": None,
        "python_tag": None,
        "wheel_version": None,
    }

    embedded = _read_embedded_build_info()
    for key in info:
        if key in embedded:
            info[key] = _normalize_optional(embedded[key])

    fallback = _parse_wheel_tag_fallback()
    for key, value in fallback.items():
        if info.get(key) is None:
            info[key] = _normalize_optional(value)

    if info["wheel_version"] is None:
        info["wheel_version"] = _package_version()

    return info
