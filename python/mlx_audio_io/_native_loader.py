from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import platform
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ._build_info import load_build_info


_REMEDIATION = (
    "Reinstall the environment and do not copy `.venv` across machines.\n"
    "Fix: rm -rf .venv && uv venv --python 3.11 && uv sync"
)

_LOCK = threading.Lock()
_CORE_MODULE: Any = None
_CORE_LOAD_ERROR: RuntimeError | None = None


def remediation_text() -> str:
    return _REMEDIATION


def _runtime_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _canonical_arch(arch: str | None) -> str | None:
    if not arch:
        return None

    a = arch.lower()
    alias = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86-64": "x86_64",
        "aarch64": "arm64",
        "arm64e": "arm64",
    }
    return alias.get(a, a)


def _extract_extension_python_tag(native_path: Path) -> str | None:
    name = native_path.name

    m = re.search(r"cpython-(\d{2,3})", name)
    if m:
        return f"cp{m.group(1)}"

    m = re.search(r"\.cp(\d{2,3})[.-]", name)
    if m:
        return f"cp{m.group(1)}"

    return None


def _parse_version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None

    parts: list[int] = []
    for piece in value.split("."):
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts) if parts else None


def _runtime_context(native_path: Path) -> dict[str, Any]:
    os_name = platform.system()
    mac_version = platform.mac_ver()[0] if os_name == "Darwin" else None
    os_version = mac_version or platform.release()

    return {
        "os_name": os_name,
        "os_version": os_version,
        "arch": platform.machine(),
        "python_tag": _runtime_python_tag(),
        "extension_python_tag": _extract_extension_python_tag(native_path),
        "macos_version": mac_version,
    }


def resolve_native_path() -> Path:
    spec = importlib.util.find_spec("mlx_audio_io._core")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Unable to locate native extension `mlx_audio_io._core`.\n"
            f"{_REMEDIATION}"
        )

    native_path = Path(spec.origin)
    if not native_path.exists():
        raise RuntimeError(
            f"Native extension path does not exist: {native_path}\n"
            f"{_REMEDIATION}"
        )

    return native_path


def verify_record_hash(native_path: Path) -> None:
    try:
        dist = importlib_metadata.distribution("mlx-audio-io")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Could not locate installed distribution metadata for `mlx-audio-io`; "
            "cannot verify binary hash against RECORD.\n"
            f"{_REMEDIATION}"
        ) from exc

    files = list(dist.files or [])
    if not files:
        raise RuntimeError(
            "Installed distribution has no file metadata; cannot verify binary hash.\n"
            f"{_REMEDIATION}"
        )

    native_real = native_path.resolve()
    entry = None
    for f in files:
        try:
            if Path(dist.locate_file(f)).resolve() == native_real:
                entry = f
                break
        except OSError:
            continue

    if entry is None:
        raise RuntimeError(
            "Native extension is missing from dist-info RECORD; refusing to load.\n"
            f"{_REMEDIATION}"
        )

    file_hash = getattr(entry, "hash", None)
    if file_hash is None or not getattr(file_hash, "value", None):
        raise RuntimeError(
            "RECORD hash entry missing for native extension; refusing to load.\n"
            f"{_REMEDIATION}"
        )

    mode = getattr(file_hash, "mode", "")
    if mode.lower() != "sha256":
        raise RuntimeError(
            f"Unsupported RECORD hash mode `{mode}` for native extension.\n"
            f"{_REMEDIATION}"
        )

    try:
        digest = hashlib.sha256(native_path.read_bytes()).digest()
    except OSError as exc:
        raise RuntimeError(
            f"Unable to read native extension for hash verification: {native_path}\n"
            f"{_REMEDIATION}"
        ) from exc

    actual = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    expected = file_hash.value
    if actual != expected:
        raise RuntimeError(
            "Native extension hash mismatch vs dist-info RECORD.\n"
            f"Expected sha256={expected}, got sha256={actual}.\n"
            f"{_REMEDIATION}"
        )


def verify_codesign(native_path: Path) -> None:
    if platform.system() != "Darwin":
        return

    try:
        result = subprocess.run(
            ["codesign", "--verify", "--verbose=2", str(native_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "`codesign` is unavailable on this macOS system; cannot verify native binary.\n"
            f"{_REMEDIATION}"
        ) from exc

    if result.returncode != 0:
        detail = "\n".join(x for x in (result.stdout.strip(), result.stderr.strip()) if x)
        if not detail:
            detail = "codesign returned non-zero status"
        raise RuntimeError(
            f"codesign verification failed for {native_path}.\n"
            f"{detail}\n"
            f"{_REMEDIATION}"
        )


def verify_compatibility(native_path: Path, build_info: dict[str, Any] | None = None) -> None:
    build = dict(load_build_info() if build_info is None else build_info)
    runtime = _runtime_context(native_path)

    expected_os = (build.get("build_os_name") or "").lower()
    runtime_os = (runtime.get("os_name") or "").lower()

    if expected_os in {"darwin", "macos", "macosx"} and runtime_os != "darwin":
        raise RuntimeError(
            f"Build OS mismatch: binary built for {build.get('build_os_name')} but running on {runtime.get('os_name')}.\n"
            f"{_REMEDIATION}"
        )
    if expected_os == "linux" and runtime_os != "linux":
        raise RuntimeError(
            f"Build OS mismatch: binary built for Linux but running on {runtime.get('os_name')}.\n"
            f"{_REMEDIATION}"
        )

    build_arch = _canonical_arch(build.get("arch"))
    runtime_arch = _canonical_arch(runtime.get("arch"))
    if build_arch and runtime_arch and build_arch != runtime_arch:
        raise RuntimeError(
            f"Architecture mismatch: built for {build.get('arch')}, runtime is {runtime.get('arch')}.\n"
            f"{_REMEDIATION}"
        )

    runtime_tag = runtime.get("python_tag")
    build_tag = build.get("python_tag")
    if build_tag and runtime_tag and build_tag != runtime_tag:
        raise RuntimeError(
            f"Python ABI/tag mismatch: build tag {build_tag}, runtime tag {runtime_tag}.\n"
            f"{_REMEDIATION}"
        )

    ext_tag = runtime.get("extension_python_tag")
    if ext_tag and runtime_tag and ext_tag != runtime_tag:
        raise RuntimeError(
            f"Native extension tag mismatch: extension tag {ext_tag}, runtime tag {runtime_tag}.\n"
            f"{_REMEDIATION}"
        )

    deployment_target = build.get("deployment_target")
    runtime_macos = runtime.get("macos_version")
    if platform.system() == "Darwin" and deployment_target:
        runtime_tuple = _parse_version_tuple(runtime_macos)
        target_tuple = _parse_version_tuple(str(deployment_target))
        if runtime_tuple and target_tuple and runtime_tuple < target_tuple:
            raise RuntimeError(
                "macOS deployment target mismatch: "
                f"binary requires macOS {deployment_target}+, runtime is {runtime_macos}.\n"
                f"{_REMEDIATION}"
            )


def run_preflight_checks(native_path: Path | None = None) -> Path:
    path = resolve_native_path() if native_path is None else native_path
    verify_record_hash(path)
    verify_codesign(path)
    verify_compatibility(path)
    return path


def load_native_module() -> Any:
    global _CORE_MODULE, _CORE_LOAD_ERROR

    if _CORE_MODULE is not None:
        return _CORE_MODULE
    if _CORE_LOAD_ERROR is not None:
        raise _CORE_LOAD_ERROR

    with _LOCK:
        if _CORE_MODULE is not None:
            return _CORE_MODULE
        if _CORE_LOAD_ERROR is not None:
            raise _CORE_LOAD_ERROR

        try:
            run_preflight_checks()
            # _core depends on MLX runtime libs; import mlx.core first so the
            # dependent symbols are available at dlopen time.
            import mlx.core  # noqa: F401

            _CORE_MODULE = importlib.import_module("mlx_audio_io._core")
            return _CORE_MODULE
        except RuntimeError as exc:
            _CORE_LOAD_ERROR = exc
            raise
        except Exception as exc:  # pragma: no cover - defensive wrapping
            _CORE_LOAD_ERROR = RuntimeError(
                "Failed to import native extension after preflight checks.\n"
                f"Original error: {exc}\n"
                f"{_REMEDIATION}"
            )
            raise _CORE_LOAD_ERROR from exc


def get_diagnostic_info() -> dict[str, Any]:
    info = load_build_info()

    runtime_tag = _runtime_python_tag()
    runtime_arch = platform.machine()
    runtime_os = platform.system()
    runtime_os_version = platform.mac_ver()[0] if runtime_os == "Darwin" else platform.release()

    native_path: str | None = None
    native_path_error: str | None = None
    ext_tag: str | None = None

    try:
        resolved = resolve_native_path()
        native_path = str(resolved)
        ext_tag = _extract_extension_python_tag(resolved)
    except RuntimeError as exc:
        native_path_error = str(exc)

    build_arch = info.get("arch")
    build_python_tag = info.get("python_tag")

    return {
        "package_version": info.get("wheel_version"),
        "native_path": native_path,
        "native_path_error": native_path_error,
        "build_os_name": info.get("build_os_name"),
        "build_os_version": info.get("build_os_version"),
        "build_deployment_target": info.get("deployment_target"),
        "build_arch": build_arch,
        "build_python_tag": build_python_tag,
        "build_wheel_version": info.get("wheel_version"),
        "runtime_os_name": runtime_os,
        "runtime_os_version": runtime_os_version,
        "runtime_arch": runtime_arch,
        "runtime_python_tag": runtime_tag,
        "native_extension_python_tag": ext_tag,
        "arch_matches": (
            _canonical_arch(build_arch) == _canonical_arch(runtime_arch)
            if build_arch and runtime_arch
            else None
        ),
        "python_tag_matches": (
            build_python_tag == runtime_tag if build_python_tag and runtime_tag else None
        ),
    }
