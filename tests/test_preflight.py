"""Unit tests for preflight checks before native import."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from pathlib import Path
import subprocess

import pytest

import mlx_audio_io._native_loader as native_loader


@dataclass
class _FakeHash:
    mode: str
    value: str


@dataclass
class _FakeFile:
    relpath: str
    real_path: Path
    hash: _FakeHash | None

    def __str__(self) -> str:
        return self.relpath


class _FakeDist:
    def __init__(self, files):
        self.files = files

    def locate_file(self, file_obj):
        return file_obj.real_path


def _sha256_b64_no_pad(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")


def test_record_hash_mismatch_raises_runtime_error(monkeypatch, tmp_path):
    native_path = tmp_path / "_core.cpython-312-darwin.so"
    native_path.write_bytes(b"native-bytes")

    bad_hash = _sha256_b64_no_pad(b"wrong-bytes")
    fake_dist = _FakeDist(
        [
            _FakeFile(
                "mlx_audio_io/_core.cpython-312-darwin.so",
                native_path,
                _FakeHash("sha256", bad_hash),
            )
        ]
    )

    monkeypatch.setattr(native_loader.importlib_metadata, "distribution", lambda _: fake_dist)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        native_loader.verify_record_hash(native_path)


def test_record_hash_missing_raises_runtime_error(monkeypatch, tmp_path):
    native_path = tmp_path / "_core.cpython-312-darwin.so"
    native_path.write_bytes(b"native-bytes")

    fake_dist = _FakeDist(
        [
            _FakeFile(
                "mlx_audio_io/_core.cpython-312-darwin.so",
                native_path,
                None,
            )
        ]
    )

    monkeypatch.setattr(native_loader.importlib_metadata, "distribution", lambda _: fake_dist)

    with pytest.raises(RuntimeError, match="RECORD hash entry missing"):
        native_loader.verify_record_hash(native_path)


def test_codesign_failure_raises_runtime_error(monkeypatch, tmp_path):
    native_path = tmp_path / "_core.cpython-312-darwin.so"
    native_path.write_bytes(b"native-bytes")

    monkeypatch.setattr(native_loader.platform, "system", lambda: "Darwin")

    def _fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["codesign"],
            returncode=1,
            stdout="invalid or modified",
            stderr="",
        )

    monkeypatch.setattr(native_loader.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="codesign verification failed"):
        native_loader.verify_codesign(native_path)


def test_arch_mismatch_raises_runtime_error(monkeypatch, tmp_path):
    native_path = tmp_path / "_core.cpython-312-darwin.so"
    native_path.write_bytes(b"native-bytes")

    monkeypatch.setattr(
        native_loader,
        "_runtime_context",
        lambda _p: {
            "os_name": "Darwin",
            "os_version": "14.2",
            "arch": "arm64",
            "python_tag": "cp312",
            "extension_python_tag": "cp312",
            "macos_version": "14.2",
        },
    )

    with pytest.raises(RuntimeError, match="Architecture mismatch"):
        native_loader.verify_compatibility(
            native_path,
            build_info={
                "build_os_name": "Darwin",
                "deployment_target": "13.0",
                "arch": "x86_64",
                "python_tag": "cp312",
            },
        )


def test_python_tag_mismatch_raises_runtime_error(monkeypatch, tmp_path):
    native_path = tmp_path / "_core.cpython-312-darwin.so"
    native_path.write_bytes(b"native-bytes")

    monkeypatch.setattr(
        native_loader,
        "_runtime_context",
        lambda _p: {
            "os_name": "Darwin",
            "os_version": "14.2",
            "arch": "arm64",
            "python_tag": "cp312",
            "extension_python_tag": "cp312",
            "macos_version": "14.2",
        },
    )

    with pytest.raises(RuntimeError, match="Python ABI/tag mismatch"):
        native_loader.verify_compatibility(
            native_path,
            build_info={
                "build_os_name": "Darwin",
                "deployment_target": "13.0",
                "arch": "arm64",
                "python_tag": "cp311",
            },
        )


def test_deployment_target_mismatch_raises_runtime_error(monkeypatch, tmp_path):
    native_path = tmp_path / "_core.cpython-312-darwin.so"
    native_path.write_bytes(b"native-bytes")

    monkeypatch.setattr(native_loader.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        native_loader,
        "_runtime_context",
        lambda _p: {
            "os_name": "Darwin",
            "os_version": "13.5",
            "arch": "arm64",
            "python_tag": "cp312",
            "extension_python_tag": "cp312",
            "macos_version": "13.5",
        },
    )

    with pytest.raises(RuntimeError, match="deployment target mismatch"):
        native_loader.verify_compatibility(
            native_path,
            build_info={
                "build_os_name": "Darwin",
                "deployment_target": "14.0",
                "arch": "arm64",
                "python_tag": "cp312",
            },
        )
