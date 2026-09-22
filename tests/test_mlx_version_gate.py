"""Tests for the MLX runtime-version gate.

mlx-audio-io links libmlx directly and shares nanobind's type registry, so a
binary is only loadable against an ABI-compatible MLX. The gate used to demand
exact string equality with the build-time version, which meant downstream
projects could not test any other MLX at all (ssmall256/mlx-audio-separator#4).
It now accepts any version the build was verified against, with an opt-in
override for experimentation.
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

import mlx_audio_io._native_loader as native_loader


@pytest.fixture
def stub_runtime(monkeypatch):
    """Neutralize the OS/arch/tag checks so only the MLX gate is exercised."""
    monkeypatch.setattr(
        native_loader,
        "_runtime_context",
        lambda native_path: {
            "os_name": platform.system(),
            "arch": platform.machine(),
            "python_tag": None,
            "extension_python_tag": None,
            "macos_version": None,
        },
    )

    def _verify(build, runtime_mlx):
        monkeypatch.setattr(native_loader, "_runtime_mlx_version", lambda: runtime_mlx)
        payload = {
            "build_os_name": platform.system(),
            "arch": platform.machine(),
            **build,
        }
        native_loader.verify_compatibility(Path("fake.so"), build_info=payload)

    return _verify


def test_exact_build_version_is_accepted(stub_runtime):
    stub_runtime({"build_mlx_version": "0.31.2"}, "0.31.2")


def test_any_verified_version_is_accepted(stub_runtime):
    build = {
        "build_mlx_version": "0.31.2",
        "compatible_mlx_versions": "0.31.2,0.31.3",
    }
    stub_runtime(build, "0.31.3")


def test_unverified_version_is_rejected(stub_runtime):
    with pytest.raises(RuntimeError) as excinfo:
        stub_runtime({"build_mlx_version": "0.31.2"}, "0.32.2")

    message = str(excinfo.value)
    assert "0.32.2" in message
    assert "0.31.2" in message
    # The message must tell the user how to proceed deliberately.
    assert native_loader._ALLOW_MISMATCH_ENV in message


def test_override_downgrades_rejection_to_warning(stub_runtime, monkeypatch):
    monkeypatch.setenv(native_loader._ALLOW_MISMATCH_ENV, "1")

    with pytest.warns(RuntimeWarning, match="0.32.2"):
        stub_runtime({"build_mlx_version": "0.31.2"}, "0.32.2")


def test_missing_mlx_is_rejected(stub_runtime):
    with pytest.raises(RuntimeError, match="MLX runtime not found"):
        stub_runtime({"build_mlx_version": "0.31.2"}, None)


def test_unknown_build_version_skips_the_gate(stub_runtime):
    """Editable checkouts have no generated build info; they must not hard-fail."""
    stub_runtime({"build_mlx_version": "unknown"}, "0.99.0")
    stub_runtime({}, "0.99.0")


def test_override_does_not_apply_to_a_missing_runtime(stub_runtime, monkeypatch):
    """The override is for ABI risk, not for running without MLX at all."""
    monkeypatch.setenv(native_loader._ALLOW_MISMATCH_ENV, "1")
    with pytest.raises(RuntimeError, match="MLX runtime not found"):
        stub_runtime({"build_mlx_version": "0.31.2"}, None)


@pytest.mark.parametrize(
    "declared,expected",
    [
        ("0.31.2", ["0.31.2"]),
        ("0.31.2,0.31.3", ["0.31.2", "0.31.3"]),
        (["0.31.3"], ["0.31.3", "0.31.2"]),
        ("", ["0.31.2"]),
        (None, ["0.31.2"]),
    ],
)
def test_compatible_versions_always_include_the_build_version(declared, expected):
    build = {"build_mlx_version": "0.31.2"}
    if declared is not None:
        build["compatible_mlx_versions"] = declared
    assert native_loader._compatible_mlx_versions(build) == expected
