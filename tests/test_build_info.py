"""Tests for build metadata fallback parsing."""

from __future__ import annotations

import mlx_audio_io._build_info as build_info


class _FakeDist:
    def __init__(self, wheel_text: str):
        self._wheel_text = wheel_text

    def read_text(self, name: str) -> str:
        if name == "WHEEL":
            return self._wheel_text
        return ""


def test_manylinux_arch_parsing(monkeypatch):
    monkeypatch.setattr(build_info, "_read_embedded_build_info", lambda: {})
    monkeypatch.setattr(
        build_info.importlib_metadata,
        "distribution",
        lambda _name: _FakeDist("Tag: cp314-cp314-manylinux_2_39_x86_64\n"),
    )
    monkeypatch.setattr(build_info, "_package_version", lambda: "1.2.3")

    info = build_info.load_build_info()

    assert info["python_tag"] == "cp314"
    assert info["build_os_name"] == "Linux"
    assert info["arch"] == "x86_64"
    assert info["wheel_version"] == "1.2.3"


def test_musllinux_aarch64_arch_parsing(monkeypatch):
    monkeypatch.setattr(build_info, "_read_embedded_build_info", lambda: {})
    monkeypatch.setattr(
        build_info.importlib_metadata,
        "distribution",
        lambda _name: _FakeDist("Tag: cp313-cp313-musllinux_1_2_aarch64\n"),
    )
    monkeypatch.setattr(build_info, "_package_version", lambda: "1.2.3")

    info = build_info.load_build_info()

    assert info["build_os_name"] == "Linux"
    assert info["arch"] == "aarch64"
