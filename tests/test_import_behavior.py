"""Tests around lazy native import behavior."""

from __future__ import annotations

import importlib
import sys


def test_import_is_pure_python_and_does_not_load_core():
    if "mlx_audio_io._core" in sys.modules:
        del sys.modules["mlx_audio_io._core"]

    module = importlib.import_module("mlx_audio_io")
    assert module is not None
    assert "mlx_audio_io._core" not in sys.modules


def test_first_native_api_call_invokes_lazy_loader(monkeypatch):
    import mlx_audio_io as aio

    calls = {"count": 0}

    class _StubCore:
        @staticmethod
        def info(path):
            return {"path": path}

    def _fake_loader():
        calls["count"] += 1
        return _StubCore()

    monkeypatch.setattr(aio, "load_native_module", _fake_loader)

    out = aio.info("dummy.wav")
    assert out == {"path": "dummy.wav"}
    assert calls["count"] == 1
