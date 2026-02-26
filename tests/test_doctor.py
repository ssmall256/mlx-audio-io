"""Tests for mlx_audio_io.doctor command behavior."""

from __future__ import annotations

import subprocess

import mlx_audio_io.doctor as doctor


def test_doctor_success_path(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "_run_child_process",
        lambda: subprocess.CompletedProcess(args=["python"], returncode=0, stdout="ok", stderr=""),
    )

    rc = doctor.main([])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Doctor check passed." in out


def test_doctor_exit_137_prints_corruption_guidance(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "_run_child_process",
        lambda: subprocess.CompletedProcess(args=["python"], returncode=137, stdout="", stderr=""),
    )

    rc = doctor.main([])
    out = capsys.readouterr().out

    assert rc == 1
    assert "Likely cause: code-sign failure or binary corruption." in out
    assert "do not copy `.venv` between machines" in out
    assert "rm -rf .venv && uv venv --python 3.11 && uv sync" in out


def test_doctor_signal_exit_prints_corruption_guidance(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "_run_child_process",
        lambda: subprocess.CompletedProcess(args=["python"], returncode=-9, stdout="", stderr=""),
    )

    rc = doctor.main([])
    out = capsys.readouterr().out

    assert rc == 1
    assert "terminated by signal" in out
    assert "Likely cause: code-sign failure or binary corruption." in out
