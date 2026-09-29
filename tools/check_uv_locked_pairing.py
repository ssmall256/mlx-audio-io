"""Exercise a fresh uv consumer with two MLX locks and the editable sdist source."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("0.31.2", "0.32.3")
SMOKE = r'''
import importlib.metadata
import tempfile
import wave
from pathlib import Path

import mlx.core as mx
import mlx_audio_io as mac

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "stereo.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0" * 16000 * 2 * 2)
    audio, sample_rate = mac.load(path)
    mx.eval(audio)
    assert isinstance(audio, mx.array)
    assert audio.shape == (16000, 2), audio.shape
    assert sample_rate == 16000, sample_rate
    info = mac.show_build_info()
    runtime = importlib.metadata.version("mlx")
    assert info["build_mlx_version"] == info["runtime_mlx_version"] == runtime, info
    print(f"| `{runtime}` | `{info['build_mlx_version']}` | `{audio.shape}` | ✅ |")
'''


def run(version: str) -> None:
    # Keep the consumer beside the checkout: macOS /tmp is a /private/tmp
    # symlink, which makes relative editable paths in a frozen lock invalid.
    with tempfile.TemporaryDirectory(prefix="mlx-audio-io-pairing-", dir=ROOT.parent) as tmp:
        project = Path(tmp)
        project.joinpath("pyproject.toml").write_text(
            "[project]\n"
            'name = "mlx-audio-io-pairing-smoke"\n'
            'version = "0.0.1"\n'
            'requires-python = ">=3.12,<3.13"\n'
            f'dependencies = ["mlx=={version}", "mlx-audio-io"]\n\n'
            "[tool.uv.sources]\n"
            f"mlx-audio-io = {{ path = {json.dumps(str(ROOT))}, editable = true }}\n\n"
            "[tool.uv.extra-build-dependencies]\n"
            'mlx-audio-io = [{ requirement = "mlx[cpu]", match-runtime = true }]\n'
        )
        env = os.environ | {"MLX_DEVICE": "cpu"}
        subprocess.run(["uv", "lock"], cwd=project, env=env, check=True)
        subprocess.run(
            ["uv", "sync", "--frozen", "--refresh-package", "mlx-audio-io", "--python", "3.12"],
            cwd=project,
            env=env,
            check=True,
        )
        subprocess.run(
            [str(project / ".venv" / "bin" / "python"), "-c", SMOKE],
            cwd=project,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    print("## Locked uv build and native array return", flush=True)
    print("| Runtime MLX | Build MLX | Stereo shape | Decode |", flush=True)
    print("|---|---|---|---|", flush=True)
    for version in VERSIONS:
        run(version)
