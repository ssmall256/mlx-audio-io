"""Build the extension once, with no MLX installed, then use it with several MLX releases.

Since 1.3.23 the extension does not link libmlx: one build must load, save and
resample under every supported MLX runtime without a rebuild.
"""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = ("0.32.0", "0.32.3")
# Every MLX release in VERSIONS ships wheels for 3.12.
PYTHON = "3.12"
SMOKE = r'''
import tempfile, wave
from pathlib import Path
import mlx.core as mx
import mlx_audio_io as mac
with tempfile.TemporaryDirectory() as d:
    src = Path(d) / "in.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(bytes(range(256)) * 250)
    audio, sr = mac.load(src, layout="channels_first")
    assert isinstance(audio, mx.array) and audio.shape == (2, 16000) and sr == 16000, (audio.shape, sr)
    out = Path(d) / "out.wav"
    mac.save(out, audio, sr, layout="channels_first", encoding="float32")
    back, _ = mac.load(out, layout="channels_first")
    assert mx.array_equal(audio, back).item()
    up = mac.resample(audio, sr, 44100, layout="channels_first")
    assert up.shape[0] == 2
print(f"| `{mx.__version__}` | load/save/resample | ✅ |")
'''


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mlx-audio-io-matrix-") as tmp:
        tmp = Path(tmp)
        # Isolated build: the backend must not need (or fetch) MLX.
        run(["uv", "build", "--wheel", "--python", PYTHON, "-o", str(tmp / "dist"), str(ROOT)])
        wheel = next((tmp / "dist").glob("*.whl"))
        print("## One build, several MLX runtimes", flush=True)
        if platform.system() == "Darwin":
            import zipfile
            with zipfile.ZipFile(wheel) as zf:
                name = next(n for n in zf.namelist() if n.startswith("mlx_audio_io/_core") and n.endswith(".so"))
                zf.extract(name, tmp / "x")
            linked = subprocess.run(["otool", "-L", str(tmp / "x" / name)], capture_output=True, text=True, check=True).stdout
            assert "libmlx" not in linked, linked
            print("extension links libmlx: no", flush=True)
        print("| Runtime MLX | Check | Result |\n|---|---|---|", flush=True)
        extra = "mlx[cpu]" if platform.system() == "Linux" else "mlx"
        for version in VERSIONS:
            venv = tmp / f"v{version}"
            run(["uv", "venv", "-q", "--python", PYTHON, str(venv)])
            env = os.environ | {"VIRTUAL_ENV": str(venv), "MLX_DEVICE": "cpu"}
            run(["uv", "pip", "install", "-q", f"{extra}=={version}"], env=env)
            run(["uv", "pip", "install", "-q", "--no-deps", str(wheel)], env=env)
            run([str(venv / "bin" / "python"), "-c", SMOKE], env=env)


if __name__ == "__main__":
    main()
