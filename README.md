# mlx-audio-io

`mlx-audio-io` is the audio data layer for [MLX](https://github.com/ml-explore/mlx): fast file decode/encode directly to and from `mlx.core.array`, with one API across macOS and Linux.

## Why This Project Exists

MLX has strong tensor and model primitives, but it does not ship a first-class, cross-platform audio file I/O layer comparable to what `torchaudio` provides in the PyTorch ecosystem.

In practice, MLX users often end up with one of these compromises:
- bridge through NumPy/SoundFile/librosa with extra copies and inconsistent format behavior
- shell out to `ffmpeg`/`ffprobe` for non-WAV workflows
- pull in parts of the PyTorch audio stack just to handle common audio containers/codecs

`mlx-audio-io` closes that gap with a native backend designed for MLX workloads:
- direct decode/encode into `mlx.core.array`
- one Python API (`load`, `save`, `info`, `stream`, `batch_load`) on both macOS and Linux
- consistent validation and error messages across platforms
- support for training/inference data access patterns (partial reads, chunked streaming, optional resampling)

## Platform Backends

- macOS backend optimized for Apple Silicon via AudioToolbox
- Linux backend with native WAV/MP3 fast paths plus libav-backed codec support (FLAC/M4A/AIFF/CAF)

The public Python API is the same on both platforms: `load`, `save`, `info`, `stream`, `batch_load`.

## Backend Feature Matrix

| Capability | macOS backend | Linux backend |
|---|---|---|
| `info(path)` | AudioToolbox-supported formats (WAV, MP3, M4A/AAC, FLAC, AIFF, CAF, etc.) | WAV, MP3, FLAC, M4A/AAC, AIFF, CAF |
| `load(path)` | AudioToolbox-supported formats + native-rate MP3 fast path | WAV, MP3, FLAC, M4A/AAC, AIFF, CAF |
| `load(..., sr=...)` | Supported, with AudioToolbox resampling | Supported (`WAV/MP3` native linear path, other supported formats via libav decode/resample) |
| `save(path, ...)` | WAV, MP3, M4A/AAC, FLAC, AIFF, CAF | WAV, MP3, M4A/AAC, FLAC, AIFF, CAF |
| `encoding` | `float32`, `pcm16`, `alac` (for `.m4a`) | `float32`, `pcm16`, `alac` (for `.m4a`) |
| `stream(path, ...)` | AudioToolbox-supported formats + native-rate MP3 path | WAV, MP3, FLAC, M4A/AAC, AIFF, CAF |
| `stream(..., sr=...)` | Supported | Supported (`WAV/MP3` native linear path, other supported formats via libav-backed chunked decode path) |

Unsupported format/encoding combinations fail with explicit `ValueError` messages.

## Installation

### End users (PyPI)

For normal use:

```bash
pip install mlx-audio-io
```

### Contributors (source checkout)

For local development and tests:

```bash
git clone https://github.com/ssmall256/mlx-audio-io.git
cd mlx-audio-io
uv sync --extra dev
```

### Hard Rule: Do Not Copy `.venv` Between Machines

Do not copy project virtual environments across machines. Native extensions can fail
integrity/code-sign checks or crash when moved between hosts.

If you already copied one, recreate it:

```bash
rm -rf .venv && uv venv --python 3.11 && uv sync
```

### Linux source build behavior

Linux source builds require libav and use direct libav-backed paths:
- Linux `info()` for non-WAV formats uses direct libav metadata.
- Linux `load()` for non-WAV formats uses direct libav decode for all `offset`/`duration` combinations.
- Linux `stream()` for non-WAV formats uses direct libav packet/frame decode.
- Linux `save()` for encoded formats (`.mp3`, `.flac`, `.m4a`, `.aiff/.aif`, `.caf`) uses direct libav encode/mux.

### Requirements

- Python 3.10+
- Runtime:
  - macOS: Apple Silicon + `mlx`
  - Linux: `mlx[cpu]` (current default)
- Source builds:
  - CMake 3.24+, C++17 toolchain, `pkg-config`
  - Linux default build: `libavformat-dev`, `libavcodec-dev`, `libavutil-dev`, `libswresample-dev`

### Linux Troubleshooting

- `ModuleNotFoundError: mlx_audio_io`
  - Install in the project environment (`uv sync`) and run via `uv run ...`.
- `ImportError` for `mlx` on Linux
  - Ensure Linux dependency is installed as `mlx[cpu]`.
- Build failures on source installs
  - Verify `build-essential`, `cmake`, `ninja-build`, and `pkg-config` are installed.
- Extended Linux format support errors (`.mp3`, `.m4a`, `.flac`, `.aiff`, `.caf`)
  - For default Linux builds, ensure runtime libav libraries are present (`libavformat`, `libavcodec`, `libavutil`, `libswresample`).
- MP3 test fixture generation failures
  - Tests that generate MP3 fixtures require `ffmpeg` or `lame` available on `PATH`.
- Native import failures or unexpected crashes
  - Run diagnostics: `python -m mlx_audio_io.doctor`
  - Check MLX runtime compatibility:
    `python -c "import mlx_audio_io as aio; print(aio.show_build_info())"`
  - If `build_mlx_version` and `runtime_mlx_version` differ, reinstall with matching deps:
    `pip install -U "mlx==<build_mlx_version>" "mlx-audio-io"`
  - Avoid `pip install --no-deps` for `mlx-audio-io` unless you manually pin a matching `mlx` version.
  - Recreate env (do not copy `.venv` between machines):
    `rm -rf .venv && uv venv --python 3.11 && uv sync`

## Quickstart

```python
from mlx_audio_io import load, save, info, stream, batch_load

# Load
x, sr = load("speech.wav")

# Resample + mono
x16, sr16 = load("speech.wav", sr=16000, mono=True)

# Metadata without decoding
meta = info("speech.wav")

# Stream in chunks
for chunk, chunk_sr in stream("long.wav", chunk_duration=2.0):
    pass

# Save WAV
save("out.wav", x, sr)
save("out_pcm16.wav", x, sr, encoding="pcm16")

# Batch load
items = batch_load(["a.wav", "b.wav"], sr=16000, mono=True)
```

Additional save examples:

```python
save("out.flac", x, sr)
save("out.mp3", x, sr, bitrate="192k")
save("out.m4a", x, sr, bitrate="256k")
save("out.m4a", x, sr, encoding="alac")
```

## API Reference

### `load`

```python
load(path, sr=None, offset=0.0, duration=None, mono=False, mono_mode="mean",
     layout="channels_last", dtype="float32", resample_quality="default")
```

Decode audio into an `mlx.core.array`. Returns `(audio, sample_rate)`.

| Parameter | Default | Description |
|---|---|---|
| `path` | — | Path to audio file |
| `sr` | `None` | Target sample rate; `None` keeps native rate |
| `offset` | `0.0` | Start position in seconds |
| `duration` | `None` | Duration in seconds; `None` reads to end |
| `mono` | `False` | Mix down to mono |
| `mono_mode` | `"mean"` | Mono fold policy: `"mean"` or `"equal_power"` |
| `layout` | `"channels_last"` | `"channels_last"` `[frames, ch]` or `"channels_first"` `[ch, frames]` |
| `dtype` | `"float32"` | `"float32"` or `"float16"` |
| `resample_quality` | `"default"` | `"default"`, `"fastest"`, `"low"`, `"medium"`, `"high"`, `"best"`, `"torchaudio_compat"` |

> On Linux WAV/MP3 fast paths, resample quality levels currently map to the same linear behavior.
> `torchaudio_compat` requires `torch` + `torchaudio` and uses `torchaudio.functional.resample`.

### `batch_load`

```python
batch_load(paths, sr=None, mono=False, mono_mode="mean", dtype="float32", num_workers=4)
```

Threaded multi-file `load()`. Returns `list[(audio, sample_rate)]`.

### `save`

```python
save(path, audio, sr, layout="channels_last", encoding="float32",
     bitrate="auto", clip=True)
```

Write audio from `mx.array` (or `numpy.ndarray`) to disk.

| Parameter | Default | Description |
|---|---|---|
| `path` | — | Output file path (format inferred from extension) |
| `audio` | — | Audio data; 1-D input is treated as mono |
| `sr` | — | Sample rate |
| `layout` | `"channels_last"` | Layout of the input array |
| `encoding` | `"float32"` | `"float32"`, `"pcm16"`, or `"alac"` (for `.m4a`) |
| `bitrate` | `"auto"` | Bitrate for lossy formats (`.m4a` AAC, `.mp3` on Linux) |
| `clip` | `True` | Clamp samples to `[-1, 1]` before encoding |

### `stream`

```python
stream(path, chunk_frames=None, chunk_duration=None, sr=None,
       mono=False, mono_mode="mean", dtype="float32", offset=0.0, duration=None)
```

Return an iterator yielding `(audio_chunk, sample_rate)`. Exactly one of `chunk_frames` or `chunk_duration` is required.

| Parameter | Default | Description |
|---|---|---|
| `path` | — | Path to audio file |
| `chunk_frames` | `None` | Chunk size in frames |
| `chunk_duration` | `None` | Chunk size in seconds |
| `sr` | `None` | Target sample rate; `None` keeps native rate |
| `mono` | `False` | Mix down to mono |
| `mono_mode` | `"mean"` | Mono fold policy: `"mean"` or `"equal_power"` |
| `dtype` | `"float32"` | `"float32"` or `"float16"` |
| `offset` | `0.0` | Start position in seconds for windowed stream |
| `duration` | `None` | Duration in seconds for windowed stream; `None` streams to end |

### `info`

```python
info(path)
```

Return `AudioInfo` metadata without decoding sample buffers.

| Field | Description |
|---|---|
| `frames` | Total number of sample frames |
| `sample_rate` | Sample rate in Hz |
| `channels` | Number of channels |
| `duration` | Duration in seconds |
| `subtype` | Sample encoding (e.g. `pcm16`, `float32`) |
| `container` | File format (e.g. `wav`, `mp3`, `m4a`) |

## Testing

Run all tests:

```bash
uv sync --extra dev
uv run python -m pytest -q
```

Run Linux supported subset:

```bash
uv run python -m pytest -q -m "not apple_only"
```

Run Apple-only subset:

```bash
uv run python -m pytest -q -m "apple_only"
```

Linux Docker run from a macOS host:

```bash
docker run --rm -it --platform linux/arm64 \
  -v "$PWD":/work -w /work \
  python:3.14-bookworm bash -lc '
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config ffmpeg \
      libavformat-dev libavcodec-dev libavutil-dev libswresample-dev &&
    python -m pip install -U pip uv &&
    uv sync --extra dev &&
    uv run python -m pytest -q -m "not apple_only"
  '
```

## Performance

Benchmark methodology, commands, and full result tables live in [`docs/benchmarking.md`](docs/benchmarking.md).

Headline numbers (194.8s stereo PCM16 WAV @ 44.1 kHz, p50 median latency):

| Task | macOS M4 Max | Linux arm64 |
|---|---|---|
| Full WAV load | **3.59 ms** — 6.9x faster than librosa | **8.41 ms** — 5.9x faster than librosa |
| WAV partial read (1 s) | **0.04 ms** — 3.4x faster than librosa | **0.05 ms** — 2.6x faster than librosa |
| WAV save (float32) | **6.98 ms** — 2.8x faster than soundfile | **31.70 ms** — 1.8x faster than soundfile |
| MP3 load (native SR) | **63.70 ms** — 1.3x faster than librosa | **80.93 ms** — on par with librosa |
| M4A/AAC load | **56.31 ms** — 2.2x faster than librosa | **89.63 ms** — 1.6x faster than librosa |
| Load + resample 16 kHz | **13.12 ms** — 4.4x faster than librosa | **10.93 ms** — 7.9x faster than librosa |

Full tables with torchaudio comparisons, M1 Max, and Linux x86_64 results are in the benchmarking doc.

## License

MIT
