# mlx-audio-io

Cross-platform native audio I/O for MLX. C++ extension via nanobind — macOS uses AudioToolbox, Linux uses libav. Decodes directly to `mx.array` with zero-copy where possible.

## Architecture

### C++ layer (`src/cpp/`)

| File | Purpose |
|------|---------|
| `bindings.cpp` | nanobind Python-C++ bridge, exports `_core` module |
| `audio_io.h/.cpp` | High-level API: `get_info`, `load_audio`, `save_audio`, `resample_audio` |
| `audio_backend.h` | Abstract backend interface |
| `audio_backend_apple.cpp` | macOS: AudioToolbox, fast-path WAV, NEON int16→float32 |
| `audio_backend_linux.cpp` | Linux: libav codecs, custom WAV parser |
| `audio_stream.h/.cpp` | Chunked streaming reader (platform-specific internals) |
| `mp3_decoder.h/.cpp` | RAII minimp3 wrapper |
| `mp3_encoder.h/.cpp` | RAII LAME wrapper |
| `tensor_utils.h` | Interleaved↔planar conversion, mono mixdown, MLX array creation |
| `raii_audio.h` | macOS RAII wrappers (ExtAudioFile, CFURL) |

### Python layer (`python/mlx_audio_io/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Public API: `load`, `save`, `info`, `stream`, `batch_load`, `resample` |
| `_native_loader.py` | Preflight checks (hash, codesign, MLX version match) |
| `doctor.py` | Diagnostic CLI (`python -m mlx_audio_io.doctor`) |

### Vendored (`src/vendor/`)

- **minimp3**: Header-only MP3 decoder
- **LAME**: Full MP3 encoder source (16 .c files)

## Key Design Decisions

- **nanobind with STABLE_ABI + LTO**: Binary compatibility across Python micro-versions
- **GIL released** on I/O-heavy operations (load, save, stream read_chunk)
- **Exact MLX version pin**: Build-time MLX version recorded; runtime rejects mismatches (prevents silent ABI crashes)
- **WAV fast-path**: Both platforms have custom little-endian parser (avoids libav/AudioToolbox overhead for common case). Supports pcm16/pcm24/pcm32, float32, and float64 (downcast to float32 since MLX has no float64 dtype). Unsupported encodings throw on both platforms.
- **Three resampling tiers**: platform-native → soxr_hq/soxr_vhq (optional libsoxr) → torchaudio_compat fallback
- **Mono mixdown modes**: `"mean"` (simple average) and `"equal_power"` (1/sqrt(N) scaling)
- **Layout flexibility**: `channels_last` (default) or `channels_first`

## Build

Requires CMake 3.24+, C++17 toolchain, pkg-config.

```bash
uv sync --extra dev              # builds C++ extension via scikit-build-core + CMake
```

**macOS extras**: AudioToolbox + Accelerate frameworks (system), optional libsoxr via Homebrew.
**Linux extras**: libavformat-dev, libavcodec-dev, libavutil-dev, libswresample-dev.

Custom `build_backend.py` handles: Homebrew cmake resolution, delocate/auditwheel wheel repair, MLX version embedding.

## Tests

```bash
uv run pytest -q                           # all tests
uv run pytest -q -m "not apple_only"       # Linux subset
uv run pytest -q -m "apple_only"           # macOS-only tests
```

Fixtures in `conftest.py` auto-generate WAV/AIFF/MP3 test files. MP3 fixtures require ffmpeg or lame on PATH.

## Diagnostics

```bash
python -m mlx_audio_io.doctor    # hash verification, codesign, version matching
```

## Ecosystem

**Dependencies**: `mlx` (exact version pin). **Consumed by**: mlx-amt-core, mlx-audio-separator, and any MLX project needing audio I/O.

## Feedback

Before finishing your task, you MUST check whether you encountered any bugs, unexpected behavior, workflow friction, missing features, or surprising performance. If so, record each one:

```bash
devfeedback mlx-audio-io "description of bug, friction, or idea"
```
