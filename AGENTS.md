# mlx-audio-io — Agent Rules

## What This Is

Cross-platform native audio I/O for MLX. C++ extension (nanobind) with macOS (AudioToolbox) and Linux (libav) backends. Decodes directly to `mx.array`.

## Hard Rules

- **Exact MLX version pin.** Build-time MLX version is recorded and enforced at runtime. Do not relax this check — ABI mismatches cause silent crashes.
- **Do not add Python-side audio decoding.** All decoding happens in C++. The Python layer is thin (API, preflight, diagnostics).
- **GIL must be released** during I/O-heavy C++ operations (load, save, read_chunk). Use `nb::gil_scoped_release`.
- **WAV fast-path must remain.** Both platforms have custom little-endian WAV parsers that bypass AudioToolbox/libav. Do not route WAV through the codec path. Supported encodings: pcm16/pcm24/pcm32, float32, float64 (downcast to float32). New decode branches follow the pcm32 pattern (separate source buffer, loop-convert, free).
- **Unsupported formats must throw, never silently return empty.** Both platforms must raise on unrecognized WAV encodings. Do not use `make_empty_result` as a fallback for decode failures.
- **LAME and minimp3 are vendored.** Do not add external MP3 library dependencies. Source lives in `src/vendor/`.
- **Platform backends are separate files.** `audio_backend_apple.cpp` and `audio_backend_linux.cpp` are compiled exclusively per platform. Do not merge them or add cross-platform ifdefs within a backend file.

## Building

```bash
uv sync --extra dev
```

This runs scikit-build-core + CMake. Requires CMake 3.24+, C++17 toolchain. macOS needs AudioToolbox + Accelerate (system frameworks). Optional: libsoxr for high-quality resampling.

Custom `build_backend.py` handles Homebrew cmake resolution and wheel repair (delocate on macOS, auditwheel on Linux).

## Testing

```bash
uv run pytest -q                           # all tests
uv run pytest -q -m "not apple_only"       # Linux-safe subset
uv run pytest -q -m "apple_only"           # macOS-only (AudioToolbox codecs)
```

Test fixtures in `conftest.py` auto-generate WAV/AIFF files. MP3 fixtures require ffmpeg or lame on PATH.

## Diagnostics

```bash
python -m mlx_audio_io.doctor
```

Run this after any build change. Checks: native binary hash, codesign, OS/arch/Python/MLX version matching.

## What Not to Touch

- Do not modify `_native_loader.py` preflight checks without understanding the failure modes (hash mismatch, codesign failure, MLX version drift).
- Do not change the nanobind module name (`_core`) — it's referenced by the native loader and doctor.
- Do not remove RAII wrappers (`ScopedExtAudioFile`, `ScopedMp3Decoder`, `ScopedMp3Encoder`). Manual resource management is not acceptable.
- Do not add new codec support without testing on both platforms.

## Wheel Verification

After building wheels: `python tools/check_wheel_linkage.py` — ensures no absolute host paths leak into the wheel.

## Feedback

Before finishing your task, you MUST check whether you encountered any bugs, unexpected behavior, workflow friction, missing features, or surprising performance. If so, record each one:

```bash
devfeedback mlx-audio-io "description of bug, friction, or idea"
```
