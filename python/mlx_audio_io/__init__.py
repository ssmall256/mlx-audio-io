"""mlx-audio-io: Native audio I/O for MLX on macOS and Linux."""

from __future__ import annotations

from typing import Any

from ._native_loader import get_diagnostic_info, load_native_module


def _get_core_module() -> Any:
    return load_native_module()


def load(
    path,
    sr=None,
    offset=0.0,
    duration=None,
    mono=False,
    layout="channels_last",
    dtype="float32",
    resample_quality="default",
):
    return _get_core_module().load(
        path,
        sr=sr,
        offset=offset,
        duration=duration,
        mono=mono,
        layout=layout,
        dtype=dtype,
        resample_quality=resample_quality,
    )


def info(path):
    return _get_core_module().info(path)


def resample(audio, in_sr, out_sr, quality="default"):
    return _get_core_module().resample(audio, in_sr, out_sr, quality=quality)


def stream(path, chunk_frames=None, chunk_duration=None, sr=None, mono=False, dtype="float32"):
    return _get_core_module().stream(
        path,
        chunk_frames=chunk_frames,
        chunk_duration=chunk_duration,
        sr=sr,
        mono=mono,
        dtype=dtype,
    )


def _maybe_convert_numpy(audio):
    try:
        import numpy as np
    except ImportError:
        return audio

    if isinstance(audio, np.ndarray):
        import mlx.core as mx

        return mx.array(audio)

    return audio


def save(path, audio, sr, layout="channels_last", encoding="float32", bitrate="auto", clip=True):
    """Save an mlx array (or numpy array) to an audio file."""
    audio = _maybe_convert_numpy(audio)
    return _get_core_module().save(
        path,
        audio,
        sr,
        layout=layout,
        encoding=encoding,
        bitrate=bitrate,
        clip=clip,
    )


def batch_load(paths, sr=None, mono=False, dtype="float32", num_workers=4):
    """Load multiple audio files in parallel using threads."""
    from concurrent.futures import ThreadPoolExecutor

    def _load_one(path):
        return load(path, sr=sr, mono=mono, dtype=dtype)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(_load_one, list(paths)))


def show_build_info() -> dict[str, Any]:
    """Return build/runtime diagnostic metadata without importing native code."""
    return get_diagnostic_info()


def __getattr__(name: str) -> Any:
    if name in {"AudioInfo", "AudioStreamReader"}:
        return getattr(_get_core_module(), name)
    raise AttributeError(f"module 'mlx_audio_io' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"AudioInfo", "AudioStreamReader"})


__all__ = [
    "load",
    "save",
    "resample",
    "info",
    "stream",
    "batch_load",
    "show_build_info",
    "AudioInfo",
    "AudioStreamReader",
]
