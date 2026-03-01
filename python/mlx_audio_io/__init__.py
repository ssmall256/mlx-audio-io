"""mlx-audio-io: Native audio I/O for MLX on macOS and Linux."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from ._native_loader import get_diagnostic_info, load_native_module

_MONO_MODE_MEAN = "mean"
_MONO_MODE_EQUAL_POWER = "equal_power"
_MONO_MODE_VALUES = {_MONO_MODE_MEAN, _MONO_MODE_EQUAL_POWER}


def _get_core_module() -> Any:
    return load_native_module()


def _normalize_mono_mode(mono_mode: str) -> str:
    mode = str(mono_mode).strip().lower()
    if mode not in _MONO_MODE_VALUES:
        raise ValueError(
            "mono_mode must be one of "
            f"{sorted(_MONO_MODE_VALUES)}, got {mono_mode!r}"
        )
    return mode


def _mixdown_channels_last(audio: mx.array, mono_mode: str) -> mx.array:
    channels = int(audio.shape[1])
    if channels <= 1:
        return audio

    if mono_mode == _MONO_MODE_EQUAL_POWER:
        scale = 1.0 / math.sqrt(float(channels))
        mixed = mx.sum(audio, axis=1, keepdims=True) * scale
    else:
        mixed = mx.mean(audio, axis=1, keepdims=True)
    return mixed.astype(audio.dtype)


def _mixdown_channels_first(audio: mx.array, mono_mode: str) -> mx.array:
    channels = int(audio.shape[0])
    if channels <= 1:
        return audio

    if mono_mode == _MONO_MODE_EQUAL_POWER:
        scale = 1.0 / math.sqrt(float(channels))
        mixed = mx.sum(audio, axis=0, keepdims=True) * scale
    else:
        mixed = mx.mean(audio, axis=0, keepdims=True)
    return mixed.astype(audio.dtype)


def load(
    path,
    sr=None,
    offset=0.0,
    duration=None,
    mono=False,
    mono_mode="mean",
    layout="channels_last",
    dtype="float32",
    resample_quality="default",
):
    mono_mode = _normalize_mono_mode(mono_mode)
    request_stereo_for_fold = bool(mono) and mono_mode == _MONO_MODE_EQUAL_POWER
    audio, out_sr = _get_core_module().load(
        path,
        sr=sr,
        offset=offset,
        duration=duration,
        mono=(False if request_stereo_for_fold else mono),
        layout=layout,
        dtype=dtype,
        resample_quality=resample_quality,
    )
    if request_stereo_for_fold:
        if layout == "channels_first":
            audio = _mixdown_channels_first(audio, mono_mode)
        else:
            audio = _mixdown_channels_last(audio, mono_mode)
    return audio, out_sr


def info(path):
    return _get_core_module().info(path)


def resample(audio, in_sr, out_sr, quality="default"):
    return _get_core_module().resample(audio, in_sr, out_sr, quality=quality)


class _WindowedStreamReader:
    """Reader shim that applies offset/duration over a native stream reader."""

    def __init__(self, base_reader, offset_s=0.0, duration_s=None):
        self._base = base_reader
        self._sample_rate = int(base_reader.sample_rate)
        self._chunk_frames = int(base_reader.chunk_frames)
        self._channels = int(base_reader.channels)
        self._frames_read = 0
        self._dtype = None
        if offset_s < 0:
            raise ValueError("offset must be >= 0")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("duration must be > 0 when provided")
        self._skip_remaining = int(math.floor(float(offset_s) * self._sample_rate))
        self._emit_remaining = (
            None
            if duration_s is None
            else int(math.floor(float(duration_s) * self._sample_rate))
        )
        if self._emit_remaining is not None and self._emit_remaining <= 0:
            raise ValueError("duration too small, results in 0 frames.")

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def channels(self):
        return self._channels

    @property
    def chunk_frames(self):
        return self._chunk_frames

    @property
    def frames_read(self):
        return self._frames_read

    def at_eof(self):
        if self._emit_remaining is not None and self._emit_remaining <= 0:
            return True
        return self._base.at_eof()

    def read_chunk(self):
        if self.at_eof():
            dtype = self._dtype or mx.float32
            return mx.zeros((0, self._channels), dtype=dtype), self._sample_rate

        while True:
            chunk, sr = self._base.read_chunk()
            if chunk.shape[0] == 0:
                dtype = self._dtype or mx.float32
                return mx.zeros((0, self._channels), dtype=dtype), self._sample_rate
            self._dtype = chunk.dtype

            start = 0
            if self._skip_remaining > 0:
                take = min(self._skip_remaining, int(chunk.shape[0]))
                self._skip_remaining -= take
                start = take
                if start >= int(chunk.shape[0]):
                    continue

            out = chunk[start:, :]
            if self._emit_remaining is not None and int(out.shape[0]) > self._emit_remaining:
                out = out[: self._emit_remaining, :]

            out_frames = int(out.shape[0])
            if out_frames == 0:
                dtype = self._dtype or mx.float32
                return mx.zeros((0, self._channels), dtype=dtype), self._sample_rate

            self._frames_read += out_frames
            if self._emit_remaining is not None:
                self._emit_remaining -= out_frames
            return out, sr

    def __iter__(self):
        return self

    def __next__(self):
        if self.at_eof():
            raise StopIteration
        chunk, sr = self.read_chunk()
        if chunk.shape[0] == 0:
            raise StopIteration
        return chunk, sr


class _MonoModeStreamReader:
    """Reader shim that applies Python-side mono fold to stream chunks."""

    def __init__(self, base_reader, mono_mode="mean"):
        self._base = base_reader
        self._sample_rate = int(base_reader.sample_rate)
        self._chunk_frames = int(base_reader.chunk_frames)
        self._frames_read = 0
        self._mono_mode = _normalize_mono_mode(mono_mode)

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def channels(self):
        return 1

    @property
    def chunk_frames(self):
        return self._chunk_frames

    @property
    def frames_read(self):
        return self._frames_read

    def at_eof(self):
        return self._base.at_eof()

    def read_chunk(self):
        chunk, sr = self._base.read_chunk()
        if int(chunk.shape[0]) == 0:
            return chunk, sr

        out = _mixdown_channels_last(chunk, self._mono_mode)
        self._frames_read += int(out.shape[0])
        return out, sr

    def __iter__(self):
        return self

    def __next__(self):
        if self.at_eof():
            raise StopIteration
        chunk, sr = self.read_chunk()
        if chunk.shape[0] == 0:
            raise StopIteration
        return chunk, sr


def stream(
    path,
    chunk_frames=None,
    chunk_duration=None,
    sr=None,
    mono=False,
    mono_mode="mean",
    dtype="float32",
    offset=0.0,
    duration=None,
):
    if (chunk_frames is None) == (chunk_duration is None):
        raise ValueError("Exactly one of chunk_frames or chunk_duration must be specified.")
    mono_mode = _normalize_mono_mode(mono_mode)
    request_stereo_for_fold = bool(mono) and mono_mode == _MONO_MODE_EQUAL_POWER
    core = _get_core_module()
    try:
        # Preferred path: native stream-level slicing support.
        reader = core.stream(
            path,
            chunk_frames=chunk_frames,
            chunk_duration=chunk_duration,
            sr=sr,
            mono=(False if request_stereo_for_fold else mono),
            offset=offset,
            duration=duration,
            dtype=dtype,
        )
        if request_stereo_for_fold:
            return _MonoModeStreamReader(reader, mono_mode=mono_mode)
        return reader
    except TypeError:
        # Compatibility fallback for older native modules.
        reader = core.stream(
            path,
            chunk_frames=chunk_frames,
            chunk_duration=chunk_duration,
            sr=sr,
            mono=(False if request_stereo_for_fold else mono),
            dtype=dtype,
        )
        if request_stereo_for_fold:
            reader = _MonoModeStreamReader(reader, mono_mode=mono_mode)
        if float(offset) == 0.0 and duration is None:
            return reader
        return _WindowedStreamReader(reader, offset_s=offset, duration_s=duration)


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


def batch_load(
    paths,
    sr=None,
    mono=False,
    mono_mode="mean",
    dtype="float32",
    num_workers=4,
):
    """Load multiple audio files in parallel using threads."""
    from concurrent.futures import ThreadPoolExecutor

    def _load_one(path):
        return load(path, sr=sr, mono=mono, mono_mode=mono_mode, dtype=dtype)

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
