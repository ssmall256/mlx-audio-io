"""mlx-audio-io: Native audio I/O for MLX on macOS and Linux."""

from __future__ import annotations

import math
import os
from typing import Any

import mlx.core as mx

from ._native_loader import get_diagnostic_info, load_native_module

_MONO_MODE_MEAN = "mean"
_MONO_MODE_EQUAL_POWER = "equal_power"
_MONO_MODE_VALUES = {_MONO_MODE_MEAN, _MONO_MODE_EQUAL_POWER}
_RESAMPLE_QUALITY_NATIVE_VALUES = {
    "default",
    "fastest",
    "low",
    "medium",
    "high",
    "best",
    "soxr_hq",
    "soxr_vhq",
}
_RESAMPLE_QUALITY_ALIASES = {
    "soxr_style": "soxr_hq",
    "soxr_compat": "soxr_hq",
}
_RESAMPLE_QUALITY_TORCHAUDIO = "torchaudio_compat"
_RESAMPLE_QUALITY_SOXR_VALUES = {"soxr_hq", "soxr_vhq"}
_LAYOUT_CHANNELS_LAST = "channels_last"
_LAYOUT_CHANNELS_FIRST = "channels_first"
_LAYOUT_VALUES = {_LAYOUT_CHANNELS_LAST, _LAYOUT_CHANNELS_FIRST}


def _normalize_layout(layout: str) -> str:
    value = str(layout).strip().lower()
    if value not in _LAYOUT_VALUES:
        raise ValueError(
            "layout must be one of "
            f"{sorted(_LAYOUT_VALUES)}, got {layout!r}"
        )
    return value


def _get_core_module() -> Any:
    return load_native_module()


def _normalize_path(path) -> str:
    return os.fspath(path)


def _normalize_mono_mode(mono_mode: str) -> str:
    mode = str(mono_mode).strip().lower()
    if mode not in _MONO_MODE_VALUES:
        raise ValueError(
            "mono_mode must be one of "
            f"{sorted(_MONO_MODE_VALUES)}, got {mono_mode!r}"
        )
    return mode


def _normalize_resample_quality(quality: str) -> str:
    q = str(quality).strip().lower()
    q = _RESAMPLE_QUALITY_ALIASES.get(q, q)
    if q in _RESAMPLE_QUALITY_NATIVE_VALUES or q == _RESAMPLE_QUALITY_TORCHAUDIO:
        return q
    allowed = sorted(_RESAMPLE_QUALITY_NATIVE_VALUES | {_RESAMPLE_QUALITY_TORCHAUDIO} | set(_RESAMPLE_QUALITY_ALIASES.keys()))
    raise ValueError(f"Invalid resample quality {quality!r}. Must be one of {allowed}.")


def supports_soxr() -> bool:
    return bool(getattr(_get_core_module(), "_HAS_SOXR", False))


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


# Compiled versions fuse the elementwise ops (sum/mean + scale + astype),
# avoiding intermediate buffers.  Most beneficial in streaming where mixdown
# runs per-chunk with fixed shapes.
_compiled_mixdown_channels_last = mx.compile(_mixdown_channels_last)
_compiled_mixdown_channels_first = mx.compile(_mixdown_channels_first)


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
    low_memory=False,
):
    """Load an audio file.

    ``low_memory=True`` routes through a bounded-scratch streaming resample
    pipeline (requires libsoxr): the file is read chunk-by-chunk at its native
    sample rate and pushed through a stateful soxr resampler directly into a
    single preallocated output buffer, instead of first materializing the full
    native-SR input array. Peak scratch RAM becomes independent of file length
    — useful for hour-scale files. Ignored when ``sr`` is ``None`` or equal to
    the file's native rate. Only the soxr quality modes are supported in this
    path; ``resample_quality`` must be ``'default'``, ``'soxr_hq'``, or
    ``'soxr_vhq'`` when ``low_memory=True``.
    """
    mono_mode = _normalize_mono_mode(mono_mode)
    resample_quality_norm = _normalize_resample_quality(resample_quality)
    request_stereo_for_fold = bool(mono) and mono_mode == _MONO_MODE_EQUAL_POWER

    # When caller asks for resampling (sr is set) with default quality,
    # auto-select the best available backend: soxr_vhq > best.
    if resample_quality_norm == "default" and sr is not None:
        resample_quality_norm = "soxr_vhq" if supports_soxr() else "best"

    # Streaming (bounded-scratch) fast path. Only takes effect when a target
    # sr is requested that differs from the file's native rate — otherwise the
    # regular load already streams. When the requested fold is equal_power
    # mono, mixdown has to happen at the full channel count so we run the
    # stream with mono=False and fold after — same pattern as the default
    # load path.
    if low_memory and sr is not None:
        if not supports_soxr():
            raise RuntimeError(
                "low_memory=True requires libsoxr support; this build has none"
            )
        if resample_quality_norm not in _RESAMPLE_QUALITY_SOXR_VALUES:
            raise ValueError(
                "low_memory=True only supports soxr quality modes "
                f"({sorted(_RESAMPLE_QUALITY_SOXR_VALUES)}); "
                f"got resample_quality={resample_quality!r}"
            )
        audio, out_sr = _get_core_module().load_streaming_resample(
            _normalize_path(path),
            target_sr=int(sr),
            offset=offset,
            duration=duration,
            mono=(False if request_stereo_for_fold else mono),
            layout=layout,
            dtype=dtype,
            quality=resample_quality_norm,
        )
        if request_stereo_for_fold:
            if layout == "channels_first":
                audio = _compiled_mixdown_channels_first(audio, mono_mode)
            else:
                audio = _compiled_mixdown_channels_last(audio, mono_mode)
        return audio, out_sr

    use_soxr_resample = (
        resample_quality_norm in _RESAMPLE_QUALITY_SOXR_VALUES and sr is not None
    )
    use_torchaudio_resample = (
        resample_quality_norm == _RESAMPLE_QUALITY_TORCHAUDIO and sr is not None
    )
    if use_soxr_resample and not supports_soxr():
        raise RuntimeError(
            f"quality={resample_quality_norm!r} requested but mlx-audio-io was built "
            "without libsoxr support"
        )
    deferred_resample = use_torchaudio_resample or use_soxr_resample
    native_load_quality = (
        "default"
        if deferred_resample or resample_quality_norm in _RESAMPLE_QUALITY_SOXR_VALUES
        else resample_quality_norm
    )

    audio, out_sr = _get_core_module().load(
        _normalize_path(path),
        sr=(None if deferred_resample else sr),
        offset=offset,
        duration=duration,
        mono=(False if request_stereo_for_fold else mono),
        layout=layout,
        dtype=dtype,
        resample_quality=native_load_quality,
    )

    # Deferred Python-level resample: resample() handles channels_first by
    # transposing internally, so pass layout through rather than re-deriving.
    if use_torchaudio_resample and int(out_sr) != int(sr):
        audio = resample(
            audio, int(out_sr), int(sr),
            quality=_RESAMPLE_QUALITY_TORCHAUDIO, layout=layout,
        )
        out_sr = int(sr)
    elif use_soxr_resample and int(out_sr) != int(sr):
        audio = resample(
            audio, int(out_sr), int(sr),
            quality=resample_quality_norm, layout=layout,
        )
        out_sr = int(sr)

    if request_stereo_for_fold:
        if layout == "channels_first":
            audio = _compiled_mixdown_channels_first(audio, mono_mode)
        else:
            audio = _compiled_mixdown_channels_last(audio, mono_mode)
    return audio, out_sr


def info(path):
    return _get_core_module().info(_normalize_path(path))


def _resample_torchaudio_compat(audio: mx.array, in_sr: int, out_sr: int) -> mx.array:
    import numpy as np

    try:
        import torch
        import torchaudio
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "quality='torchaudio_compat' requires torch and torchaudio to be installed"
        ) from exc

    x = audio.astype(mx.float32) if isinstance(audio, mx.array) else mx.array(audio, dtype=mx.float32)
    if x.ndim == 1:
        samples = np.asarray(x, dtype=np.float32)
        tx = torch.from_numpy(samples)[None, :]
        with torch.no_grad():
            ty = torchaudio.functional.resample(tx, int(in_sr), int(out_sr))
        return mx.array(ty[0].cpu().numpy().astype(np.float32), dtype=mx.float32)
    if x.ndim == 2:
        frames_channels = np.asarray(x, dtype=np.float32)
        tx = torch.from_numpy(frames_channels.T).contiguous()
        with torch.no_grad():
            ty = torchaudio.functional.resample(tx, int(in_sr), int(out_sr))
        return mx.array(ty.T.cpu().numpy().astype(np.float32), dtype=mx.float32)
    raise ValueError(
        "audio must be 1D [frames] or 2D [frames, channels] for torchaudio_compat "
        f"resampling, got shape={tuple(x.shape)}"
    )


def resample(audio, in_sr, out_sr, quality="default", layout="channels_last"):
    """Resample audio between sample rates.

    ``layout`` describes the axis order of the input (and thus the output):
    ``"channels_last"`` means ``[frames, channels]``; ``"channels_first"``
    means ``[channels, frames]``. 1D input is always treated as a mono frame
    sequence regardless of ``layout``.
    """
    quality_norm = _normalize_resample_quality(quality)
    layout_norm = _normalize_layout(layout)
    if quality_norm in _RESAMPLE_QUALITY_SOXR_VALUES and not supports_soxr():
        raise RuntimeError(
            f"quality={quality_norm!r} requested but mlx-audio-io was built "
            "without libsoxr support"
        )

    transpose_2d = (
        layout_norm == _LAYOUT_CHANNELS_FIRST
        and isinstance(audio, mx.array)
        and audio.ndim == 2
    )
    if transpose_2d:
        # swapaxes alone returns a non-contiguous view; the C++ resamplers
        # read linearly from data<float>() and would misinterpret the memory
        # layout. Force a contiguous copy in the transposed orientation.
        audio = mx.contiguous(mx.swapaxes(audio, 0, 1))

    if quality_norm == _RESAMPLE_QUALITY_TORCHAUDIO:
        result = _resample_torchaudio_compat(audio, int(in_sr), int(out_sr))
    else:
        result = _get_core_module().resample(audio, in_sr, out_sr, quality=quality_norm)

    if transpose_2d and result.ndim == 2:
        # Contiguous copy so downstream native calls (save, resample again,
        # etc.) that read linearly see the correct channels_first layout.
        result = mx.contiguous(mx.swapaxes(result, 0, 1))
    return result


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

        out = _compiled_mixdown_channels_last(chunk, self._mono_mode)
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
            _normalize_path(path),
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
            _normalize_path(path),
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
        _normalize_path(path),
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
    "supports_soxr",
    "AudioInfo",
    "AudioStreamReader",
]
