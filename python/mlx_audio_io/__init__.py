"""mlx-audio-io: Native audio I/O for MLX on macOS and Linux."""

# Import mlx.core first to register nanobind type casters
import mlx.core  # noqa: F401

from ._core import AudioInfo, AudioStreamReader, info, load, resample, stream
from ._core import save as _save_core

try:
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def save(path, audio, sr, layout="channels_last", encoding="float32", bitrate="auto", clip=True):
    """Save an mlx array (or numpy array) to an audio file.

    Accepts mlx.core.array or numpy.ndarray. numpy arrays are
    automatically converted to mlx arrays before saving.
    """
    if _HAS_NUMPY and isinstance(audio, _np.ndarray):
        audio = mlx.core.array(audio)
    return _save_core(path, audio, sr, layout=layout, encoding=encoding, bitrate=bitrate, clip=clip)


def batch_load(paths, sr=None, mono=False, dtype="float32", num_workers=4):
    """Load multiple audio files in parallel using threads.

    The C++ load function releases the GIL, so threads provide
    true parallelism for I/O-bound decoding.

    Args:
        paths: Iterable of file paths.
        sr: Target sample rate (None for native).
        mono: If True, mix down to mono.
        dtype: Output dtype — 'float32' or 'float16'.
        num_workers: Number of threads (default 4).

    Returns:
        List of (array, sample_rate) tuples, one per path.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _load_one(path):
        return load(path, sr=sr, mono=mono, dtype=dtype)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(_load_one, list(paths)))


__all__ = [
    "load",
    "save",
    "resample",
    "info",
    "stream",
    "batch_load",
    "AudioInfo",
    "AudioStreamReader",
]
