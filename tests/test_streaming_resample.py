"""Tests for the ``low_memory=True`` (bounded-scratch streaming resample) path.

These verify that the streaming pipeline produces output equivalent to the
single-shot soxr path, preserves layout/mono/dtype/window semantics, and
rejects invalid configurations.
"""

import math
import os
import struct

import mlx.core as mx
import numpy as np
import pytest

import mlx_audio_io as mac


_HAS_SOXR = mac.supports_soxr()

requires_soxr = pytest.mark.skipif(
    not _HAS_SOXR, reason="libsoxr not built into this wheel"
)


def _write_wav_pcm16(path, sample_rate, channels, duration_s, freq=440.0):
    num_frames = int(sample_rate * duration_s)
    samples = []
    for i in range(num_frames):
        val = math.sin(2.0 * math.pi * freq * i / sample_rate)
        # Add slight stereo decorrelation so channel-vs-frame bugs are catchable.
        for c in range(channels):
            v = val * (1.0 if c == 0 else 0.5)
            sample = max(-32768, min(32767, int(v * 32767)))
            samples.append(sample)

    data = struct.pack(f"<{len(samples)}h", *samples)
    bits_per_sample = 16
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data_size = len(data)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)


@pytest.fixture(scope="module")
def long_stereo_48k(tmp_path_factory):
    path = tmp_path_factory.mktemp("streaming_resample") / "stereo_48k_3s.wav"
    _write_wav_pcm16(str(path), sample_rate=48000, channels=2, duration_s=3.0)
    return str(path)


@pytest.fixture(scope="module")
def long_mono_44k1(tmp_path_factory):
    path = tmp_path_factory.mktemp("streaming_resample") / "mono_44k1_3s.wav"
    _write_wav_pcm16(str(path), sample_rate=44100, channels=1, duration_s=3.0)
    return str(path)


def _assert_arrays_close(streamed, single_shot, atol=1e-4, rtol=1e-4):
    a = np.array(streamed)
    b = np.array(single_shot)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    # soxr is deterministic for identical input; allow tiny numerical slack
    # since chunk boundaries affect internal filter state accumulation order.
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


class TestEquivalence:
    """Streaming output must match the single-shot soxr path."""

    @requires_soxr
    def test_stereo_channels_last(self, long_stereo_48k):
        streamed, sr_s = mac.load(
            long_stereo_48k, sr=44100,
            resample_quality="soxr_hq", low_memory=True,
        )
        single, sr = mac.load(
            long_stereo_48k, sr=44100, resample_quality="soxr_hq",
        )
        assert sr_s == sr == 44100
        _assert_arrays_close(streamed, single)

    @requires_soxr
    def test_stereo_channels_first(self, long_stereo_48k):
        streamed, _ = mac.load(
            long_stereo_48k, sr=44100, layout="channels_first",
            resample_quality="soxr_hq", low_memory=True,
        )
        single, _ = mac.load(
            long_stereo_48k, sr=44100, layout="channels_first",
            resample_quality="soxr_hq",
        )
        assert streamed.shape[0] == 2  # channels axis
        _assert_arrays_close(streamed, single)

    @requires_soxr
    def test_mono_file(self, long_mono_44k1):
        streamed, _ = mac.load(
            long_mono_44k1, sr=22050,
            resample_quality="soxr_hq", low_memory=True,
        )
        single, _ = mac.load(
            long_mono_44k1, sr=22050, resample_quality="soxr_hq",
        )
        _assert_arrays_close(streamed, single)

    @requires_soxr
    def test_mono_mixdown_mean(self, long_stereo_48k):
        streamed, _ = mac.load(
            long_stereo_48k, sr=44100, mono=True, mono_mode="mean",
            resample_quality="soxr_hq", low_memory=True,
        )
        single, _ = mac.load(
            long_stereo_48k, sr=44100, mono=True, mono_mode="mean",
            resample_quality="soxr_hq",
        )
        _assert_arrays_close(streamed, single)

    @requires_soxr
    def test_mono_mixdown_equal_power(self, long_stereo_48k):
        streamed, _ = mac.load(
            long_stereo_48k, sr=44100, mono=True, mono_mode="equal_power",
            resample_quality="soxr_hq", low_memory=True,
        )
        single, _ = mac.load(
            long_stereo_48k, sr=44100, mono=True, mono_mode="equal_power",
            resample_quality="soxr_hq",
        )
        _assert_arrays_close(streamed, single)

    @requires_soxr
    def test_offset_duration_window(self, long_stereo_48k):
        streamed, _ = mac.load(
            long_stereo_48k, sr=44100, offset=0.5, duration=1.0,
            resample_quality="soxr_hq", low_memory=True,
        )
        single, _ = mac.load(
            long_stereo_48k, sr=44100, offset=0.5, duration=1.0,
            resample_quality="soxr_hq",
        )
        # Duration windows can differ slightly at the tail depending on where
        # the resampler flush lands; allow a small length difference.
        a = np.array(streamed)
        b = np.array(single)
        n = min(a.shape[0], b.shape[0])
        assert abs(a.shape[0] - b.shape[0]) <= 4
        np.testing.assert_allclose(a[:n], b[:n], atol=1e-4, rtol=1e-4)

    @requires_soxr
    def test_float16_dtype(self, long_stereo_48k):
        streamed, _ = mac.load(
            long_stereo_48k, sr=44100, dtype="float16",
            resample_quality="soxr_hq", low_memory=True,
        )
        assert streamed.dtype == mx.float16
        single, _ = mac.load(
            long_stereo_48k, sr=44100, dtype="float16",
            resample_quality="soxr_hq",
        )
        # float16 has limited precision; compare with coarser tolerance.
        _assert_arrays_close(streamed, single, atol=5e-3, rtol=5e-3)

    @requires_soxr
    def test_soxr_vhq(self, long_stereo_48k):
        streamed, _ = mac.load(
            long_stereo_48k, sr=44100,
            resample_quality="soxr_vhq", low_memory=True,
        )
        single, _ = mac.load(
            long_stereo_48k, sr=44100, resample_quality="soxr_vhq",
        )
        _assert_arrays_close(streamed, single)


class TestPassthrough:
    """When no resampling is needed, streaming path falls back cleanly."""

    @requires_soxr
    def test_same_sr_passthrough(self, long_stereo_48k):
        streamed, sr = mac.load(
            long_stereo_48k, sr=48000,
            resample_quality="soxr_hq", low_memory=True,
        )
        single, _ = mac.load(long_stereo_48k, sr=48000)
        assert sr == 48000
        _assert_arrays_close(streamed, single, atol=1e-6, rtol=1e-6)


class TestRejection:
    """Invalid configurations must fail fast with a clear error."""

    @requires_soxr
    def test_rejects_non_soxr_quality(self, long_stereo_48k):
        with pytest.raises(ValueError, match="soxr"):
            mac.load(
                long_stereo_48k, sr=44100,
                resample_quality="best", low_memory=True,
            )

    @requires_soxr
    def test_rejects_torchaudio_compat(self, long_stereo_48k):
        with pytest.raises(ValueError, match="soxr"):
            mac.load(
                long_stereo_48k, sr=44100,
                resample_quality="torchaudio_compat", low_memory=True,
            )

    @pytest.mark.skipif(_HAS_SOXR, reason="build has soxr; skip no-soxr branch")
    def test_rejects_when_soxr_missing(self, long_stereo_48k):
        with pytest.raises(RuntimeError, match="libsoxr"):
            mac.load(
                long_stereo_48k, sr=44100,
                resample_quality="soxr_hq", low_memory=True,
            )

    @requires_soxr
    def test_low_memory_without_sr_is_ignored(self, long_stereo_48k):
        # Without a target sr, low_memory is a no-op — should not raise even
        # if soxr quality was specified.
        audio, sr = mac.load(
            long_stereo_48k, low_memory=True,
        )
        assert sr == 48000
        assert audio.shape[0] > 0
