"""Tests for mlx_audio_io.resample()."""

import mlx.core as mx
import numpy as np
import pytest

import mlx_audio_io as mac

try:
    import torch
    import torchaudio
except Exception:  # pragma: no cover - optional dependency
    torch = None
    torchaudio = None


_HAS_TORCHAUDIO = torch is not None and torchaudio is not None


def _make_sine(sr, duration=0.1, channels=1, freq=440.0):
    """Create a sine wave mlx array (frames, channels)."""
    frames = int(sr * duration)
    t = np.linspace(0, duration, frames, endpoint=False, dtype=np.float32)
    mono = np.sin(2 * np.pi * freq * t)
    if channels == 1:
        return mx.array(mono.reshape(-1, 1))
    return mx.array(np.column_stack([mono] * channels))


class TestNoOp:
    def test_same_sr_returns_unchanged(self):
        audio = _make_sine(16000)
        result = mac.resample(audio, 16000, 16000)
        np.testing.assert_array_equal(np.array(result), np.array(audio))

    def test_same_sr_1d(self):
        audio = mx.array(np.sin(np.linspace(0, 1, 1600, dtype=np.float32)))
        result = mac.resample(audio, 16000, 16000)
        assert result.ndim == 1
        np.testing.assert_array_equal(np.array(result), np.array(audio))


class TestFrameCount:
    def test_upsample_doubles_frames(self):
        audio = _make_sine(16000, duration=0.5)
        result = mac.resample(audio, 16000, 32000)
        expected = int(0.5 * 32000)
        assert abs(result.shape[0] - expected) <= 2

    def test_downsample_reduces_frames(self):
        audio = _make_sine(44100, duration=0.5)
        result = mac.resample(audio, 44100, 16000)
        expected = int(0.5 * 16000)
        assert abs(result.shape[0] - expected) <= 2


class TestShape:
    def test_stereo_preserved(self):
        audio = _make_sine(44100, channels=2)
        result = mac.resample(audio, 44100, 16000)
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_mono_1d_stays_1d(self):
        audio = mx.array(np.sin(np.linspace(0, 1, 4410, dtype=np.float32)))
        assert audio.ndim == 1
        result = mac.resample(audio, 44100, 16000)
        assert result.ndim == 1

    def test_mono_2d_stays_2d(self):
        audio = _make_sine(44100, channels=1)
        assert audio.ndim == 2
        result = mac.resample(audio, 44100, 16000)
        assert result.ndim == 2
        assert result.shape[1] == 1


class TestQuality:
    @pytest.mark.parametrize("q", ["default", "fastest", "low", "medium", "high", "best"])
    def test_quality_levels(self, q):
        audio = _make_sine(44100)
        result = mac.resample(audio, 44100, 16000, quality=q)
        assert result.shape[0] > 0

    def test_invalid_quality_raises(self):
        audio = _make_sine(44100)
        with pytest.raises(ValueError):
            mac.resample(audio, 44100, 16000, quality="ultra")


@pytest.mark.skipif(not _HAS_TORCHAUDIO, reason="torch/torchaudio not installed")
class TestTorchAudioCompat:
    def test_quality_mode_is_accepted(self):
        audio = _make_sine(44100, channels=2)
        result = mac.resample(audio, 44100, 16000, quality="torchaudio_compat")
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_matches_direct_torchaudio_stereo(self):
        audio = _make_sine(44100, duration=0.4, channels=2, freq=523.25)
        got = np.asarray(mac.resample(audio, 44100, 16000, quality="torchaudio_compat"))

        ref_in = torch.from_numpy(np.asarray(audio, dtype=np.float32).T).contiguous()
        with torch.no_grad():
            ref = torchaudio.functional.resample(ref_in, 44100, 16000).T.cpu().numpy().astype(np.float32)

        np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-6)

    def test_matches_direct_torchaudio_1d(self):
        mono = mx.array(np.sin(np.linspace(0, 1, 4410, dtype=np.float32)))
        got = np.asarray(mac.resample(mono, 44100, 16000, quality="torchaudio_compat"))

        ref_in = torch.from_numpy(np.asarray(mono, dtype=np.float32))[None, :]
        with torch.no_grad():
            ref = torchaudio.functional.resample(ref_in, 44100, 16000)[0].cpu().numpy().astype(np.float32)

        np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-6)


class TestValidation:
    def test_invalid_in_sr_raises(self):
        audio = _make_sine(44100)
        with pytest.raises(ValueError):
            mac.resample(audio, 0, 16000)

    def test_invalid_out_sr_raises(self):
        audio = _make_sine(44100)
        with pytest.raises(ValueError):
            mac.resample(audio, 44100, -1)

    def test_negative_sr_raises(self):
        audio = _make_sine(44100)
        with pytest.raises(ValueError):
            mac.resample(audio, -44100, 16000)


class TestZeroFrames:
    def test_zero_frames_2d(self):
        audio = mx.zeros((0, 1))
        result = mac.resample(audio, 44100, 16000)
        assert result.shape[0] == 0
        assert result.ndim == 2

    def test_zero_frames_1d(self):
        audio = mx.array(np.array([], dtype=np.float32))
        result = mac.resample(audio, 44100, 16000)
        assert result.shape[0] == 0
        assert result.ndim == 1


class TestFloat16:
    def test_float16_input_produces_valid_output(self):
        audio = mx.array(np.sin(np.linspace(0, 1, 4410, dtype=np.float32)).reshape(-1, 1))
        audio_f16 = audio.astype(mx.float16)
        result = mac.resample(audio_f16, 44100, 16000)
        assert result.shape[0] > 0
        assert result.dtype == mx.float32


class TestRoundtrip:
    def test_roundtrip_vs_load(self, pcm16_stereo_44k1):
        """resample(load(file), 44100, 16000) ≈ load(file, sr=16000)."""
        audio_native, sr = mac.load(pcm16_stereo_44k1)
        assert sr == 44100

        resampled = mac.resample(audio_native, 44100, 16000)
        loaded_16k, sr2 = mac.load(pcm16_stereo_44k1, sr=16000)
        assert sr2 == 16000

        # Frame counts should be very close
        min_frames = min(resampled.shape[0], loaded_16k.shape[0])
        assert abs(resampled.shape[0] - loaded_16k.shape[0]) <= 2

        # Values should be reasonably close (different resampling paths)
        r = np.array(resampled[:min_frames])
        ref = np.array(loaded_16k[:min_frames])
        np.testing.assert_allclose(r, ref, atol=0.05)
