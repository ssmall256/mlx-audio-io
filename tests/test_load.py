"""Tests for mlx_audio_io.load()."""

import math
import os
import tempfile

import mlx.core as mx
import pytest
from mlx_audio_io import batch_load, load, save, supports_soxr

pytestmark = pytest.mark.linux_mvp
_HAS_SOXR_NATIVE = supports_soxr()


class TestLoadBasic:
    def test_shape_mono_file(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k)
        assert sr == 16000
        assert audio.dtype == mx.float32
        assert audio.ndim == 2
        assert audio.shape == (16000, 1)

    def test_shape_stereo_file(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1)
        assert sr == 44100
        assert audio.shape == (44100, 2)

    def test_float32_file(self, float32_stereo_48k):
        audio, sr = load(float32_stereo_48k)
        assert sr == 48000
        assert audio.shape == (48000, 2)
        assert audio.dtype == mx.float32

    def test_float64_file(self, float64_stereo_48k):
        audio, sr = load(float64_stereo_48k)
        assert sr == 48000
        assert audio.shape == (48000, 2)
        assert audio.dtype == mx.float32

    def test_float64_values_in_range(self, float64_stereo_48k):
        audio, sr = load(float64_stereo_48k)
        mx.eval(audio)
        assert mx.all(audio >= -1.0).item()
        assert mx.all(audio <= 1.0).item()

    def test_extensible_float32_file(self, extensible_float32_stereo_48k):
        audio, sr = load(extensible_float32_stereo_48k)
        assert sr == 48000
        assert audio.shape == (48000, 2)
        assert audio.dtype == mx.float32

    def test_extensible_float32_values_match_float32(
        self, float32_stereo_48k, extensible_float32_stereo_48k
    ):
        ref, _ = load(float32_stereo_48k)
        ext, _ = load(extensible_float32_stereo_48k)
        mx.eval(ref, ext)
        assert mx.max(mx.abs(ref - ext)).item() < 1e-6


class TestLoadAiff:
    def test_load_aiff(self, pcm16_stereo_44k1_aiff):
        audio, sr = load(pcm16_stereo_44k1_aiff)
        assert sr == 44100
        assert audio.dtype == mx.float32
        assert audio.shape == (44100, 2)

    def test_load_aiff_mono(self, pcm16_stereo_44k1_aiff):
        audio, sr = load(pcm16_stereo_44k1_aiff, mono=True)
        assert sr == 44100
        assert audio.shape == (44100, 1)

    def test_load_aiff_offset_duration_resample(self, pcm16_stereo_44k1_aiff):
        audio, sr = load(
            pcm16_stereo_44k1_aiff, offset=0.25, duration=0.5, sr=16000
        )
        assert sr == 16000
        assert abs(audio.shape[0] - 8000) <= 2
        assert audio.shape[1] == 2

    @pytest.mark.apple_only
    def test_load_resample_offset_tracks_expected_window_center(self):
        native_sr = 44100
        target_sr = 16000
        offset = 0.25
        duration = 0.5
        frames = native_sr

        t = mx.arange(frames) / native_sr
        ramp = mx.reshape((2.0 * t) - 1.0, [frames, 1]).astype(mx.float32)
        mx.eval(ramp)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, ramp, native_sr)
            sliced, sr = load(path, sr=target_sr, offset=offset, duration=duration)
            mx.eval(sliced)

            assert sr == target_sr
            assert abs(sliced.shape[0] - int(math.floor(duration * target_sr))) <= 2

            observed_center = mx.mean(sliced[:, 0]).item()
            expected_center = 2.0 * (offset + (duration / 2.0)) - 1.0
            assert abs(observed_center - expected_center) < 0.08
        finally:
            os.unlink(path)


class TestLoadExtendedOffsets:
    def test_load_flac_offset_duration(self):
        sr = 24000
        frames = sr * 2
        t = mx.arange(frames) / sr
        audio = mx.reshape(mx.sin(2.0 * 3.141592653589793 * 220.0 * t), [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            sliced, sliced_sr = load(path, offset=0.5, duration=0.75)
            assert sliced_sr == sr
            assert sliced.shape[1] == 1
            assert abs(sliced.shape[0] - int(0.75 * sr)) <= 2
        finally:
            os.unlink(path)


class TestLoadLayouts:
    def test_channels_last(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, layout="channels_last")
        assert audio.shape == (44100, 2)

    def test_channels_first(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, layout="channels_first")
        assert audio.shape == (2, 44100)

    def test_channels_first_mono_file(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, layout="channels_first")
        assert audio.shape == (1, 16000)

    def test_invalid_layout(self, pcm16_mono_16k):
        with pytest.raises(ValueError):
            load(pcm16_mono_16k, layout="invalid")

    def test_channels_first_with_resample(self, float32_stereo_48k):
        # Regression: prior to the fix, deferred (Python-level) resample was
        # fed channels_first audio directly. The resampler treated shape[0]
        # (channels=2) as frames and shape[1] (frames=48000) as channels,
        # allocating ~(2*ratio+256) * 48000 * 4 bytes and producing nonsense
        # output. For long files this ballooned to hundreds of GB.
        audio, sr = load(float32_stereo_48k, sr=44100, layout="channels_first")
        assert sr == 44100
        assert audio.ndim == 2
        assert audio.shape[0] == 2
        assert abs(audio.shape[1] - 44100) <= 2

    def test_channels_first_resample_matches_channels_last(self, float32_stereo_48k):
        last, sr_last = load(float32_stereo_48k, sr=44100, layout="channels_last")
        first, sr_first = load(float32_stereo_48k, sr=44100, layout="channels_first")
        assert sr_last == sr_first == 44100
        mx.eval(last, first)
        # channels_first should be the transpose of channels_last
        assert first.shape == (last.shape[1], last.shape[0])
        diff = mx.max(mx.abs(first - mx.swapaxes(last, 0, 1))).item()
        assert diff < 1e-5


class TestLoadMono:
    def test_mono_from_stereo(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, mono=True)
        assert audio.shape == (44100, 1)

    def test_mono_from_mono(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, mono=True)
        assert audio.shape == (16000, 1)

    def test_mono_channels_first(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, mono=True, layout="channels_first")
        assert audio.shape == (1, 44100)

    def test_mono_values_are_average(self, pcm16_stereo_44k1):
        """Mono should be average of channels (for identical channels, same as original)."""
        stereo, _ = load(pcm16_stereo_44k1)
        mono, _ = load(pcm16_stereo_44k1, mono=True)
        mx.eval(stereo, mono)
        # Since both channels have the same sine, mono should be close to either channel
        left = stereo[:, 0]
        mx.eval(left)
        max_diff = mx.max(mx.abs(mono[:, 0] - left)).item()
        assert max_diff < 1e-4

    def test_mono_mode_equal_power_scales_stereo_mix(self, pcm16_stereo_44k1):
        stereo, _ = load(pcm16_stereo_44k1)
        mono_mean, _ = load(pcm16_stereo_44k1, mono=True, mono_mode="mean")
        mono_equal_power, _ = load(
            pcm16_stereo_44k1, mono=True, mono_mode="equal_power"
        )
        mx.eval(stereo, mono_mean, mono_equal_power)

        left = stereo[:, 0]
        expected_equal_power = left * math.sqrt(2.0)
        mx.eval(expected_equal_power)

        diff_mean = mx.max(mx.abs(mono_mean[:, 0] - left)).item()
        diff_equal_power = mx.max(
            mx.abs(mono_equal_power[:, 0] - expected_equal_power)
        ).item()
        assert diff_mean < 1e-4
        assert diff_equal_power < 2e-3

    def test_invalid_mono_mode_raises(self, pcm16_mono_16k):
        with pytest.raises(ValueError, match="mono_mode"):
            load(pcm16_mono_16k, mono=True, mono_mode="invalid")


class TestLoadOffsetDuration:
    def test_offset_zero(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, offset=0.0)
        assert audio.shape[0] == 16000

    def test_offset_half(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, offset=0.5)
        assert audio.shape[0] == 8000

    def test_duration(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, duration=0.5)
        assert audio.shape[0] == 8000

    def test_offset_plus_duration(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, offset=0.25, duration=0.5)
        assert audio.shape[0] == 8000

    def test_duration_exceeds_remainder(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, offset=0.5, duration=2.0)
        assert audio.shape[0] == 8000

    def test_offset_beyond_end(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, offset=5.0)
        assert audio.shape[0] == 0
        assert audio.shape[1] == 1

    def test_negative_offset_raises(self, pcm16_mono_16k):
        with pytest.raises(ValueError):
            load(pcm16_mono_16k, offset=-1.0)

    def test_negative_duration_raises(self, pcm16_mono_16k):
        with pytest.raises(ValueError):
            load(pcm16_mono_16k, duration=-1.0)

    def test_zero_duration_raises(self, pcm16_mono_16k):
        with pytest.raises(ValueError):
            load(pcm16_mono_16k, duration=0.0)


class TestLoadResample:
    def test_resample_up(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, sr=32000)
        assert sr == 32000
        # Should have roughly double the frames
        assert abs(audio.shape[0] - 32000) <= 1

    def test_resample_down(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, sr=16000)
        assert sr == 16000
        assert abs(audio.shape[0] - 16000) <= 1


class TestLoadFloat16:
    def test_dtype_and_shape(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, dtype="float16")
        assert audio.dtype == mx.float16
        assert audio.shape == (16000, 1)
        assert sr == 16000

    def test_mono(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, dtype="float16", mono=True)
        assert audio.dtype == mx.float16
        assert audio.shape == (44100, 1)

    def test_channels_first(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, dtype="float16", layout="channels_first")
        assert audio.dtype == mx.float16
        assert audio.shape == (2, 44100)

    def test_values_close_to_float32(self, pcm16_mono_16k):
        f32, _ = load(pcm16_mono_16k, dtype="float32")
        f16, _ = load(pcm16_mono_16k, dtype="float16")
        mx.eval(f32, f16)
        max_diff = mx.max(mx.abs(f32 - f16.astype(mx.float32))).item()
        assert max_diff < 2e-3

    def test_empty_array_dtype(self, pcm16_mono_16k):
        audio, sr = load(pcm16_mono_16k, offset=999.0, dtype="float16")
        assert audio.dtype == mx.float16
        assert audio.shape[0] == 0


class TestLoadResampleQuality:
    def test_default_produces_valid_output(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, sr=16000, resample_quality="default")
        assert sr == 16000
        assert abs(audio.shape[0] - 16000) <= 1

    def test_fastest_produces_valid_output(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, sr=16000, resample_quality="fastest")
        assert sr == 16000
        assert abs(audio.shape[0] - 16000) <= 1

    def test_best_produces_valid_output(self, pcm16_stereo_44k1):
        audio, sr = load(pcm16_stereo_44k1, sr=16000, resample_quality="best")
        assert sr == 16000
        assert abs(audio.shape[0] - 16000) <= 1

    def test_ignored_when_no_resample(self, pcm16_mono_16k):
        """quality setting should not error when sr matches native."""
        audio, sr = load(pcm16_mono_16k, resample_quality="best")
        assert sr == 16000
        assert audio.shape[0] == 16000

    def test_invalid_raises(self, pcm16_mono_16k):
        with pytest.raises(ValueError):
            load(pcm16_mono_16k, resample_quality="invalid")

    def test_all_levels(self, pcm16_stereo_44k1):
        for level in ("default", "fastest", "low", "medium", "high", "best"):
            audio, sr = load(pcm16_stereo_44k1, sr=16000, resample_quality=level)
            assert sr == 16000
            assert audio.shape[0] > 0

    def test_default_auto_selects_soxr_vhq_when_available(self, pcm16_stereo_44k1):
        """When sr is set and resample_quality='default', load() should auto-select
        soxr_vhq if soxr is available, producing results different from 'fastest'."""
        audio_default, sr_d = load(pcm16_stereo_44k1, sr=16000)
        audio_fastest, sr_f = load(pcm16_stereo_44k1, sr=16000, resample_quality="fastest")
        assert sr_d == 16000
        assert sr_f == 16000
        assert audio_default.shape == audio_fastest.shape
        if _HAS_SOXR_NATIVE:
            # soxr_vhq should produce numerically different output from fastest
            diff = float(mx.max(mx.abs(audio_default - audio_fastest)))
            assert diff > 1e-6, f"default should differ from fastest when soxr available, max_diff={diff}"

    def test_default_no_resample_unchanged(self, pcm16_mono_16k):
        """When sr is None, default quality should not trigger soxr auto-selection."""
        audio, sr = load(pcm16_mono_16k, resample_quality="default")
        assert sr == 16000
        assert audio.shape[0] == 16000

    def test_soxr_quality_mode(self, pcm16_stereo_44k1):
        if _HAS_SOXR_NATIVE:
            audio, sr = load(pcm16_stereo_44k1, sr=16000, resample_quality="soxr_hq")
            assert sr == 16000
            assert audio.shape[0] > 0
        else:
            with pytest.raises(RuntimeError, match="without libsoxr support"):
                load(pcm16_stereo_44k1, sr=16000, resample_quality="soxr_hq")


class TestLoadErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load("/nonexistent/path.wav")

    def test_unsupported_dtype(self, pcm16_mono_16k):
        with pytest.raises(ValueError):
            load(pcm16_mono_16k, dtype="int16")

    def test_non_positive_sr(self, pcm16_mono_16k):
        with pytest.raises(ValueError, match="sr must be > 0"):
            load(pcm16_mono_16k, sr=0)


class TestBatchLoad:
    def test_results_match_sequential(self, pcm16_mono_16k, pcm16_stereo_44k1):
        """batch_load results should match individual load calls."""
        paths = [pcm16_mono_16k, pcm16_stereo_44k1]
        results = batch_load(paths)
        assert len(results) == 2

        for i, path in enumerate(paths):
            expected_audio, expected_sr = load(path)
            mx.eval(expected_audio, results[i][0])
            assert results[i][1] == expected_sr
            assert results[i][0].shape == expected_audio.shape
            max_diff = mx.max(mx.abs(results[i][0] - expected_audio)).item()
            assert max_diff < 1e-5

    def test_correct_count(self, pcm16_mono_16k):
        paths = [pcm16_mono_16k] * 3
        results = batch_load(paths)
        assert len(results) == 3
