"""Tests for mlx_audio_io.save()."""

import math
import os
from pathlib import Path
import tempfile

import mlx.core as mx
import pytest
from mlx_audio_io import info, load, save

pytestmark = pytest.mark.linux_mvp


class TestSaveBasic:
    def test_roundtrip_sine(self):
        """Save a sine wave, load it back, check values are close."""
        sr = 16000
        frames = sr  # 1 second
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)

    def test_roundtrip_stereo(self):
        sr = 44100
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.stack([sine, sine], axis=1)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 2)
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)

    def test_save_accepts_pathlike(self):
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pathlike.wav"
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)


class TestSaveChannelsFirst:
    def test_channels_first_roundtrip(self):
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.stack([sine, sine], axis=0)  # [2, frames]
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, layout="channels_first")
            loaded, loaded_sr = load(path, layout="channels_first")
            mx.eval(loaded)
            assert loaded.shape == (2, frames)
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)


class TestSaveAutoLayout:
    def test_auto_channels_first_stereo(self):
        """Channels-first (2, frames) should be automatically detected with layout='auto'."""
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        sine1 = mx.sin(2.0 * math.pi * 440.0 * t)
        sine2 = mx.sin(2.0 * math.pi * 880.0 * t)
        audio = mx.stack([sine1, sine2], axis=0)  # [2, frames]
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            # Default is layout="auto"
            save(path, audio, sr)
            meta = info(path)
            assert meta.channels == 2
            assert meta.frames == frames

            loaded, loaded_sr = load(path, layout="channels_first")
            mx.eval(loaded)
            assert loaded.shape == (2, frames)
            assert loaded_sr == sr
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)

    def test_auto_channels_first_mono(self):
        """Channels-first (1, frames) should be automatically detected with layout='auto'."""
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        audio = mx.reshape(mx.sin(2.0 * math.pi * 440.0 * t), [1, frames])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            meta = info(path)
            assert meta.channels == 1
            assert meta.frames == frames

            loaded, loaded_sr = load(path, layout="channels_first")
            mx.eval(loaded)
            assert loaded.shape == (1, frames)
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)

    def test_auto_channels_last(self):
        """Channels-last (frames, 2) should be automatically detected with layout='auto'."""
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        sine1 = mx.sin(2.0 * math.pi * 440.0 * t)
        sine2 = mx.sin(2.0 * math.pi * 880.0 * t)
        audio = mx.stack([sine1, sine2], axis=1)  # [frames, 2]
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            meta = info(path)
            assert meta.channels == 2
            assert meta.frames == frames

            loaded, loaded_sr = load(path, layout="channels_last")
            mx.eval(loaded)
            assert loaded.shape == (frames, 2)
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)

    def test_auto_1d(self):
        """1D audio (frames,) should be saved as mono."""
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        audio = mx.sin(2.0 * math.pi * 440.0 * t)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            meta = info(path)
            assert meta.channels == 1
            assert meta.frames == frames

            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded.shape == (frames, 1)
            max_diff = mx.max(mx.abs(loaded[:, 0] - audio)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)


class TestSaveClip:
    def test_clip_true(self):
        sr = 16000
        audio = mx.array([[2.0], [-2.0], [0.5]], dtype=mx.float32)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, clip=True)
            loaded, _ = load(path)
            mx.eval(loaded)
            assert loaded[0, 0].item() == pytest.approx(1.0, abs=1e-5)
            assert loaded[1, 0].item() == pytest.approx(-1.0, abs=1e-5)
            assert loaded[2, 0].item() == pytest.approx(0.5, abs=1e-5)
        finally:
            os.unlink(path)

    def test_clip_false(self):
        sr = 16000
        audio = mx.array([[2.0], [-2.0]], dtype=mx.float32)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, clip=False)
            loaded, _ = load(path)
            mx.eval(loaded)
            assert loaded[0, 0].item() == pytest.approx(2.0, abs=1e-5)
            assert loaded[1, 0].item() == pytest.approx(-2.0, abs=1e-5)
        finally:
            os.unlink(path)


class TestSavePcm16:
    def test_roundtrip_pcm16(self):
        """Save pcm16, load back, verify shape/sr and quantization error."""
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="pcm16")
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-4  # PCM16 quantization
        finally:
            os.unlink(path)

    def test_pcm16_metadata(self):
        """Save pcm16, verify info() reports correct subtype."""
        sr = 44100
        frames = 1000
        audio = mx.zeros([frames, 2])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="pcm16")
            meta = info(path)
            assert meta.subtype == "pcm16"
            assert meta.sample_rate == sr
            assert meta.channels == 2
            assert meta.frames == frames
        finally:
            os.unlink(path)


class TestSavePcm24:
    def test_save_pcm24_wav(self):
        """Save pcm24 WAV, load back, verify subtype and round-trip."""
        sr = 44100
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="pcm24")
            meta = info(path)
            assert meta.subtype == "pcm24"
            assert meta.sample_rate == sr
            assert meta.channels == 1
            assert meta.frames == frames
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-6  # 24-bit quantization
        finally:
            os.unlink(path)

    def test_save_pcm24_aiff(self):
        """Save pcm24 AIFF, load back, verify subtype and round-trip."""
        sr = 44100
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="pcm24")
            meta = info(path)
            assert meta.subtype == "pcm24"
            assert meta.sample_rate == sr
            assert meta.channels == 1
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-6  # 24-bit quantization
        finally:
            os.unlink(path)


class TestSaveFloat16:
    def test_roundtrip_float16(self):
        """Save float16 audio, load back, verify shape and values."""
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        audio = audio.astype(mx.float16)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)
            # float16 -> float32 conversion loses some precision
            max_diff = mx.max(mx.abs(loaded - audio.astype(mx.float32))).item()
            assert max_diff < 1e-3
        finally:
            os.unlink(path)

    def test_float16_stereo_channels_first(self):
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.stack([sine, sine], axis=0).astype(mx.float16)  # [2, frames]
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, layout="channels_first")
            loaded, loaded_sr = load(path, layout="channels_first")
            mx.eval(loaded)
            assert loaded.shape == (2, frames)
        finally:
            os.unlink(path)

    def test_float16_pcm16_encoding(self):
        sr = 16000
        frames = 1000
        audio = mx.zeros([frames, 1]).astype(mx.float16)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="pcm16")
            meta = info(path)
            assert meta.subtype == "pcm16"
        finally:
            os.unlink(path)

    def test_int32_rejected(self):
        audio = mx.zeros([100, 1]).astype(mx.int32)
        mx.eval(audio)
        with pytest.raises(ValueError):
            save("/tmp/test.wav", audio, 16000)


class TestSave1D:
    def test_roundtrip_1d_sine(self):
        """1D array should be treated as mono."""
        sr = 16000
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        mx.eval(sine)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, sine, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)
            max_diff = mx.max(mx.abs(loaded[:, 0] - sine)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)


class TestSaveErrors:
    def test_wrong_ndim(self):
        audio = mx.zeros([2, 3, 4], dtype=mx.float32)
        mx.eval(audio)
        with pytest.raises(ValueError):
            save("/tmp/test.wav", audio, 16000)

    def test_unsupported_encoding(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError):
            save("/tmp/test.wav", audio, 16000, encoding="pcm8")

    def test_invalid_layout(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError):
            save("/tmp/test.wav", audio, 16000, layout="invalid")

    def test_saved_file_metadata(self):
        sr = 44100
        frames = 1000
        audio = mx.zeros([frames, 2])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            meta = info(path)
            assert meta.sample_rate == sr
            assert meta.channels == 2
            assert meta.frames == frames
            assert meta.subtype == "float32"
        finally:
            os.unlink(path)

    def test_unsupported_extension(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="Unsupported output format"):
            save("/tmp/test.ogg", audio, 16000)

    def test_non_positive_sr(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="sr must be > 0"):
            save("/tmp/test.wav", audio, 0)

    def test_wav_bitrate_rejected(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="bitrate is only supported"):
            save("/tmp/test.wav", audio, 16000, bitrate="128k")

    def test_flac_bitrate_rejected(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="bitrate is not supported"):
            save("/tmp/test.flac", audio, 16000, bitrate="128k")


class TestSaveM4A:
    def test_roundtrip_m4a(self):
        """Save M4A, load back, verify shape/sr (lossy — larger tolerance)."""
        sr = 44100
        frames = sr * 2  # 2 seconds
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            # AAC encoder adds priming/remainder frames — allow tolerance
            assert abs(loaded.shape[0] - frames) < 2048
            assert loaded.shape[1] == 1
        finally:
            os.unlink(path)

    def test_m4a_metadata(self):
        sr = 44100
        frames = sr
        audio = mx.zeros([frames, 2])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            meta = info(path)
            assert meta.container == "m4a"
            assert meta.subtype == "aac"
            assert meta.sample_rate == sr
            assert meta.channels == 2
        finally:
            os.unlink(path)

    def test_m4a_bitrate(self):
        sr = 44100
        frames = sr
        audio = mx.zeros([frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, bitrate="128k")
            meta = info(path)
            assert meta.container == "m4a"
            assert meta.subtype == "aac"
        finally:
            os.unlink(path)

    def test_m4a_float16_input(self):
        sr = 44100
        frames = sr
        audio = mx.zeros([frames, 1]).astype(mx.float16)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            meta = info(path)
            assert meta.container == "m4a"
        finally:
            os.unlink(path)

    def test_m4a_stereo(self):
        sr = 44100
        frames = sr * 2
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.stack([sine, sine], axis=1)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape[1] == 2
        finally:
            os.unlink(path)

    def test_m4a_invalid_bitrate(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="Invalid bitrate"):
            save("/tmp/test.m4a", audio, 44100, bitrate="pcm16")

    def test_m4a_invalid_encoding(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="Unsupported encoding"):
            save("/tmp/test.m4a", audio, 44100, encoding="pcm24")

    def test_m4a_alac_rejects_bitrate(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="bitrate is not supported when encoding='alac'"):
            save("/tmp/test.m4a", audio, 44100, encoding="alac", bitrate="128k")


class TestSaveMP3:
    def test_roundtrip_mp3(self):
        sr = 44100
        frames = sr * 2
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape[1] == 1
            # MP3 encoder delay/padding can shift frame count.
            assert abs(loaded.shape[0] - frames) < 4096
        finally:
            os.unlink(path)

    def test_mp3_metadata(self):
        sr = 44100
        frames = sr
        audio = mx.zeros([frames, 2])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, bitrate="192k")
            meta = info(path)
            assert meta.container == "mp3"
            assert meta.subtype == "mp3"
            assert meta.sample_rate == sr
            assert meta.channels == 2
        finally:
            os.unlink(path)

    def test_mp3_invalid_bitrate(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="Invalid bitrate"):
            save("/tmp/test.mp3", audio, 44100, bitrate="pcm16")

    def test_mp3_invalid_encoding(self):
        audio = mx.zeros([100, 1])
        mx.eval(audio)
        with pytest.raises(ValueError, match="Unsupported encoding"):
            save("/tmp/test.mp3", audio, 44100, encoding="alac")


class TestSaveStrided:
    """save() must honour strides: data<>() alone reads raw memory linearly."""

    @staticmethod
    def _interleaved(frames=4800, channels=2):
        audio = mx.random.uniform(-0.5, 0.5, (frames, channels), key=mx.random.key(7))
        mx.eval(audio)
        return audio

    @staticmethod
    def _roundtrip(audio, layout, suffix=".wav"):
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            path = f.name
        try:
            save(path, audio, 48000, layout=layout)
            loaded, _ = load(path)
            mx.eval(loaded)
            return loaded
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("suffix", [".wav", ".flac", ".caf"])
    def test_channels_first_transposed_view(self, suffix):
        """A [channels, frames] .T view of interleaved memory is column-major."""
        expected = self._interleaved()
        view = expected.T
        mx.eval(view)
        loaded = self._roundtrip(view, "channels_first", suffix)
        assert mx.max(mx.abs(loaded - expected)).item() < 1e-4

    @pytest.mark.parametrize("suffix", [".wav", ".flac", ".caf"])
    def test_channels_last_transposed_view(self, suffix):
        """A [frames, channels] .T view of planar memory is column-major."""
        expected = self._interleaved()
        planar = mx.contiguous(expected.T)
        view = planar.T
        mx.eval(view)
        loaded = self._roundtrip(view, "channels_last", suffix)
        assert mx.max(mx.abs(loaded - expected)).item() < 1e-4

    def test_auto_layout_transposed_view(self):
        expected = self._interleaved()
        loaded = self._roundtrip(expected.T, "auto")
        assert loaded.shape == expected.shape
        assert mx.max(mx.abs(loaded - expected)).item() < 1e-6

    def test_strided_slice(self):
        """A column slice is neither row- nor column-contiguous."""
        wide = self._interleaved(channels=4)
        view = wide[:, 1:3]
        mx.eval(view)
        loaded = self._roundtrip(view, "channels_last")
        assert mx.max(mx.abs(loaded - view)).item() < 1e-6

    def test_stem_slice_of_batched_output(self):
        """A stem taken from a [stems, channels, frames] batch is an offset view."""
        stems = mx.random.uniform(-0.5, 0.5, (4, 2, 4800), key=mx.random.key(3))
        mx.eval(stems)
        for index in range(stems.shape[0]):
            loaded = self._roundtrip(stems[index], "channels_first")
            assert mx.max(mx.abs(loaded - stems[index].T)).item() < 1e-6

    def test_mp3_transposed_view_matches_contiguous(self):
        expected = self._interleaved(frames=48000)
        contiguous = self._roundtrip(expected, "channels_last", ".mp3")
        strided = self._roundtrip(expected.T, "channels_first", ".mp3")
        assert mx.max(mx.abs(contiguous - strided)).item() < 1e-6


@pytest.mark.apple_only
class TestSaveFLACEncoding:
    """AudioToolbox FLAC: bit depth, pcm24, and streams shorter than a packet."""

    @staticmethod
    def _roundtrip(frames, encoding="pcm16"):
        audio = mx.random.uniform(-0.5, 0.5, (2, frames), key=mx.random.key(frames))
        mx.eval(audio)
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            path = f.name
        try:
            save(path, audio, 44100, layout="channels_first", encoding=encoding)
            loaded, _ = load(path, layout="channels_first")
            mx.eval(loaded)
            return audio, loaded
        finally:
            os.unlink(path)

    @pytest.mark.parametrize("frames", [192, 1000, 4095, 4607, 4608, 9001])
    def test_short_streams_round_trip(self, frames):
        """Under one 4608-frame packet the encoder used to write a bare header."""
        audio, loaded = self._roundtrip(frames)
        assert loaded.shape == audio.shape
        assert mx.max(mx.abs(loaded - audio)).item() < 1e-4

    def test_shorter_than_smallest_block_is_rejected(self):
        with pytest.raises(ValueError, match="at least 192 frames"):
            self._roundtrip(191)

    def test_pcm16_writes_16_bit_samples(self):
        audio, loaded = self._roundtrip(44100, "pcm16")
        error = mx.max(mx.abs(loaded - audio)).item()
        assert 1e-6 < error < 1e-4

    @pytest.mark.parametrize("encoding", ["pcm24", "float32"])
    def test_pcm24_and_float32_write_24_bit_samples(self, encoding):
        audio, loaded = self._roundtrip(44100, encoding)
        assert mx.max(mx.abs(loaded - audio)).item() < 1e-6


class TestSaveFLACCompression:
    @staticmethod
    def _write(path, audio, compression):
        save(path, audio, 44100, layout="channels_first", encoding="pcm16",
             flac_compression=compression)
        return os.path.getsize(path)

    def test_fast_is_lossless_and_not_smaller(self, tmp_path):
        t = mx.arange(44100 * 5) / 44100
        tone = 0.3 * mx.sin(2 * math.pi * 220 * t) + 0.05 * mx.random.uniform(
            -1, 1, t.shape, key=mx.random.key(9))
        audio = mx.stack([tone, tone * 0.5])
        mx.eval(audio)
        default_size = self._write(str(tmp_path / "default.flac"), audio, "default")
        fast_size = self._write(str(tmp_path / "fast.flac"), audio, "fast")
        default_audio, _ = load(str(tmp_path / "default.flac"), layout="channels_first")
        fast_audio, _ = load(str(tmp_path / "fast.flac"), layout="channels_first")
        assert mx.array_equal(default_audio, fast_audio).item()
        assert fast_size >= default_size

    def test_invalid_value_is_rejected(self, tmp_path):
        audio = mx.zeros((2, 4800))
        with pytest.raises(ValueError, match="flac_compression"):
            save(str(tmp_path / "x.flac"), audio, 48000, layout="channels_first",
                 flac_compression="best")

    def test_other_formats_ignore_it(self, tmp_path):
        audio = mx.zeros((2, 4800))
        save(str(tmp_path / "x.wav"), audio, 48000, layout="channels_first", flac_compression="fast")
        assert os.path.getsize(tmp_path / "x.wav") > 0


@pytest.mark.apple_only
class TestSaveMP3Gapless:
    """The LAME tag must record encoder delay and padding so decoders trim them."""

    @pytest.mark.parametrize("frames", [100, 1000, 9000, 88200])
    @pytest.mark.parametrize("bitrate", ["auto", "320k"])
    def test_length_and_alignment_round_trip(self, frames, bitrate):
        clicks = [frames // 4, frames // 2, 3 * frames // 4]
        audio = mx.zeros((frames,)).at[mx.array(clicks)].add(0.8)
        audio = mx.stack([audio, audio])
        mx.eval(audio)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        try:
            save(path, audio, 44100, layout="channels_first", bitrate=bitrate)
            loaded, _ = load(path, layout="channels_first")
            mx.eval(loaded)
        finally:
            os.unlink(path)
        assert loaded.shape == audio.shape
        window = max(frames // 8, 1)
        for click in clicks:
            segment = mx.abs(loaded[0, click - window : click + window])
            assert int(mx.argmax(segment).item()) + click - window == click


class TestSaveNumpy:
    def test_roundtrip_numpy(self):
        """Save from numpy array, load back, check values."""
        np = pytest.importorskip("numpy")
        sr = 16000
        frames = sr
        t = np.arange(frames, dtype=np.float32) / sr
        sine = np.sin(2.0 * np.pi * 440.0 * t).reshape(frames, 1).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, sine, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)
            max_diff = mx.max(mx.abs(loaded - mx.array(sine))).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)

    def test_numpy_1d(self):
        """1D numpy array should work as mono."""
        np = pytest.importorskip("numpy")
        sr = 16000
        sine = np.sin(np.linspace(0, 2 * np.pi * 440, sr, dtype=np.float32))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name

        try:
            save(path, sine, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (sr, 1)
        finally:
            os.unlink(path)


class TestSaveFLAC:
    def test_roundtrip(self):
        """Save FLAC, load back, verify lossless roundtrip."""
        sr = 44100
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape == (frames, 1)
            # FLAC is lossless but float->int->float adds quantization
            max_diff = mx.max(mx.abs(loaded - audio)).item()
            assert max_diff < 1e-4
        finally:
            os.unlink(path)

    def test_metadata(self):
        sr = 44100
        frames = sr  # 1 second — FLAC encoder needs enough frames for a block
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.stack([sine, sine], axis=1)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            meta = info(path)
            assert meta.container == "flac"
            assert meta.subtype == "flac"
            assert meta.sample_rate == sr
            assert meta.channels == 2
        finally:
            os.unlink(path)

    def test_stereo(self):
        sr = 44100
        frames = sr
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.stack([sine, sine], axis=1)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr)
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            assert loaded.shape[1] == 2
        finally:
            os.unlink(path)


class TestSaveALAC:
    def test_roundtrip(self):
        """Save ALAC, load back, verify lossless roundtrip."""
        sr = 44100
        frames = sr * 2  # 2 seconds
        t = mx.arange(frames) / sr
        sine = mx.sin(2.0 * math.pi * 440.0 * t)
        audio = mx.reshape(sine, [frames, 1])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="alac")
            loaded, loaded_sr = load(path)
            mx.eval(loaded)
            assert loaded_sr == sr
            # M4A container may add priming frames
            assert abs(loaded.shape[0] - frames) < 2048
            assert loaded.shape[1] == 1
        finally:
            os.unlink(path)

    def test_metadata(self):
        sr = 44100
        frames = sr
        audio = mx.zeros([frames, 2])
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="alac")
            meta = info(path)
            assert meta.container == "m4a"
            assert meta.subtype == "alac"
            assert meta.sample_rate == sr
            assert meta.channels == 2
        finally:
            os.unlink(path)


class TestSaveAIFFCAF:
    def test_aiff_float32_metadata(self):
        sr = 32000
        frames = sr
        audio = mx.zeros([frames, 2], dtype=mx.float32)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="float32")
            meta = info(path)
            assert meta.container == "aiff"
            assert meta.subtype == "float32"
            assert meta.sample_rate == sr
            assert meta.channels == 2
        finally:
            os.unlink(path)

    def test_caf_pcm16_metadata(self):
        sr = 22050
        frames = sr
        audio = mx.zeros([frames, 1], dtype=mx.float32)
        mx.eval(audio)

        with tempfile.NamedTemporaryFile(suffix=".caf", delete=False) as f:
            path = f.name

        try:
            save(path, audio, sr, encoding="pcm16")
            meta = info(path)
            assert meta.container == "caf"
            assert meta.subtype == "pcm16"
            assert meta.sample_rate == sr
            assert meta.channels == 1
        finally:
            os.unlink(path)


class TestSaveWavQuantizationIsExact:
    """The fused strided WAV writer quantizes exactly as documented.

    Every sample is checked against a Python model of the rule: x <= -1 maps
    to negative full scale, x >= 1 to positive full scale, everything else to
    round-half-even(float32(x * full_scale)). Exactly -1 and exact .5 ties are
    the cases a vectorized path gets wrong first.
    """

    @staticmethod
    def _signal(n):
        import random
        import struct

        rng = random.Random(7)
        edges = [-2.0, -1.0000001, -1.0, -0.99999, -0.5, -1e-9, 0.0, 1e-9, 0.5, 0.99999, 1.0, 1.0000001, 2.0]
        ties = [(k + 0.5) / 32767.0 for k in range(-40, 40)]
        values = edges + ties + [rng.uniform(-1.2, 1.2) for _ in range(n)]
        # Round-trip through float32 so the model sees exactly what MLX stores.
        return [struct.unpack("f", struct.pack("f", v))[0] for v in values]

    @staticmethod
    def _expected(x, full):
        import struct

        if x <= -1.0:
            return -full - 1
        if x >= 1.0:
            return full
        product = struct.unpack("f", struct.pack("f", x * float(full)))[0]
        return round(product)

    @staticmethod
    def _read_ints(path, width):
        import wave

        with wave.open(str(path), "rb") as w:
            raw = w.readframes(w.getnframes())
        if width == 2:
            return [int.from_bytes(raw[i:i + 2], "little", signed=True) for i in range(0, len(raw), 2)]
        return [int.from_bytes(raw[i:i + 3], "little", signed=True) for i in range(0, len(raw), 3)]

    @pytest.mark.parametrize("encoding,width,full", [("pcm16", 2, 32767), ("pcm24", 3, 8388607)])
    @pytest.mark.parametrize("channels", [1, 2, 3])
    @pytest.mark.parametrize("view", ["channels_first", "channels_last", "transposed_view", "column_slice"])
    @pytest.mark.parametrize("clip", [True, False])
    def test_every_sample_matches_the_rule(self, tmp_path, encoding, width, full, channels, view, clip):
        base = self._signal(301)
        planar = mx.array([[v * (1.0 - 0.05 * c) for v in base] for c in range(channels)], dtype=mx.float32)
        if view == "channels_first":
            arr, layout = planar, "channels_first"
        elif view == "channels_last":
            arr, layout = mx.contiguous(planar.T), "channels_last"
        elif view == "transposed_view":
            arr, layout = planar.T, "channels_last"
        else:
            arr, layout = planar[:, 3:-5], "channels_first"
        mx.eval(arr)
        path = tmp_path / "q.wav"
        save(str(path), arr, 44100, layout=layout, encoding=encoding, clip=clip)

        frames = arr if layout == "channels_last" else arr.T
        expected = [self._expected(x, full) for x in frames.reshape(-1).tolist()]
        assert self._read_ints(path, width) == expected
