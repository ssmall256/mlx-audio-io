"""Tests for mlx_audio_io.save()."""

import math
import os
import sys
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
            save("/tmp/test.wav", audio, 16000, encoding="pcm24")

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
