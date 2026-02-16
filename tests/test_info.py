"""Tests for mlx_audio_io.info()."""

import pytest
from mlx_audio_io import info

pytestmark = pytest.mark.linux_mvp


class TestInfo:
    def test_pcm16_mono_16k(self, pcm16_mono_16k):
        meta = info(pcm16_mono_16k)
        assert meta.frames == 16000
        assert meta.sample_rate == 16000
        assert meta.channels == 1
        assert abs(meta.duration - 1.0) < 1e-6
        assert meta.subtype == "pcm16"
        assert meta.container == "wav"

    def test_pcm16_stereo_44k1(self, pcm16_stereo_44k1):
        meta = info(pcm16_stereo_44k1)
        assert meta.frames == 44100
        assert meta.sample_rate == 44100
        assert meta.channels == 2
        assert abs(meta.duration - 1.0) < 1e-6
        assert meta.subtype == "pcm16"
        assert meta.container == "wav"

    def test_float32_stereo_48k(self, float32_stereo_48k):
        meta = info(float32_stereo_48k)
        assert meta.frames == 48000
        assert meta.sample_rate == 48000
        assert meta.channels == 2
        assert abs(meta.duration - 1.0) < 1e-6
        assert meta.subtype == "float32"
        assert meta.container == "wav"

    def test_aiff_stereo(self, pcm16_stereo_44k1_aiff):
        meta = info(pcm16_stereo_44k1_aiff)
        assert meta.frames == 44100
        assert meta.sample_rate == 44100
        assert meta.channels == 2
        assert abs(meta.duration - 1.0) < 1e-6
        assert meta.subtype == "pcm16"
        assert meta.container == "aiff"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            info("/nonexistent/path/to/file.wav")

    def test_repr(self, pcm16_mono_16k):
        meta = info(pcm16_mono_16k)
        r = repr(meta)
        assert "AudioInfo" in r
        assert "16000" in r
