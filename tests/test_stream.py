"""Tests for mlx_audio_io.stream()."""

import math
import os
import tempfile

import mlx.core as mx
import pytest
from mlx_audio_io import load, save, stream

pytestmark = pytest.mark.linux_mvp


def _make_test_wav(sr=16000, duration=1.0, channels=1):
    """Create a temporary WAV file with a sine wave and return its path."""
    frames = int(sr * duration)
    t = mx.arange(frames) / sr
    sine = mx.sin(2.0 * math.pi * 440.0 * t)
    if channels == 1:
        audio = mx.reshape(sine, [frames, 1])
    else:
        audio = mx.stack([sine] * channels, axis=1)
    mx.eval(audio)

    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    path = f.name
    f.close()
    save(path, audio, sr)
    return path


class TestStreamConcatenated:
    def test_concatenated_matches_load(self):
        """Stream all chunks, concatenate, compare to load()."""
        path = _make_test_wav(sr=16000, duration=1.0)
        try:
            full, full_sr = load(path)
            mx.eval(full)

            chunks = []
            for chunk, sr in stream(path, chunk_frames=5000):
                mx.eval(chunk)
                chunks.append(chunk)

            concatenated = mx.concatenate(chunks, axis=0)
            mx.eval(concatenated)

            assert concatenated.shape == full.shape
            max_diff = mx.max(mx.abs(concatenated - full)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)


class TestStreamChunkSizes:
    def test_chunk_sizes(self):
        """16000 frames / 5000 = chunks of [5000, 5000, 5000, 1000]."""
        path = _make_test_wav(sr=16000, duration=1.0)  # 16000 frames
        try:
            sizes = []
            for chunk, sr in stream(path, chunk_frames=5000):
                mx.eval(chunk)
                sizes.append(chunk.shape[0])

            assert sizes == [5000, 5000, 5000, 1000]
        finally:
            os.unlink(path)

    def test_chunk_duration(self):
        """0.5s at 16kHz = 8000 frames per chunk."""
        path = _make_test_wav(sr=16000, duration=1.0)
        try:
            sizes = []
            for chunk, sr in stream(path, chunk_duration=0.5):
                mx.eval(chunk)
                sizes.append(chunk.shape[0])

            assert sizes == [8000, 8000]
        finally:
            os.unlink(path)


class TestStreamOptions:
    def test_mono(self):
        path = _make_test_wav(sr=16000, duration=0.5, channels=2)
        try:
            for chunk, sr in stream(path, chunk_frames=4000, mono=True):
                mx.eval(chunk)
                assert chunk.shape[1] == 1
        finally:
            os.unlink(path)

    def test_resample(self):
        path = _make_test_wav(sr=44100, duration=0.5)
        try:
            chunks = []
            for chunk, sr in stream(path, chunk_frames=8000, sr=16000):
                mx.eval(chunk)
                chunks.append(chunk)
                assert sr == 16000

            total_frames = sum(c.shape[0] for c in chunks)
            # 0.5s at 16kHz = 8000 frames
            assert total_frames == 8000
        finally:
            os.unlink(path)

    def test_float16(self):
        path = _make_test_wav(sr=16000, duration=0.5)
        try:
            for chunk, sr in stream(path, chunk_frames=4000, dtype="float16"):
                mx.eval(chunk)
                assert chunk.dtype == mx.float16
        finally:
            os.unlink(path)

    def test_stream_m4a(self):
        wav_path = _make_test_wav(sr=44100, duration=1.0)
        m4a = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
        m4a_path = m4a.name
        m4a.close()
        try:
            audio, sr = load(wav_path)
            save(m4a_path, audio, sr)

            chunks = []
            for chunk, chunk_sr in stream(m4a_path, chunk_frames=8000):
                mx.eval(chunk)
                chunks.append(chunk)
                assert chunk_sr == 44100

            total_frames = sum(c.shape[0] for c in chunks)
            assert abs(total_frames - 44100) < 2048
        finally:
            os.unlink(wav_path)
            os.unlink(m4a_path)

    def test_stream_flac_resample_mono(self):
        wav_path = _make_test_wav(sr=48000, duration=1.0, channels=2)
        flac = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        flac_path = flac.name
        flac.close()
        try:
            audio, sr = load(wav_path)
            save(flac_path, audio, sr)

            chunks = []
            for chunk, chunk_sr in stream(flac_path, chunk_frames=4000, sr=16000, mono=True):
                mx.eval(chunk)
                chunks.append(chunk)
                assert chunk_sr == 16000
                assert chunk.shape[1] == 1

            total_frames = sum(c.shape[0] for c in chunks)
            assert abs(total_frames - 16000) < 64
        finally:
            os.unlink(wav_path)
            os.unlink(flac_path)


class TestStreamOffsetDuration:
    def test_stream_window_matches_load(self):
        path = _make_test_wav(sr=16000, duration=1.0)
        try:
            expected, expected_sr = load(path, offset=0.25, duration=0.5)
            mx.eval(expected)

            chunks = []
            for chunk, sr in stream(path, chunk_frames=3000, offset=0.25, duration=0.5):
                mx.eval(chunk)
                assert sr == expected_sr
                chunks.append(chunk)

            concatenated = mx.concatenate(chunks, axis=0)
            mx.eval(concatenated)
            assert concatenated.shape == expected.shape
            max_diff = mx.max(mx.abs(concatenated - expected)).item()
            assert max_diff < 1e-5
        finally:
            os.unlink(path)

    def test_chunk_duration_with_window(self):
        path = _make_test_wav(sr=16000, duration=1.0)
        try:
            sizes = []
            for chunk, sr in stream(path, chunk_duration=0.2, offset=0.2, duration=0.5):
                mx.eval(chunk)
                sizes.append(chunk.shape[0])
            assert sizes == [3200, 3200, 1600]
        finally:
            os.unlink(path)


class TestStreamErrors:
    def test_both_args_raises(self):
        path = _make_test_wav(sr=16000, duration=0.5)
        try:
            with pytest.raises(ValueError, match="Exactly one"):
                stream(path, chunk_frames=1000, chunk_duration=0.5)
        finally:
            os.unlink(path)

    def test_neither_arg_raises(self):
        path = _make_test_wav(sr=16000, duration=0.5)
        try:
            with pytest.raises(ValueError, match="Exactly one"):
                stream(path)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            stream("/nonexistent/file.wav", chunk_frames=1000)


class TestStreamProperties:
    def test_reader_properties(self):
        path = _make_test_wav(sr=16000, duration=0.5, channels=2)
        try:
            reader = stream(path, chunk_frames=4000)
            assert reader.sample_rate == 16000
            assert reader.channels == 2
            assert reader.chunk_frames == 4000
            assert not reader.at_eof()
        finally:
            os.unlink(path)

    def test_stereo_channels_preserved(self):
        path = _make_test_wav(sr=16000, duration=0.5, channels=2)
        try:
            for chunk, sr in stream(path, chunk_frames=4000):
                mx.eval(chunk)
                assert chunk.shape[1] == 2
        finally:
            os.unlink(path)


class TestStreamProgress:
    def test_frames_read_increments(self):
        """frames_read should increment after each read_chunk."""
        path = _make_test_wav(sr=16000, duration=1.0)
        try:
            reader = stream(path, chunk_frames=5000)
            assert reader.frames_read == 0

            reader.read_chunk()
            assert reader.frames_read == 5000

            reader.read_chunk()
            assert reader.frames_read == 10000
        finally:
            os.unlink(path)

    def test_frames_read_total(self):
        """Total frames_read should match file length."""
        path = _make_test_wav(sr=16000, duration=1.0)
        try:
            reader = stream(path, chunk_frames=5000)
            for _ in reader:
                pass
            assert reader.frames_read == 16000
        finally:
            os.unlink(path)
