"""Test fixtures: generate WAV test assets using stdlib only."""

import math
import os
import shutil
import struct
import subprocess
import sys

import pytest

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


def pytest_collection_modifyitems(config, items):
    """Apply platform-specific skips for marker-tagged tests."""
    is_macos = sys.platform == "darwin"

    for item in items:
        if "apple_only" in item.keywords and not is_macos:
            item.add_marker(
                pytest.mark.skip(reason="apple_only test requires macOS AudioToolbox backend")
            )


def _write_wav_pcm16(path, sample_rate, channels, duration_s, freq=440.0):
    """Write a PCM16 WAV file with a sine tone."""
    num_frames = int(sample_rate * duration_s)
    samples = []
    for i in range(num_frames):
        val = math.sin(2.0 * math.pi * freq * i / sample_rate)
        sample = int(val * 32767)
        sample = max(-32768, min(32767, sample))
        for _ in range(channels):
            samples.append(sample)

    data = struct.pack(f"<{len(samples)}h", *samples)
    bits_per_sample = 16
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data_size = len(data)

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # format tag: PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)


def _write_wav_float32(path, sample_rate, channels, duration_s, freq=440.0):
    """Write a float32 WAV file (IEEE_FLOAT format tag 3) with fact chunk."""
    num_frames = int(sample_rate * duration_s)
    samples = []
    for i in range(num_frames):
        val = math.sin(2.0 * math.pi * freq * i / sample_rate)
        for _ in range(channels):
            samples.append(val)

    data = struct.pack(f"<{len(samples)}f", *samples)
    bits_per_sample = 32
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data_size = len(data)

    # fmt chunk: 18 bytes (extensible with cbSize=0)
    fmt_chunk_size = 18
    # fact chunk: 12 bytes total (4 tag + 4 size + 4 data)
    fact_data = struct.pack("<I", num_frames)
    fact_chunk_size = 4

    riff_size = (
        4  # 'WAVE'
        + 8 + fmt_chunk_size   # fmt chunk
        + 8 + fact_chunk_size  # fact chunk
        + 8 + data_size        # data chunk
    )

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        # fmt chunk (format tag 3 = IEEE_FLOAT)
        f.write(b"fmt ")
        f.write(struct.pack("<I", fmt_chunk_size))
        f.write(struct.pack("<H", 3))   # format tag: IEEE_FLOAT
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(struct.pack("<H", 0))   # cbSize
        # fact chunk
        f.write(b"fact")
        f.write(struct.pack("<I", fact_chunk_size))
        f.write(fact_data)
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)


def _write_wav_float64(path, sample_rate, channels, duration_s, freq=440.0):
    """Write a float64 WAV file (IEEE_FLOAT format tag 3) with fact chunk."""
    num_frames = int(sample_rate * duration_s)
    samples = []
    for i in range(num_frames):
        val = math.sin(2.0 * math.pi * freq * i / sample_rate)
        for _ in range(channels):
            samples.append(val)

    data = struct.pack(f"<{len(samples)}d", *samples)
    bits_per_sample = 64
    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    data_size = len(data)

    # fmt chunk: 18 bytes (extensible with cbSize=0)
    fmt_chunk_size = 18
    # fact chunk: 12 bytes total (4 tag + 4 size + 4 data)
    fact_data = struct.pack("<I", num_frames)
    fact_chunk_size = 4

    riff_size = (
        4  # 'WAVE'
        + 8 + fmt_chunk_size   # fmt chunk
        + 8 + fact_chunk_size  # fact chunk
        + 8 + data_size        # data chunk
    )

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", riff_size))
        f.write(b"WAVE")
        # fmt chunk (format tag 3 = IEEE_FLOAT)
        f.write(b"fmt ")
        f.write(struct.pack("<I", fmt_chunk_size))
        f.write(struct.pack("<H", 3))   # format tag: IEEE_FLOAT
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(struct.pack("<H", 0))   # cbSize
        # fact chunk
        f.write(b"fact")
        f.write(struct.pack("<I", fact_chunk_size))
        f.write(fact_data)
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data)


def _write_aiff(path, sample_rate, channels, duration_s, freq=440.0):
    """Write a PCM16 AIFF file (big-endian) with a sine tone."""
    num_frames = int(sample_rate * duration_s)
    samples = []
    for i in range(num_frames):
        val = math.sin(2.0 * math.pi * freq * i / sample_rate)
        sample = int(val * 32767)
        sample = max(-32768, min(32767, sample))
        for _ in range(channels):
            samples.append(sample)

    data = struct.pack(f">{len(samples)}h", *samples)

    bits_per_sample = 16
    # 80-bit IEEE 754 extended float for sample rate
    # Simple encoding for common rates
    def _encode_extended(rate):
        """Encode an integer sample rate as 80-bit IEEE 754 extended."""
        if rate == 0:
            return b"\x00" * 10
        exp = 16383 + 63  # bias + 63 for the integer position
        mantissa = rate
        # Normalize: shift left until bit 63 is set
        while mantissa < (1 << 63):
            mantissa <<= 1
            exp -= 1
        return struct.pack(">HQ", exp, mantissa)

    sr_bytes = _encode_extended(sample_rate)
    data_size = len(data)

    # COMM chunk: 26 bytes payload
    comm_payload = struct.pack(">hIh", channels, num_frames, bits_per_sample) + sr_bytes
    # SSND chunk: 8-byte header (offset + blockSize) + data
    ssnd_payload_size = 8 + data_size

    form_size = (
        4                         # 'AIFF'
        + 8 + len(comm_payload)   # COMM chunk
        + 8 + ssnd_payload_size   # SSND chunk
    )

    with open(path, "wb") as f:
        f.write(b"FORM")
        f.write(struct.pack(">I", form_size))
        f.write(b"AIFF")
        # COMM chunk
        f.write(b"COMM")
        f.write(struct.pack(">I", len(comm_payload)))
        f.write(comm_payload)
        # SSND chunk
        f.write(b"SSND")
        f.write(struct.pack(">I", ssnd_payload_size))
        f.write(struct.pack(">II", 0, 0))  # offset, blockSize
        f.write(data)


@pytest.fixture(scope="session", autouse=True)
def generate_test_assets():
    """Generate test assets before any tests run."""
    os.makedirs(ASSETS_DIR, exist_ok=True)

    _write_wav_pcm16(
        os.path.join(ASSETS_DIR, "pcm16_mono_16k.wav"),
        sample_rate=16000, channels=1, duration_s=1.0,
    )
    _write_wav_pcm16(
        os.path.join(ASSETS_DIR, "pcm16_stereo_44k1.wav"),
        sample_rate=44100, channels=2, duration_s=1.0,
    )
    _write_wav_float32(
        os.path.join(ASSETS_DIR, "float32_stereo_48k.wav"),
        sample_rate=48000, channels=2, duration_s=1.0,
    )
    _write_wav_float64(
        os.path.join(ASSETS_DIR, "float64_stereo_48k.wav"),
        sample_rate=48000, channels=2, duration_s=1.0,
    )
    _write_aiff(
        os.path.join(ASSETS_DIR, "pcm16_stereo_44k1.aiff"),
        sample_rate=44100, channels=2, duration_s=1.0,
    )

    yield


@pytest.fixture
def pcm16_mono_16k():
    return os.path.join(ASSETS_DIR, "pcm16_mono_16k.wav")


@pytest.fixture
def pcm16_stereo_44k1():
    return os.path.join(ASSETS_DIR, "pcm16_stereo_44k1.wav")


@pytest.fixture
def float32_stereo_48k():
    return os.path.join(ASSETS_DIR, "float32_stereo_48k.wav")


@pytest.fixture
def float64_stereo_48k():
    return os.path.join(ASSETS_DIR, "float64_stereo_48k.wav")


@pytest.fixture
def pcm16_stereo_44k1_aiff():
    return os.path.join(ASSETS_DIR, "pcm16_stereo_44k1.aiff")


def _encode_mp3(wav_path, mp3_path):
    """Encode a WAV to MP3 using ffmpeg or lame. Returns True on success."""
    if shutil.which("ffmpeg"):
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-q:a", "2", mp3_path],
            capture_output=True,
        )
        return result.returncode == 0
    if shutil.which("lame"):
        result = subprocess.run(
            ["lame", "--quiet", "-V", "2", wav_path, mp3_path],
            capture_output=True,
        )
        return result.returncode == 0
    return False


@pytest.fixture(scope="session")
def mp3_mono_44k1():
    """MP3 fixture generated from WAV via ffmpeg/lame."""
    wav_path = os.path.join(ASSETS_DIR, "pcm16_mono_44k1_for_mp3.wav")
    mp3_path = os.path.join(ASSETS_DIR, "mono_44k1.mp3")
    if not os.path.exists(mp3_path):
        _write_wav_pcm16(wav_path, sample_rate=44100, channels=1, duration_s=1.0)
        if not _encode_mp3(wav_path, mp3_path):
            pytest.skip("ffmpeg or lame not available for MP3 generation")
    return mp3_path


@pytest.fixture(scope="session")
def mp3_stereo_44k1():
    """Stereo MP3 fixture."""
    wav_path = os.path.join(ASSETS_DIR, "pcm16_stereo_44k1_for_mp3.wav")
    mp3_path = os.path.join(ASSETS_DIR, "stereo_44k1.mp3")
    if not os.path.exists(mp3_path):
        _write_wav_pcm16(wav_path, sample_rate=44100, channels=2, duration_s=1.0)
        if not _encode_mp3(wav_path, mp3_path):
            pytest.skip("ffmpeg or lame not available for MP3 generation")
    return mp3_path
