#!/usr/bin/env python3
"""Benchmark mlx-audio-io vs librosa/soundfile/torchaudio.

Run: uv sync --extra benchmark && uv run python tests/benchmark.py
     uv run python tests/benchmark.py /path/to/your/file.wav
"""

import argparse
import math
import os
import platform
import struct
import subprocess
import tempfile
import time
import warnings

import mlx.core as mx
import numpy as np
import psutil

try:
    import torch
    import torchaudio

    HAS_TORCHAUDIO = True
    TORCHAUDIO_IMPORT_ERROR = None
except Exception as e:
    torch = None
    torchaudio = None
    HAS_TORCHAUDIO = False
    TORCHAUDIO_IMPORT_ERROR = f"{type(e).__name__}: {e}"


def get_machine_info():
    """Print machine info for reproducibility."""
    chip = "Unknown"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True,
        )
        chip = result.stdout.strip()
    except Exception:
        pass

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"Machine: {chip}")
    print(f"RAM: {ram_gb:.0f} GB")
    print(f"Platform: {platform.platform()}")
    if platform.system() == "Darwin":
        print(f"macOS: {platform.mac_ver()[0]}")
    print(f"Python: {platform.python_version()}")
    if HAS_TORCHAUDIO:
        print(f"torchaudio: {torchaudio.__version__}")
    else:
        print(f"torchaudio: unavailable ({TORCHAUDIO_IMPORT_ERROR})")
    print()


def generate_bench_wav(path, sr=44100, channels=2, duration_s=10.0):
    """Generate a PCM16 stereo WAV for benchmarking."""
    num_frames = int(sr * duration_s)
    samples = []
    for i in range(num_frames):
        val = math.sin(2.0 * math.pi * 440.0 * i / sr)
        sample = int(val * 32767)
        sample = max(-32768, min(32767, sample))
        for _ in range(channels):
            samples.append(sample)

    data = struct.pack(f"<{len(samples)}h", *samples)
    bits = 16
    block_align = channels * (bits // 8)
    byte_rate = sr * block_align

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)


def bench(fn, warmup=3, runs=20):
    """Run fn with warmup, return median and p90 times in ms."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    p50 = times[len(times) // 2]
    p90_idx = int(len(times) * 0.9)
    p90 = times[p90_idx]
    return p50, p90


def main():
    import librosa
    import soundfile as sf
    from mlx_audio_io import info, load, save

    parser = argparse.ArgumentParser(description="Benchmark mlx-audio-io")
    parser.add_argument("wav", nargs="?", help="Path to a .wav file to benchmark")
    args = parser.parse_args()

    get_machine_info()

    tmpdir = tempfile.mkdtemp()
    generated = False
    cleanup_files = []

    if args.wav:
        wav_path = os.path.abspath(args.wav)
        meta = info(wav_path)
        native_sr = meta.sample_rate
        print(f"Benchmark file: {wav_path}")
        print(f"  {meta.duration:.1f}s, {meta.channels}ch, {meta.subtype} @ {native_sr}Hz")
    else:
        wav_path = os.path.join(tmpdir, "bench.wav")
        generate_bench_wav(wav_path, sr=44100, channels=2, duration_s=10.0)
        native_sr = 44100
        generated = True
        print("Benchmark file: 10s stereo PCM16 @ 44.1kHz (generated)")

    print("=" * 65)

    # 1. Full load (no resample)
    print("\n1. Full load (native SR)")

    def mlx_load_full():
        a, s = load(wav_path)
        mx.eval(a)

    def librosa_load_full():
        a, s = librosa.load(wav_path, sr=None, mono=False)

    p50, p90 = bench(mlx_load_full)
    print(f"   mlx-audio-io:  p50={p50:7.2f}ms  p90={p90:7.2f}ms")
    p50_lib, p90_lib = bench(librosa_load_full)
    print(f"   librosa:          p50={p50_lib:7.2f}ms  p90={p90_lib:7.2f}ms")

    def ta_load_full():
        torchaudio.load(wav_path)

    p50_ta = None
    if HAS_TORCHAUDIO:
        try:
            p50_ta, p90_ta = bench(ta_load_full)
            print(f"   torchaudio:       p50={p50_ta:7.2f}ms  p90={p90_ta:7.2f}ms")
        except Exception as e:
            print(f"   torchaudio:       N/A ({type(e).__name__}: {e})")
    else:
        print(f"   torchaudio:       N/A ({TORCHAUDIO_IMPORT_ERROR})")
    print(f"   speedup vs librosa: {p50_lib / p50:.1f}x", end="")
    if p50_ta is not None:
        print(f"  vs torchaudio: {p50_ta / p50:.1f}x")
    else:
        print()


    # 2. Load with resample
    print("\n2. Load with resample to 16kHz")

    def mlx_load_resample():
        a, s = load(wav_path, sr=16000)
        mx.eval(a)

    def librosa_load_resample():
        a, s = librosa.load(wav_path, sr=16000, mono=False)

    p50, p90 = bench(mlx_load_resample)
    print(f"   mlx-audio-io:  p50={p50:7.2f}ms  p90={p90:7.2f}ms")
    p50_lib, p90_lib = bench(librosa_load_resample)
    print(f"   librosa:          p50={p50_lib:7.2f}ms  p90={p90_lib:7.2f}ms")

    def ta_load_resample():
        waveform, sr_orig = torchaudio.load(wav_path)
        torchaudio.functional.resample(waveform, sr_orig, 16000)

    p50_ta = None
    if HAS_TORCHAUDIO:
        try:
            p50_ta, p90_ta = bench(ta_load_resample)
            print(f"   torchaudio:       p50={p50_ta:7.2f}ms  p90={p90_ta:7.2f}ms")
        except Exception as e:
            print(f"   torchaudio:       N/A ({type(e).__name__}: {e})")
    else:
        print(f"   torchaudio:       N/A ({TORCHAUDIO_IMPORT_ERROR})")
    print(f"   speedup vs librosa: {p50_lib / p50:.1f}x", end="")
    if p50_ta is not None:
        print(f"  vs torchaudio: {p50_ta / p50:.1f}x")
    else:
        print()


    # 3. Partial load (offset + duration)
    print("\n3. Partial load (offset=2s, duration=1s)")

    def mlx_load_partial():
        a, s = load(wav_path, offset=2.0, duration=1.0)
        mx.eval(a)

    def librosa_load_partial():
        a, s = librosa.load(wav_path, sr=None, offset=2.0, duration=1.0, mono=False)

    p50, p90 = bench(mlx_load_partial)
    print(f"   mlx-audio-io:  p50={p50:7.2f}ms  p90={p90:7.2f}ms")
    p50_lib, p90_lib = bench(librosa_load_partial)
    print(f"   librosa:          p50={p50_lib:7.2f}ms  p90={p90_lib:7.2f}ms")

    partial_offset_frames = int(2.0 * native_sr)
    partial_num_frames = int(1.0 * native_sr)

    def ta_load_partial():
        torchaudio.load(wav_path, frame_offset=partial_offset_frames,
                        num_frames=partial_num_frames)

    p50_ta = None
    if HAS_TORCHAUDIO:
        try:
            p50_ta, p90_ta = bench(ta_load_partial)
            print(f"   torchaudio:       p50={p50_ta:7.2f}ms  p90={p90_ta:7.2f}ms")
        except Exception as e:
            print(f"   torchaudio:       N/A ({type(e).__name__}: {e})")
    else:
        print(f"   torchaudio:       N/A ({TORCHAUDIO_IMPORT_ERROR})")

    print(f"   speedup vs librosa: {p50_lib / p50:.1f}x", end="")
    if p50_ta is not None:
        print(f"  vs torchaudio: {p50_ta / p50:.1f}x")
    else:
        print()

    # 4. Save
    print("\n4. Save (float32 WAV)")
    audio_for_save, save_sr = load(wav_path)
    mx.eval(audio_for_save)
    np_audio = np.array(audio_for_save)

    out_mlx = os.path.join(tmpdir, "out_mlx.wav")
    out_sf = os.path.join(tmpdir, "out_sf.wav")
    cleanup_files.extend([out_mlx, out_sf])

    # Warm the data pages so the first timed method doesn't pay for page faults
    warmup_path = os.path.join(tmpdir, "warmup.wav")
    save(warmup_path, audio_for_save, save_sr)
    os.unlink(warmup_path)

    def mlx_save():
        save(out_mlx, audio_for_save, save_sr)

    def sf_save():
        sf.write(out_sf, np_audio, save_sr, subtype="FLOAT")

    can_ta_save = HAS_TORCHAUDIO and torch is not None
    if can_ta_save:
        # torchaudio expects [channels, frames]
        ta_tensor = torch.from_numpy(np_audio.T.copy())
        out_ta = os.path.join(tmpdir, "out_ta.wav")
        cleanup_files.append(out_ta)

        def ta_save():
            torchaudio.save(out_ta, ta_tensor, save_sr)

    p50_sf, p90_sf = bench(sf_save)
    print(f"   soundfile:        p50={p50_sf:7.2f}ms  p90={p90_sf:7.2f}ms  jitter={p90_sf/p50_sf:.2f}x")
    p50_ta = None
    if can_ta_save:
        try:
            p50_ta, p90_ta = bench(ta_save)
            print(f"   torchaudio:       p50={p50_ta:7.2f}ms  p90={p90_ta:7.2f}ms  jitter={p90_ta/p50_ta:.2f}x")
        except Exception as e:
            print(f"   torchaudio:       N/A ({type(e).__name__}: {e})")
    else:
        print(f"   torchaudio:       N/A ({TORCHAUDIO_IMPORT_ERROR})")
    p50, p90 = bench(mlx_save)
    print(f"   mlx-audio-io:  p50={p50:7.2f}ms  p90={p90:7.2f}ms  jitter={p90/p50:.2f}x")

    print(f"   speedup vs soundfile: {p50_sf / p50:.1f}x", end="")
    if p50_ta is not None:
        print(f"  vs torchaudio: {p50_ta / p50:.1f}x")
    else:
        print()


    # 5. Memory (best-effort RSS delta)
    print("\n5. Memory usage (RSS delta, best-effort)")
    import gc

    gc.collect()
    rss_before = psutil.Process().memory_info().rss

    audio_mem, _ = load(wav_path)
    mx.eval(audio_mem)

    gc.collect()
    rss_after = psutil.Process().memory_info().rss
    delta_mb = (rss_after - rss_before) / (1024 * 1024)
    meta = info(wav_path)
    expected_mb = (meta.frames * meta.channels * 4) / (1024 * 1024)
    print(f"   RSS delta:        {delta_mb:.1f} MB")
    print(f"   Expected buffer:  {expected_mb:.1f} MB")

    # 6. Compressed formats (MP3, M4A)
    print("\n6. Compressed formats")
    mp3_path = os.path.join(tmpdir, "bench.mp3")
    m4a_path = os.path.join(tmpdir, "bench.m4a")

    compressed_files = {}
    # afconvert cannot encode MP3 — try lame, then ffmpeg
    mp3_created = False
    for mp3_cmd in [
        ["lame", "--quiet", "-V", "2", wav_path, mp3_path],
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-q:a", "2", mp3_path],
    ]:
        try:
            subprocess.run(mp3_cmd, capture_output=True, check=True)
            mp3_created = True
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    if mp3_created:
        compressed_files["mp3"] = mp3_path
        cleanup_files.append(mp3_path)
    else:
        print("   Skipping MP3: neither lame nor ffmpeg found")

    m4a_created = False
    for m4a_cmd in [
        ["afconvert", "-f", "m4af", "-d", "aac", wav_path, m4a_path],
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, "-c:a", "aac", "-b:a", "256k", m4a_path],
    ]:
        try:
            subprocess.run(m4a_cmd, capture_output=True, check=True)
            compressed_files["m4a"] = m4a_path
            cleanup_files.append(m4a_path)
            m4a_created = True
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    if not m4a_created:
        print("   Skipping M4A: neither afconvert nor ffmpeg found")

    for fmt, fpath in compressed_files.items():
        print(f"\n   {fmt.upper()} load:")

        def mlx_load_compressed(p=fpath):
            a, s = load(p)
            mx.eval(a)

        def librosa_load_compressed(p=fpath):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                librosa.load(p, sr=None, mono=False)

        p50_m, p90_m = bench(mlx_load_compressed)
        print(f"     mlx-audio-io:  p50={p50_m:7.2f}ms  p90={p90_m:7.2f}ms")

        p50_l, p90_l = bench(librosa_load_compressed)
        print(f"     librosa:          p50={p50_l:7.2f}ms  p90={p90_l:7.2f}ms")

        p50_t = None
        if HAS_TORCHAUDIO:
            def ta_load_compressed(p=fpath):
                torchaudio.load(p)

            try:
                p50_t, p90_t = bench(ta_load_compressed)
                print(f"     torchaudio:       p50={p50_t:7.2f}ms  p90={p90_t:7.2f}ms")
            except Exception as e:
                print(f"     torchaudio:       N/A ({type(e).__name__}: {e})")
                p50_t = None
        else:
            print(f"     torchaudio:       N/A ({TORCHAUDIO_IMPORT_ERROR})")

        print(f"     speedup vs librosa: {p50_l / p50_m:.1f}x", end="")
        if p50_t is not None:
            print(f"  vs torchaudio: {p50_t / p50_m:.1f}x")
        else:
            print()

    # Cleanup
    for f in cleanup_files:
        if os.path.exists(f):
            os.unlink(f)
    if generated and os.path.exists(wav_path):
        os.unlink(wav_path)
    if os.path.isdir(tmpdir) and not os.listdir(tmpdir):
        os.rmdir(tmpdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
