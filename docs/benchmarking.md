# Benchmarking

This document is the source-of-truth template for publishing performance numbers.
The README keeps only a short summary and links here for full detail.

### Benchmark Protocol

- Input: same WAV file for all libraries on the same machine.
- Warmup/runs: `3` warmups, `20` measured runs (current `tests/benchmark.py` defaults).
- Metrics: report `p50` (median) and `p90` latency in milliseconds.
- Baselines: `librosa` and `torchaudio` for load paths, `soundfile` and `torchaudio` for save.
- Reproducibility: include host metadata and commit SHA next to every published table.

Metadata block template:

```text
Date: YYYY-MM-DD
Commit: <git-sha>
Machine: <cpu/model>
RAM: <size>
OS: <os-version>
Python: <version>
MLX: <version>
```

### Run Commands

macOS/local benchmark run:

```bash
uv sync --extra benchmark
uv run python tests/benchmark.py /path/to/file.wav
```

Linux benchmark run (Docker on macOS host):

```bash
docker run --rm -it --platform linux/arm64 \
  -v "$PWD":/work -w /work \
  python:3.14-bookworm bash -lc '
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build pkg-config ffmpeg \
      libavformat-dev libavcodec-dev libavutil-dev libswresample-dev &&
    python -m pip install -U pip uv &&
    uv sync --extra benchmark &&
    uv run python tests/benchmark.py /work/tests/assets/pcm16_stereo_44k1.wav
  '
```

If no file is passed to `tests/benchmark.py`, it generates a 10s stereo PCM16 WAV input.

### macOS Benchmark Table (M4 Max)

Caption: Lower is better. `p50` is median latency, `p90` shows tail latency. Speedup columns are computed as `baseline_p50 / mlx_p50`.

Note: The very large `vs torchaudio` ratios on partial WAV reads are overhead-dominated latency results, not linear decode-throughput multipliers.

Run metadata:

```text
Machine: Apple M4 Max
RAM: 128 GB
OS: macOS 26.3
Python: 3.12.11
torchaudio: 2.10.0
Input: 194.8s stereo PCM16 WAV @ 44.1 kHz
```

| Task | mlx p50 (ms) | mlx p90 (ms) | librosa p50 (ms) | torchaudio p50 (ms) | Speedup vs librosa | Speedup vs torchaudio |
|---|---:|---:|---:|---:|---:|---:|
| Full load (native SR) | 3.59 | 3.93 | 24.65 | 34.70 | 6.9x | 9.7x |
| Load + resample to 16 kHz | 13.12 | 13.19 | 58.35 | 56.46 | 4.4x | 4.3x |
| Partial load (offset=2s, duration=1s) | 0.04 | 0.05 | 0.15 | 32.38 | 3.4x | 725.5x |
| Save WAV (float32) | 6.98 | 13.90 | 19.83 (`soundfile`) | 15.88 | 2.8x (`soundfile`) | 2.3x |
| MP3 load (native SR) | 63.70 | 64.24 | 85.13 | 114.74 | 1.3x | 1.8x |
| M4A/AAC load | 56.31 | 56.69 | 121.38 | 64.36 | 2.2x | 1.1x |

### macOS Benchmark Table (M1 Max)

Caption: Lower is better. `p50` is median latency, `p90` shows tail latency. Speedup columns are computed as `baseline_p50 / mlx_p50`.

Note: The very large `vs torchaudio` ratios on partial WAV reads are overhead-dominated latency results, not linear decode-throughput multipliers.

Run metadata:

```text
Machine: Apple M1 Max
RAM: 32 GB
OS: macOS 26.3
Python: 3.12.12
torchaudio: 2.10.0
Input: 194.8s stereo PCM16 WAV @ 44.1 kHz
```

| Task | mlx p50 (ms) | mlx p90 (ms) | librosa p50 (ms) | torchaudio p50 (ms) | Speedup vs librosa | Speedup vs torchaudio |
|---|---:|---:|---:|---:|---:|---:|
| Full load (native SR) | 7.29 | 7.44 | 33.47 | 51.86 | 4.6x | 7.1x |
| Load + resample to 16 kHz | 18.95 | 19.27 | 88.86 | 90.88 | 4.7x | 4.8x |
| Partial load (offset=2s, duration=1s) | 0.06 | 0.07 | 0.24 | 51.54 | 3.9x | 835.8x |
| Save WAV (float32) | 12.63 | 30.66 | 49.33 (`soundfile`) | 25.21 | 3.9x (`soundfile`) | 2.0x |
| MP3 load (native SR) | 96.69 | 97.24 | 124.01 | 166.41 | 1.3x | 1.7x |
| M4A/AAC load | 83.64 | 84.48 | 191.41 | 100.19 | 2.3x | 1.2x |

### Linux Benchmark Table (arm64 Docker on M4 Max host)

Caption: Linux supports WAV/MP3/FLAC/M4A/AIFF/CAF with native WAV/MP3 fast paths. TorchCodec unavailable on linux/arm64.

Run metadata:

```text
Platform: Linux-6.17.8-orbstack-00308-g8f9c941121b1-aarch64-with-glibc2.36
RAM: 16 GB
Python: 3.14.3
torchaudio: 2.10.0 (TorchCodec unavailable on linux/arm64)
Input: 194.8s stereo PCM16 WAV @ 44.1 kHz
```

| Task | mlx p50 (ms) | mlx p90 (ms) | librosa p50 (ms) | torchaudio p50 (ms) | Speedup vs librosa | Speedup vs torchaudio |
|---|---:|---:|---:|---:|---:|---:|
| Full load (native SR) | 8.41 | 8.96 | 49.95 | N/A | 5.9x | N/A |
| Load + resample to 16 kHz | 10.93 | 10.99 | 86.50 | N/A | 7.9x | N/A |
| Partial load (offset=2s, duration=1s) | 0.05 | 0.06 | 0.14 | N/A | 2.6x | N/A |
| Save WAV (float32) | 31.70 | 32.15 | 56.25 (`soundfile`) | N/A | 1.8x (`soundfile`) | N/A |
| MP3 load (native SR) | 80.93 | 81.46 | 84.14 | N/A | 1.0x | N/A |
| M4A/AAC load | 89.63 | 90.10 | 141.48 | N/A | 1.6x | N/A |

### Linux Benchmark Table (amd64 Docker on M4 Max host)

Caption: Linux x86_64 under emulation. Absolute latencies reflect Rosetta/QEMU overhead; speedup ratios are the meaningful comparison.

Run metadata:

```text
Platform: Linux-6.17.8-orbstack-00308-g8f9c941121b1-x86_64-with-glibc2.36
RAM: 16 GB
Python: 3.14.3
torchaudio: 2.10.0+cu128
Input: 194.8s stereo PCM16 WAV @ 44.1 kHz
```

| Task | mlx p50 (ms) | mlx p90 (ms) | librosa p50 (ms) | torchaudio p50 (ms) | Speedup vs librosa | Speedup vs torchaudio |
|---|---:|---:|---:|---:|---:|---:|
| Full load (native SR) | 10.76 | 12.34 | 47.18 | 44.29 | 4.4x | 4.1x |
| Load + resample to 16 kHz | 12.16 | 12.33 | 92.92 | 90.40 | 7.6x | 7.4x |
| Partial load (offset=2s, duration=1s) | 0.06 | 0.06 | 0.25 | 44.46 | 4.4x | 793.9x |
| Save WAV (float32) | 39.50 | 39.93 | 83.61 (`soundfile`) | 50.58 | 2.1x (`soundfile`) | 1.3x |
| MP3 load (native SR) | 119.78 | 121.09 | 116.69 | 137.41 | 1.0x | 1.1x |
| M4A/AAC load | 106.49 | 107.91 | 307.89 | 104.83 | 2.9x | 1.0x |

### Notes for publishing results

- Keep macOS and Linux results in separate tables.
- Do not compare macOS vs Linux directly in a single speedup number.
- If a baseline library is unavailable on a host, mark that cell as `N/A`.

### Interpretation notes (important)

- The `700x+` torchaudio ratios on partial WAV reads are real measurements, but they are **overhead-dominated latency results**, not a claim of `700x` codec throughput.
- In our macOS runs, torchaudio WAV timings are nearly flat across very different workloads (about `32-57ms` for full load, resample, and 1s partial slice), which indicates a large fixed per-call cost.
- `mlx-audio-io` has a direct WAV fast path for `offset`/`duration` that can return tiny slices in tens of microseconds, so dividing a mostly fixed torchaudio cost by a very small mlx latency produces very large ratios.
- A better proxy for codec/decode throughput is the compressed-format rows (MP3/M4A), where the ratios are much smaller (`~1.0–2.2x`) and closer to true decode work differences.
- Sub-millisecond results are sensitive to cache and timer effects. Treat the extreme ratios as evidence of lower overhead for small WAV slice reads, not universal end-to-end speedups.
- Linux `arm64` torchaudio values are `N/A` because torchaudio 2.10 depends on TorchCodec APIs, and `torchcodec` wheels are currently unavailable for that target.
- Linux `amd64` results run under Rosetta/QEMU emulation on an Apple Silicon host. Absolute latencies are inflated; the speedup ratios are the meaningful comparison.
- Linux container RSS deltas can appear as `0.0 MB` due to allocator/page accounting behavior; use that row as a coarse signal only.
