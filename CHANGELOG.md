# Changelog

All notable changes to this project are documented in this file.

Entries before 1.3.12 were reconstructed from the commit history after the fact,
so they summarise what shipped rather than what was announced at the time.

## 1.3.14 - 2026-09-23

### Fixed

- `tests/test_build_pairing.py` imports `build_backend`, which lives at the
  repository root rather than in the installed package, so it only imported
  when something else happened to put the root on `sys.path`. That stopped
  being true once the project was installed non-editable, and every CI job
  collected with `ModuleNotFoundError: No module named 'build_backend'`.
  `pythonpath = ["."]` in the pytest configuration fixes it.
- `uv.lock` refreshed to the released stack.

## 1.3.13 - 2026-09-23

### Changed

- **The build now fails fast on a known-bad MLX/nanobind pair.** pip resolves
  `[build-system] requires` in an isolated environment and installs them before
  a backend hook runs, so a backend cannot substitute a different pair; what it
  can do is refuse. `get_requires_for_build_wheel` now validates and raises,
  naming the runtime error a mismatched binary would have produced and the
  commands that fix it, and CMake fails rather than warning.
- **To build against a specific MLX**, install the build dependencies yourself
  and pass `--no-build-isolation`; the README has the exact commands. Verified
  end to end against a TestPyPI sdist with MLX pinned to 0.31.2.
- Release workflows set the version from the workflow input before building,
  wait up to 10 minutes for a new file to reach every CDN mirror, and retry the
  install. Because this package ships an sdist, the wait polls the index rather
  than using `pip download`, which would compile the extension once to read
  metadata and again on install.

## 1.3.12 - 2026-09-22

### Added

- **MLX 0.32.x support.** Verified against 0.32.2: the extension rebuilds cleanly,
  all four suites in the MLX audio stack pass on both 0.31.2 and 0.32.2, and Demucs
  output is bit-identical across them (max abs diff 0.000e+00 on every stem).
  `mlx` widens to `>=0.31.2,<0.33` for build and runtime.
- `MLX_AUDIO_IO_BUILD_MLX` declares the MLX you intend to build against. The
  build fails if it is handed a different one, rather than silently producing a
  binary you cannot load. It cannot change pip's isolated resolution — nothing in
  a backend can — so to actually build against a specific MLX, install the build
  dependencies yourself and pass `--no-build-isolation`. The README and the
  loader's error message both give the exact commands.
- The build records the nanobind it used (`build_nanobind_version`), and the loader
  rejects a binary whose nanobind does not match its MLX.

### Fixed

- **The build could pick an MLX and a nanobind that never occur together.** This
  package publishes an sdist, so `pip install` compiles the extension inside an
  isolated build environment that pip resolves independently of the environment
  being installed into. `mlx` and `nanobind` are both ranges in
  `[build-system] requires` and were resolved independently of each other, so pip
  could hand the build MLX 0.32.x with nanobind 2.12.0. That compiles and links
  cleanly and then fails on every call with `Unable to convert function return
  value to a Python type` — an error that never mentions nanobind. The build now
  refuses that combination, from `get_requires_for_build_wheel` and again from
  CMake, naming the error the broken binary would have produced and the commands
  that fix it. A backend cannot *correct* the pair — pip installs
  `[build-system] requires` before calling the hook and rejects any conflicting
  pin it returns — so refusing is the strongest thing available, and it beats
  shipping a binary that imports and then fails.
- **`pip install` could produce a binary that cannot load.** Same mechanism, other
  axis: a user pinned to MLX 0.31.2 could get a build environment with 0.32.x and
  a binary that fails `dlopen` with an undefined `mlx::core::astype` symbol. The
  version gate already caught this, but its suggested fix was to upgrade their MLX
  to match the build — backwards for someone who pinned deliberately. It now leads
  with rebuilding against their MLX and gives the exact command.
- The CMake nanobind/MLX pairing check was a `message(WARNING)`. Under
  `pip install`, CMake output is captured and shown only on failure, so the warning
  was invisible to exactly the person who needed it, and the result was a
  guaranteed runtime failure rather than a degraded one. It is now a
  `FATAL_ERROR`; `-DMLX_AUDIO_IO_ALLOW_NANOBIND_MISMATCH=ON` overrides it.
- A `dlopen` failure was reported with "do not copy `.venv` across machines" and
  `rm -rf .venv && uv sync`, which is unrelated to ABI drift and whose suggested
  fix reproduces it. An undefined `mlx::core` symbol now gets rebuild guidance.
- `stream()` caught `TypeError` to fall back for older native modules with a
  narrower signature. A nanobind registry mismatch also raises `TypeError`, so the
  fallback swallowed it, retried into the same failure, and discarded the
  informative first traceback. Narrowed to the signature case.
- `deployment_target` is now recorded in the build metadata (`13.0`), so the
  runtime macOS version check that reads it is active. It previously recorded
  an empty string because `CMAKE_OSX_DEPLOYMENT_TARGET` was set without
  `FORCE` over a cache entry scikit-build-core pre-seeds.
- `load_build_info()` filtered build metadata through a hardcoded key set and
  silently dropped anything not listed, including the new compatible-version list.
- Editable checkouts reported every build field as `unknown`, because
  `importlib.resources` resolves `mlx_audio_io` to the source tree, which holds only
  `_build_info.json.in`. The generated file sits beside the compiled `_core` module
  and is now read from there. **This had quietly disabled the MLX version gate in
  every development checkout.**

### Changed

- The loader accepts any MLX in a build-recorded list of verified versions instead
  of requiring character-for-character equality with the build-time version. The
  constraint itself is real and stays: the extension links `libmlx`, passes
  `mlx::core::array` by value and shares nanobind's type registry through
  `NB_DOMAIN`, and MLX has no stable C++ ABI or soname versioning —
  `StreamOrDevice` gained a variant alternative in 0.32.0, remangling every
  operation that takes a stream. What changed is granularity.
  `MLX_AUDIO_IO_COMPATIBLE_MLX_VERSIONS` defaults to the build-time version, so
  behaviour is unchanged unless a maintainer opts in, and
  `MLX_AUDIO_IO_ALLOW_MLX_MISMATCH=1` downgrades rejection to a `RuntimeWarning`.
  This was reported by a user who could not evaluate other MLX releases while
  investigating [mlx-audio-separator#4](https://github.com/ssmall256/mlx-audio-separator/issues/4).
- `nanobind` widens to `>=2.12.0,<2.16`. Known pairings, from MLX's own
  `CMakeLists.txt` `FetchContent ... GIT_TAG`: MLX 0.31.x → nanobind 2.12.0,
  MLX 0.32.x → nanobind 2.15.0.
- scikit-build-core is imported lazily by the build backend, so the
  requirement-pairing logic is testable without a build toolchain present.
- README corrected: the install snippet said 1.3.11, the MLX range said `<0.32`,
  and the claim that the project ships one wheel line per MLX minor was never true
  of an sdist-only release. Troubleshooting is no longer filed under "Linux".

## 1.3.11 - 2026-06-13

### Added

- Streaming resample: `load(low_memory=True)` reads in chunks at the native rate
  through a stateful libsoxr resampler into a preallocated buffer, so peak scratch
  memory is independent of file length. Requires soxr and one of
  `soxr_hq`/`soxr_vhq`.
- `resample()` takes a `layout` argument (`channels_last` default, or
  `channels_first`). The array is transposed to contiguous before the native call,
  avoiding the trap where a `swapaxes` view is read linearly by `data<float>()`.
- `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`) parsing in both backends, with SubFormat GUID
  validation.
- float64 (IEEE double) WAV reads on all three decode paths, downcast to float32
  since MLX has no float64.

### Fixed

- The macOS WAV fast path silently returned an empty array for unsupported
  encodings. It now raises, matching Linux.

### Changed

- Built with C++20 and pinned to MLX 0.31.2 with nanobind 2.12.0.
- `./dev` script added, and pytest rebuilds the C++ extension when sources are
  stale — previously a source edit could be tested against a stale binary.

## 1.3.10 - 2026-03-13

### Changed

- MLX compatibility policy pinned and documented, with packaging tests enforcing it.

## 1.3.9 - 2026-03-06

### Changed

- `load(sr=...)` selects `soxr_vhq` automatically when soxr is available and
  `resample_quality="default"`, falling back to `best` otherwise. Callers no longer
  need to probe `supports_soxr()` themselves.

## 1.3.8 - 2026-03-06

### Added

- True libsoxr resampling modes, wired into `load()` and `resample()` on both the
  Apple and Linux backends, with a `soxr_hq` alias.
- `torchaudio_compat` resample quality, for bit-comparable output against
  torchaudio (requires torch/torchaudio).
- `mx.compile`-fused mixdown kernels.

### Fixed

- libsoxr wheel packaging hardened with automatic repair, release linkage gates and
  bundled third-party notices.
- Linux CI builds fixed with `MLX_DISABLE_COMPILE=1`.

### Changed

- numpy is imported lazily at package import.
- Minimum MLX raised to 0.31.0.

## 1.3.7 - 2026-03-01

### Added

- `mono_mode` compatibility option on the load and stream paths.

## 1.3.6 - 2026-03-01

### Fixed

- Apple backend seek semantics for resampled offsets, with conformance tests.

## 1.3.5 - 2026-03-01

### Changed

- MLX pin relaxed to allow 0.31.

## 1.3.4 - 2026-03-01

### Added

- Native stream windowing (`offset` / `duration` at the stream level).
