# mlx-audio-io TODO

## Near Term

- Consider adding `squeeze_mono=True` to `load()`, `stream()`, and `batch_load()`.
  Keep the current default behavior unchanged: `mono=True` returns an explicit
  singleton channel axis, e.g. `(frames, 1)` for `channels_last` and
  `(1, frames)` for `channels_first`. The new option should only squeeze that
  singleton channel axis when callers explicitly opt in.
