# Third-Party Notices

This package may bundle third-party binary components inside wheels.

## libsoxr

- Component: SoX Resampler Library (`libsoxr`)
- Upstream: https://sourceforge.net/projects/soxr/
- Copyright:
  - Copyright (c) 2007-2018 Rob Sykes
- License:
  - GNU Lesser General Public License v2.1 or later (`LGPL-2.1-or-later`)

When `mlx-audio-io` is built with `libsoxr` support and wheel repair is enabled,
the wheel may include a copied `libsoxr` dynamic library.

Included license files:

- `mlx_audio_io/licenses/libsoxr/LICENCE`
- `mlx_audio_io/licenses/libsoxr/COPYING.LGPL`
- `mlx_audio_io/licenses/libsoxr/AUTHORS`
