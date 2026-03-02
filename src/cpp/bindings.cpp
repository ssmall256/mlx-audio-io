#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include "audio_io.h"
#include "audio_stream.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    m.doc() = "mlx-audio-io: Native audio I/O for MLX on macOS and Linux";
#if MLX_AUDIO_IO_ENABLE_SOXR
    m.attr("_HAS_SOXR") = true;
#else
    m.attr("_HAS_SOXR") = false;
#endif

    // Register custom exceptions
    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const mlx_audio::file_not_found_error& e) {
            PyErr_SetString(PyExc_FileNotFoundError, e.what());
        } catch (const mlx_audio::value_error& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

    // AudioInfo class
    nb::class_<mlx_audio::AudioFileInfo>(m, "AudioInfo")
        .def_ro("frames", &mlx_audio::AudioFileInfo::frames)
        .def_ro("sample_rate", &mlx_audio::AudioFileInfo::sample_rate)
        .def_ro("channels", &mlx_audio::AudioFileInfo::channels)
        .def_ro("duration", &mlx_audio::AudioFileInfo::duration)
        .def_ro("subtype", &mlx_audio::AudioFileInfo::subtype)
        .def_ro("container", &mlx_audio::AudioFileInfo::container)
        .def("__repr__", [](const mlx_audio::AudioFileInfo& info) {
            return "AudioInfo(frames=" + std::to_string(info.frames) +
                   ", sample_rate=" + std::to_string(info.sample_rate) +
                   ", channels=" + std::to_string(info.channels) +
                   ", duration=" + std::to_string(info.duration) +
                   ", subtype='" + info.subtype + "'" +
                   ", container='" + info.container + "')";
        });

    // AudioStreamReader class
    nb::class_<mlx_audio::AudioStreamReader>(m, "AudioStreamReader")
        .def_prop_ro("sample_rate", &mlx_audio::AudioStreamReader::sample_rate)
        .def_prop_ro("channels", &mlx_audio::AudioStreamReader::channels)
        .def_prop_ro("chunk_frames", &mlx_audio::AudioStreamReader::chunk_frames)
        .def_prop_ro("frames_read", &mlx_audio::AudioStreamReader::frames_read)
        .def("at_eof", &mlx_audio::AudioStreamReader::at_eof)
        .def("read_chunk", &mlx_audio::AudioStreamReader::read_chunk,
             nb::call_guard<nb::gil_scoped_release>())
        .def("__iter__", [](mlx_audio::AudioStreamReader& self) -> mlx_audio::AudioStreamReader& {
            return self;
        }, nb::rv_policy::reference)
        .def("__next__", [](mlx_audio::AudioStreamReader& self) {
            if (self.at_eof()) {
                throw nb::stop_iteration();
            }
            auto result = [&] {
                nb::gil_scoped_release release;
                return self.read_chunk();
            }();
            if (result.first.shape(0) == 0) {
                throw nb::stop_iteration();
            }
            return result;
        });

    // info()
    m.def("info", &mlx_audio::get_info,
          "path"_a,
          R"(Get metadata about an audio file without decoding samples.

Format support is backend-dependent:
- macOS backend: WAV, MP3, AAC/M4A, FLAC, AIFF, CAF, and more via AudioToolbox.
- Linux backend: WAV, MP3, FLAC, M4A, AIFF, CAF.

Args:
    path: Path to the audio file.

Returns:
    AudioInfo with frames, sample_rate, channels, duration, subtype, container.

Raises:
    FileNotFoundError: If the file does not exist.
)");

    // load()
    m.def("load", &mlx_audio::load_audio,
          "path"_a,
          "sr"_a = nb::none(),
          "offset"_a = 0.0,
          "duration"_a = nb::none(),
          "mono"_a = false,
          "layout"_a = "channels_last",
          "dtype"_a = "float32",
          "resample_quality"_a = "default",
          nb::call_guard<nb::gil_scoped_release>(),
          R"(Load audio from a file into an mlx.core.array.

Format support is backend-dependent:
- macOS backend: WAV, MP3, AAC/M4A, FLAC, AIFF, CAF, and more via AudioToolbox.
- Linux backend: WAV, MP3, FLAC, M4A, AIFF, CAF.

Args:
    path: Path to the audio file.
    sr: Target sample rate. None to use native rate.
    offset: Start time in seconds (default 0.0).
    duration: Duration in seconds. None to read to end.
    mono: If True, mix down to mono.
    layout: 'channels_last' [frames, channels] or 'channels_first' [channels, frames].
    dtype: Output dtype — 'float32' (default) or 'float16'.
    resample_quality: Resampler quality when sr differs from native rate.
        'default', 'fastest', 'low', 'medium', 'high', or 'best'.
        Python wrapper also supports 'soxr_hq' and 'soxr_vhq'.
        Ignored when no resampling occurs.

Returns:
    Tuple of (audio array, sample rate).

Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: For invalid arguments.
)");

    // save()
    m.def("save", &mlx_audio::save_audio,
          "path"_a,
          "audio"_a,
          "sr"_a,
          "layout"_a = "channels_last",
          "encoding"_a = "float32",
          "bitrate"_a = "auto",
          "clip"_a = true,
          nb::call_guard<nb::gil_scoped_release>(),
          R"(Save an mlx.core.array to an audio file.

Output support is backend-dependent:
- macOS backend: WAV (.wav), MP3 (.mp3), M4A/AAC (.m4a), FLAC (.flac), AIFF (.aiff), CAF (.caf).
- Linux backend: WAV (.wav), MP3 (.mp3), M4A/AAC (.m4a), FLAC (.flac), AIFF (.aiff), CAF (.caf).
The format is determined by the file extension.

The GIL is released during save, so this function can run on a background
thread without blocking the Python interpreter. Any pending mlx.core.eval()
is performed internally — callers do not need to evaluate the array first.

Args:
    path: Output file path. Extension determines format.
    audio: 1D or 2D float32/float16 mlx array. 1D treated as mono.
    sr: Sample rate.
    layout: 'channels_last' or 'channels_first'.
    encoding: Sample encoding — 'float32' (default), 'pcm16', 'pcm24', or 'alac' (Apple Lossless for .m4a).
    bitrate: Lossy encode bitrate — 'auto' (default), '128k', '192k', '256k', '320k'.
             Used for .m4a AAC and .mp3 output.
    clip: If True, clamp samples to [-1, 1] before writing.

Raises:
    ValueError: For invalid arguments.
)");

    // resample()
    m.def("resample", &mlx_audio::resample_audio,
          "audio"_a,
          "in_sr"_a,
          "out_sr"_a,
          "quality"_a = "default",
          nb::call_guard<nb::gil_scoped_release>(),
          R"(Resample an in-memory audio array to a different sample rate.

Args:
    audio: 1D (frames,) or 2D (frames, channels) float32/float16 mlx array.
    in_sr: Source sample rate (must be > 0).
    out_sr: Target sample rate (must be > 0).
    quality: Resampler quality — 'default', 'fastest', 'low', 'medium', 'high', or 'best'.
        Also supports 'soxr_hq' and 'soxr_vhq' when built with libsoxr.
        On macOS, maps to AudioConverter quality levels.
        On Linux, default quality modes use linear interpolation.

Returns:
    Resampled mlx array with same ndim and channel count.
    Returns the input unchanged when in_sr == out_sr.

Raises:
    ValueError: For invalid arguments.
)");

    // stream() factory function
    m.def("stream",
          [](const std::string& path,
             std::optional<int> chunk_frames,
             std::optional<double> chunk_duration,
             std::optional<int> sr,
             bool mono,
             double offset,
             std::optional<double> duration,
             const std::string& dtype) -> mlx_audio::AudioStreamReader {
              // Validate: exactly one of chunk_frames / chunk_duration
              if (chunk_frames.has_value() == chunk_duration.has_value()) {
                  throw mlx_audio::value_error(
                      "Exactly one of chunk_frames or chunk_duration must be specified.");
              }

              int frames;
              if (chunk_frames.has_value()) {
                  frames = chunk_frames.value();
              } else {
                  // Need to determine effective SR to convert duration to frames.
                  // If sr is given, use that; otherwise peek at the file's native SR.
                  // For MP3, probe directly so platforms without full info()
                  // support can still stream with chunk_duration.
                  int effective_sr;
                  if (sr.has_value()) {
                      effective_sr = sr.value();
                  } else if (mlx_audio::is_mp3_path(path)) {
                      mlx_audio::ScopedMp3Decoder probe;
                      probe.open(path);
                      effective_sr = probe.sample_rate();
                  } else {
                      auto info = mlx_audio::get_info(path);
                      effective_sr = info.sample_rate;
                  }
                  frames = static_cast<int>(
                      std::floor(chunk_duration.value() * effective_sr));
                  if (frames <= 0) {
                      throw mlx_audio::value_error(
                          "chunk_duration too small, results in 0 frames.");
                  }
              }

              return mlx_audio::AudioStreamReader(
                  path, frames, sr, mono, offset, duration, dtype);
          },
          "path"_a,
          "chunk_frames"_a = nb::none(),
          "chunk_duration"_a = nb::none(),
          "sr"_a = nb::none(),
          "mono"_a = false,
          "offset"_a = 0.0,
          "duration"_a = nb::none(),
          "dtype"_a = "float32",
          R"(Create a streaming reader that yields audio chunks.

Returns an iterator of (array, sample_rate) tuples. Each array has shape
[chunk_frames, channels] (channels_last). The final chunk may have fewer frames.

Exactly one of chunk_frames or chunk_duration must be specified.

Args:
    path: Path to the audio file.
    chunk_frames: Number of frames per chunk.
    chunk_duration: Duration of each chunk in seconds.
    sr: Target sample rate. None to use native rate.
    mono: If True, mix down to mono.
    offset: Start time in seconds (default 0.0).
    duration: Duration in seconds. None to read to end.
    dtype: Output dtype — 'float32' (default) or 'float16'.

Returns:
    AudioStreamReader iterator yielding (array, sample_rate) tuples.

Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: For invalid arguments.
)");

}
