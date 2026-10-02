#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <cstring>

#include "audio_backend.h"
#include "audio_buffer.h"
#include "audio_io.h"
#include "audio_stream.h"
#include "mp3_decoder.h"
#include "dlpack/dlpack.h"

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

    // AudioBuffer class
    nb::class_<mlx_audio::AudioBuffer>(m, "AudioBuffer")
        .def_prop_ro("ndim", [](const mlx_audio::AudioBuffer& b) { return b.ndim; })
        .def_prop_ro("shape", [](const mlx_audio::AudioBuffer& b) {
            if (b.ndim == 1) {
                return nb::make_tuple(b.shape[0]);
            }
            return nb::make_tuple(b.shape[0], b.shape[1]);
        })
        .def_prop_ro("dtype", [](const mlx_audio::AudioBuffer& b) { return b.dtype; })
        .def("__dlpack_device__", [](const mlx_audio::AudioBuffer&) {
            return nb::make_tuple(1, 0); // (kDLCPU, 0)
        })
        .def("__dlpack__", [](const mlx_audio::AudioBuffer& self, nb::args, nb::kwargs) {
            auto* ctx = new mlx_audio::DLPackContext();
            ctx->storage = self.storage;
            ctx->shape[0] = self.shape[0];
            ctx->shape[1] = self.shape[1];
            ctx->strides[0] = self.strides[0];
            ctx->strides[1] = self.strides[1];

            auto* managed = new DLManagedTensor();
            managed->manager_ctx = ctx;
            managed->dl_tensor.data = const_cast<void*>(self.data());
            managed->dl_tensor.device = DLDevice{kDLCPU, 0};
            managed->dl_tensor.ndim = self.ndim;
            managed->dl_tensor.dtype = (self.dtype == "float16")
                ? DLDataType{kDLFloat, 16, 1}
                : DLDataType{kDLFloat, 32, 1};
            managed->dl_tensor.shape = ctx->shape;
            managed->dl_tensor.strides = ctx->strides;
            managed->dl_tensor.byte_offset = 0;

            managed->deleter = [](DLManagedTensor* self_tensor) {
                if (!self_tensor) return;
                auto* ctx_ptr = static_cast<mlx_audio::DLPackContext*>(self_tensor->manager_ctx);
                delete ctx_ptr;
                delete self_tensor;
            };

            auto capsule_destructor = [](PyObject* cap) {
                const char* name = PyCapsule_GetName(cap);
                if (name && std::strcmp(name, "dltensor") == 0) {
                    auto* dlm = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
                    if (dlm && dlm->deleter) {
                        dlm->deleter(dlm);
                    }
                }
            };

            PyObject* cap = PyCapsule_New(managed, "dltensor", capsule_destructor);
            if (!cap) {
                managed->deleter(managed);
                throw std::runtime_error("Failed to create DLPack capsule");
            }
            return nb::steal(cap);
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
            if (result.first.frames() == 0) {
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
          R"(Load audio from a file into an AudioBuffer.

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
    Tuple of (AudioBuffer, sample rate).

Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: For invalid arguments.
)");

    // save()
    m.def("save", [](
        const std::string& path,
        nb::handle audio_obj,
        int sr,
        const std::string& layout,
        const std::string& encoding,
        const std::string& bitrate,
        bool clip,
        const std::string& flac_compression) {

        Py_buffer view;
        if (PyObject_GetBuffer(audio_obj.ptr(), &view, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
            throw mlx_audio::value_error("save() requires an object supporting the Python buffer protocol");
        }

        struct PyBufferGuard {
            Py_buffer* b;
            ~PyBufferGuard() { PyBuffer_Release(b); }
        } guard{&view};

        if (view.itemsize != sizeof(float)) {
            throw mlx_audio::value_error("save() requires float32 audio samples");
        }

        int64_t frames = 0;
        int channels = 1;
        int64_t stride_frame = 1;
        int64_t stride_chan = 1;

        if (view.ndim == 1) {
            frames = view.shape[0];
            channels = 1;
            stride_frame = view.strides[0] / sizeof(float);
            stride_chan = 1;
        } else if (view.ndim == 2) {
            if (layout == "channels_first") {
                channels = static_cast<int>(view.shape[0]);
                frames = view.shape[1];
                stride_chan = view.strides[0] / sizeof(float);
                stride_frame = view.strides[1] / sizeof(float);
            } else {
                frames = view.shape[0];
                channels = static_cast<int>(view.shape[1]);
                stride_frame = view.strides[0] / sizeof(float);
                stride_chan = view.strides[1] / sizeof(float);
            }
        } else {
            throw mlx_audio::value_error("save() requires 1D or 2D audio array, got " + std::to_string(view.ndim) + "D");
        }

        const float* data = static_cast<const float*>(view.buf);

        {
            nb::gil_scoped_release release;
            mlx_audio::save_audio(
                path,
                data,
                frames,
                channels,
                stride_frame,
                stride_chan,
                sr,
                layout,
                encoding,
                bitrate,
                clip,
                flac_compression
            );
        }
    },
    "path"_a,
    "audio"_a,
    "sr"_a,
    "layout"_a = "channels_last",
    "encoding"_a = "float32",
    "bitrate"_a = "auto",
    "clip"_a = true,
    "flac_compression"_a = "default",
    R"(Save an audio buffer to a file.

Output support is backend-dependent:
- macOS backend: WAV (.wav), MP3 (.mp3), M4A/AAC (.m4a), FLAC (.flac), AIFF (.aiff), CAF (.caf).
- Linux backend: WAV (.wav), MP3 (.mp3), M4A/AAC (.m4a), FLAC (.flac), AIFF (.aiff), CAF (.caf).
The format is determined by the file extension.

The GIL is released during save, so this function can run on a background
thread without blocking the Python interpreter.

Args:
    path: Output file path. Extension determines format.
    audio: 1D or 2D float32 array/buffer. 1D treated as mono.
    sr: Sample rate.
    layout: 'channels_last' or 'channels_first'.
    encoding: Sample encoding — 'float32' (default), 'pcm16', 'pcm24', or 'alac'.
    bitrate: Lossy encode bitrate — 'auto' (default), '128k', '192k', '256k', '320k'.
    clip: If True, clamp samples to [-1, 1] before writing.
    flac_compression: 'default', or 'fast'.

Raises:
    ValueError: For invalid arguments.
)");

    // load_streaming_resample() — bounded-scratch streaming resample.
    m.def("load_streaming_resample",
          &mlx_audio::backend_load_audio_streaming_resample,
          "path"_a,
          "target_sr"_a,
          "offset"_a = 0.0,
          "duration"_a = nb::none(),
          "mono"_a = false,
          "layout"_a = "channels_last",
          "dtype"_a = "float32",
          "quality"_a = "soxr_hq",
          nb::call_guard<nb::gil_scoped_release>(),
          R"(Load and resample an audio file using a bounded-scratch streaming pipeline.

Reads the file in chunks at its native sample rate and pushes each chunk
through a stateful libsoxr resampler directly into a single preallocated
output buffer. Peak scratch memory is bounded to a single chunk plus the
soxr filter state, independent of file length.

Requires a build with libsoxr support. For files whose native SR already
matches target_sr, the call transparently falls back to the regular load().

Args:
    path: Path to the audio file.
    target_sr: Output sample rate (must be > 0).
    offset: Start time in seconds.
    duration: Duration in seconds. None to read to end.
    mono: If True, mix down to mono before resampling (saves work).
    layout: 'channels_last' or 'channels_first'.
    dtype: Output dtype — 'float32' or 'float16'.
    quality: 'soxr_hq' (default) or 'soxr_vhq'.

Returns:
    Tuple of (AudioBuffer, target_sr).

Raises:
    ValueError: If libsoxr support was not compiled in.
)");

    // resample()
    m.def("resample", [](
        nb::handle audio_obj,
        int in_sr,
        int out_sr,
        const std::string& quality) -> mlx_audio::AudioBuffer {

        Py_buffer view;
        if (PyObject_GetBuffer(audio_obj.ptr(), &view, PyBUF_STRIDES | PyBUF_FORMAT) != 0) {
            throw mlx_audio::value_error("resample() requires an object supporting the Python buffer protocol");
        }

        struct PyBufferGuard {
            Py_buffer* b;
            ~PyBufferGuard() { PyBuffer_Release(b); }
        } guard{&view};

        if (view.itemsize != sizeof(float)) {
            throw mlx_audio::value_error("resample() requires float32 audio samples");
        }

        int64_t frames = 0;
        int channels = 1;
        int64_t stride_frame = 1;
        int64_t stride_chan = 1;
        int ndim = static_cast<int>(view.ndim);

        if (ndim == 1) {
            frames = view.shape[0];
            channels = 1;
            stride_frame = view.strides[0] / sizeof(float);
            stride_chan = 1;
        } else if (ndim == 2) {
            frames = view.shape[0];
            channels = static_cast<int>(view.shape[1]);
            stride_frame = view.strides[0] / sizeof(float);
            stride_chan = view.strides[1] / sizeof(float);
        } else {
            throw mlx_audio::value_error("resample() requires 1D or 2D audio array, got " + std::to_string(ndim) + "D");
        }

        const float* data = static_cast<const float*>(view.buf);

        mlx_audio::AudioBuffer result;
        {
            nb::gil_scoped_release release;
            result = mlx_audio::resample_audio(
                data,
                frames,
                channels,
                stride_frame,
                stride_chan,
                in_sr,
                out_sr,
                quality
            );
        }
        if (ndim == 1) {
            result.ndim = 1;
            result.shape[0] = result.frames();
            result.shape[1] = 1;
            result.strides[0] = 1;
            result.strides[1] = 1;
        } else {
            result.ndim = 2;
            result.shape[0] = result.frames();
            result.shape[1] = channels;
            result.strides[0] = channels;
            result.strides[1] = 1;
        }
        return result;
    },
    "audio"_a,
    "in_sr"_a,
    "out_sr"_a,
    "quality"_a = "default",
    R"(Resample an in-memory audio array to a different sample rate.

Args:
    audio: 1D (frames,) or 2D (frames, channels) float32 buffer.
    in_sr: Source sample rate (must be > 0).
    out_sr: Target sample rate (must be > 0).
    quality: Resampler quality — 'default', 'fastest', 'low', 'medium', 'high', or 'best'.
        Also supports 'soxr_hq' and 'soxr_vhq' when built with libsoxr.
        On macOS, maps to AudioConverter quality levels.
        On Linux, default quality modes use linear interpolation.

Returns:
    Resampled AudioBuffer with same ndim and channel count.
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

Returns an iterator of (AudioBuffer, sample_rate) tuples. Each buffer has shape
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
    AudioStreamReader iterator yielding (AudioBuffer, sample_rate) tuples.

Raises:
    FileNotFoundError: If the file does not exist.
    ValueError: For invalid arguments.
)");

}
