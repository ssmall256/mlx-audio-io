// Streaming resample path: read a file chunk-by-chunk at its native sample
// rate and push each chunk through a stateful libsoxr resampler directly into
// a single preallocated output buffer. Bounds peak scratch memory to one
// chunk plus soxr filter state, rather than materializing the full native-SR
// input as the single-shot path does.

#include "audio_backend.h"
#include "audio_stream.h"
#include "internal_utils.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#if MLX_AUDIO_IO_ENABLE_SOXR
#include <soxr.h>
#endif

namespace mlx_audio {

namespace {

#if MLX_AUDIO_IO_ENABLE_SOXR

soxr_quality_spec_t make_soxr_quality(const std::string& quality) {
    if (quality == "soxr_vhq") {
        return soxr_quality_spec(SOXR_VHQ, 0);
    }
    // default to HQ for anything else (including "default")
    return soxr_quality_spec(SOXR_HQ, 0);
}

// Mix an interleaved [frames, channels] chunk to mono in-place into dst.
void mix_to_mono(const float* src, int64_t frames, int channels, float* dst) {
    if (channels == 1) {
        std::memcpy(dst, src, static_cast<size_t>(frames) * sizeof(float));
        return;
    }
    const float scale = 1.0f / static_cast<float>(channels);
    for (int64_t i = 0; i < frames; ++i) {
        float s = 0.0f;
        const float* row = src + i * channels;
        for (int c = 0; c < channels; ++c) {
            s += row[c];
        }
        dst[i] = s * scale;
    }
}

// Stream the file at native SR, resample chunk-by-chunk through soxr,
// write directly into a preallocated output buffer.
std::pair<mlx::core::array, int> load_streaming_resample_impl(
    const std::string& path,
    int target_sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& quality) {
    internal::check_file_exists(path);

    auto info = backend_get_info(path);
    const int native_sr = info.sample_rate;
    const int native_channels = info.channels;
    const int64_t total_frames_native = info.frames;

    // If the caller actually asked for the native SR, the single-shot path
    // is both simpler and already streaming-sized (direct fread for WAV,
    // chunked ExtAudioFile/libav otherwise). Fall through to it.
    if (native_sr == target_sr) {
        return backend_load_audio(
            path, std::optional<int>(target_sr), offset, duration,
            mono, layout, dtype, "default");
    }

    // Compute the native-SR window that will be read.
    int64_t start_frame_native =
        static_cast<int64_t>(std::floor(offset * native_sr));
    if (start_frame_native < 0) start_frame_native = 0;
    if (total_frames_native > 0 && start_frame_native >= total_frames_native) {
        return tensor_utils::make_empty_audio_result(
            target_sr, native_channels, mono, layout, dtype);
    }

    int64_t frames_native;
    if (duration.has_value()) {
        frames_native =
            static_cast<int64_t>(std::floor(duration.value() * native_sr));
        if (frames_native < 0) frames_native = 0;
        if (total_frames_native > 0) {
            const int64_t remaining = total_frames_native - start_frame_native;
            frames_native = std::min(frames_native, remaining);
        }
    } else if (total_frames_native > 0) {
        frames_native = total_frames_native - start_frame_native;
    } else {
        // Unknown total frames (e.g. some MP3s) — we'll read until EOF.
        frames_native = -1;
    }

    if (frames_native == 0) {
        return tensor_utils::make_empty_audio_result(
            target_sr, native_channels, mono, layout, dtype);
    }

    const int resample_channels = mono ? 1 : native_channels;

    // Preallocate the output buffer. For known input length, size it from the
    // ratio plus a small margin for filter ringing; for unknown length fall
    // back to a growable vector (rare path — compressed files without length
    // metadata).
    const bool have_length = frames_native > 0;
    int64_t out_capacity = 0;
    std::unique_ptr<float, void(*)(void*)> out_buf(
        nullptr, &internal::aligned_free);

    if (have_length) {
        const int64_t estimated = static_cast<int64_t>(std::ceil(
            static_cast<double>(frames_native) * target_sr / native_sr));
        out_capacity = estimated + 256;  // ringing + rounding margin
        size_t out_bytes =
            static_cast<size_t>(std::max<int64_t>(out_capacity, 1)) *
            static_cast<size_t>(resample_channels) * sizeof(float);
        out_buf.reset(static_cast<float*>(internal::aligned_alloc_64(out_bytes)));
    }
    std::vector<float> out_fallback;  // used only when !have_length

    // Create soxr
    soxr_quality_spec_t q_spec = make_soxr_quality(quality);
    soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
    soxr_runtime_spec_t runtime_spec = soxr_runtime_spec(1);
    soxr_error_t soxr_err = nullptr;
    soxr_t soxr = soxr_create(
        static_cast<double>(native_sr),
        static_cast<double>(target_sr),
        static_cast<unsigned>(resample_channels),
        &soxr_err,
        &io_spec,
        &q_spec,
        &runtime_spec);
    if (soxr == nullptr || soxr_err != nullptr) {
        throw std::runtime_error(
            std::string("Failed to create libsoxr streaming resampler: ") +
            (soxr_err ? soxr_err : "unknown"));
    }
    struct SoxrGuard {
        soxr_t handle;
        ~SoxrGuard() { if (handle) soxr_delete(handle); }
    };
    SoxrGuard soxr_guard{soxr};

    // Use the existing cross-platform stream reader at native SR.
    // channels_last, no mono (we handle per-chunk mixdown here).
    constexpr int kChunkFrames = 65536;
    AudioStreamReader reader(
        path, kChunkFrames,
        /*sr*/ std::optional<int>(native_sr),
        /*mono*/ false,
        offset,
        duration,
        /*dtype*/ "float32");

    // Per-chunk mono scratch, allocated lazily if needed.
    std::unique_ptr<float, void(*)(void*)> mono_scratch(
        nullptr, &internal::aligned_free);
    auto ensure_mono_scratch = [&](int64_t frames) {
        if (!mono && native_channels == 1) return;  // unused
        if (!mono_scratch) {
            mono_scratch.reset(static_cast<float*>(
                internal::aligned_alloc_64(
                    static_cast<size_t>(kChunkFrames) * sizeof(float))));
        }
        (void)frames;
    };

    int64_t total_out = 0;  // used with have_length path

    while (true) {
        auto chunk_result = reader.read_chunk();
        mlx::core::array& chunk = chunk_result.first;
        const int64_t chunk_frames = chunk.shape(0);
        if (chunk_frames == 0) break;

        // Ensure data is materialized and contiguous.
        mlx::core::eval(chunk);
        const float* chunk_data = chunk.data<float>();
        const float* in_ptr = chunk_data;

        if (mono && native_channels > 1) {
            ensure_mono_scratch(chunk_frames);
            mix_to_mono(chunk_data, chunk_frames, native_channels,
                        mono_scratch.get());
            in_ptr = mono_scratch.get();
        } else if (mono && native_channels == 1) {
            // already mono
            in_ptr = chunk_data;
        }

        size_t out_done = 0;
        if (have_length) {
            const int64_t room = out_capacity - total_out;
            if (room > 0) {
                soxr_err = soxr_process(
                    soxr,
                    in_ptr, static_cast<size_t>(chunk_frames), nullptr,
                    out_buf.get() + total_out * resample_channels,
                    static_cast<size_t>(room),
                    &out_done);
                if (soxr_err != nullptr) {
                    throw std::runtime_error(
                        std::string("libsoxr streaming process failed: ") +
                        soxr_err);
                }
                total_out += static_cast<int64_t>(out_done);
            }
        } else {
            // Grow into out_fallback. Allocate a per-iteration staging buffer
            // sized to the maximum possible soxr output for this chunk.
            const int64_t stage_capacity =
                static_cast<int64_t>(std::ceil(
                    static_cast<double>(chunk_frames) * target_sr / native_sr)) + 256;
            std::unique_ptr<float, void(*)(void*)> stage(
                static_cast<float*>(internal::aligned_alloc_64(
                    static_cast<size_t>(stage_capacity) *
                    resample_channels * sizeof(float))),
                &internal::aligned_free);
            soxr_err = soxr_process(
                soxr,
                in_ptr, static_cast<size_t>(chunk_frames), nullptr,
                stage.get(), static_cast<size_t>(stage_capacity),
                &out_done);
            if (soxr_err != nullptr) {
                throw std::runtime_error(
                    std::string("libsoxr streaming process failed: ") +
                    soxr_err);
            }
            if (out_done > 0) {
                const size_t cur = out_fallback.size();
                out_fallback.resize(cur + out_done * resample_channels);
                std::memcpy(out_fallback.data() + cur, stage.get(),
                            out_done * resample_channels * sizeof(float));
            }
        }
    }

    // Flush trailing filter state.
    for (;;) {
        size_t out_done = 0;
        if (have_length) {
            const int64_t room = out_capacity - total_out;
            if (room <= 0) break;
            soxr_err = soxr_process(
                soxr, nullptr, 0, nullptr,
                out_buf.get() + total_out * resample_channels,
                static_cast<size_t>(room),
                &out_done);
            if (soxr_err != nullptr) {
                throw std::runtime_error(
                    std::string("libsoxr streaming flush failed: ") + soxr_err);
            }
            total_out += static_cast<int64_t>(out_done);
            if (out_done == 0) break;
        } else {
            constexpr size_t kFlushChunk = 8192;
            std::unique_ptr<float, void(*)(void*)> stage(
                static_cast<float*>(internal::aligned_alloc_64(
                    kFlushChunk * resample_channels * sizeof(float))),
                &internal::aligned_free);
            soxr_err = soxr_process(
                soxr, nullptr, 0, nullptr,
                stage.get(), kFlushChunk, &out_done);
            if (soxr_err != nullptr) {
                throw std::runtime_error(
                    std::string("libsoxr streaming flush failed: ") + soxr_err);
            }
            if (out_done == 0) break;
            const size_t cur = out_fallback.size();
            out_fallback.resize(cur + out_done * resample_channels);
            std::memcpy(out_fallback.data() + cur, stage.get(),
                        out_done * resample_channels * sizeof(float));
        }
    }

    float* final_buf = nullptr;
    int64_t final_frames = 0;
    if (have_length) {
        final_buf = out_buf.release();
        final_frames = total_out;
    } else {
        // Transfer fallback vector contents to an aligned buffer so
        // wrap_interleaved_audio_buffer can take ownership of it.
        final_frames =
            static_cast<int64_t>(out_fallback.size()) / resample_channels;
        size_t bytes =
            static_cast<size_t>(std::max<int64_t>(final_frames, 1)) *
            resample_channels * sizeof(float);
        final_buf = static_cast<float*>(internal::aligned_alloc_64(bytes));
        if (final_frames > 0) {
            std::memcpy(final_buf, out_fallback.data(),
                        static_cast<size_t>(final_frames) * resample_channels *
                        sizeof(float));
        }
    }

    // wrap_interleaved_audio_buffer handles layout / dtype conversion. mono
    // is already applied, so pass mono=false.
    return tensor_utils::wrap_interleaved_audio_buffer(
        final_buf, final_frames, resample_channels, target_sr,
        /*mono*/ false, layout, dtype);
}

#endif  // MLX_AUDIO_IO_ENABLE_SOXR

}  // namespace

std::pair<mlx::core::array, int> backend_load_audio_streaming_resample(
    const std::string& path,
    int target_sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& quality) {
#if MLX_AUDIO_IO_ENABLE_SOXR
    return load_streaming_resample_impl(
        path, target_sr, offset, duration, mono, layout, dtype, quality);
#else
    (void)path; (void)target_sr; (void)offset; (void)duration;
    (void)mono; (void)layout; (void)dtype; (void)quality;
    throw value_error(
        "low_memory streaming resample requires libsoxr support; "
        "this build does not include libsoxr");
#endif
}

}  // namespace mlx_audio
