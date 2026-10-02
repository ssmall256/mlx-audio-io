#pragma once

#include <string>
#include <utility>

#include "audio_buffer.h"
#include "internal_utils.h"

namespace mlx_audio::tensor_utils {

/// Return an empty AudioBuffer with the right shape/dtype for a result tuple.
inline std::pair<AudioBuffer, int> make_empty_audio_result(
    int out_sr,
    int channels,
    bool mono,
    const std::string& layout,
    const std::string& dtype) {
    int out_channels = mono ? 1 : channels;
    return {AudioBuffer::make_empty(out_channels, layout, dtype), out_sr};
}

/// Apply mono mixdown, channels_first deinterleave, and wrap in AudioBuffer.
/// Takes ownership of buffer.
inline std::pair<AudioBuffer, int> wrap_interleaved_audio_buffer(
    float* buffer,
    int64_t actual_frames,
    int channels,
    int out_sr,
    bool mono,
    const std::string& layout,
    const std::string& dtype) {
    int out_channels = channels;
    if (mono && channels > 1) {
        buffer = internal::mono_mixdown(buffer, channels, actual_frames);
        out_channels = 1;
    }

    if (layout == "channels_first" && out_channels > 1) {
        size_t planar_bytes = static_cast<size_t>(actual_frames) * out_channels * sizeof(float);
        float* planar_buf = static_cast<float*>(internal::aligned_alloc_64(planar_bytes));

        for (int c = 0; c < out_channels; ++c) {
            internal::strided_copy(
                buffer + c,
                out_channels,
                planar_buf + c * actual_frames,
                1,
                static_cast<int>(actual_frames));
        }

        std::free(buffer);
        buffer = planar_buf;
    }

    int64_t d0, d1, s0, s1;
    if (layout == "channels_first" && out_channels > 1) {
        d0 = out_channels;
        d1 = actual_frames;
        s0 = actual_frames;
        s1 = 1;
    } else if (out_channels == 1 && layout == "channels_first") {
        d0 = 1;
        d1 = actual_frames;
        s0 = actual_frames;
        s1 = 1;
    } else {
        d0 = actual_frames;
        d1 = out_channels;
        s0 = out_channels;
        s1 = 1;
    }

    size_t total_bytes = static_cast<size_t>(actual_frames) * out_channels * sizeof(float);
    auto storage = std::make_shared<AudioStorage>(buffer, total_bytes, /*take_ownership=*/true);
    AudioBuffer audio_buf(std::move(storage), d0, d1, s0, s1, 2, "float32");

    return {std::move(audio_buf), out_sr};
}

}  // namespace mlx_audio::tensor_utils
