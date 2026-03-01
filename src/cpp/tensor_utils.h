#pragma once

#include <string>
#include <utility>

#include "internal_utils.h"

namespace mlx_audio::tensor_utils {

/// Return an empty array with the right shape/dtype for a result tuple.
inline std::pair<mlx::core::array, int> make_empty_audio_result(
    int out_sr,
    int channels,
    bool mono,
    const std::string& layout,
    const std::string& dtype) {
    int out_channels = mono ? 1 : channels;
    auto target_dtype = (dtype == "float16") ? mlx::core::float16 : mlx::core::float32;
    mlx::core::Shape shape;
    if (layout == "channels_last") {
        shape = {0, static_cast<int32_t>(out_channels)};
    } else {
        shape = {static_cast<int32_t>(out_channels), 0};
    }
    return {mlx::core::array(std::initializer_list<int>{}, shape, target_dtype), out_sr};
}

/// Apply mono mixdown, channels_first deinterleave, shape, and dtype to
/// an interleaved float32 buffer. Takes ownership of buffer.
inline std::pair<mlx::core::array, int> wrap_interleaved_audio_buffer(
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

    mlx::core::Shape shape;
    if (layout == "channels_first" && out_channels > 1) {
        shape = {static_cast<int32_t>(out_channels), static_cast<int32_t>(actual_frames)};
    } else if (out_channels == 1 && layout == "channels_first") {
        shape = {1, static_cast<int32_t>(actual_frames)};
    } else {
        shape = {static_cast<int32_t>(actual_frames), static_cast<int32_t>(out_channels)};
    }

    // Copy into MLX-owned storage, then release native buffer immediately.
    // This avoids cross-runtime lifetime issues when returning many chunks.
    auto arr = mlx::core::array(buffer, std::move(shape), mlx::core::float32);
    std::free(buffer);

    if (dtype == "float16") {
        arr = mlx::core::astype(arr, mlx::core::float16);
    }

    return {std::move(arr), out_sr};
}

}  // namespace mlx_audio::tensor_utils
