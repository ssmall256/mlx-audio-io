#pragma once

#include "audio_io.h"

namespace mlx_audio {

AudioFileInfo backend_get_info(const std::string& path);

std::pair<mlx::core::array, int> backend_load_audio(
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& resample_quality);

void backend_save_audio(
    const std::string& path,
    mlx::core::array audio,
    int sr,
    const std::string& layout,
    const std::string& encoding,
    const std::string& bitrate,
    bool clip);

mlx::core::array backend_resample_audio(
    mlx::core::array audio,
    int in_sr,
    int out_sr,
    const std::string& quality);

// Streaming (bounded-scratch) load + resample. Reads the file in chunks at
// native SR and pushes each chunk through a stateful libsoxr resampler into
// a single preallocated output buffer. Requires libsoxr.
std::pair<mlx::core::array, int> backend_load_audio_streaming_resample(
    const std::string& path,
    int target_sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& quality);

}  // namespace mlx_audio
