#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace mlx_audio {

/// Custom exception types for nanobind mapping.
class file_not_found_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class value_error : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/// Metadata about an audio file.
struct AudioFileInfo {
    int64_t frames;
    int sample_rate;
    int channels;
    double duration;
    std::string subtype;
    std::string container;
};

}  // namespace mlx_audio

#include "audio_buffer.h"

namespace mlx_audio {

/// Get metadata about an audio file without decoding samples.
AudioFileInfo get_info(const std::string& path);

/// Load audio from a file into an AudioBuffer.
/// Returns (audio_buffer, output_sample_rate).
std::pair<AudioBuffer, int> load_audio(
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& resample_quality);

/// Save audio samples from raw float memory to an audio file.
void save_audio(
    const std::string& path,
    const float* data,
    int64_t frames,
    int channels,
    int64_t stride_frame,
    int64_t stride_chan,
    int sr,
    const std::string& layout,
    const std::string& encoding,
    const std::string& bitrate,
    bool clip,
    const std::string& flac_compression);

/// Resample in-memory audio samples to a different sample rate.
AudioBuffer resample_audio(
    const float* in_data,
    int64_t in_frames,
    int channels,
    int64_t stride_frame,
    int64_t stride_chan,
    int in_sr,
    int out_sr,
    const std::string& quality);

}  // namespace mlx_audio
