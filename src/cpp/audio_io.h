#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <mlx/mlx.h>

namespace mlx_audio {

/// Metadata about an audio file.
struct AudioFileInfo {
    int64_t frames;
    int sample_rate;
    int channels;
    double duration;
    std::string subtype;
    std::string container;
};

/// Custom exception types for nanobind mapping.
class file_not_found_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class value_error : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/// Get metadata about a WAV file without decoding samples.
AudioFileInfo get_info(const std::string& path);

/// Load audio from a WAV file into an mlx::core::array.
/// Returns (array, output_sample_rate).
std::pair<mlx::core::array, int> load_audio(
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& resample_quality);

/// Save an mlx::core::array to an audio file.
void save_audio(
    const std::string& path,
    mlx::core::array audio,
    int sr,
    const std::string& layout,
    const std::string& encoding,
    const std::string& bitrate,
    bool clip);

/// Resample an in-memory audio array to a different sample rate.
mlx::core::array resample_audio(
    mlx::core::array audio,
    int in_sr,
    int out_sr,
    const std::string& quality);

}  // namespace mlx_audio
