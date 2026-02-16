#include "audio_backend.h"

#include <utility>

namespace mlx_audio {

AudioFileInfo get_info(const std::string& path) {
    return backend_get_info(path);
}

std::pair<mlx::core::array, int> load_audio(
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& resample_quality) {
    return backend_load_audio(path, sr, offset, duration, mono, layout, dtype, resample_quality);
}

void save_audio(
    const std::string& path,
    mlx::core::array audio,
    int sr,
    const std::string& layout,
    const std::string& encoding,
    const std::string& bitrate,
    bool clip) {
    backend_save_audio(path, std::move(audio), sr, layout, encoding, bitrate, clip);
}

}  // namespace mlx_audio
