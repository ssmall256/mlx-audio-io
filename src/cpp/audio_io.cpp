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
    bool clip,
    const std::string& flac_compression) {
    if (flac_compression != "default" && flac_compression != "fast") {
        throw value_error(
            "flac_compression must be 'default' or 'fast', got '" + flac_compression + "'");
    }
    backend_save_audio(path, std::move(audio), sr, layout, encoding, bitrate, clip, flac_compression);
}

mlx::core::array resample_audio(
    mlx::core::array audio,
    int in_sr,
    int out_sr,
    const std::string& quality) {
    return backend_resample_audio(std::move(audio), in_sr, out_sr, quality);
}

}  // namespace mlx_audio
