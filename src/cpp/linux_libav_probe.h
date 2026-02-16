#pragma once

#include <string>

namespace mlx_audio {

struct LinuxLibavProbeResult {
    bool libav_enabled = false;
    std::string avformat_version;
    std::string avcodec_version;
    std::string avutil_version;
    std::string swresample_version;
};

LinuxLibavProbeResult linux_libav_probe();

}  // namespace mlx_audio

