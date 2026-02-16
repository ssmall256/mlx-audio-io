#include "linux_libav_probe.h"

#if defined(__linux__) && defined(MLX_AUDIO_IO_ENABLE_LIBAV)
extern "C" {
#include <libavcodec/version.h>
#include <libavformat/version.h>
#include <libavutil/version.h>
#include <libswresample/version.h>
}
#endif

namespace mlx_audio {
namespace {

std::string version_to_string(unsigned int version_int) {
    unsigned int major = (version_int >> 16) & 0xFFu;
    unsigned int minor = (version_int >> 8) & 0xFFu;
    unsigned int micro = version_int & 0xFFu;
    return std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(micro);
}

}  // namespace

LinuxLibavProbeResult linux_libav_probe() {
    LinuxLibavProbeResult out;
#if defined(__linux__) && defined(MLX_AUDIO_IO_ENABLE_LIBAV)
    out.libav_enabled = true;
    out.avformat_version = version_to_string(LIBAVFORMAT_VERSION_INT);
    out.avcodec_version = version_to_string(LIBAVCODEC_VERSION_INT);
    out.avutil_version = version_to_string(LIBAVUTIL_VERSION_INT);
    out.swresample_version = version_to_string(LIBSWRESAMPLE_VERSION_INT);
#endif
    return out;
}

}  // namespace mlx_audio

