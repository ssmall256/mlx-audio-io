#pragma once

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <AudioToolbox/AudioToolbox.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

#include "audio_io.h"
#include "raii_audio.h"

namespace mlx_audio {
namespace internal {

/// Allocate 64-byte aligned memory.
inline void* aligned_alloc_64(size_t bytes) {
    void* ptr = nullptr;
    if (bytes == 0) return nullptr;
    int rc = posix_memalign(&ptr, 64, bytes);
    if (rc != 0 || ptr == nullptr) {
        throw std::runtime_error("Failed to allocate " + std::to_string(bytes) +
                                 " bytes of aligned memory");
    }
    return ptr;
}

/// Free function suitable for mlx::core::array deleter.
inline void aligned_free(void* ptr) { std::free(ptr); }

/// Strided float copy using cblas_scopy.
inline void strided_copy(const float* src, int src_stride,
                         float* dst, int dst_stride, int n) {
#if defined(__APPLE__)
    cblas_scopy(n, src, src_stride, dst, dst_stride);
#else
    for (int i = 0; i < n; ++i) {
        dst[i * dst_stride] = src[i * src_stride];
    }
#endif
}

#if defined(__APPLE__)
/// Create a CFURLRef from a file path string.
inline ScopedCFURL make_url(const std::string& path) {
    CFStringRef cf_path = CFStringCreateWithCString(
        kCFAllocatorDefault, path.c_str(), kCFStringEncodingUTF8);
    if (!cf_path) {
        throw std::runtime_error("Failed to create CFString from path: " + path);
    }
    CFURLRef url = CFURLCreateWithFileSystemPath(
        kCFAllocatorDefault, cf_path, kCFURLPOSIXPathStyle, false);
    CFRelease(cf_path);
    if (!url) {
        throw std::runtime_error("Failed to create CFURL from path: " + path);
    }
    return ScopedCFURL(url);
}
#endif

/// Check that a file exists; throw file_not_found_error if not.
inline void check_file_exists(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        throw file_not_found_error("File not found: " + path);
    }
}

#if defined(__APPLE__)
/// Set up the client format for reading: float32, interleaved, at target_sr.
inline AudioStreamBasicDescription make_client_format(int target_sr, int channels) {
    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = static_cast<Float64>(target_sr);
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags =
        kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    fmt.mBitsPerChannel = 32;
    fmt.mChannelsPerFrame = static_cast<UInt32>(channels);
    fmt.mBytesPerFrame = static_cast<UInt32>(channels * sizeof(float));
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerPacket = fmt.mBytesPerFrame;
    return fmt;
}
#endif

/// Mix interleaved multi-channel buffer down to mono.
/// Returns a new mono buffer; caller must free the original buffer.
inline float* mono_mixdown(float* buffer, int channels, int64_t frames) {
    size_t mono_bytes = static_cast<size_t>(frames) * sizeof(float);
    float* mono_buf = static_cast<float*>(aligned_alloc_64(
        mono_bytes > 0 ? mono_bytes : 64));

    if (channels == 2) {
#if defined(__APPLE__)
        // Stereo fast path: (L + R) / 2 using vDSP
        vDSP_vadd(
            buffer, 2,           // L channel, stride 2
            buffer + 1, 2,       // R channel, stride 2
            mono_buf, 1,         // output, stride 1
            static_cast<vDSP_Length>(frames));
        float half = 0.5f;
        vDSP_vsmul(
            mono_buf, 1, &half,
            mono_buf, 1,
            static_cast<vDSP_Length>(frames));
#else
        for (int64_t i = 0; i < frames; ++i) {
            mono_buf[i] = 0.5f * (buffer[2 * i] + buffer[2 * i + 1]);
        }
#endif
    } else {
#if defined(__APPLE__)
        // General case: average all channels
        strided_copy(buffer, channels, mono_buf, 1,
                     static_cast<int>(frames));
        for (int c = 1; c < channels; ++c) {
            vDSP_vadd(
                mono_buf, 1,
                buffer + c, channels,
                mono_buf, 1,
                static_cast<vDSP_Length>(frames));
        }
        float scale = 1.0f / static_cast<float>(channels);
        vDSP_vsmul(
            mono_buf, 1, &scale,
            mono_buf, 1,
            static_cast<vDSP_Length>(frames));
#else
        for (int64_t i = 0; i < frames; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < channels; ++c) {
                sum += buffer[i * channels + c];
            }
            mono_buf[i] = sum / static_cast<float>(channels);
        }
#endif
    }

    std::free(buffer);
    return mono_buf;
}

/// Optimal read size for compressed formats.
constexpr uint32_t kReadChunkFrames = 65536;

}  // namespace internal
}  // namespace mlx_audio
