#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace mlx_audio {

/// Forward-declared opaque type wrapping lame_global_flags.
struct Mp3EncImpl;

/// RAII wrapper for LAME MP3 encoder.
/// Non-copyable, movable.
class ScopedMp3Encoder {
  public:
    ScopedMp3Encoder();
    ~ScopedMp3Encoder();

    ScopedMp3Encoder(const ScopedMp3Encoder&) = delete;
    ScopedMp3Encoder& operator=(const ScopedMp3Encoder&) = delete;
    ScopedMp3Encoder(ScopedMp3Encoder&&) noexcept;
    ScopedMp3Encoder& operator=(ScopedMp3Encoder&&) noexcept;

    /// Configure and initialize the encoder. Must be called before encode().
    /// bitrate_kbps=0 means use LAME's VBR default (~190 kbps).
    /// Throws on error.
    void init(int sample_rate, int channels, int bitrate_kbps = 0);

    /// Encode interleaved float32 PCM frames.
    /// Appends encoded MP3 bytes to internal buffer.
    /// pcm: pointer to interleaved float32 samples in [-1, 1]
    /// num_frames: number of frames (not samples)
    void encode(const float* pcm, int num_frames);

    /// Flush remaining encoder buffers and finalize.
    void flush();

    /// Get the accumulated MP3 output data.
    const unsigned char* data() const;
    size_t size() const;

  private:
    std::unique_ptr<Mp3EncImpl> impl_;
};

}  // namespace mlx_audio
