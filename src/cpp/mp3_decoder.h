#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace mlx_audio {

/// Forward-declared opaque type wrapping mp3dec_ex_t.
struct Mp3DecImpl;

/// RAII wrapper for minimp3 extended decoder.
/// Non-copyable, movable.
class ScopedMp3Decoder {
  public:
    ScopedMp3Decoder();
    ~ScopedMp3Decoder();

    ScopedMp3Decoder(const ScopedMp3Decoder&) = delete;
    ScopedMp3Decoder& operator=(const ScopedMp3Decoder&) = delete;
    ScopedMp3Decoder(ScopedMp3Decoder&&) noexcept;
    ScopedMp3Decoder& operator=(ScopedMp3Decoder&&) noexcept;

    /// Open an MP3 file. Throws on error.
    void open(const std::string& path);

    /// Native sample rate.
    int sample_rate() const;

    /// Number of channels.
    int channels() const;

    /// Total decoded frames (samples / channels).
    int64_t total_frames() const;

    /// Seek to a sample position (in interleaved samples, not frames).
    void seek(int64_t sample_pos);

    /// Read up to num_samples interleaved float32 samples into buf.
    /// Returns the number of samples actually read.
    int64_t read(float* buf, int64_t num_samples);

  private:
    std::unique_ptr<Mp3DecImpl> impl_;
};

/// Check if a path has an .mp3 extension (case-insensitive).
bool is_mp3_path(const std::string& path);

}  // namespace mlx_audio
