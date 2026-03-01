#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <mlx/mlx.h>

#include "mp3_decoder.h"
#include "raii_audio.h"

namespace mlx_audio {

/// Chunked audio file reader that yields fixed-size mlx arrays.
/// Not thread-safe — do not share instances across threads without
/// external synchronization.
class AudioStreamReader {
  public:
    AudioStreamReader(const std::string& path, int chunk_frames,
                      std::optional<int> sr, bool mono,
                      double offset,
                      std::optional<double> duration,
                      const std::string& dtype);
    ~AudioStreamReader();
    AudioStreamReader(const AudioStreamReader&) = delete;
    AudioStreamReader& operator=(const AudioStreamReader&) = delete;
    AudioStreamReader(AudioStreamReader&& other) noexcept;
    AudioStreamReader& operator=(AudioStreamReader&& other) noexcept;

    /// Read the next chunk. Returns (array, sample_rate).
    /// The final chunk may have fewer frames than chunk_frames().
    /// Returns a 0-frame array once EOF is reached.
    std::pair<mlx::core::array, int> read_chunk();

    bool at_eof() const { return eof_; }
    int sample_rate() const { return out_sr_; }
    int channels() const { return out_channels_; }
    int chunk_frames() const { return chunk_frames_; }
    int64_t frames_read() const { return frames_read_; }

  private:
#if defined(__APPLE__)
    ScopedExtAudioFile ext_file_;
#else
    bool is_wav_ = false;
    std::FILE* wav_file_ = nullptr;
    int wav_bits_per_sample_ = 0;
    int wav_format_tag_ = 0;
    int64_t wav_total_frames_ = 0;
    bool is_predecoded_ = false;
    std::optional<mlx::core::array> predecoded_audio_;
    const float* predecoded_data_ = nullptr;
    int64_t predecoded_total_frames_ = 0;

    bool is_libav_stream_ = false;
    void* libav_stream_ = nullptr;
#endif
    int chunk_frames_;
    int out_sr_;
    int native_channels_;
    int out_channels_;
    bool mono_;
    bool eof_ = false;
    int64_t frames_read_ = 0;
    int64_t max_frames_to_emit_ = -1;
    bool is_mp3_ = false;
    std::optional<ScopedMp3Decoder> mp3_dec_;
    std::string dtype_;
};

}  // namespace mlx_audio
