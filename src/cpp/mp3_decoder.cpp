#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "minimp3_ex.h"

#include "mp3_decoder.h"

#include <stdexcept>

namespace mlx_audio {

struct Mp3DecImpl {
    mp3dec_ex_t dec = {};
    bool opened = false;
};

ScopedMp3Decoder::ScopedMp3Decoder()
    : impl_(std::make_unique<Mp3DecImpl>()) {}

ScopedMp3Decoder::~ScopedMp3Decoder() {
    if (impl_ && impl_->opened) {
        mp3dec_ex_close(&impl_->dec);
    }
}

ScopedMp3Decoder::ScopedMp3Decoder(ScopedMp3Decoder&&) noexcept = default;
ScopedMp3Decoder& ScopedMp3Decoder::operator=(ScopedMp3Decoder&&) noexcept = default;

void ScopedMp3Decoder::open(const std::string& path) {
    int err = mp3dec_ex_open(&impl_->dec, path.c_str(), MP3D_SEEK_TO_SAMPLE);
    if (err != 0) {
        throw std::runtime_error(
            "minimp3: failed to open '" + path + "' (error " + std::to_string(err) + ")");
    }
    impl_->opened = true;
}

int ScopedMp3Decoder::sample_rate() const {
    return impl_->dec.info.hz;
}

int ScopedMp3Decoder::channels() const {
    return impl_->dec.info.channels;
}

int64_t ScopedMp3Decoder::total_frames() const {
    if (impl_->dec.info.channels == 0) return 0;
    return static_cast<int64_t>(impl_->dec.samples) / impl_->dec.info.channels;
}

void ScopedMp3Decoder::seek(int64_t sample_pos) {
    int err = mp3dec_ex_seek(&impl_->dec, static_cast<uint64_t>(sample_pos));
    if (err != 0) {
        throw std::runtime_error(
            "minimp3: seek failed (error " + std::to_string(err) + ")");
    }
}

int64_t ScopedMp3Decoder::read(float* buf, int64_t num_samples) {
    size_t got = mp3dec_ex_read(&impl_->dec, buf, static_cast<size_t>(num_samples));
    return static_cast<int64_t>(got);
}

bool is_mp3_path(const std::string& path) {
    if (path.size() < 4) return false;
    auto ext = path.substr(path.size() - 4);
    return (ext == ".mp3" || ext == ".MP3" || ext == ".Mp3" || ext == ".mP3");
}

}  // namespace mlx_audio
