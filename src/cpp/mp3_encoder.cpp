#include "mp3_encoder.h"
#include "lame.h"

#include <stdexcept>
#include <vector>

namespace mlx_audio {

struct Mp3EncImpl {
    lame_global_flags* gfp = nullptr;
    std::vector<unsigned char> mp3_buffer;
    int channels = 0;
    bool initialized = false;
};

ScopedMp3Encoder::ScopedMp3Encoder()
    : impl_(std::make_unique<Mp3EncImpl>()) {}

ScopedMp3Encoder::~ScopedMp3Encoder() {
    if (impl_ && impl_->gfp) {
        lame_close(impl_->gfp);
    }
}

ScopedMp3Encoder::ScopedMp3Encoder(ScopedMp3Encoder&&) noexcept = default;
ScopedMp3Encoder& ScopedMp3Encoder::operator=(ScopedMp3Encoder&&) noexcept = default;

void ScopedMp3Encoder::init(int sample_rate, int channels, int bitrate_kbps) {
    impl_->gfp = lame_init();
    if (!impl_->gfp) {
        throw std::runtime_error("lame_init() failed");
    }

    lame_set_in_samplerate(impl_->gfp, sample_rate);
    lame_set_out_samplerate(impl_->gfp, sample_rate);
    lame_set_num_channels(impl_->gfp, channels);

    if (channels == 1) {
        lame_set_mode(impl_->gfp, MONO);
    } else {
        lame_set_mode(impl_->gfp, JOINT_STEREO);
    }

    if (bitrate_kbps > 0) {
        // CBR when explicit bitrate is requested
        lame_set_VBR(impl_->gfp, vbr_off);
        lame_set_brate(impl_->gfp, bitrate_kbps);
    } else {
        // Default: VBR quality ~2 (high quality, ~190 kbps)
        lame_set_VBR(impl_->gfp, vbr_default);
        lame_set_VBR_quality(impl_->gfp, 2.0f);
    }

    // Near-best quality, good speed trade-off
    lame_set_quality(impl_->gfp, 2);

    int ret = lame_init_params(impl_->gfp);
    if (ret < 0) {
        lame_close(impl_->gfp);
        impl_->gfp = nullptr;
        throw std::runtime_error(
            "lame_init_params() failed (error " + std::to_string(ret) + ")");
    }
    impl_->channels = channels;
    impl_->initialized = true;
}

void ScopedMp3Encoder::encode(const float* pcm, int num_frames) {
    if (!impl_->initialized) {
        throw std::runtime_error("MP3 encoder not initialized");
    }
    // LAME worst case: 1.25 * num_frames + 7200
    int mp3buf_size = static_cast<int>(1.25 * num_frames) + 7200;
    size_t old_size = impl_->mp3_buffer.size();
    impl_->mp3_buffer.resize(old_size + mp3buf_size);

    int bytes_written;
    if (impl_->channels == 1) {
        // Mono: use non-interleaved API (pass same buffer as L and R)
        bytes_written = lame_encode_buffer_ieee_float(
            impl_->gfp,
            pcm,
            pcm,  // ignored for mono mode
            num_frames,
            impl_->mp3_buffer.data() + old_size,
            mp3buf_size);
    } else {
        // Stereo: use interleaved API
        bytes_written = lame_encode_buffer_interleaved_ieee_float(
            impl_->gfp,
            pcm,
            num_frames,
            impl_->mp3_buffer.data() + old_size,
            mp3buf_size);
    }

    if (bytes_written < 0) {
        throw std::runtime_error(
            "lame_encode_buffer failed (error " + std::to_string(bytes_written) + ")");
    }
    impl_->mp3_buffer.resize(old_size + bytes_written);
}

void ScopedMp3Encoder::flush() {
    if (!impl_->initialized) return;

    // 7200 bytes is LAME's recommended minimum for flush
    size_t old_size = impl_->mp3_buffer.size();
    impl_->mp3_buffer.resize(old_size + 7200);

    int bytes_written = lame_encode_flush(
        impl_->gfp,
        impl_->mp3_buffer.data() + old_size,
        7200);

    if (bytes_written < 0) {
        throw std::runtime_error(
            "lame_encode_flush failed (error " + std::to_string(bytes_written) + ")");
    }
    impl_->mp3_buffer.resize(old_size + bytes_written);
}

const unsigned char* ScopedMp3Encoder::data() const {
    return impl_->mp3_buffer.data();
}

size_t ScopedMp3Encoder::size() const {
    return impl_->mp3_buffer.size();
}

}  // namespace mlx_audio
