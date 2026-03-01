#include "audio_stream.h"
#include "audio_backend.h"
#include "internal_utils.h"
#include "tensor_utils.h"

#if defined(__APPLE__)
#include "osstatus.h"

#include <AudioToolbox/AudioToolbox.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#if !defined(__APPLE__)
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}
#endif

namespace mlx_audio {

using internal::aligned_alloc_64;
using internal::check_file_exists;
using internal::kReadChunkFrames;

#if defined(__APPLE__)
using internal::make_client_format;
using internal::make_url;
#endif

#if !defined(__APPLE__)
struct LibavStreamState {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    SwrContext* swr_ctx = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;

    int stream_index = -1;
    int native_sr = 0;
    int out_sr = 0;
    int native_channels = 0;

    bool flush_sent = false;
    bool decoder_eof = false;

    std::vector<float> pending_interleaved;
    size_t pending_pos = 0;

    ~LibavStreamState() {
        if (frame) {
            av_frame_free(&frame);
            frame = nullptr;
        }
        if (packet) {
            av_packet_free(&packet);
            packet = nullptr;
        }
        if (swr_ctx) {
            swr_free(&swr_ctx);
            swr_ctx = nullptr;
        }
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
            codec_ctx = nullptr;
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
            format_ctx = nullptr;
        }
    }
};
#endif

namespace {

#if !defined(__APPLE__)
struct WavInfo {
    int sample_rate;
    int channels;
    int bits_per_sample;
    int format_tag;       // 1 = PCM, 3 = IEEE float
    int64_t data_offset;  // byte offset of PCM data in file
    int64_t data_bytes;   // byte size of data chunk
    int64_t total_frames; // data_bytes / block_align
};

std::string lowercase_extension(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return {};
    std::string ext = path.substr(dot);
    for (char& c : ext) c = static_cast<char>(std::tolower(c));
    return ext;
}

bool is_libav_stream_path(const std::string& path) {
    const auto ext = lowercase_extension(path);
    return ext == ".flac" || ext == ".m4a" || ext == ".aiff" ||
           ext == ".aif" || ext == ".caf";
}

bool is_wav_path(const std::string& path) {
    if (path.size() < 4) return false;
    auto ext = path.substr(path.size() - 4);
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    return ext == ".wav";
}

bool parse_wav_header(const std::string& path, WavInfo& out) {
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path.c_str(), "rb"), fclose);
    if (!f) return false;

    uint8_t riff[12];
    if (fread(riff, 1, 12, f.get()) != 12) return false;
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        return false;
    }

    bool found_fmt = false;
    bool found_data = false;

    while (!found_data) {
        uint8_t chunk_hdr[8];
        if (fread(chunk_hdr, 1, 8, f.get()) != 8) break;

        uint32_t chunk_size;
        memcpy(&chunk_size, chunk_hdr + 4, 4);

        if (memcmp(chunk_hdr, "fmt ", 4) == 0) {
            if (chunk_size < 16) return false;

            uint8_t fmt[16];
            if (fread(fmt, 1, 16, f.get()) != 16) return false;

            uint16_t format_tag;
            uint16_t num_channels;
            uint16_t bits_per_sample;
            uint16_t block_align;
            uint32_t sample_rate;

            memcpy(&format_tag, fmt + 0, 2);
            memcpy(&num_channels, fmt + 2, 2);
            memcpy(&sample_rate, fmt + 4, 4);
            memcpy(&block_align, fmt + 12, 2);
            memcpy(&bits_per_sample, fmt + 14, 2);

            if (format_tag != 1 && format_tag != 3) return false;

            out.format_tag = static_cast<int>(format_tag);
            out.sample_rate = static_cast<int>(sample_rate);
            out.channels = static_cast<int>(num_channels);
            out.bits_per_sample = static_cast<int>(bits_per_sample);

            found_fmt = true;

            long remaining = static_cast<long>(chunk_size) - 16;
            if (remaining > 0) fseek(f.get(), remaining, SEEK_CUR);
        } else if (memcmp(chunk_hdr, "data", 4) == 0) {
            if (!found_fmt) return false;

            out.data_offset = static_cast<int64_t>(ftell(f.get()));
            out.data_bytes = static_cast<int64_t>(chunk_size);
            int block_align_bytes = out.channels * (out.bits_per_sample / 8);
            out.total_frames =
                (block_align_bytes > 0) ? (out.data_bytes / block_align_bytes) : 0;

            found_data = true;
        } else {
            long skip = static_cast<long>(chunk_size);
            if (skip & 1) skip++;
            fseek(f.get(), skip, SEEK_CUR);
        }
    }

    return found_fmt && found_data;
}

std::string av_error_to_string(int errnum) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> buf{};
    if (av_strerror(errnum, buf.data(), buf.size()) == 0) {
        return std::string(buf.data());
    }
    return "ffmpeg_error(" + std::to_string(errnum) + ")";
}

int codecctx_num_channels(const AVCodecContext* codec_ctx) {
    if (!codec_ctx) return 0;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    return codec_ctx->ch_layout.nb_channels;
#else
    return codec_ctx->channels;
#endif
}

SwrContext* create_libav_stream_swr(
    const AVCodecContext* codec_ctx,
    int in_channels,
    int out_sr) {
    SwrContext* swr_raw = nullptr;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    AVChannelLayout in_layout;
    AVChannelLayout out_layout;
    std::memset(&in_layout, 0, sizeof(in_layout));
    std::memset(&out_layout, 0, sizeof(out_layout));

    int rc = 0;
    if (codec_ctx->ch_layout.nb_channels > 0) {
        rc = av_channel_layout_copy(&in_layout, &codec_ctx->ch_layout);
        if (rc < 0) {
            throw std::runtime_error(
                "Failed to create libav stream resampler: " + av_error_to_string(rc));
        }
    } else {
        av_channel_layout_default(&in_layout, in_channels);
    }
    av_channel_layout_default(&out_layout, in_channels);

    rc = swr_alloc_set_opts2(
        &swr_raw,
        &out_layout,
        AV_SAMPLE_FMT_FLT,
        out_sr,
        &in_layout,
        codec_ctx->sample_fmt,
        codec_ctx->sample_rate,
        0,
        nullptr);
    av_channel_layout_uninit(&in_layout);
    av_channel_layout_uninit(&out_layout);
    if (rc < 0 || !swr_raw) {
        throw std::runtime_error(
            "Failed to create libav stream resampler: " + av_error_to_string(rc));
    }
#else
    int64_t ch_layout = codec_ctx->channel_layout;
    if (ch_layout == 0) {
        ch_layout = av_get_default_channel_layout(in_channels);
    }
    swr_raw = swr_alloc_set_opts(
        nullptr,
        ch_layout,
        AV_SAMPLE_FMT_FLT,
        out_sr,
        ch_layout,
        codec_ctx->sample_fmt,
        codec_ctx->sample_rate,
        0,
        nullptr);
    if (!swr_raw) {
        throw std::runtime_error("Failed to create libav stream resampler");
    }
#endif

    int init_rc = swr_init(swr_raw);
    if (init_rc < 0) {
        swr_free(&swr_raw);
        throw std::runtime_error(
            "Failed to initialize libav stream resampler: " +
            av_error_to_string(init_rc));
    }
    return swr_raw;
}

void seek_libav_stream_state(
    LibavStreamState* st,
    double offset_seconds) {
    if (!st || offset_seconds <= 0.0) return;
    AVStream* stream = st->format_ctx->streams[st->stream_index];
    int64_t target_us = static_cast<int64_t>(std::llround(offset_seconds * AV_TIME_BASE));
    int64_t target_ts = av_rescale_q(
        target_us,
        AVRational{1, AV_TIME_BASE},
        stream->time_base);
    int rc = av_seek_frame(
        st->format_ctx,
        st->stream_index,
        target_ts,
        AVSEEK_FLAG_BACKWARD);
    if (rc < 0) {
        throw value_error(
            "Failed to seek stream() on Linux backend via libav: " +
            av_error_to_string(rc));
    }

    avformat_flush(st->format_ctx);
    avcodec_flush_buffers(st->codec_ctx);
    if (st->swr_ctx) {
        swr_close(st->swr_ctx);
        int init_rc = swr_init(st->swr_ctx);
        if (init_rc < 0) {
            throw value_error(
                "Failed to reset stream() resampler on Linux backend via libav: " +
                av_error_to_string(init_rc));
        }
    }

    st->flush_sent = false;
    st->decoder_eof = false;
    st->pending_interleaved.clear();
    st->pending_pos = 0;
}

LibavStreamState* create_libav_stream_state(
    const std::string& path,
    std::optional<int> sr,
    double offset_seconds) {
    auto* st = new LibavStreamState();
    try {
        int rc = avformat_open_input(&st->format_ctx, path.c_str(), nullptr, nullptr);
        if (rc < 0 || !st->format_ctx) {
            throw value_error(
                "Failed to open input format for stream() on Linux backend via libav: " +
                path + " (" + av_error_to_string(rc) + ")");
        }

        rc = avformat_find_stream_info(st->format_ctx, nullptr);
        if (rc < 0) {
            throw value_error(
                "Failed to read stream info for stream() on Linux backend via libav: " +
                path + " (" + av_error_to_string(rc) + ")");
        }

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 18, 100)
        AVCodec* decoder_for_find = nullptr;
#else
        const AVCodec* decoder_for_find = nullptr;
#endif
        st->stream_index = av_find_best_stream(
            st->format_ctx,
            AVMEDIA_TYPE_AUDIO,
            -1,
            -1,
            &decoder_for_find,
            0);
        const AVCodec* decoder = decoder_for_find;
        if (st->stream_index < 0 || !decoder) {
            throw value_error(
                "Unsupported input format for stream() on Linux backend. Supported: "
                ".wav, .mp3, .flac, .m4a, .aiff, .caf");
        }

        st->codec_ctx = avcodec_alloc_context3(decoder);
        if (!st->codec_ctx) {
            throw std::runtime_error("Failed to allocate AVCodecContext for stream()");
        }

        AVStream* stream = st->format_ctx->streams[st->stream_index];
        rc = avcodec_parameters_to_context(st->codec_ctx, stream->codecpar);
        if (rc < 0) {
            throw value_error(
                "Failed to initialize decoder for stream() on Linux backend via libav: " +
                path + " (" + av_error_to_string(rc) + ")");
        }

        rc = avcodec_open2(st->codec_ctx, decoder, nullptr);
        if (rc < 0) {
            throw value_error(
                "Failed to open decoder for stream() on Linux backend via libav: " +
                path + " (" + av_error_to_string(rc) + ")");
        }

        st->native_sr = st->codec_ctx->sample_rate;
        st->native_channels = codecctx_num_channels(st->codec_ctx);
        st->out_sr = sr.value_or(st->native_sr);
        if (st->native_sr <= 0 || st->native_channels <= 0 || st->out_sr <= 0) {
            throw value_error("Failed to initialize libav stream decoder metadata for: " + path);
        }

        st->swr_ctx = create_libav_stream_swr(st->codec_ctx, st->native_channels, st->out_sr);
        st->packet = av_packet_alloc();
        st->frame = av_frame_alloc();
        if (!st->packet || !st->frame) {
            throw std::runtime_error("Failed to allocate libav packet/frame for stream()");
        }
        seek_libav_stream_state(st, offset_seconds);
    } catch (...) {
        delete st;
        throw;
    }
    return st;
}

void append_stream_frame_converted(LibavStreamState* st, const AVFrame* src_frame) {
    if (!st || !src_frame || src_frame->nb_samples <= 0) return;

    int64_t delay = swr_get_delay(st->swr_ctx, st->native_sr);
    int out_samples = static_cast<int>(av_rescale_rnd(
        delay + src_frame->nb_samples,
        st->out_sr,
        st->native_sr,
        AV_ROUND_UP));
    if (out_samples <= 0) return;

    std::vector<float> chunk(
        static_cast<size_t>(out_samples) * static_cast<size_t>(st->native_channels));
    uint8_t* out_data[1] = {
        reinterpret_cast<uint8_t*>(chunk.data())
    };
    const bool planar = av_sample_fmt_is_planar(
        static_cast<AVSampleFormat>(src_frame->format)) != 0;
    const int plane_count = planar ? st->native_channels : 1;
    std::vector<const uint8_t*> in_data(static_cast<size_t>(plane_count));
    for (int i = 0; i < plane_count; ++i) {
        in_data[static_cast<size_t>(i)] = src_frame->extended_data[i];
    }

    int converted = swr_convert(
        st->swr_ctx,
        out_data,
        out_samples,
        in_data.data(),
        src_frame->nb_samples);
    if (converted < 0) {
        throw std::runtime_error(
            "Failed to convert audio frame in stream() on Linux backend via libav: " +
            av_error_to_string(converted));
    }
    if (converted == 0) return;

    chunk.resize(
        static_cast<size_t>(converted) * static_cast<size_t>(st->native_channels));
    st->pending_interleaved.insert(
        st->pending_interleaved.end(), chunk.begin(), chunk.end());
}

void drain_decoder_frames(LibavStreamState* st) {
    while (true) {
        int rc = avcodec_receive_frame(st->codec_ctx, st->frame);
        if (rc == AVERROR(EAGAIN)) return;
        if (rc == AVERROR_EOF) {
            st->decoder_eof = true;
            return;
        }
        if (rc < 0) {
            throw std::runtime_error(
                "Failed to decode audio frame in stream() on Linux backend via libav: " +
                av_error_to_string(rc));
        }
        append_stream_frame_converted(st, st->frame);
        av_frame_unref(st->frame);
    }
}

void fill_libav_stream_pending(LibavStreamState* st, size_t target_samples) {
    while ((st->pending_interleaved.size() - st->pending_pos) < target_samples &&
           !st->decoder_eof) {
        drain_decoder_frames(st);
        if ((st->pending_interleaved.size() - st->pending_pos) >= target_samples ||
            st->decoder_eof) {
            break;
        }

        if (!st->flush_sent) {
            int rc = av_read_frame(st->format_ctx, st->packet);
            if (rc == AVERROR_EOF) {
                rc = avcodec_send_packet(st->codec_ctx, nullptr);
                if (rc < 0 && rc != AVERROR_EOF) {
                    throw std::runtime_error(
                        "Failed to flush stream decoder on Linux backend via libav: " +
                        av_error_to_string(rc));
                }
                st->flush_sent = true;
                continue;
            }
            if (rc < 0) {
                throw std::runtime_error(
                    "Failed to read audio packet in stream() on Linux backend via libav: " +
                    av_error_to_string(rc));
            }

            if (st->packet->stream_index != st->stream_index) {
                av_packet_unref(st->packet);
                continue;
            }

            rc = avcodec_send_packet(st->codec_ctx, st->packet);
            av_packet_unref(st->packet);
            if (rc < 0 && rc != AVERROR(EAGAIN) && rc != AVERROR_EOF) {
                throw std::runtime_error(
                    "Failed to send audio packet to decoder in stream() on Linux backend via libav: " +
                    av_error_to_string(rc));
            }
        } else {
            drain_decoder_frames(st);
            if (!st->decoder_eof) {
                st->decoder_eof = true;
            }
            break;
        }
    }
}

float* decode_wav_chunk(
    FILE* f,
    int channels,
    int format_tag,
    int bits_per_sample,
    int64_t frames_to_read,
    int64_t* actual_frames_out) {
    size_t out_bytes = static_cast<size_t>(std::max<int64_t>(frames_to_read, 1)) *
                       channels * sizeof(float);
    float* buffer = static_cast<float*>(aligned_alloc_64(out_bytes));

    int block_align = channels * (bits_per_sample / 8);
    int64_t read_bytes = frames_to_read * block_align;
    int64_t actual_frames = frames_to_read;

    if (format_tag == 3 && bits_per_sample == 32) {
        size_t got = fread(buffer, 1, static_cast<size_t>(read_bytes), f);
        actual_frames = static_cast<int64_t>(got) / (channels * sizeof(float));
    } else if (format_tag == 1 && bits_per_sample == 16) {
        size_t sample_count = static_cast<size_t>(frames_to_read) * channels;
        int16_t* pcm_buf = static_cast<int16_t*>(
            aligned_alloc_64(sample_count * sizeof(int16_t)));

        size_t got = fread(pcm_buf, 1, static_cast<size_t>(read_bytes), f);
        actual_frames = static_cast<int64_t>(got) / (channels * sizeof(int16_t));
        size_t actual_samples = static_cast<size_t>(actual_frames) * channels;

        constexpr float kScale = 1.0f / 32768.0f;
        for (size_t i = 0; i < actual_samples; ++i) {
            buffer[i] = static_cast<float>(pcm_buf[i]) * kScale;
        }

        std::free(pcm_buf);
    } else if (format_tag == 1 && bits_per_sample == 24) {
        size_t sample_count = static_cast<size_t>(frames_to_read) * channels;
        size_t raw_bytes = sample_count * 3;
        uint8_t* raw = static_cast<uint8_t*>(aligned_alloc_64(raw_bytes));

        size_t got = fread(raw, 1, raw_bytes, f);
        size_t actual_samples = got / 3;
        actual_frames = static_cast<int64_t>(actual_samples) / channels;

        constexpr float kInv = 1.0f / 8388608.0f;
        for (size_t i = 0; i < actual_samples; ++i) {
            int32_t val = static_cast<int32_t>(raw[i * 3]) |
                          (static_cast<int32_t>(raw[i * 3 + 1]) << 8) |
                          (static_cast<int32_t>(raw[i * 3 + 2]) << 16);
            if (val & 0x800000) val |= 0xFF000000;
            buffer[i] = static_cast<float>(val) * kInv;
        }

        std::free(raw);
    } else if (format_tag == 1 && bits_per_sample == 32) {
        size_t sample_count = static_cast<size_t>(frames_to_read) * channels;
        int32_t* pcm_buf = static_cast<int32_t*>(
            aligned_alloc_64(sample_count * sizeof(int32_t)));

        size_t got = fread(pcm_buf, 1, static_cast<size_t>(read_bytes), f);
        actual_frames = static_cast<int64_t>(got) / (channels * sizeof(int32_t));
        size_t actual_samples = static_cast<size_t>(actual_frames) * channels;

        constexpr float kScale = 1.0f / 2147483648.0f;
        for (size_t i = 0; i < actual_samples; ++i) {
            buffer[i] = static_cast<float>(pcm_buf[i]) * kScale;
        }

        std::free(pcm_buf);
    } else {
        std::free(buffer);
        throw value_error(
            "Unsupported WAV encoding in stream() on Linux backend: format_tag=" +
            std::to_string(format_tag) + ", bits_per_sample=" +
            std::to_string(bits_per_sample));
    }

    *actual_frames_out = actual_frames;
    return buffer;
}

#endif

}  // namespace

AudioStreamReader::AudioStreamReader(
    const std::string& path,
    int chunk_frames,
    std::optional<int> sr,
    bool mono,
    double offset,
    std::optional<double> duration,
    const std::string& dtype)
    : chunk_frames_(chunk_frames), mono_(mono), dtype_(dtype) {
    if (dtype != "float32" && dtype != "float16") {
        throw value_error("Unsupported dtype '" + dtype + "'. Must be 'float32' or 'float16'.");
    }
    if (chunk_frames <= 0) {
        throw value_error("chunk_frames must be > 0, got " + std::to_string(chunk_frames));
    }
    if (sr.has_value() && sr.value() <= 0) {
        throw value_error("sr must be > 0");
    }
    if (offset < 0.0) {
        throw value_error("offset must be >= 0, got " + std::to_string(offset));
    }
    if (duration.has_value() && duration.value() <= 0.0) {
        throw value_error("duration must be > 0, got " + std::to_string(duration.value()));
    }

    check_file_exists(path);

#if !defined(__APPLE__)
    bool is_mp3_file = is_mp3_path(path);
    bool is_wav_file = is_wav_path(path);
    bool is_libav_file = is_libav_stream_path(path);

    if (!is_mp3_file && !is_wav_file) {
        if (!is_libav_file) {
            throw value_error(
                "Unsupported input format for stream() on Linux backend. Supported: "
                ".wav, .mp3, .flac, .m4a, .aiff, .caf");
        }

        auto* st = create_libav_stream_state(path, sr, offset);
        libav_stream_ = st;
        is_libav_stream_ = true;
        out_sr_ = st->out_sr;
        native_channels_ = st->native_channels;
        out_channels_ = mono_ ? 1 : native_channels_;
        if (duration.has_value()) {
            max_frames_to_emit_ = static_cast<int64_t>(std::floor(duration.value() * out_sr_));
            if (max_frames_to_emit_ <= 0) {
                eof_ = true;
            }
        }
        return;
    }

    if (sr.has_value()) {
        int native_sr = 0;
        if (is_mp3_file) {
            ScopedMp3Decoder probe;
            probe.open(path);
            native_sr = probe.sample_rate();
        } else if (is_wav_file) {
            WavInfo wav;
            if (!parse_wav_header(path, wav)) {
                throw value_error("Failed to parse WAV header: " + path);
            }
            native_sr = wav.sample_rate;
        } else {
            throw value_error(
                "Unsupported input format for stream() on Linux backend. Supported: .wav, .mp3");
        }

        if (sr.value() != native_sr) {
            auto predecoded = backend_load_audio(
                path,
                sr,
                offset,
                duration,
                mono_,
                "channels_last",
                "float32",
                "default");

            predecoded_audio_.emplace(std::move(predecoded.first));
            mlx::core::eval(*predecoded_audio_);

            if (predecoded_audio_->ndim() != 2) {
                throw std::runtime_error("Unexpected predecoded audio shape for stream()");
            }

            predecoded_data_ = predecoded_audio_->data<float>();
            predecoded_total_frames_ = predecoded_audio_->shape(0);
            out_sr_ = predecoded.second;
            out_channels_ = static_cast<int>(predecoded_audio_->shape(1));
            native_channels_ = out_channels_;
            max_frames_to_emit_ = predecoded_total_frames_;
            is_predecoded_ = true;
            eof_ = (predecoded_total_frames_ == 0);
            return;
        }
    }
#endif

    // Try MP3 fast path
    if (is_mp3_path(path)) {
        // Open once — reuse for both SR check and streaming
        mp3_dec_.emplace();
        mp3_dec_->open(path);
        bool needs_resample = sr.has_value() && sr.value() != mp3_dec_->sample_rate();
        if (!needs_resample) {
            is_mp3_ = true;
            native_channels_ = mp3_dec_->channels();
            out_sr_ = mp3_dec_->sample_rate();
            out_channels_ = mono_ ? 1 : native_channels_;
            int64_t total_frames = mp3_dec_->total_frames();
            int64_t start_frame = static_cast<int64_t>(std::floor(offset * out_sr_));
            if (start_frame >= total_frames) {
                eof_ = true;
                max_frames_to_emit_ = 0;
                return;
            }
            if (start_frame > 0) {
                mp3_dec_->seek(start_frame * native_channels_);
            }
            max_frames_to_emit_ = total_frames - start_frame;
            if (duration.has_value()) {
                int64_t duration_frames =
                    static_cast<int64_t>(std::floor(duration.value() * out_sr_));
                max_frames_to_emit_ = std::min<int64_t>(
                    max_frames_to_emit_, std::max<int64_t>(0, duration_frames));
            }
            if (max_frames_to_emit_ <= 0) {
                eof_ = true;
            }
            return;
        }
        // Needs resampling — discard the decoder and fall through
        mp3_dec_.reset();
    }

#if defined(__APPLE__)
    auto url = make_url(path);
    OSStatus status = ExtAudioFileOpenURL(url.get(), ext_file_.ptr());
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to open audio file '" + path +
            "': OSStatus " + osstatus_to_string(status));
    }

    AudioStreamBasicDescription native_fmt = {};
    UInt32 prop_size = sizeof(native_fmt);
    status = ExtAudioFileGetProperty(
        ext_file_.get(), kExtAudioFileProperty_FileDataFormat, &prop_size, &native_fmt);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to get file format: OSStatus " + osstatus_to_string(status));
    }

    int native_sr = static_cast<int>(native_fmt.mSampleRate);
    native_channels_ = static_cast<int>(native_fmt.mChannelsPerFrame);
    out_sr_ = sr.value_or(native_sr);
    out_channels_ = mono_ ? 1 : native_channels_;

    auto client_fmt = make_client_format(out_sr_, native_channels_);
    status = ExtAudioFileSetProperty(
        ext_file_.get(),
        kExtAudioFileProperty_ClientDataFormat,
        sizeof(client_fmt),
        &client_fmt);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to set client format: OSStatus " + osstatus_to_string(status));
    }

    SInt64 native_frames = 0;
    prop_size = sizeof(native_frames);
    status = ExtAudioFileGetProperty(
        ext_file_.get(),
        kExtAudioFileProperty_FileLengthFrames,
        &prop_size,
        &native_frames);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to get frame count: OSStatus " + osstatus_to_string(status));
    }

    // ExtAudioFileSeek uses source/native frame positions. Compute seek in native
    // domain, then derive the emitted frame budget in output-SR domain.
    int64_t start_frame_native = static_cast<int64_t>(std::floor(offset * native_sr));
    if (start_frame_native >= static_cast<int64_t>(native_frames)) {
        eof_ = true;
        max_frames_to_emit_ = 0;
        return;
    }

    int64_t remaining_native = static_cast<int64_t>(native_frames) - start_frame_native;
    int64_t remaining_out = 0;
    if (out_sr_ == native_sr) {
        remaining_out = remaining_native;
    } else {
        remaining_out = static_cast<int64_t>(
            std::ceil(static_cast<double>(remaining_native) * out_sr_ / native_sr));
    }

    max_frames_to_emit_ = remaining_out;
    if (duration.has_value()) {
        int64_t duration_frames =
            static_cast<int64_t>(std::floor(duration.value() * out_sr_));
        max_frames_to_emit_ = std::min<int64_t>(
            max_frames_to_emit_, std::max<int64_t>(0, duration_frames));
    }
    if (max_frames_to_emit_ <= 0) {
        eof_ = true;
        return;
    }

    if (start_frame_native > 0) {
        status = ExtAudioFileSeek(ext_file_.get(), start_frame_native);
        if (status != noErr) {
            throw std::runtime_error(
                "Failed to seek stream reader: OSStatus " + osstatus_to_string(status));
        }
    }
#else
    if (!is_wav_path(path)) {
        throw value_error(
            "Unsupported input format for stream() on Linux backend. Supported: .wav, .mp3");
    }

    WavInfo wav;
    if (!parse_wav_header(path, wav)) {
        throw value_error("Failed to parse WAV header: " + path);
    }

    wav_file_ = fopen(path.c_str(), "rb");
    if (!wav_file_) {
        throw std::runtime_error("Failed to open WAV file: " + path);
    }
    int64_t start_frame = static_cast<int64_t>(std::floor(offset * wav.sample_rate));
    if (start_frame >= wav.total_frames) {
        eof_ = true;
        max_frames_to_emit_ = 0;
        fseek(wav_file_, static_cast<long>(wav.data_offset + wav.data_bytes), SEEK_SET);
    } else {
        int64_t emit_frames = wav.total_frames - start_frame;
        if (duration.has_value()) {
            int64_t duration_frames =
                static_cast<int64_t>(std::floor(duration.value() * wav.sample_rate));
            emit_frames = std::min<int64_t>(
                emit_frames, std::max<int64_t>(0, duration_frames));
        }
        wav_total_frames_ = emit_frames;
        max_frames_to_emit_ = emit_frames;
        int block_align = wav.channels * (wav.bits_per_sample / 8);
        int64_t byte_offset = wav.data_offset + start_frame * block_align;
        if (fseek(wav_file_, static_cast<long>(byte_offset), SEEK_SET) != 0) {
            fclose(wav_file_);
            wav_file_ = nullptr;
            throw std::runtime_error("Failed to seek in WAV file: " + path);
        }
        if (emit_frames <= 0) {
            eof_ = true;
        }
    }

    is_wav_ = true;
    wav_bits_per_sample_ = wav.bits_per_sample;
    wav_format_tag_ = wav.format_tag;
    if (max_frames_to_emit_ < 0) {
        wav_total_frames_ = wav.total_frames;
    }

    native_channels_ = wav.channels;
    out_sr_ = wav.sample_rate;
    out_channels_ = mono_ ? 1 : native_channels_;
#endif
}

AudioStreamReader::~AudioStreamReader() {
#if !defined(__APPLE__)
    if (is_libav_stream_ && libav_stream_) {
        delete static_cast<LibavStreamState*>(libav_stream_);
        libav_stream_ = nullptr;
        is_libav_stream_ = false;
    }
    if (wav_file_) {
        fclose(wav_file_);
        wav_file_ = nullptr;
    }
#endif
}

AudioStreamReader::AudioStreamReader(AudioStreamReader&& other) noexcept
#if defined(__APPLE__)
    : ext_file_(std::move(other.ext_file_))
#else
    : is_wav_(other.is_wav_),
      wav_file_(other.wav_file_),
      wav_bits_per_sample_(other.wav_bits_per_sample_),
      wav_format_tag_(other.wav_format_tag_),
      wav_total_frames_(other.wav_total_frames_),
      is_predecoded_(other.is_predecoded_),
      predecoded_audio_(std::move(other.predecoded_audio_)),
      predecoded_data_(other.predecoded_data_),
      predecoded_total_frames_(other.predecoded_total_frames_),
      is_libav_stream_(other.is_libav_stream_),
      libav_stream_(other.libav_stream_)
#endif
      ,
      chunk_frames_(other.chunk_frames_),
      out_sr_(other.out_sr_),
      native_channels_(other.native_channels_),
      out_channels_(other.out_channels_),
      mono_(other.mono_),
      eof_(other.eof_),
      frames_read_(other.frames_read_),
      max_frames_to_emit_(other.max_frames_to_emit_),
      is_mp3_(other.is_mp3_),
      mp3_dec_(std::move(other.mp3_dec_)),
      dtype_(std::move(other.dtype_)) {
#if !defined(__APPLE__)
    other.is_wav_ = false;
    other.wav_file_ = nullptr;
    other.wav_bits_per_sample_ = 0;
    other.wav_format_tag_ = 0;
    other.wav_total_frames_ = 0;
    other.is_predecoded_ = false;
    other.predecoded_audio_.reset();
    other.predecoded_data_ = nullptr;
    other.predecoded_total_frames_ = 0;
    other.is_libav_stream_ = false;
    other.libav_stream_ = nullptr;
#endif
    other.chunk_frames_ = 0;
    other.out_sr_ = 0;
    other.native_channels_ = 0;
    other.out_channels_ = 0;
    other.mono_ = false;
    other.eof_ = true;
    other.frames_read_ = 0;
    other.max_frames_to_emit_ = -1;
    other.is_mp3_ = false;
}

AudioStreamReader& AudioStreamReader::operator=(AudioStreamReader&& other) noexcept {
    if (this == &other) {
        return *this;
    }

#if defined(__APPLE__)
    ext_file_ = std::move(other.ext_file_);
#else
    if (is_libav_stream_ && libav_stream_) {
        delete static_cast<LibavStreamState*>(libav_stream_);
        libav_stream_ = nullptr;
        is_libav_stream_ = false;
    }
    if (wav_file_) {
        fclose(wav_file_);
    }
    is_wav_ = other.is_wav_;
    wav_file_ = other.wav_file_;
    wav_bits_per_sample_ = other.wav_bits_per_sample_;
    wav_format_tag_ = other.wav_format_tag_;
    wav_total_frames_ = other.wav_total_frames_;
    is_predecoded_ = other.is_predecoded_;
    predecoded_audio_ = std::move(other.predecoded_audio_);
    predecoded_data_ = other.predecoded_data_;
    predecoded_total_frames_ = other.predecoded_total_frames_;
    is_libav_stream_ = other.is_libav_stream_;
    libav_stream_ = other.libav_stream_;
#endif

    chunk_frames_ = other.chunk_frames_;
    out_sr_ = other.out_sr_;
    native_channels_ = other.native_channels_;
    out_channels_ = other.out_channels_;
    mono_ = other.mono_;
    eof_ = other.eof_;
    frames_read_ = other.frames_read_;
    max_frames_to_emit_ = other.max_frames_to_emit_;
    is_mp3_ = other.is_mp3_;
    mp3_dec_ = std::move(other.mp3_dec_);
    dtype_ = std::move(other.dtype_);

#if !defined(__APPLE__)
    other.is_wav_ = false;
    other.wav_file_ = nullptr;
    other.wav_bits_per_sample_ = 0;
    other.wav_format_tag_ = 0;
    other.wav_total_frames_ = 0;
    other.is_predecoded_ = false;
    other.predecoded_audio_.reset();
    other.predecoded_data_ = nullptr;
    other.predecoded_total_frames_ = 0;
    other.is_libav_stream_ = false;
    other.libav_stream_ = nullptr;
#endif
    other.chunk_frames_ = 0;
    other.out_sr_ = 0;
    other.native_channels_ = 0;
    other.out_channels_ = 0;
    other.mono_ = false;
    other.eof_ = true;
    other.frames_read_ = 0;
    other.max_frames_to_emit_ = -1;
    other.is_mp3_ = false;

    return *this;
}

std::pair<mlx::core::array, int> AudioStreamReader::read_chunk() {
    if (eof_) {
        return tensor_utils::make_empty_audio_result(
            out_sr_, out_channels_, false, "channels_last", dtype_);
    }
    int64_t window_remaining = -1;
    if (max_frames_to_emit_ >= 0) {
        window_remaining = max_frames_to_emit_ - frames_read_;
        if (window_remaining <= 0) {
            eof_ = true;
            return tensor_utils::make_empty_audio_result(
                out_sr_, out_channels_, false, "channels_last", dtype_);
        }
    }

#if !defined(__APPLE__)
    if (is_libav_stream_) {
        if (!libav_stream_) {
            throw std::runtime_error("Invalid libav stream reader state on Linux backend");
        }
        auto* st = static_cast<LibavStreamState*>(libav_stream_);
        int64_t chunk_limit = chunk_frames_;
        if (window_remaining >= 0) {
            chunk_limit = std::min<int64_t>(chunk_limit, window_remaining);
        }
        size_t target_samples =
            static_cast<size_t>(chunk_limit) * static_cast<size_t>(st->native_channels);

        fill_libav_stream_pending(st, target_samples);

        size_t available_samples = st->pending_interleaved.size() - st->pending_pos;
        int64_t available_frames =
            static_cast<int64_t>(available_samples / static_cast<size_t>(st->native_channels));
        int64_t frames_to_emit = std::min<int64_t>(chunk_limit, available_frames);

        if (frames_to_emit <= 0) {
            eof_ = true;
            is_libav_stream_ = false;
            delete st;
            libav_stream_ = nullptr;
            return tensor_utils::make_empty_audio_result(
                out_sr_, out_channels_, false, "channels_last", dtype_);
        }

        size_t samples_to_emit =
            static_cast<size_t>(frames_to_emit) * static_cast<size_t>(st->native_channels);
        size_t bytes_to_emit = samples_to_emit * sizeof(float);
        float* buffer = static_cast<float*>(aligned_alloc_64(bytes_to_emit));
        std::memcpy(
            buffer,
            st->pending_interleaved.data() + st->pending_pos,
            bytes_to_emit);
        st->pending_pos += samples_to_emit;
        frames_read_ += frames_to_emit;

        if (st->pending_pos == st->pending_interleaved.size()) {
            st->pending_interleaved.clear();
            st->pending_pos = 0;
        } else if (st->pending_pos > (1u << 20)) {
            st->pending_interleaved.erase(
                st->pending_interleaved.begin(),
                st->pending_interleaved.begin() + static_cast<std::ptrdiff_t>(st->pending_pos));
            st->pending_pos = 0;
        }

        if (st->decoder_eof &&
            (st->pending_interleaved.size() - st->pending_pos) <
                static_cast<size_t>(st->native_channels)) {
            eof_ = true;
            is_libav_stream_ = false;
            delete st;
            libav_stream_ = nullptr;
        } else if (max_frames_to_emit_ >= 0 && frames_read_ >= max_frames_to_emit_) {
            eof_ = true;
            is_libav_stream_ = false;
            delete st;
            libav_stream_ = nullptr;
        }

        return tensor_utils::wrap_interleaved_audio_buffer(
            buffer,
            frames_to_emit,
            st->native_channels,
            out_sr_,
            mono_,
            "channels_last",
            dtype_);
    }

    if (is_predecoded_) {
        int64_t remaining = predecoded_total_frames_ - frames_read_;
        if (window_remaining >= 0) {
            remaining = std::min<int64_t>(remaining, window_remaining);
        }
        if (remaining <= 0) {
            eof_ = true;
            return tensor_utils::make_empty_audio_result(
                out_sr_, out_channels_, false, "channels_last", dtype_);
        }

        int64_t frames_to_read = std::min<int64_t>(chunk_frames_, remaining);
        size_t sample_count =
            static_cast<size_t>(frames_to_read) * static_cast<size_t>(out_channels_);
        float* buffer = static_cast<float*>(aligned_alloc_64(sample_count * sizeof(float)));
        const float* src = predecoded_data_ + frames_read_ * out_channels_;
        std::memcpy(buffer, src, sample_count * sizeof(float));

        frames_read_ += frames_to_read;
        if (frames_read_ >= predecoded_total_frames_ ||
            (max_frames_to_emit_ >= 0 && frames_read_ >= max_frames_to_emit_)) {
            eof_ = true;
        }

        return tensor_utils::wrap_interleaved_audio_buffer(
            buffer, frames_to_read, out_channels_, out_sr_, false, "channels_last", dtype_);
    }
#endif

    if (is_mp3_) {
        int64_t chunk_limit = chunk_frames_;
        if (window_remaining >= 0) {
            chunk_limit = std::min<int64_t>(chunk_limit, window_remaining);
        }
        size_t buffer_bytes = static_cast<size_t>(chunk_limit) * native_channels_ * sizeof(float);
        float* buffer = static_cast<float*>(aligned_alloc_64(buffer_bytes));

        int64_t samples_wanted = chunk_limit * native_channels_;
        int64_t samples_read = mp3_dec_->read(buffer, samples_wanted);
        int64_t actual_frames = (native_channels_ > 0) ? samples_read / native_channels_ : 0;

        if (actual_frames == 0) {
            std::free(buffer);
            eof_ = true;
            return tensor_utils::make_empty_audio_result(
                out_sr_, out_channels_, false, "channels_last", dtype_);
        }

        frames_read_ += actual_frames;
        if (max_frames_to_emit_ >= 0 && frames_read_ >= max_frames_to_emit_) {
            eof_ = true;
        }

        return tensor_utils::wrap_interleaved_audio_buffer(
            buffer, actual_frames, native_channels_, out_sr_, mono_, "channels_last", dtype_);
    }

#if defined(__APPLE__)
    int64_t chunk_limit = chunk_frames_;
    if (window_remaining >= 0) {
        chunk_limit = std::min<int64_t>(chunk_limit, window_remaining);
    }
    size_t buffer_bytes = static_cast<size_t>(chunk_limit) * native_channels_ * sizeof(float);
    float* buffer = static_cast<float*>(aligned_alloc_64(buffer_bytes));

    UInt32 frames_remaining = static_cast<UInt32>(chunk_limit);
    UInt32 total_read = 0;
    float* write_ptr = buffer;

    while (frames_remaining > 0) {
        UInt32 read_count = std::min(frames_remaining, static_cast<UInt32>(kReadChunkFrames));

        AudioBufferList abl;
        abl.mNumberBuffers = 1;
        abl.mBuffers[0].mNumberChannels = static_cast<UInt32>(native_channels_);
        abl.mBuffers[0].mDataByteSize =
            read_count * static_cast<UInt32>(native_channels_) * sizeof(float);
        abl.mBuffers[0].mData = write_ptr;

        OSStatus status = ExtAudioFileRead(ext_file_.get(), &read_count, &abl);
        if (status != noErr) {
            std::free(buffer);
            throw std::runtime_error(
                "Failed to read audio: OSStatus " + osstatus_to_string(status));
        }
        if (read_count == 0) {
            eof_ = true;
            break;
        }

        total_read += read_count;
        frames_remaining -= read_count;
        write_ptr += static_cast<size_t>(read_count) * native_channels_;
    }

    int64_t actual_frames = static_cast<int64_t>(total_read);
    frames_read_ += actual_frames;
    if (max_frames_to_emit_ >= 0 && frames_read_ >= max_frames_to_emit_) {
        eof_ = true;
    }

    if (actual_frames == 0) {
        std::free(buffer);
        eof_ = true;
        return tensor_utils::make_empty_audio_result(
            out_sr_, out_channels_, false, "channels_last", dtype_);
    }

    return tensor_utils::wrap_interleaved_audio_buffer(
        buffer, actual_frames, native_channels_, out_sr_, mono_, "channels_last", dtype_);
#else
    if (!is_wav_ || !wav_file_) {
        throw std::runtime_error("Invalid stream reader state on Linux backend");
    }

    int64_t remaining = wav_total_frames_ - frames_read_;
    if (window_remaining >= 0) {
        remaining = std::min<int64_t>(remaining, window_remaining);
    }
    if (remaining <= 0) {
        eof_ = true;
        return tensor_utils::make_empty_audio_result(
            out_sr_, out_channels_, false, "channels_last", dtype_);
    }

    int64_t frames_to_read = std::min<int64_t>(chunk_frames_, remaining);
    int64_t actual_frames = 0;
    float* buffer = decode_wav_chunk(
        wav_file_,
        native_channels_,
        wav_format_tag_,
        wav_bits_per_sample_,
        frames_to_read,
        &actual_frames);

    if (actual_frames == 0) {
        std::free(buffer);
        eof_ = true;
        return tensor_utils::make_empty_audio_result(
            out_sr_, out_channels_, false, "channels_last", dtype_);
    }

    frames_read_ += actual_frames;
    if (frames_read_ >= wav_total_frames_ ||
        (max_frames_to_emit_ >= 0 && frames_read_ >= max_frames_to_emit_)) {
        eof_ = true;
    }

    return tensor_utils::wrap_interleaved_audio_buffer(
        buffer, actual_frames, native_channels_, out_sr_, mono_, "channels_last", dtype_);
#endif
}

}  // namespace mlx_audio
