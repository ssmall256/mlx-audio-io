#include "audio_backend.h"

#include "internal_utils.h"
#include "mp3_decoder.h"
#include "tensor_utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/error.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

// WAV fast path assumes little-endian layout.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "WAV parser assumes little-endian byte order");

namespace mlx_audio {

using internal::aligned_alloc_64;
using internal::check_file_exists;

namespace {

struct WavInfo {
    int sample_rate;
    int channels;
    int bits_per_sample;
    int format_tag;       // 1 = PCM, 3 = IEEE float
    int64_t data_offset;  // byte offset of PCM data in file
    int64_t data_bytes;   // byte size of data chunk
    int64_t total_frames; // data_bytes / block_align
};

bool parse_wav_header(const std::string& path, WavInfo& out);
std::pair<mlx::core::array, int> load_wav(
    const WavInfo& wav,
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype);

std::string lowercase_extension(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return {};
    std::string ext = path.substr(dot);
    for (char& c : ext) c = static_cast<char>(std::tolower(c));
    return ext;
}

bool is_libav_input_path(const std::string& path) {
    const auto ext = lowercase_extension(path);
    return ext == ".flac" || ext == ".m4a" || ext == ".aiff" ||
           ext == ".aif" || ext == ".caf";
}

bool is_libav_output_path(const std::string& path) {
    const auto ext = lowercase_extension(path);
    return ext == ".mp3" || ext == ".flac" || ext == ".m4a" || ext == ".aiff" ||
           ext == ".aif" || ext == ".caf";
}

struct FfprobeInfo {
    int sample_rate = 0;
    int channels = 0;
    double duration = 0.0;
    int64_t frames = 0;
    std::string codec_name;
};

std::string av_error_to_string(int errnum) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> buf{};
    if (av_strerror(errnum, buf.data(), buf.size()) == 0) {
        return std::string(buf.data());
    }
    return "ffmpeg_error(" + std::to_string(errnum) + ")";
}

struct AvFormatInputDeleter {
    void operator()(AVFormatContext* ctx) const {
        if (ctx) {
            avformat_close_input(&ctx);
        }
    }
};

struct AvFormatOutputDeleter {
    void operator()(AVFormatContext* ctx) const {
        if (!ctx) return;
        if (!(ctx->oformat->flags & AVFMT_NOFILE) && ctx->pb) {
            avio_closep(&ctx->pb);
        }
        avformat_free_context(ctx);
    }
};

struct AvCodecContextDeleter {
    void operator()(AVCodecContext* ctx) const {
        if (ctx) {
            avcodec_free_context(&ctx);
        }
    }
};

struct AvPacketDeleter {
    void operator()(AVPacket* pkt) const {
        if (pkt) {
            av_packet_free(&pkt);
        }
    }
};

struct AvFrameDeleter {
    void operator()(AVFrame* frame) const {
        if (frame) {
            av_frame_free(&frame);
        }
    }
};

struct SwrContextDeleter {
    void operator()(SwrContext* ctx) const {
        if (ctx) {
            swr_free(&ctx);
        }
    }
};

using UniqueAVFormatInput = std::unique_ptr<AVFormatContext, AvFormatInputDeleter>;
using UniqueAVFormatOutput = std::unique_ptr<AVFormatContext, AvFormatOutputDeleter>;
using UniqueAVCodecContext = std::unique_ptr<AVCodecContext, AvCodecContextDeleter>;
using UniqueAVPacket = std::unique_ptr<AVPacket, AvPacketDeleter>;
using UniqueAVFrame = std::unique_ptr<AVFrame, AvFrameDeleter>;
using UniqueSwrContext = std::unique_ptr<SwrContext, SwrContextDeleter>;

int codecpar_num_channels(const AVCodecParameters* codecpar) {
    if (!codecpar) return 0;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    return codecpar->ch_layout.nb_channels;
#else
    return codecpar->channels;
#endif
}

int codecctx_num_channels(const AVCodecContext* codec_ctx) {
    if (!codec_ctx) return 0;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    return codec_ctx->ch_layout.nb_channels;
#else
    return codec_ctx->channels;
#endif
}

FfprobeInfo libav_audio_info(const std::string& path) {
    AVFormatContext* raw_ctx = nullptr;
    int rc = avformat_open_input(&raw_ctx, path.c_str(), nullptr, nullptr);
    if (rc < 0 || !raw_ctx) {
        throw value_error(
            "Unsupported input format on Linux backend. Supported formats: "
            ".wav, .mp3, .flac, .m4a, .aiff, .caf");
    }
    UniqueAVFormatInput format_ctx(raw_ctx);

    rc = avformat_find_stream_info(format_ctx.get(), nullptr);
    if (rc < 0) {
        throw value_error(
            "Failed to query audio stream metadata via libav for: " + path +
            " (" + av_error_to_string(rc) + ")");
    }

    int stream_index = av_find_best_stream(
        format_ctx.get(),
        AVMEDIA_TYPE_AUDIO,
        -1,
        -1,
        nullptr,
        0);
    if (stream_index < 0) {
        throw value_error(
            "Unsupported input format on Linux backend. Supported formats: "
            ".wav, .mp3, .flac, .m4a, .aiff, .caf");
    }

    AVStream* stream = format_ctx->streams[stream_index];
    AVCodecParameters* codecpar = stream->codecpar;
    FfprobeInfo info;
    info.sample_rate = codecpar->sample_rate;
    info.channels = codecpar_num_channels(codecpar);
    info.codec_name = avcodec_get_name(codecpar->codec_id);

    if (stream->duration > 0 && stream->time_base.den > 0) {
        info.duration = stream->duration * av_q2d(stream->time_base);
    } else if (format_ctx->duration > 0) {
        info.duration = static_cast<double>(format_ctx->duration) / AV_TIME_BASE;
    }

    if (stream->nb_frames > 0) {
        info.frames = stream->nb_frames;
    } else if (info.sample_rate > 0 && info.duration > 0.0) {
        info.frames = static_cast<int64_t>(
            std::llround(static_cast<long double>(info.duration) * info.sample_rate));
    }

    if (info.sample_rate <= 0 || info.channels <= 0) {
        throw value_error(
            "Failed to query audio stream metadata via libav for: " + path);
    }
    return info;
}

struct LibavDecodeContext {
    UniqueAVFormatInput format_ctx;
    UniqueAVCodecContext codec_ctx;
    int stream_index = -1;
};

LibavDecodeContext open_libav_decode_context(const std::string& path) {
    AVFormatContext* raw_ctx = nullptr;
    int rc = avformat_open_input(&raw_ctx, path.c_str(), nullptr, nullptr);
    if (rc < 0 || !raw_ctx) {
        throw value_error(
            "Failed to decode input format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }
    UniqueAVFormatInput format_ctx(raw_ctx);

    rc = avformat_find_stream_info(format_ctx.get(), nullptr);
    if (rc < 0) {
        throw value_error(
            "Failed to decode input format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 18, 100)
    AVCodec* decoder_for_find = nullptr;
#else
    const AVCodec* decoder_for_find = nullptr;
#endif
    int stream_index = av_find_best_stream(
        format_ctx.get(),
        AVMEDIA_TYPE_AUDIO,
        -1,
        -1,
        &decoder_for_find,
        0);
    const AVCodec* decoder = decoder_for_find;
    if (stream_index < 0 || !decoder) {
        throw value_error(
            "Failed to decode input format on Linux backend via libav: " + path);
    }

    AVCodecContext* raw_codec_ctx = avcodec_alloc_context3(decoder);
    if (!raw_codec_ctx) {
        throw std::runtime_error("Failed to allocate AVCodecContext");
    }
    UniqueAVCodecContext codec_ctx(raw_codec_ctx);

    AVStream* stream = format_ctx->streams[stream_index];
    rc = avcodec_parameters_to_context(codec_ctx.get(), stream->codecpar);
    if (rc < 0) {
        throw value_error(
            "Failed to decode input format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }

    rc = avcodec_open2(codec_ctx.get(), decoder, nullptr);
    if (rc < 0) {
        throw value_error(
            "Failed to decode input format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }

    return {std::move(format_ctx), std::move(codec_ctx), stream_index};
}

UniqueSwrContext create_libav_swr_context(
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
            throw value_error(
                "Failed to create libav resampler: " + av_error_to_string(rc));
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
        throw value_error(
            "Failed to create libav resampler: " + av_error_to_string(rc));
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
        throw value_error("Failed to create libav resampler");
    }
#endif

    UniqueSwrContext swr_ctx(swr_raw);
    int init_rc = swr_init(swr_ctx.get());
    if (init_rc < 0) {
        throw value_error(
            "Failed to initialize libav resampler: " +
            av_error_to_string(init_rc));
    }
    return swr_ctx;
}

std::pair<mlx::core::array, int> load_via_libav(
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype) {
    auto ctx = open_libav_decode_context(path);
    int native_sr = ctx.codec_ctx->sample_rate;
    int native_channels = codecctx_num_channels(ctx.codec_ctx.get());
    if (native_sr <= 0 || native_channels <= 0) {
        throw value_error(
            "Failed to decode input format on Linux backend via libav: " + path);
    }

    int out_sr = sr.value_or(native_sr);
    if (out_sr <= 0) {
        throw value_error("sr must be > 0");
    }

    auto swr_ctx = create_libav_swr_context(ctx.codec_ctx.get(), native_channels, out_sr);

    UniqueAVPacket packet(av_packet_alloc());
    UniqueAVFrame frame(av_frame_alloc());
    if (!packet || !frame) {
        throw std::runtime_error("Failed to allocate libav packet/frame");
    }

    std::vector<float> interleaved;
    // Estimate total samples from stream duration to avoid repeated reallocation.
    // Falls back to a modest default if duration metadata is unavailable.
    AVStream* av_stream = ctx.format_ctx->streams[ctx.stream_index];
    double est_duration = 0.0;
    if (av_stream->duration > 0 && av_stream->time_base.den > 0) {
        est_duration = av_stream->duration * av_q2d(av_stream->time_base);
    } else if (ctx.format_ctx->duration > 0) {
        est_duration = static_cast<double>(ctx.format_ctx->duration) / AV_TIME_BASE;
    }
    if (est_duration > 0.0) {
        size_t est_samples = static_cast<size_t>(
            std::ceil(est_duration * out_sr) * native_channels * 1.05);
        interleaved.reserve(est_samples);
    } else {
        interleaved.reserve(65536);
    }

    // Reusable buffers for the decode/convert loop — avoids per-frame heap churn
    std::vector<float> conv_chunk;
    std::vector<const uint8_t*> conv_in_data;

    auto append_converted_frame = [&](const AVFrame* src_frame) {
        if (!src_frame || src_frame->nb_samples <= 0) return;
        int64_t delay = swr_get_delay(swr_ctx.get(), native_sr);
        int out_samples = static_cast<int>(av_rescale_rnd(
            delay + src_frame->nb_samples,
            out_sr,
            native_sr,
            AV_ROUND_UP));
        if (out_samples <= 0) return;

        size_t needed = static_cast<size_t>(out_samples) * static_cast<size_t>(native_channels);
        conv_chunk.resize(needed);
        uint8_t* out_data[1] = {
            reinterpret_cast<uint8_t*>(conv_chunk.data())
        };
        const bool planar = av_sample_fmt_is_planar(
            static_cast<AVSampleFormat>(src_frame->format)) != 0;
        const int plane_count = planar ? native_channels : 1;
        conv_in_data.resize(static_cast<size_t>(plane_count));
        for (int i = 0; i < plane_count; ++i) {
            conv_in_data[static_cast<size_t>(i)] = src_frame->extended_data[i];
        }
        int converted = swr_convert(
            swr_ctx.get(),
            out_data,
            out_samples,
            conv_in_data.data(),
            src_frame->nb_samples);
        if (converted < 0) {
            throw value_error(
                "Failed to decode input format on Linux backend via libav: " + path +
                " (" + av_error_to_string(converted) + ")");
        }
        if (converted == 0) return;
        size_t emit = static_cast<size_t>(converted) * static_cast<size_t>(native_channels);
        interleaved.insert(interleaved.end(), conv_chunk.begin(), conv_chunk.begin() + emit);
    };

    while (true) {
        int rc = av_read_frame(ctx.format_ctx.get(), packet.get());
        if (rc == AVERROR_EOF) break;
        if (rc < 0) {
            throw value_error(
                "Failed to decode input format on Linux backend via libav: " + path +
                " (" + av_error_to_string(rc) + ")");
        }

        if (packet->stream_index != ctx.stream_index) {
            av_packet_unref(packet.get());
            continue;
        }

        rc = avcodec_send_packet(ctx.codec_ctx.get(), packet.get());
        av_packet_unref(packet.get());
        if (rc < 0 && rc != AVERROR(EAGAIN) && rc != AVERROR_EOF) {
            throw value_error(
                "Failed to decode input format on Linux backend via libav: " + path +
                " (" + av_error_to_string(rc) + ")");
        }

        while (true) {
            rc = avcodec_receive_frame(ctx.codec_ctx.get(), frame.get());
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
                break;
            }
            if (rc < 0) {
                throw value_error(
                    "Failed to decode input format on Linux backend via libav: " + path +
                    " (" + av_error_to_string(rc) + ")");
            }
            append_converted_frame(frame.get());
            av_frame_unref(frame.get());
        }
    }

    int rc = avcodec_send_packet(ctx.codec_ctx.get(), nullptr);
    if (rc < 0 && rc != AVERROR_EOF) {
        throw value_error(
            "Failed to decode input format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }
    while (true) {
        rc = avcodec_receive_frame(ctx.codec_ctx.get(), frame.get());
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
        if (rc < 0) {
            throw value_error(
                "Failed to decode input format on Linux backend via libav: " + path +
                " (" + av_error_to_string(rc) + ")");
        }
        append_converted_frame(frame.get());
        av_frame_unref(frame.get());
    }

    while (true) {
        int out_samples = static_cast<int>(av_rescale_rnd(
            swr_get_delay(swr_ctx.get(), native_sr),
            out_sr,
            native_sr,
            AV_ROUND_UP));
        if (out_samples <= 0) break;
        size_t needed = static_cast<size_t>(out_samples) * static_cast<size_t>(native_channels);
        conv_chunk.resize(needed);
        uint8_t* out_data[1] = {
            reinterpret_cast<uint8_t*>(conv_chunk.data())
        };
        int converted = swr_convert(
            swr_ctx.get(),
            out_data,
            out_samples,
            nullptr,
            0);
        if (converted < 0) {
            throw value_error(
                "Failed to decode input format on Linux backend via libav: " + path +
                " (" + av_error_to_string(converted) + ")");
        }
        if (converted == 0) break;
        size_t emit = static_cast<size_t>(converted) * static_cast<size_t>(native_channels);
        interleaved.insert(interleaved.end(), conv_chunk.begin(), conv_chunk.begin() + emit);
    }

    if (interleaved.empty()) {
        return tensor_utils::make_empty_audio_result(
            out_sr, native_channels, mono, layout, dtype);
    }

    int64_t total_frames = static_cast<int64_t>(interleaved.size() / native_channels);
    int64_t start_frame = static_cast<int64_t>(std::floor(offset * out_sr));
    if (start_frame >= total_frames) {
        return tensor_utils::make_empty_audio_result(
            out_sr, native_channels, mono, layout, dtype);
    }

    int64_t frames = total_frames - start_frame;
    if (duration.has_value()) {
        int64_t duration_frames =
            static_cast<int64_t>(std::floor(duration.value() * out_sr));
        duration_frames = std::max<int64_t>(0, duration_frames);
        frames = std::min<int64_t>(frames, duration_frames);
    }
    if (frames <= 0) {
        return tensor_utils::make_empty_audio_result(
            out_sr, native_channels, mono, layout, dtype);
    }

    size_t out_bytes = static_cast<size_t>(frames) * native_channels * sizeof(float);
    float* buffer = static_cast<float*>(aligned_alloc_64(out_bytes));
    const float* src =
        interleaved.data() + static_cast<size_t>(start_frame) * native_channels;
    std::memcpy(buffer, src, out_bytes);

    return tensor_utils::wrap_interleaved_audio_buffer(
        buffer, frames, native_channels, out_sr, mono, layout, dtype);
}

int parse_libav_bitrate_bps(const std::string& bitrate, const std::string& label) {
    if (bitrate.empty()) return 0;
    if (bitrate.size() < 2 || bitrate.back() != 'k') {
        throw value_error("Invalid bitrate/subtype '" + bitrate + "' for " + label + ".");
    }
    int kbps = 0;
    try {
        kbps = std::stoi(bitrate.substr(0, bitrate.size() - 1));
    } catch (...) {
        throw value_error("Invalid bitrate/subtype '" + bitrate + "' for " + label + ".");
    }
    if (kbps <= 0) {
        throw value_error("Invalid bitrate/subtype '" + bitrate + "' for " + label + ".");
    }
    return kbps * 1000;
}

AVCodecID codec_id_for_transcode_codec(const std::string& codec_name) {
    if (codec_name == "mp3" || codec_name == "libmp3lame") return AV_CODEC_ID_MP3;
    if (codec_name == "flac") return AV_CODEC_ID_FLAC;
    if (codec_name == "aac") return AV_CODEC_ID_AAC;
    if (codec_name == "alac") return AV_CODEC_ID_ALAC;
    if (codec_name == "pcm_f32be") return AV_CODEC_ID_PCM_F32BE;
    if (codec_name == "pcm_s16be") return AV_CODEC_ID_PCM_S16BE;
    if (codec_name == "pcm_f32le") return AV_CODEC_ID_PCM_F32LE;
    if (codec_name == "pcm_s16le") return AV_CODEC_ID_PCM_S16LE;
    return AV_CODEC_ID_NONE;
}

AVSampleFormat choose_encoder_sample_fmt(
    const AVCodec* encoder,
    AVSampleFormat preferred) {
    if (!encoder->sample_fmts) {
        return preferred == AV_SAMPLE_FMT_NONE ? AV_SAMPLE_FMT_FLT : preferred;
    }
    if (preferred != AV_SAMPLE_FMT_NONE) {
        for (const AVSampleFormat* p = encoder->sample_fmts;
             *p != AV_SAMPLE_FMT_NONE; ++p) {
            if (*p == preferred) {
                return preferred;
            }
        }
    }
    return encoder->sample_fmts[0];
}

UniqueSwrContext create_libav_encode_swr_context(
    const AVCodecContext* codec_ctx,
    int channels,
    int sample_rate) {
    SwrContext* swr_raw = nullptr;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    AVChannelLayout in_layout;
    AVChannelLayout out_layout;
    std::memset(&in_layout, 0, sizeof(in_layout));
    std::memset(&out_layout, 0, sizeof(out_layout));

    av_channel_layout_default(&in_layout, channels);
    int rc = av_channel_layout_copy(&out_layout, &codec_ctx->ch_layout);
    if (rc < 0) {
        av_channel_layout_uninit(&in_layout);
        throw value_error(
            "Failed to initialize output encoder on Linux backend via libav (" +
            av_error_to_string(rc) + ")");
    }

    rc = swr_alloc_set_opts2(
        &swr_raw,
        &out_layout,
        codec_ctx->sample_fmt,
        sample_rate,
        &in_layout,
        AV_SAMPLE_FMT_FLT,
        sample_rate,
        0,
        nullptr);
    av_channel_layout_uninit(&in_layout);
    av_channel_layout_uninit(&out_layout);
    if (rc < 0 || !swr_raw) {
        throw value_error(
            "Failed to initialize output encoder on Linux backend via libav (" +
            av_error_to_string(rc) + ")");
    }
#else
    int64_t out_ch_layout = codec_ctx->channel_layout;
    if (out_ch_layout == 0) {
        out_ch_layout = av_get_default_channel_layout(channels);
    }
    int64_t in_ch_layout = av_get_default_channel_layout(channels);
    swr_raw = swr_alloc_set_opts(
        nullptr,
        out_ch_layout,
        codec_ctx->sample_fmt,
        sample_rate,
        in_ch_layout,
        AV_SAMPLE_FMT_FLT,
        sample_rate,
        0,
        nullptr);
    if (!swr_raw) {
        throw value_error("Failed to initialize output encoder on Linux backend via libav");
    }
#endif

    UniqueSwrContext swr_ctx(swr_raw);
    int init_rc = swr_init(swr_ctx.get());
    if (init_rc < 0) {
        throw value_error(
            "Failed to initialize output encoder on Linux backend via libav (" +
            av_error_to_string(init_rc) + ")");
    }
    return swr_ctx;
}

UniqueAVFrame allocate_libav_encode_frame(const AVCodecContext* codec_ctx, int nb_samples) {
    AVFrame* raw_frame = av_frame_alloc();
    if (!raw_frame) {
        throw std::runtime_error("Failed to allocate encoder frame");
    }
    UniqueAVFrame frame(raw_frame);
    frame->format = codec_ctx->sample_fmt;
    frame->sample_rate = codec_ctx->sample_rate;
    frame->nb_samples = nb_samples;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    int layout_rc = av_channel_layout_copy(&frame->ch_layout, &codec_ctx->ch_layout);
    if (layout_rc < 0) {
        throw value_error(
            "Failed to allocate output encoder frame on Linux backend via libav (" +
            av_error_to_string(layout_rc) + ")");
    }
#else
    frame->channel_layout = codec_ctx->channel_layout;
    frame->channels = codec_ctx->channels;
#endif

    int rc = av_frame_get_buffer(frame.get(), 0);
    if (rc < 0) {
        throw value_error(
            "Failed to allocate output encoder frame on Linux backend via libav (" +
            av_error_to_string(rc) + ")");
    }
    rc = av_frame_make_writable(frame.get());
    if (rc < 0) {
        throw value_error(
            "Failed to allocate output encoder frame on Linux backend via libav (" +
            av_error_to_string(rc) + ")");
    }
    return frame;
}

void drain_libav_encoder_packets(
    AVCodecContext* codec_ctx,
    AVFormatContext* out_ctx,
    AVStream* stream,
    AVPacket* packet,
    const std::string& path) {
    while (true) {
        int rc = avcodec_receive_packet(codec_ctx, packet);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
            return;
        }
        if (rc < 0) {
            throw value_error(
                "Failed to encode output format on Linux backend via libav: " + path +
                " (" + av_error_to_string(rc) + ")");
        }

        av_packet_rescale_ts(packet, codec_ctx->time_base, stream->time_base);
        packet->stream_index = stream->index;

        rc = av_interleaved_write_frame(out_ctx, packet);
        av_packet_unref(packet);
        if (rc < 0) {
            throw value_error(
                "Failed to encode output format on Linux backend via libav: " + path +
                " (" + av_error_to_string(rc) + ")");
        }
    }
}

void save_via_libav(
    const std::string& path,
    const float* write_data,
    int frames,
    int channels,
    int sr,
    const std::string& transcode_codec,
    const std::string& transcode_bitrate,
    AVSampleFormat preferred_sample_fmt) {
    const AVCodecID codec_id = codec_id_for_transcode_codec(transcode_codec);
    if (codec_id == AV_CODEC_ID_NONE) {
        throw value_error(
            "No libav codec mapping for '" + transcode_codec +
            "' when encoding: " + path);
    }

    const AVCodec* encoder = avcodec_find_encoder(codec_id);
    if (!encoder) {
        throw value_error(
            "libav encoder not found for codec '" + transcode_codec +
            "' (ensure the required encoder is installed): " + path);
    }

    const std::string ext = lowercase_extension(path);
    const char* muxer_name = (ext == ".m4a") ? "ipod" : nullptr;

    AVFormatContext* raw_out_ctx = nullptr;
    int rc = avformat_alloc_output_context2(&raw_out_ctx, nullptr, muxer_name, path.c_str());
    if (rc < 0 || !raw_out_ctx) {
        throw value_error(
            "Failed to encode output format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }
    UniqueAVFormatOutput out_ctx(raw_out_ctx);

    AVStream* stream = avformat_new_stream(out_ctx.get(), nullptr);
    if (!stream) {
        throw std::runtime_error("Failed to allocate output stream");
    }

    AVCodecContext* raw_codec_ctx = avcodec_alloc_context3(encoder);
    if (!raw_codec_ctx) {
        throw std::runtime_error("Failed to allocate output codec context");
    }
    UniqueAVCodecContext codec_ctx(raw_codec_ctx);

    codec_ctx->codec_id = codec_id;
    codec_ctx->sample_rate = sr;
    codec_ctx->time_base = AVRational{1, sr};
    stream->time_base = codec_ctx->time_base;
#if LIBAVUTIL_VERSION_MAJOR >= 57
    av_channel_layout_default(&codec_ctx->ch_layout, channels);
#else
    codec_ctx->channels = channels;
    codec_ctx->channel_layout = av_get_default_channel_layout(channels);
#endif
    codec_ctx->sample_fmt = choose_encoder_sample_fmt(encoder, preferred_sample_fmt);

    if (!transcode_bitrate.empty()) {
        const std::string label = (ext == ".mp3") ? "MP3" : "M4A";
        codec_ctx->bit_rate = parse_libav_bitrate_bps(transcode_bitrate, label);
    }

    if (out_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    rc = avcodec_open2(codec_ctx.get(), encoder, nullptr);
    if (rc < 0) {
        throw value_error(
            "Failed to encode output format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }

    rc = avcodec_parameters_from_context(stream->codecpar, codec_ctx.get());
    if (rc < 0) {
        throw value_error(
            "Failed to encode output format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }

    if (!(out_ctx->oformat->flags & AVFMT_NOFILE)) {
        rc = avio_open(&out_ctx->pb, path.c_str(), AVIO_FLAG_WRITE);
        if (rc < 0) {
            throw value_error(
                "Failed to encode output format on Linux backend via libav: " + path +
                " (" + av_error_to_string(rc) + ")");
        }
    }

    rc = avformat_write_header(out_ctx.get(), nullptr);
    if (rc < 0) {
        throw value_error(
            "Failed to encode output format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }

    auto swr_ctx = create_libav_encode_swr_context(codec_ctx.get(), channels, sr);
    UniqueAVPacket packet(av_packet_alloc());
    if (!packet) {
        throw std::runtime_error("Failed to allocate encoder packet");
    }

    const int frame_chunk = (codec_ctx->frame_size > 0) ? codec_ctx->frame_size : 4096;
    int64_t cursor = 0;
    int64_t next_pts = 0;

    // Allocate a single reusable frame for the encode loop
    auto out_frame = allocate_libav_encode_frame(codec_ctx.get(), frame_chunk);

    while (cursor < frames) {
        int in_samples = static_cast<int>(std::min<int64_t>(frame_chunk, frames - cursor));

        rc = av_frame_make_writable(out_frame.get());
        if (rc < 0) {
            throw value_error(
                "Failed to make encoder frame writable on Linux backend via libav (" +
                av_error_to_string(rc) + ")");
        }

        const uint8_t* in_data[1] = {
            reinterpret_cast<const uint8_t*>(write_data + cursor * channels)
        };
        int converted = swr_convert(
            swr_ctx.get(),
            out_frame->data,
            in_samples,
            in_data,
            in_samples);
        if (converted < 0) {
            throw value_error(
                "Failed to encode output format on Linux backend via libav: " + path +
                " (" + av_error_to_string(converted) + ")");
        }
        if (converted == 0) {
            cursor += in_samples;
            continue;
        }

        out_frame->nb_samples = converted;
        out_frame->pts = next_pts;
        next_pts += converted;

        rc = avcodec_send_frame(codec_ctx.get(), out_frame.get());
        if (rc < 0) {
            throw value_error(
                "Failed to encode output format on Linux backend via libav: " + path +
                " (" + av_error_to_string(rc) + ")");
        }
        drain_libav_encoder_packets(
            codec_ctx.get(), out_ctx.get(), stream, packet.get(), path);

        cursor += in_samples;
    }

    while (true) {
        int flush_samples = static_cast<int>(av_rescale_rnd(
            swr_get_delay(swr_ctx.get(), sr),
            codec_ctx->sample_rate,
            sr,
            AV_ROUND_UP));
        if (flush_samples <= 0) break;
        flush_samples = std::min(flush_samples, frame_chunk);

        rc = av_frame_make_writable(out_frame.get());
        if (rc < 0) {
            throw value_error(
                "Failed to make encoder frame writable on Linux backend via libav (" +
                av_error_to_string(rc) + ")");
        }

        int converted = swr_convert(
            swr_ctx.get(),
            out_frame->data,
            flush_samples,
            nullptr,
            0);
        if (converted < 0) {
            throw value_error(
                "Failed to encode output format on Linux backend via libav: " + path +
                " (" + av_error_to_string(converted) + ")");
        }
        if (converted == 0) break;

        out_frame->nb_samples = converted;
        out_frame->pts = next_pts;
        next_pts += converted;

        rc = avcodec_send_frame(codec_ctx.get(), out_frame.get());
        if (rc < 0) {
            throw value_error(
                "Failed to encode output format on Linux backend via libav: " + path +
                " (" + av_error_to_string(rc) + ")");
        }
        drain_libav_encoder_packets(
            codec_ctx.get(), out_ctx.get(), stream, packet.get(), path);
    }

    rc = avcodec_send_frame(codec_ctx.get(), nullptr);
    if (rc < 0) {
        throw value_error(
            "Failed to encode output format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }
    drain_libav_encoder_packets(codec_ctx.get(), out_ctx.get(), stream, packet.get(), path);

    rc = av_write_trailer(out_ctx.get());
    if (rc < 0) {
        throw value_error(
            "Failed to encode output format on Linux backend via libav: " + path +
            " (" + av_error_to_string(rc) + ")");
    }
}

std::string map_codec_name_to_subtype(const std::string& codec_name) {
    if (codec_name == "mp3") return "mp3";
    if (codec_name == "aac") return "aac";
    if (codec_name == "alac") return "alac";
    if (codec_name == "flac") return "flac";

    if (codec_name.rfind("pcm_s16", 0) == 0) return "pcm16";
    if (codec_name.rfind("pcm_s24", 0) == 0) return "pcm24";
    if (codec_name.rfind("pcm_s32", 0) == 0) return "pcm32";
    if (codec_name.rfind("pcm_u8", 0) == 0) return "pcm8";
    if (codec_name.rfind("pcm_f32", 0) == 0) return "float32";

    return codec_name.empty() ? "unknown" : codec_name;
}

std::string container_from_path(const std::string& path) {
    const auto ext = lowercase_extension(path);
    if (ext == ".wav") return "wav";
    if (ext == ".mp3") return "mp3";
    if (ext == ".flac") return "flac";
    if (ext == ".m4a") return "m4a";
    if (ext == ".aiff" || ext == ".aif") return "aiff";
    if (ext == ".caf") return "caf";
    return "unknown";
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

bool is_mp3_path_ci(const std::string& path) {
    return is_mp3_path(path);
}

int parse_resample_quality(const std::string& quality) {
    if (quality == "default") return -1;
    if (quality == "fastest") return 0;
    if (quality == "low") return 1;
    if (quality == "medium") return 2;
    if (quality == "high") return 3;
    if (quality == "best") return 4;

    throw value_error(
        "Invalid resample_quality '" + quality +
        "'. Must be 'default', 'fastest', 'low', 'medium', 'high', or 'best'.");
}

std::pair<float*, int64_t> resample_linear_interleaved(
    float* in,
    int64_t in_frames,
    int channels,
    int in_sr,
    int out_sr) {
    if (out_sr == in_sr || in_frames == 0) {
        return {in, in_frames};
    }

    int64_t out_frames = static_cast<int64_t>(
        std::llround(static_cast<long double>(in_frames) * out_sr / in_sr));
    if (out_frames <= 0) {
        out_frames = 0;
    }

    size_t out_bytes = static_cast<size_t>(std::max<int64_t>(out_frames, 1)) *
                       channels * sizeof(float);
    float* out = static_cast<float*>(aligned_alloc_64(out_bytes));

    if (out_frames == 0) {
        std::free(in);
        return {out, 0};
    }

    if (in_frames == 1) {
        for (int64_t i = 0; i < out_frames; ++i) {
            for (int c = 0; c < channels; ++c) {
                out[i * channels + c] = in[c];
            }
        }
        std::free(in);
        return {out, out_frames};
    }

    const double step = static_cast<double>(in_sr) / static_cast<double>(out_sr);

    for (int64_t i = 0; i < out_frames; ++i) {
        double src_pos = i * step;
        int64_t idx = static_cast<int64_t>(src_pos);
        if (idx >= in_frames - 1) {
            idx = in_frames - 2;
            src_pos = static_cast<double>(idx + 1);
        }
        double frac = src_pos - static_cast<double>(idx);

        const float* s0 = in + idx * channels;
        const float* s1 = s0 + channels;
        float* dst = out + i * channels;

        for (int c = 0; c < channels; ++c) {
            dst[c] = static_cast<float>(s0[c] + (s1[c] - s0[c]) * frac);
        }
    }

    std::free(in);
    return {out, out_frames};
}

std::pair<mlx::core::array, int> load_wav(
    const WavInfo& wav,
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype) {
    int64_t start_frame = static_cast<int64_t>(std::floor(offset * wav.sample_rate));
    if (start_frame >= wav.total_frames) {
        return tensor_utils::make_empty_audio_result(
            sr.value_or(wav.sample_rate), wav.channels, mono, layout, dtype);
    }

    int64_t frames_to_read;
    if (duration.has_value()) {
        frames_to_read =
            static_cast<int64_t>(std::floor(duration.value() * wav.sample_rate));
        frames_to_read = std::max<int64_t>(0, frames_to_read);
        if (start_frame + frames_to_read > wav.total_frames) {
            frames_to_read = wav.total_frames - start_frame;
        }
    } else {
        frames_to_read = wav.total_frames - start_frame;
    }

    int block_align = wav.channels * (wav.bits_per_sample / 8);
    int64_t read_bytes = frames_to_read * block_align;

    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path.c_str(), "rb"), fclose);
    if (!f) {
        throw std::runtime_error("Failed to open WAV file: " + path);
    }

    int64_t byte_offset = wav.data_offset + start_frame * block_align;
    if (fseek(f.get(), static_cast<long>(byte_offset), SEEK_SET) != 0) {
        throw std::runtime_error("Failed to seek in WAV file: " + path);
    }

    size_t out_bytes = static_cast<size_t>(std::max<int64_t>(frames_to_read, 1)) *
                       wav.channels * sizeof(float);
    float* buffer = static_cast<float*>(aligned_alloc_64(out_bytes));

    int64_t actual_frames = frames_to_read;

    if (wav.format_tag == 3 && wav.bits_per_sample == 32) {
        size_t got = fread(buffer, 1, static_cast<size_t>(read_bytes), f.get());
        actual_frames = static_cast<int64_t>(got) / (wav.channels * sizeof(float));
    } else if (wav.format_tag == 1 && wav.bits_per_sample == 16) {
        // Read int16 data into the tail of the float buffer, convert forward.
        // float32 is 4 bytes, int16 is 2 bytes, so int16 data fits in the
        // upper half of the buffer without overlap issues.
        size_t sample_count = static_cast<size_t>(frames_to_read) * wav.channels;
        int16_t* pcm_region = reinterpret_cast<int16_t*>(
            reinterpret_cast<char*>(buffer) + sample_count * sizeof(float)
            - sample_count * sizeof(int16_t));

        size_t got = fread(pcm_region, 1, static_cast<size_t>(read_bytes), f.get());
        actual_frames = static_cast<int64_t>(got) / (wav.channels * sizeof(int16_t));
        size_t actual_samples = static_cast<size_t>(actual_frames) * wav.channels;

        constexpr float kScale = 1.0f / 32768.0f;
        for (size_t i = 0; i < actual_samples; ++i) {
            buffer[i] = static_cast<float>(pcm_region[i]) * kScale;
        }
    } else if (wav.format_tag == 1 && wav.bits_per_sample == 24) {
        size_t sample_count = static_cast<size_t>(frames_to_read) * wav.channels;
        size_t raw_bytes = sample_count * 3;
        uint8_t* raw = static_cast<uint8_t*>(aligned_alloc_64(raw_bytes));

        size_t got = fread(raw, 1, raw_bytes, f.get());
        size_t actual_samples = got / 3;
        actual_frames = static_cast<int64_t>(actual_samples) / wav.channels;

        constexpr float kInv = 1.0f / 8388608.0f;
        for (size_t i = 0; i < actual_samples; ++i) {
            int32_t val = static_cast<int32_t>(raw[i * 3]) |
                          (static_cast<int32_t>(raw[i * 3 + 1]) << 8) |
                          (static_cast<int32_t>(raw[i * 3 + 2]) << 16);
            if (val & 0x800000) val |= 0xFF000000;
            buffer[i] = static_cast<float>(val) * kInv;
        }

        std::free(raw);
    } else if (wav.format_tag == 1 && wav.bits_per_sample == 32) {
        // Read int32 directly into the float buffer (same width), convert in-place.
        size_t got = fread(buffer, 1, static_cast<size_t>(read_bytes), f.get());
        actual_frames = static_cast<int64_t>(got) / (wav.channels * sizeof(int32_t));
        size_t actual_samples = static_cast<size_t>(actual_frames) * wav.channels;

        int32_t* pcm_buf = reinterpret_cast<int32_t*>(buffer);
        constexpr float kScale = 1.0f / 2147483648.0f;
        for (size_t i = 0; i < actual_samples; ++i) {
            buffer[i] = static_cast<float>(pcm_buf[i]) * kScale;
        }
    } else {
        std::free(buffer);
        throw value_error(
            "Unsupported WAV encoding on Linux backend: format_tag=" +
            std::to_string(wav.format_tag) + ", bits_per_sample=" +
            std::to_string(wav.bits_per_sample));
    }

    int out_sr = sr.value_or(wav.sample_rate);
    if (out_sr <= 0) {
        std::free(buffer);
        throw value_error("sr must be > 0");
    }

    if (out_sr != wav.sample_rate) {
        auto resampled =
            resample_linear_interleaved(buffer, actual_frames, wav.channels, wav.sample_rate, out_sr);
        buffer = resampled.first;
        actual_frames = resampled.second;
    }

    return tensor_utils::wrap_interleaved_audio_buffer(
        buffer, actual_frames, wav.channels, out_sr, mono, layout, dtype);
}

std::pair<mlx::core::array, int> load_mp3(
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype) {
    ScopedMp3Decoder dec;
    dec.open(path);

    int native_sr = dec.sample_rate();
    int channels = dec.channels();
    int64_t total_frames = dec.total_frames();

    int64_t start_frame = static_cast<int64_t>(std::floor(offset * native_sr));
    if (start_frame >= total_frames) {
        return tensor_utils::make_empty_audio_result(
            sr.value_or(native_sr), channels, mono, layout, dtype);
    }

    int64_t frames_to_read;
    if (duration.has_value()) {
        frames_to_read = static_cast<int64_t>(std::floor(duration.value() * native_sr));
        frames_to_read = std::max<int64_t>(0, frames_to_read);
        if (start_frame + frames_to_read > total_frames) {
            frames_to_read = total_frames - start_frame;
        }
    } else {
        frames_to_read = total_frames - start_frame;
    }

    if (start_frame > 0) {
        dec.seek(start_frame * channels);
    }

    size_t buffer_bytes = static_cast<size_t>(std::max<int64_t>(frames_to_read, 1)) *
                          channels * sizeof(float);
    float* buffer = static_cast<float*>(aligned_alloc_64(buffer_bytes));

    int64_t samples_wanted = frames_to_read * channels;
    int64_t samples_read = dec.read(buffer, samples_wanted);
    int64_t actual_frames = (channels > 0) ? samples_read / channels : 0;

    int out_sr = sr.value_or(native_sr);
    if (out_sr <= 0) {
        std::free(buffer);
        throw value_error("sr must be > 0");
    }

    if (out_sr != native_sr) {
        auto resampled =
            resample_linear_interleaved(buffer, actual_frames, channels, native_sr, out_sr);
        buffer = resampled.first;
        actual_frames = resampled.second;
    }

    return tensor_utils::wrap_interleaved_audio_buffer(
        buffer, actual_frames, channels, out_sr, mono, layout, dtype);
}

std::string wav_subtype_from_info(const WavInfo& wav) {
    if (wav.format_tag == 3 && wav.bits_per_sample == 32) return "float32";
    if (wav.format_tag == 1 && wav.bits_per_sample == 8) return "pcm8";
    if (wav.format_tag == 1 && wav.bits_per_sample == 16) return "pcm16";
    if (wav.format_tag == 1 && wav.bits_per_sample == 24) return "pcm24";
    if (wav.format_tag == 1 && wav.bits_per_sample == 32) return "pcm32";
    return "unknown";
}

void write_u16_le(FILE* f, uint16_t value) {
    fwrite(&value, sizeof(value), 1, f);
}

void write_u32_le(FILE* f, uint32_t value) {
    fwrite(&value, sizeof(value), 1, f);
}

void write_wav_header(
    FILE* f,
    int sr,
    int channels,
    int bits_per_sample,
    bool is_float,
    uint32_t data_size) {
    const uint16_t format_tag = is_float ? 3 : 1;
    const uint16_t block_align =
        static_cast<uint16_t>(channels * (bits_per_sample / 8));
    const uint32_t byte_rate =
        static_cast<uint32_t>(sr) * static_cast<uint32_t>(block_align);

    fwrite("RIFF", 1, 4, f);
    write_u32_le(f, 36u + data_size);
    fwrite("WAVE", 1, 4, f);

    fwrite("fmt ", 1, 4, f);
    write_u32_le(f, 16u);
    write_u16_le(f, format_tag);
    write_u16_le(f, static_cast<uint16_t>(channels));
    write_u32_le(f, static_cast<uint32_t>(sr));
    write_u32_le(f, byte_rate);
    write_u16_le(f, block_align);
    write_u16_le(f, static_cast<uint16_t>(bits_per_sample));

    fwrite("data", 1, 4, f);
    write_u32_le(f, data_size);
}

void write_interleaved_wav(
    const std::string& path,
    const float* write_data,
    int frames,
    int channels,
    int sr,
    const std::string& wav_encoding) {
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path.c_str(), "wb"), fclose);
    if (!f) {
        throw std::runtime_error("Failed to open output file for writing: " + path);
    }

    if (wav_encoding == "float32") {
        uint32_t data_size = static_cast<uint32_t>(
            static_cast<uint64_t>(frames) * channels * sizeof(float));
        write_wav_header(f.get(), sr, channels, 32, true, data_size);

        size_t wrote = fwrite(
            write_data, sizeof(float), static_cast<size_t>(frames) * channels, f.get());
        size_t expected = static_cast<size_t>(frames) * channels;
        if (wrote != expected) {
            throw std::runtime_error("Failed to write float32 WAV payload");
        }
        return;
    }

    uint32_t data_size = static_cast<uint32_t>(
        static_cast<uint64_t>(frames) * channels * sizeof(int16_t));
    write_wav_header(f.get(), sr, channels, 16, false, data_size);

    size_t total = static_cast<size_t>(frames) * channels;
    std::unique_ptr<int16_t, decltype(&std::free)> pcm(
        static_cast<int16_t*>(aligned_alloc_64(total * sizeof(int16_t))), std::free);

    for (size_t i = 0; i < total; ++i) {
        float x = write_data[i];
        if (x <= -1.0f) {
            pcm.get()[i] = static_cast<int16_t>(-32768);
        } else if (x >= 1.0f) {
            pcm.get()[i] = static_cast<int16_t>(32767);
        } else {
            pcm.get()[i] = static_cast<int16_t>(std::lrint(x * 32767.0f));
        }
    }

    size_t wrote = fwrite(pcm.get(), sizeof(int16_t), total, f.get());
    if (wrote != total) {
        throw std::runtime_error("Failed to write pcm16 WAV payload");
    }
}

bool is_supported_m4a_bitrate(const std::string& bitrate) {
    return bitrate == "auto" || bitrate == "128k" || bitrate == "192k" ||
           bitrate == "256k" || bitrate == "320k";
}

bool is_supported_mp3_bitrate(const std::string& bitrate) {
    return bitrate == "auto" || bitrate == "128k" || bitrate == "192k" ||
           bitrate == "256k" || bitrate == "320k";
}

}  // namespace

AudioFileInfo backend_get_info(const std::string& path) {
    check_file_exists(path);

    if (is_wav_path(path)) {
        WavInfo wav;
        if (!parse_wav_header(path, wav)) {
            throw value_error("Failed to parse WAV header: " + path);
        }

        AudioFileInfo info;
        info.frames = wav.total_frames;
        info.sample_rate = wav.sample_rate;
        info.channels = wav.channels;
        info.duration = (wav.sample_rate > 0)
                            ? static_cast<double>(wav.total_frames) / wav.sample_rate
                            : 0.0;
        info.subtype = wav_subtype_from_info(wav);
        info.container = "wav";
        return info;
    }

    if (is_mp3_path_ci(path)) {
        ScopedMp3Decoder dec;
        dec.open(path);

        AudioFileInfo info;
        info.frames = dec.total_frames();
        info.sample_rate = dec.sample_rate();
        info.channels = dec.channels();
        info.duration = (info.sample_rate > 0)
                            ? static_cast<double>(info.frames) / info.sample_rate
                            : 0.0;
        info.subtype = "mp3";
        info.container = "mp3";
        return info;
    }

    if (is_libav_input_path(path)) {
        const auto probe = libav_audio_info(path);

        AudioFileInfo info;
        info.frames = probe.frames;
        info.sample_rate = probe.sample_rate;
        info.channels = probe.channels;
        info.duration = probe.duration;
        info.subtype = map_codec_name_to_subtype(probe.codec_name);
        info.container = container_from_path(path);
        return info;
    }

    throw value_error(
        "Unsupported input format on Linux backend. Supported formats: "
        ".wav, .mp3, .flac, .m4a, .aiff, .caf");
}

std::pair<mlx::core::array, int> backend_load_audio(
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& resample_quality) {
    if (dtype != "float32" && dtype != "float16") {
        throw value_error("Unsupported dtype '" + dtype + "'. Must be 'float32' or 'float16'.");
    }
    if (layout != "channels_last" && layout != "channels_first") {
        throw value_error(
            "Invalid layout '" + layout +
            "'. Must be 'channels_last' or 'channels_first'.");
    }
    if (offset < 0.0) {
        throw value_error("offset must be >= 0, got " + std::to_string(offset));
    }
    if (duration.has_value() && duration.value() <= 0.0) {
        throw value_error("duration must be > 0, got " + std::to_string(duration.value()));
    }
    if (sr.has_value() && sr.value() <= 0) {
        throw value_error("sr must be > 0");
    }

    parse_resample_quality(resample_quality);

    check_file_exists(path);

    if (is_wav_path(path)) {
        WavInfo wav;
        if (!parse_wav_header(path, wav)) {
            throw value_error("Failed to parse WAV header: " + path);
        }
        return load_wav(wav, path, sr, offset, duration, mono, layout, dtype);
    }

    if (is_mp3_path_ci(path)) {
        return load_mp3(path, sr, offset, duration, mono, layout, dtype);
    }

    if (is_libav_input_path(path)) {
        return load_via_libav(path, sr, offset, duration, mono, layout, dtype);
    }

    throw value_error(
        "Unsupported input format on Linux backend. Supported formats: "
        ".wav, .mp3, .flac, .m4a, .aiff, .caf");
}

void backend_save_audio(
    const std::string& path,
    mlx::core::array audio,
    int sr,
    const std::string& layout,
    const std::string& encoding,
    const std::string& bitrate,
    bool clip) {
    if (layout != "channels_last" && layout != "channels_first") {
        throw value_error(
            "Invalid layout '" + layout +
            "'. Must be 'channels_last' or 'channels_first'.");
    }
    if (audio.dtype() != mlx::core::float32 && audio.dtype() != mlx::core::float16) {
        throw value_error("audio must be float32 or float16");
    }
    if (sr <= 0) {
        throw value_error("sr must be > 0");
    }

    const std::string ext = lowercase_extension(path);
    const bool output_is_wav = (ext == ".wav");
    const bool output_is_libav = is_libav_output_path(path);
    if (!output_is_wav && !output_is_libav) {
        throw value_error(
            "Unsupported output format on Linux backend. Supported formats: "
            ".wav, .mp3, .flac, .m4a, .aiff, .caf");
    }

    if (audio.dtype() == mlx::core::float16) {
        audio = mlx::core::astype(audio, mlx::core::float32);
    }

    if (audio.ndim() == 1) {
        audio = mlx::core::reshape(audio, {audio.shape(0), 1});
    } else if (audio.ndim() != 2) {
        throw value_error(
            "audio must be 1D or 2D, got ndim=" + std::to_string(audio.ndim()));
    }

    int frames;
    int channels;
    if (layout == "channels_last") {
        frames = audio.shape(0);
        channels = audio.shape(1);
    } else {
        channels = audio.shape(0);
        frames = audio.shape(1);
    }

    mlx::core::eval(audio);

    const float* data = audio.data<float>();

    std::unique_ptr<float, decltype(&std::free)> interleaved(nullptr, std::free);
    const float* write_data = data;

    if (layout == "channels_first" && channels > 1) {
        size_t bytes = static_cast<size_t>(frames) * channels * sizeof(float);
        interleaved.reset(static_cast<float*>(aligned_alloc_64(bytes)));

        for (int c = 0; c < channels; ++c) {
            internal::strided_copy(
                data + c * frames,
                1,
                interleaved.get() + c,
                channels,
                frames);
        }

        write_data = interleaved.get();
    }

    // Clip to [-1, 1]. If we already have a temp buffer (from channels_first
    // interleaving), clip in-place to avoid a second allocation.
    std::unique_ptr<float, decltype(&std::free)> clipped(nullptr, std::free);
    if (clip && interleaved) {
        size_t total = static_cast<size_t>(frames) * channels;
        for (size_t i = 0; i < total; ++i) {
            interleaved.get()[i] = std::max(-1.0f, std::min(1.0f, interleaved.get()[i]));
        }
    } else if (clip) {
        size_t total = static_cast<size_t>(frames) * channels;
        clipped.reset(static_cast<float*>(aligned_alloc_64(total * sizeof(float))));
        for (size_t i = 0; i < total; ++i) {
            clipped.get()[i] = std::max(-1.0f, std::min(1.0f, write_data[i]));
        }
        write_data = clipped.get();
    }

    if (output_is_wav) {
        if (bitrate != "auto") {
            throw value_error("bitrate is only supported for encoded formats, not WAV");
        }

        std::string wav_encoding = encoding;
        if (wav_encoding == "auto") {
            wav_encoding = "float32";
        }
        if (wav_encoding != "float32" && wav_encoding != "pcm16") {
            throw value_error(
                "Unsupported encoding '" + encoding +
                "' for .wav on Linux backend. Use 'float32' or 'pcm16'.");
        }

        write_interleaved_wav(path, write_data, frames, channels, sr, wav_encoding);
        return;
    }

    std::string transcode_codec;
    std::string transcode_bitrate;
    AVSampleFormat libav_preferred_fmt = AV_SAMPLE_FMT_NONE;

    if (ext == ".mp3") {
        if (!is_supported_mp3_bitrate(bitrate)) {
            throw value_error(
                "Invalid bitrate/subtype '" + bitrate +
                "' for MP3. Use 'auto', '128k', '192k', '256k', or '320k'.");
        }
        if (encoding != "auto" && encoding != "float32" && encoding != "pcm16") {
            throw value_error(
                "Unsupported encoding '" + encoding +
                "' for .mp3 on Linux backend. Use 'auto', 'float32', or 'pcm16'.");
        }
        transcode_codec = "libmp3lame";
        if (bitrate != "auto") {
            transcode_bitrate = bitrate;
        }
        libav_preferred_fmt =
            (encoding == "pcm16") ? AV_SAMPLE_FMT_S16P : AV_SAMPLE_FMT_FLTP;
    } else if (ext == ".flac") {
        if (bitrate != "auto") {
            throw value_error("bitrate is not supported for .flac on Linux backend");
        }
        if (encoding != "auto" && encoding != "float32" && encoding != "pcm16") {
            throw value_error(
                "Unsupported encoding '" + encoding +
                "' for .flac on Linux backend. Use 'auto', 'float32', or 'pcm16'.");
        }
        transcode_codec = "flac";
        libav_preferred_fmt =
            (encoding == "pcm16") ? AV_SAMPLE_FMT_S16 : AV_SAMPLE_FMT_S32;
    } else if (ext == ".m4a") {
        if (!is_supported_m4a_bitrate(bitrate)) {
            throw value_error(
                "Invalid bitrate/subtype '" + bitrate +
                "' for M4A. Use 'auto', '128k', '192k', '256k', or '320k'.");
        }

        if (encoding == "alac") {
            if (bitrate != "auto") {
                throw value_error(
                    "bitrate is not supported when encoding='alac' for .m4a");
            }
            transcode_codec = "alac";
            libav_preferred_fmt = AV_SAMPLE_FMT_S16P;
        } else if (encoding == "auto" || encoding == "float32" || encoding == "pcm16") {
            transcode_codec = "aac";
            if (bitrate != "auto") {
                transcode_bitrate = bitrate;
            }
            libav_preferred_fmt = AV_SAMPLE_FMT_FLTP;
        } else {
            throw value_error(
                "Unsupported encoding '" + encoding +
                "' for .m4a on Linux backend. Use 'auto', 'float32', 'pcm16', or 'alac'.");
        }
    } else if (ext == ".aiff" || ext == ".aif") {
        if (bitrate != "auto") {
            throw value_error("bitrate is not supported for .aiff on Linux backend");
        }
        if (encoding == "auto" || encoding == "float32") {
            transcode_codec = "pcm_f32be";
            libav_preferred_fmt = AV_SAMPLE_FMT_FLT;
        } else if (encoding == "pcm16") {
            transcode_codec = "pcm_s16be";
            libav_preferred_fmt = AV_SAMPLE_FMT_S16;
        } else {
            throw value_error(
                "Unsupported encoding '" + encoding +
                "' for .aiff on Linux backend. Use 'auto', 'float32', or 'pcm16'.");
        }
    } else if (ext == ".caf") {
        if (bitrate != "auto") {
            throw value_error("bitrate is not supported for .caf on Linux backend");
        }
        if (encoding == "auto" || encoding == "float32") {
            transcode_codec = "pcm_f32le";
            libav_preferred_fmt = AV_SAMPLE_FMT_FLT;
        } else if (encoding == "pcm16") {
            transcode_codec = "pcm_s16le";
            libav_preferred_fmt = AV_SAMPLE_FMT_S16;
        } else {
            throw value_error(
                "Unsupported encoding '" + encoding +
                "' for .caf on Linux backend. Use 'auto', 'float32', or 'pcm16'.");
        }
    } else {
        throw value_error(
            "Unsupported output format on Linux backend. Supported formats: "
            ".wav, .mp3, .flac, .m4a, .aiff, .caf");
    }

    save_via_libav(
        path,
        write_data,
        frames,
        channels,
        sr,
        transcode_codec,
        transcode_bitrate,
        libav_preferred_fmt);
}

}  // namespace mlx_audio
