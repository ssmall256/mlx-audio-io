#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include "audio_io.h"
#include "internal_utils.h"

namespace mlx_audio {
namespace internal {

inline void write_u16_le(FILE* f, uint16_t value) {
    fwrite(&value, sizeof(value), 1, f);
}

inline void write_u32_le(FILE* f, uint32_t value) {
    fwrite(&value, sizeof(value), 1, f);
}

inline void write_wav_header(
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

inline void write_interleaved_wav(
    const std::string& path,
    const float* write_data,
    int frames,
    int channels,
    int sr,
    const std::string& wav_encoding,
    bool clip = false) {
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path.c_str(), "wb"), fclose);
    if (!f) {
        throw std::runtime_error("Failed to open output file for writing: " + path);
    }

    if (wav_encoding == "float32") {
        uint32_t data_size = static_cast<uint32_t>(
            static_cast<uint64_t>(frames) * channels * sizeof(float));
        write_wav_header(f.get(), sr, channels, 32, true, data_size);

        size_t total = static_cast<size_t>(frames) * channels;
        if (!clip) {
            size_t wrote = fwrite(write_data, sizeof(float), total, f.get());
            if (wrote != total) {
                throw std::runtime_error("Failed to write float32 WAV payload");
            }
        } else {
            constexpr size_t kChunk = 16384;
            float chunk_buf[kChunk];
            size_t i = 0;
            while (i < total) {
                size_t n = std::min(kChunk, total - i);
#if defined(__APPLE__)
                float lo = -1.0f, hi = 1.0f;
                vDSP_vclip(write_data + i, 1, &lo, &hi, chunk_buf, 1, static_cast<vDSP_Length>(n));
#else
                for (size_t j = 0; j < n; ++j) {
                    chunk_buf[j] = std::max(-1.0f, std::min(1.0f, write_data[i + j]));
                }
#endif
                size_t wrote = fwrite(chunk_buf, sizeof(float), n, f.get());
                if (wrote != n) {
                    throw std::runtime_error("Failed to write float32 WAV payload");
                }
                i += n;
            }
        }
        return;
    }

    if (wav_encoding == "pcm16") {
        uint32_t data_size = static_cast<uint32_t>(
            static_cast<uint64_t>(frames) * channels * sizeof(int16_t));
        write_wav_header(f.get(), sr, channels, 16, false, data_size);

        size_t total = static_cast<size_t>(frames) * channels;
        constexpr size_t kChunk = 16384;
        int16_t chunk_buf[kChunk];
        size_t i = 0;
        while (i < total) {
            size_t n = std::min(kChunk, total - i);
            for (size_t j = 0; j < n; ++j) {
                float x = write_data[i + j];
                if (x <= -1.0f) {
                    chunk_buf[j] = static_cast<int16_t>(-32768);
                } else if (x >= 1.0f) {
                    chunk_buf[j] = static_cast<int16_t>(32767);
                } else {
                    chunk_buf[j] = static_cast<int16_t>(std::lrint(x * 32767.0f));
                }
            }
            size_t wrote = fwrite(chunk_buf, sizeof(int16_t), n, f.get());
            if (wrote != n) {
                throw std::runtime_error("Failed to write pcm16 WAV payload");
            }
            i += n;
        }
        return;
    }

    if (wav_encoding == "pcm24") {
        uint32_t data_size = static_cast<uint32_t>(
            static_cast<uint64_t>(frames) * channels * 3);
        write_wav_header(f.get(), sr, channels, 24, false, data_size);

        size_t total = static_cast<size_t>(frames) * channels;
        constexpr size_t kPcm24Chunk = 8192;
        uint8_t chunk_buf[kPcm24Chunk * 3];
        size_t buf_idx = 0;

        for (size_t i = 0; i < total; ++i) {
            float x = write_data[i];
            int32_t s;
            if (x <= -1.0f) {
                s = -8388608;
            } else if (x >= 1.0f) {
                s = 8388607;
            } else {
                s = static_cast<int32_t>(std::lrint(x * 8388607.0f));
            }
            chunk_buf[buf_idx++] = static_cast<uint8_t>(s & 0xFF);
            chunk_buf[buf_idx++] = static_cast<uint8_t>((s >> 8) & 0xFF);
            chunk_buf[buf_idx++] = static_cast<uint8_t>((s >> 16) & 0xFF);

            if (buf_idx == sizeof(chunk_buf)) {
                size_t wrote = fwrite(chunk_buf, 1, buf_idx, f.get());
                if (wrote != buf_idx) {
                    throw std::runtime_error("Failed to write pcm24 WAV payload");
                }
                buf_idx = 0;
            }
        }
        if (buf_idx > 0) {
            size_t wrote = fwrite(chunk_buf, 1, buf_idx, f.get());
            if (wrote != buf_idx) {
                throw std::runtime_error("Failed to write pcm24 WAV payload");
            }
        }
        return;
    }

    throw value_error("Unsupported wav encoding: " + wav_encoding);
}

}  // namespace internal
}  // namespace mlx_audio
