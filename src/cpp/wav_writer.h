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
#if defined(__ARM_NEON)
#include <arm_neon.h>
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

// --- Fused strided WAV writer -------------------------------------------------
//
// Reads the caller's samples through their strides (planar channels_first,
// interleaved channels_last, or any other layout) straight into a small
// interleaved chunk, quantizes it there, and writes it: no heap copy of the
// whole signal, no separate clip pass. Output is bit-identical to interleaving
// first and then quantizing the whole buffer: pcm16/pcm24 map x <= -1 to the
// negative full scale, x >= 1 to the positive full scale and round the rest to
// nearest-even, so clipping to [-1, 1] beforehand cannot change them.

inline int16_t quantize_pcm16(float x) {
    if (x <= -1.0f) {
        return static_cast<int16_t>(-32768);
    }
    if (x >= 1.0f) {
        return static_cast<int16_t>(32767);
    }
    return static_cast<int16_t>(std::lrint(x * 32767.0f));
}

inline int32_t quantize_pcm24(float x) {
    if (x <= -1.0f) {
        return -8388608;
    }
    if (x >= 1.0f) {
        return 8388607;
    }
    return static_cast<int32_t>(std::lrint(x * 8388607.0f));
}

#if defined(__ARM_NEON)
// Eight samples to pcm16 exactly as quantize_pcm16 does: FCVTNS rounds to
// nearest-even like lrint, the saturating narrow handles x >= 1 (and x < -1),
// and the select keeps x == -1 at -32768 where the narrow alone gives -32767.
inline int16x8_t quantize_pcm16x8(float32x4_t a, float32x4_t b) {
    const float32x4_t scale = vdupq_n_f32(32767.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    const int32x4_t neg_full = vdupq_n_s32(-32768);
    int32x4_t ia = vbslq_s32(vcleq_f32(a, neg_one), neg_full, vcvtnq_s32_f32(vmulq_f32(a, scale)));
    int32x4_t ib = vbslq_s32(vcleq_f32(b, neg_one), neg_full, vcvtnq_s32_f32(vmulq_f32(b, scale)));
    return vcombine_s16(vqmovn_s32(ia), vqmovn_s32(ib));
}
#endif

// Copy `n` frames starting at frame `f0` into `out`, interleaved.
inline void gather_interleaved(
    const float* data,
    size_t f0,
    size_t n,
    int channels,
    int64_t stride_frame,
    int64_t stride_chan,
    float* out) {
    const bool interleaved =
        channels == 1 ? stride_frame == 1 : (stride_chan == 1 && stride_frame == channels);
    if (interleaved) {
        std::copy_n(data + f0 * channels, n * channels, out);
        return;
    }
    for (int c = 0; c < channels; ++c) {
        const float* src = data + c * stride_chan + static_cast<int64_t>(f0) * stride_frame;
        for (size_t f = 0; f < n; ++f) {
            out[f * channels + c] = src[static_cast<int64_t>(f) * stride_frame];
        }
    }
}

inline void quantize_pcm16_block(const float* in, int16_t* out, size_t n) {
    size_t i = 0;
#if defined(__ARM_NEON)
    for (; i + 8 <= n; i += 8) {
        vst1q_s16(out + i, quantize_pcm16x8(vld1q_f32(in + i), vld1q_f32(in + i + 4)));
    }
#endif
    for (; i < n; ++i) {
        out[i] = quantize_pcm16(in[i]);
    }
}

inline void clip_block(float* buf, size_t n) {
#if defined(__APPLE__)
    float lo = -1.0f, hi = 1.0f;
    vDSP_vclip(buf, 1, &lo, &hi, buf, 1, static_cast<vDSP_Length>(n));
#else
    for (size_t j = 0; j < n; ++j) {
        buf[j] = std::max(-1.0f, std::min(1.0f, buf[j]));
    }
#endif
}

inline void write_wav_strided(
    const std::string& path,
    const float* data,
    int64_t frames,
    int channels,
    int64_t stride_frame,
    int64_t stride_chan,
    int sr,
    const std::string& wav_encoding,
    bool clip) {
    if (wav_encoding != "float32" && wav_encoding != "pcm16" && wav_encoding != "pcm24") {
        throw value_error("Unsupported wav encoding: " + wav_encoding);
    }
    if (channels <= 0 || frames < 0) {
        throw value_error("WAV output needs at least one channel");
    }
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path.c_str(), "wb"), fclose);
    if (!f) {
        throw std::runtime_error("Failed to open output file for writing: " + path);
    }
    const size_t total_frames = static_cast<size_t>(frames);
    const bool contiguous_interleaved =
        channels == 1 ? stride_frame == 1 : (stride_chan == 1 && stride_frame == channels);
    const int bytes_per_sample = wav_encoding == "float32" ? 4 : (wav_encoding == "pcm16" ? 2 : 3);
    const uint32_t data_size = static_cast<uint32_t>(
        static_cast<uint64_t>(total_frames) * channels * bytes_per_sample);
    write_wav_header(f.get(), sr, channels, bytes_per_sample * 8, wav_encoding == "float32", data_size);

    // Chunk of frames sized so the float staging buffer stays ~64 KB.
    const size_t chunk_frames = std::max<size_t>(1, 16384 / static_cast<size_t>(channels));
    std::unique_ptr<float[]> staging(new float[chunk_frames * channels]);
    auto put = [&](const void* buf, size_t bytes) {
        if (fwrite(buf, 1, bytes, f.get()) != bytes) {
            throw std::runtime_error("Failed to write " + wav_encoding + " WAV payload");
        }
    };

    if (wav_encoding == "float32") {
        if (contiguous_interleaved && !clip) {
            put(data, total_frames * channels * sizeof(float));
            return;
        }
        for (size_t f0 = 0; f0 < total_frames; f0 += chunk_frames) {
            size_t n = std::min(chunk_frames, total_frames - f0);
            gather_interleaved(data, f0, n, channels, stride_frame, stride_chan, staging.get());
            if (clip) {
                clip_block(staging.get(), n * channels);
            }
            put(staging.get(), n * channels * sizeof(float));
        }
        return;
    }

    if (wav_encoding == "pcm16") {
        std::unique_ptr<int16_t[]> out(new int16_t[chunk_frames * channels]);
#if defined(__ARM_NEON)
        // Planar stereo (channels_first rows): quantize both channels in
        // registers and interleave them with a two-way store.
        if (channels == 2 && stride_frame == 1) {
            const float* left = data;
            const float* right = data + stride_chan;
            for (size_t f0 = 0; f0 < total_frames; f0 += chunk_frames) {
                size_t n = std::min(chunk_frames, total_frames - f0);
                size_t i = 0;
                for (; i + 8 <= n; i += 8) {
                    const size_t f = f0 + i;
                    int16x8x2_t lr;
                    lr.val[0] = quantize_pcm16x8(vld1q_f32(left + f), vld1q_f32(left + f + 4));
                    lr.val[1] = quantize_pcm16x8(vld1q_f32(right + f), vld1q_f32(right + f + 4));
                    vst2q_s16(out.get() + 2 * i, lr);
                }
                for (; i < n; ++i) {
                    out[2 * i] = quantize_pcm16(left[f0 + i]);
                    out[2 * i + 1] = quantize_pcm16(right[f0 + i]);
                }
                put(out.get(), n * 2 * sizeof(int16_t));
            }
            return;
        }
#endif
        for (size_t f0 = 0; f0 < total_frames; f0 += chunk_frames) {
            size_t n = std::min(chunk_frames, total_frames - f0);
            const float* src = data + f0 * channels;
            if (!contiguous_interleaved) {
                gather_interleaved(data, f0, n, channels, stride_frame, stride_chan, staging.get());
                src = staging.get();
            }
            quantize_pcm16_block(src, out.get(), n * channels);
            put(out.get(), n * channels * sizeof(int16_t));
        }
        return;
    }

    // pcm24
    std::unique_ptr<uint8_t[]> out(new uint8_t[chunk_frames * channels * 3]);
    for (size_t f0 = 0; f0 < total_frames; f0 += chunk_frames) {
        size_t n = std::min(chunk_frames, total_frames - f0);
        const float* src = data + f0 * channels;
        if (!contiguous_interleaved) {
            gather_interleaved(data, f0, n, channels, stride_frame, stride_chan, staging.get());
            src = staging.get();
        }
        uint8_t* p = out.get();
        for (size_t j = 0; j < n * channels; ++j) {
            int32_t s = quantize_pcm24(src[j]);
            *p++ = static_cast<uint8_t>(s & 0xFF);
            *p++ = static_cast<uint8_t>((s >> 8) & 0xFF);
            *p++ = static_cast<uint8_t>((s >> 16) & 0xFF);
        }
        put(out.get(), n * channels * 3);
    }
}

}  // namespace internal
}  // namespace mlx_audio
