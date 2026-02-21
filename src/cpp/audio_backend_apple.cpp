#include "audio_backend.h"
#include "internal_utils.h"
#include "mp3_decoder.h"
#include "mp3_encoder.h"
#include "osstatus.h"
#include "raii_audio.h"

#include <Accelerate/Accelerate.h>
#include <AudioToolbox/AudioToolbox.h>

#include <arm_neon.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/stat.h>

// WAV fast path assumes little-endian (ARM64 on macOS is LE).
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "WAV parser assumes little-endian byte order");

namespace mlx_audio {

// Bring shared helpers into scope
using internal::aligned_alloc_64;
using internal::aligned_free;
using internal::strided_copy;
using internal::mono_mixdown;
using internal::make_url;
using internal::check_file_exists;
using internal::make_client_format;
using internal::kReadChunkFrames;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Map an AudioFileTypeID to a container string.
std::string file_type_to_container(UInt32 file_type) {
    switch (file_type) {
        case kAudioFileWAVEType: return "wav";
        case kAudioFileMP3Type:  return "mp3";
        case kAudioFileM4AType:  return "m4a";
        case kAudioFileAIFFType: return "aiff";
        case kAudioFileAIFCType: return "aiff";
        case kAudioFileFLACType: return "flac";
        case kAudioFileCAFType:  return "caf";
        default:                 return "unknown";
    }
}

/// Map an AudioStreamBasicDescription to a subtype string.
std::string format_to_subtype(const AudioStreamBasicDescription& fmt) {
    if (fmt.mFormatID == kAudioFormatLinearPCM) {
        bool is_float = (fmt.mFormatFlags & kAudioFormatFlagIsFloat) != 0;
        int bits = static_cast<int>(fmt.mBitsPerChannel);

        if (is_float) {
            if (bits == 32) return "float32";
            if (bits == 64) return "float64";
            return "float" + std::to_string(bits);
        } else {
            if (bits == 16) return "pcm16";
            if (bits == 24) return "pcm24";
            if (bits == 32) return "pcm32";
            if (bits == 8) return "pcm8";
            return "pcm" + std::to_string(bits);
        }
    }

    switch (fmt.mFormatID) {
        case kAudioFormatMPEGLayer3:    return "mp3";
        case kAudioFormatMPEG4AAC:      return "aac";
        case kAudioFormatAppleLossless: return "alac";
        case kAudioFormatFLAC:          return "flac";
        case kAudioFormatOpus:          return "opus";
        default:                        return "unknown";
    }
}

/// Parse a resample quality string to an AudioConverterQuality constant.
/// Returns -1 for "default" (meaning: don't override).
int parse_resample_quality(const std::string& quality) {
    if (quality == "default")  return -1;
    if (quality == "fastest")  return kAudioConverterQuality_Min;
    if (quality == "low")      return kAudioConverterQuality_Low;
    if (quality == "medium")   return kAudioConverterQuality_Medium;
    if (quality == "high")     return kAudioConverterQuality_High;
    if (quality == "best")     return kAudioConverterQuality_Max;
    throw value_error(
        "Invalid resample_quality '" + quality +
        "'. Must be 'default', 'fastest', 'low', 'medium', 'high', or 'best'.");
}

/// Output format info parsed from file extension.
struct OutputFormat {
    AudioFileTypeID file_type;
    enum Kind { PCM, AAC, FLAC, ALAC, MP3 } kind;
};

/// Map file extension to AudioFileTypeID and format kind.
OutputFormat parse_output_format(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) {
        throw value_error("Cannot determine format: no file extension in '" + path + "'");
    }
    std::string ext = path.substr(dot);
    // Lowercase the extension
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));

    if (ext == ".wav")  return {kAudioFileWAVEType, OutputFormat::PCM};
    if (ext == ".m4a")  return {kAudioFileM4AType, OutputFormat::AAC};
    if (ext == ".aiff" || ext == ".aif") return {kAudioFileAIFFType, OutputFormat::PCM};
    if (ext == ".caf")  return {kAudioFileCAFType, OutputFormat::PCM};
    if (ext == ".flac") return {kAudioFileFLACType, OutputFormat::FLAC};
    if (ext == ".mp3")  return {kAudioFileMP3Type, OutputFormat::MP3};
    throw value_error(
        "Unsupported output format '" + ext +
        "'. Supported: .wav, .mp3, .m4a, .aiff, .caf, .flac");
}

/// Parse MP3 bitrate string to kbps. Returns 0 for "auto".
int parse_mp3_bitrate_kbps(const std::string& s) {
    if (s == "auto") return 0;
    if (s.size() >= 2 && (s.back() == 'k' || s.back() == 'K')) {
        try {
            int kbps = std::stoi(s.substr(0, s.size() - 1));
            if (kbps > 0) return kbps;
        } catch (...) {}
    }
    throw value_error(
        "Invalid bitrate/subtype '" + s +
        "' for MP3. Use 'auto', '128k', '192k', '256k', or '320k'.");
}

/// Encode and write MP3 via vendored LAME encoder.
void save_mp3_lame(
    const std::string& path,
    const float* write_data,
    int frames,
    int channels,
    int sr,
    const std::string& bitrate) {

    int bitrate_kbps = parse_mp3_bitrate_kbps(bitrate);

    ScopedMp3Encoder encoder;
    encoder.init(sr, channels, bitrate_kbps);

    // Encode in chunks to limit peak memory
    constexpr int kChunkFrames = 131072;
    int cursor = 0;
    while (cursor < frames) {
        int chunk = std::min(kChunkFrames, frames - cursor);
        encoder.encode(write_data + static_cast<size_t>(cursor) * channels, chunk);
        cursor += chunk;
    }
    encoder.flush();

    // Write accumulated MP3 bytes to file
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path.c_str(), "wb"), fclose);
    if (!f) {
        throw std::runtime_error("Failed to open output file for writing: " + path);
    }
    size_t wrote = fwrite(encoder.data(), 1, encoder.size(), f.get());
    if (wrote != encoder.size()) {
        throw std::runtime_error("Failed to write MP3 data to: " + path);
    }
}

/// Create an AAC AudioStreamBasicDescription.
AudioStreamBasicDescription make_aac_format(int sr, int channels) {
    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = static_cast<Float64>(sr);
    fmt.mFormatID = kAudioFormatMPEG4AAC;
    fmt.mChannelsPerFrame = static_cast<UInt32>(channels);
    // Zero out the rest — AudioToolbox fills them from the encoder
    fmt.mBytesPerFrame = 0;
    fmt.mBitsPerChannel = 0;
    fmt.mBytesPerPacket = 0;
    fmt.mFramesPerPacket = 0;
    fmt.mFormatFlags = 0;
    return fmt;
}

/// Create a FLAC AudioStreamBasicDescription.
AudioStreamBasicDescription make_flac_format(int sr, int channels) {
    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = static_cast<Float64>(sr);
    fmt.mFormatID = kAudioFormatFLAC;
    fmt.mChannelsPerFrame = static_cast<UInt32>(channels);
    fmt.mBytesPerFrame = 0;
    fmt.mBitsPerChannel = 0;
    fmt.mBytesPerPacket = 0;
    fmt.mFramesPerPacket = 0;
    fmt.mFormatFlags = 0;
    return fmt;
}

/// Create an ALAC AudioStreamBasicDescription.
AudioStreamBasicDescription make_alac_format(int sr, int channels) {
    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = static_cast<Float64>(sr);
    fmt.mFormatID = kAudioFormatAppleLossless;
    fmt.mChannelsPerFrame = static_cast<UInt32>(channels);
    fmt.mBytesPerFrame = 0;
    fmt.mBitsPerChannel = 0;
    fmt.mBytesPerPacket = 0;
    fmt.mFramesPerPacket = 0;
    fmt.mFormatFlags = 0;
    return fmt;
}

/// Parse a bitrate string like "128k" to bits per second. Returns 0 for "auto".
UInt32 parse_bitrate(const std::string& s) {
    if (s == "auto") return 0;
    // Accept e.g. "128k", "192k", "256k", "320k"
    if (s.size() >= 2 && (s.back() == 'k' || s.back() == 'K')) {
        try {
            unsigned long kbps = std::stoul(s.substr(0, s.size() - 1));
            return static_cast<UInt32>(kbps) * 1000u;
        } catch (...) {}
    }
    throw value_error(
        "Invalid bitrate/subtype '" + s +
        "' for M4A. Use 'auto', '128k', '192k', '256k', or '320k'.");
}

/// Return an empty array with the right shape/dtype for offset-beyond-end.
std::pair<mlx::core::array, int> make_empty_result(
    int out_sr, int channels, bool mono,
    const std::string& layout, const std::string& dtype) {
    int out_channels = mono ? 1 : channels;
    auto target_dtype = (dtype == "float16") ? mlx::core::float16 : mlx::core::float32;
    mlx::core::Shape shape;
    if (layout == "channels_last") {
        shape = {0, static_cast<int32_t>(out_channels)};
    } else {
        shape = {static_cast<int32_t>(out_channels), 0};
    }
    return {mlx::core::array(
        std::initializer_list<int>{}, shape, target_dtype), out_sr};
}

/// Apply mono mixdown, channels_first deinterleave, shape, and dtype to
/// an interleaved float32 buffer. Takes ownership of buffer.
std::pair<mlx::core::array, int> wrap_audio_buffer(
    float* buffer, int64_t actual_frames, int channels, int out_sr,
    bool mono, const std::string& layout, const std::string& dtype) {

    int out_channels = channels;
    if (mono && channels > 1) {
        buffer = mono_mixdown(buffer, channels, actual_frames);
        out_channels = 1;
    }

    if (layout == "channels_first" && out_channels > 1) {
        size_t planar_bytes =
            static_cast<size_t>(actual_frames) * out_channels * sizeof(float);
        float* planar_buf = static_cast<float*>(aligned_alloc_64(planar_bytes));

        for (int c = 0; c < out_channels; ++c) {
            strided_copy(
                buffer + c,
                out_channels,
                planar_buf + c * actual_frames,
                1,
                static_cast<int>(actual_frames));
        }

        std::free(buffer);
        buffer = planar_buf;
    }

    mlx::core::Shape shape;
    if (layout == "channels_first" && out_channels > 1) {
        shape = {static_cast<int32_t>(out_channels),
                 static_cast<int32_t>(actual_frames)};
    } else if (out_channels == 1 && layout == "channels_first") {
        shape = {1, static_cast<int32_t>(actual_frames)};
    } else {
        shape = {static_cast<int32_t>(actual_frames),
                 static_cast<int32_t>(out_channels)};
    }

    auto arr = mlx::core::array(
        static_cast<void*>(buffer),
        std::move(shape),
        mlx::core::float32,
        aligned_free);

    if (dtype == "float16") {
        arr = mlx::core::astype(arr, mlx::core::float16);
    }

    return {std::move(arr), out_sr};
}

// ---------------------------------------------------------------------------
// WAV fast path — direct fread, bypassing AudioToolbox
// ---------------------------------------------------------------------------

/// Parsed WAV header info for the fast path.
struct WavInfo {
    int sample_rate;
    int channels;
    int bits_per_sample;
    int format_tag;       // 1 = PCM, 3 = IEEE float
    int64_t data_offset;  // byte offset of PCM data in file
    int64_t data_bytes;   // byte size of data chunk
    int64_t total_frames; // data_bytes / block_align
};

/// Try to parse a WAV header. Returns true on success.
/// Only succeeds for standard RIFF/WAVE with PCM (tag 1) or IEEE float (tag 3).
bool parse_wav_header(const std::string& path, WavInfo& out) {
    std::unique_ptr<FILE, decltype(&fclose)> f(fopen(path.c_str(), "rb"), fclose);
    if (!f) return false;

    // Read RIFF header (12 bytes)
    uint8_t riff[12];
    if (fread(riff, 1, 12, f.get()) != 12) return false;
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        return false;
    }

    bool found_fmt = false, found_data = false;

    // Walk chunks
    while (!found_data) {
        uint8_t chunk_hdr[8];
        if (fread(chunk_hdr, 1, 8, f.get()) != 8) break;

        uint32_t chunk_size;
        memcpy(&chunk_size, chunk_hdr + 4, 4);

        if (memcmp(chunk_hdr, "fmt ", 4) == 0) {
            if (chunk_size < 16) return false;
            uint8_t fmt[16];
            if (fread(fmt, 1, 16, f.get()) != 16) return false;

            uint16_t format_tag, num_channels, block_align, bits_per_sample;
            uint32_t sample_rate;
            memcpy(&format_tag, fmt + 0, 2);
            memcpy(&num_channels, fmt + 2, 2);
            memcpy(&sample_rate, fmt + 4, 4);
            // byte_rate at +8 (skip)
            memcpy(&block_align, fmt + 12, 2);
            memcpy(&bits_per_sample, fmt + 14, 2);

            // Only support PCM int or IEEE float
            if (format_tag != 1 && format_tag != 3) return false;

            out.format_tag = format_tag;
            out.sample_rate = static_cast<int>(sample_rate);
            out.channels = static_cast<int>(num_channels);
            out.bits_per_sample = static_cast<int>(bits_per_sample);
            found_fmt = true;

            // Skip remainder of fmt chunk
            long remaining = static_cast<long>(chunk_size) - 16;
            if (remaining > 0) fseek(f.get(), remaining, SEEK_CUR);

        } else if (memcmp(chunk_hdr, "data", 4) == 0) {
            if (!found_fmt) return false;
            out.data_offset = ftell(f.get());
            out.data_bytes = static_cast<int64_t>(chunk_size);
            int block_align = out.channels * (out.bits_per_sample / 8);
            out.total_frames = (block_align > 0) ? out.data_bytes / block_align : 0;
            found_data = true;
        } else {
            // Skip unknown chunk (pad to even)
            long skip = static_cast<long>(chunk_size);
            if (skip & 1) skip++;
            fseek(f.get(), skip, SEEK_CUR);
        }
    }

    return found_fmt && found_data;
}

/// Check if a path has a .wav extension (case-insensitive).
bool is_wav_path(const std::string& path) {
    if (path.size() < 4) return false;
    auto ext = path.substr(path.size() - 4);
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    return ext == ".wav";
}

/// Fast WAV load via direct fread, bypassing AudioToolbox.
std::pair<mlx::core::array, int> load_wav_fast(
    const WavInfo& wav,
    const std::string& path,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype) {

    int64_t start_frame = static_cast<int64_t>(std::floor(offset * wav.sample_rate));

    if (start_frame >= wav.total_frames) {
        return make_empty_result(wav.sample_rate, wav.channels, mono, layout, dtype);
    }

    int64_t frames_to_read;
    if (duration.has_value()) {
        frames_to_read = static_cast<int64_t>(
            std::floor(duration.value() * wav.sample_rate));
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

    // Seek directly to the sample data
    int64_t byte_offset = wav.data_offset + start_frame * block_align;
    if (fseek(f.get(), static_cast<long>(byte_offset), SEEK_SET) != 0) {
        throw std::runtime_error("Failed to seek in WAV file: " + path);
    }

    // Allocate output float32 buffer
    size_t out_bytes = static_cast<size_t>(frames_to_read) * wav.channels * sizeof(float);
    float* buffer = static_cast<float*>(aligned_alloc_64(out_bytes > 0 ? out_bytes : 64));

    int64_t actual_frames = frames_to_read;

    if (wav.format_tag == 3 && wav.bits_per_sample == 32) {
        // IEEE float32 — read directly into buffer
        size_t got = fread(buffer, 1, static_cast<size_t>(read_bytes), f.get());
        actual_frames = static_cast<int64_t>(got) / (wav.channels * sizeof(float));
    } else if (wav.format_tag == 1 && wav.bits_per_sample == 16) {
        // PCM16 — read into the tail of the float buffer, convert in-place.
        // float32 is 4 bytes per sample, int16 is 2 bytes, so the int16 data
        // fits in the upper half of the buffer. We convert back-to-front to
        // avoid overwriting unprocessed samples.
        size_t sample_count = static_cast<size_t>(frames_to_read) * wav.channels;
        // Place int16 data at the end of the float buffer
        int16_t* pcm_region = reinterpret_cast<int16_t*>(
            reinterpret_cast<char*>(buffer) + sample_count * sizeof(float)
            - sample_count * sizeof(int16_t));
        size_t got = fread(pcm_region, 1, static_cast<size_t>(read_bytes), f.get());
        actual_frames = static_cast<int64_t>(got) / (wav.channels * sizeof(int16_t));
        size_t actual_samples = static_cast<size_t>(actual_frames) * wav.channels;

        // Convert int16 -> float32 (safe: src and dst don't overlap destructively
        // because we placed int16 data at the tail and float32 writes start at
        // the head, consuming 4 bytes per sample vs 2 bytes read)
        vDSP_vflt16(pcm_region, 1, buffer, 1,
                     static_cast<vDSP_Length>(actual_samples));
        // Scale to [-1, 1]
        float scale = 1.0f / 32768.0f;
        vDSP_vsmul(buffer, 1, &scale, buffer, 1,
                    static_cast<vDSP_Length>(actual_samples));
    } else if (wav.format_tag == 1 && wav.bits_per_sample == 24) {
        // PCM24 — read raw bytes, unpack, convert (NEON-accelerated)
        size_t sample_count = static_cast<size_t>(frames_to_read) * wav.channels;
        size_t raw_bytes = sample_count * 3;
        uint8_t* raw = static_cast<uint8_t*>(aligned_alloc_64(raw_bytes));
        size_t got = fread(raw, 1, raw_bytes, f.get());
        size_t actual_samples = got / 3;
        actual_frames = static_cast<int64_t>(actual_samples) / wav.channels;

        float inv = 1.0f / 8388608.0f;  // 2^23
        float32x4_t vscale = vdupq_n_f32(inv);

        // Process 4 samples at a time (12 bytes -> 4 floats)
        size_t i = 0;
        size_t neon_end = (actual_samples >= 3) ? actual_samples - 3 : 0;
        for (; i < neon_end; i += 4) {
            const uint8_t* p = raw + i * 3;
            // Load 4 x 24-bit samples into int32 lanes
            int32x4_t vals = {
                static_cast<int32_t>(p[0])  | (static_cast<int32_t>(p[1])  << 8) | (static_cast<int32_t>(p[2])  << 16),
                static_cast<int32_t>(p[3])  | (static_cast<int32_t>(p[4])  << 8) | (static_cast<int32_t>(p[5])  << 16),
                static_cast<int32_t>(p[6])  | (static_cast<int32_t>(p[7])  << 8) | (static_cast<int32_t>(p[8])  << 16),
                static_cast<int32_t>(p[9])  | (static_cast<int32_t>(p[10]) << 8) | (static_cast<int32_t>(p[11]) << 16),
            };
            // Sign-extend from 24 bits: shift left 8 then arithmetic right 8
            vals = vshrq_n_s32(vshlq_n_s32(vals, 8), 8);
            // Convert to float and scale
            float32x4_t fvals = vmulq_f32(vcvtq_f32_s32(vals), vscale);
            vst1q_f32(buffer + i, fvals);
        }
        // Scalar tail
        for (; i < actual_samples; ++i) {
            int32_t val = static_cast<int32_t>(raw[i * 3])
                        | (static_cast<int32_t>(raw[i * 3 + 1]) << 8)
                        | (static_cast<int32_t>(raw[i * 3 + 2]) << 16);
            if (val & 0x800000) val |= 0xFF000000;
            buffer[i] = static_cast<float>(val) * inv;
        }
        std::free(raw);
    } else if (wav.format_tag == 1 && wav.bits_per_sample == 32) {
        // PCM32 — read directly into buffer (int32 and float32 are same size),
        // then convert in-place.
        size_t got = fread(buffer, 1, static_cast<size_t>(read_bytes), f.get());
        actual_frames = static_cast<int64_t>(got) / (wav.channels * sizeof(int32_t));
        size_t actual_samples = static_cast<size_t>(actual_frames) * wav.channels;

        // Reinterpret the buffer as int32 for vDSP conversion
        vDSP_vflt32(reinterpret_cast<int32_t*>(buffer), 1, buffer, 1,
                     static_cast<vDSP_Length>(actual_samples));
        float scale = 1.0f / 2147483648.0f;  // 2^31
        vDSP_vsmul(buffer, 1, &scale, buffer, 1,
                    static_cast<vDSP_Length>(actual_samples));
    } else {
        // Unsupported bit depth — should not happen (parse_wav_header validated)
        std::free(buffer);
        return make_empty_result(wav.sample_rate, wav.channels, mono, layout, dtype);
    }

    return wrap_audio_buffer(buffer, actual_frames, wav.channels,
                             wav.sample_rate, mono, layout, dtype);
}

/// Fast MP3 decode at native sample rate using minimp3.
/// Accepts a pre-opened decoder to avoid redundant file opens.
std::pair<mlx::core::array, int> load_mp3_fast(
    ScopedMp3Decoder& dec,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype) {

    int native_sr = dec.sample_rate();
    int channels = dec.channels();
    int64_t total_frames = dec.total_frames();

    int64_t start_frame = static_cast<int64_t>(std::floor(offset * native_sr));

    if (start_frame >= total_frames) {
        return make_empty_result(native_sr, channels, mono, layout, dtype);
    }

    int64_t frames_to_read;
    if (duration.has_value()) {
        frames_to_read = static_cast<int64_t>(std::floor(duration.value() * native_sr));
        if (start_frame + frames_to_read > total_frames) {
            frames_to_read = total_frames - start_frame;
        }
    } else {
        frames_to_read = total_frames - start_frame;
    }

    // Seek (minimp3 seeks in samples, not frames)
    if (start_frame > 0) {
        dec.seek(start_frame * channels);
    }

    // Allocate and read
    size_t buffer_bytes = static_cast<size_t>(frames_to_read) * channels * sizeof(float);
    float* buffer = static_cast<float*>(aligned_alloc_64(
        buffer_bytes > 0 ? buffer_bytes : 64));

    int64_t samples_wanted = frames_to_read * channels;
    int64_t samples_read = dec.read(buffer, samples_wanted);
    int64_t actual_frames = (channels > 0) ? samples_read / channels : 0;

    return wrap_audio_buffer(buffer, actual_frames, channels,
                             native_sr, mono, layout, dtype);
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// get_info
// ---------------------------------------------------------------------------

AudioFileInfo backend_get_info(const std::string& path) {
    check_file_exists(path);

    auto url = make_url(path);
    ScopedExtAudioFile ext_file;
    OSStatus status = ExtAudioFileOpenURL(url.get(), ext_file.ptr());
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to open audio file '" + path +
            "': OSStatus " + osstatus_to_string(status));
    }

    // Get the underlying AudioFileID to read container type
    AudioFileID audio_file = nullptr;
    UInt32 prop_size = sizeof(audio_file);
    status = ExtAudioFileGetProperty(
        ext_file.get(), kExtAudioFileProperty_AudioFile, &prop_size, &audio_file);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to get AudioFileID: OSStatus " + osstatus_to_string(status));
    }

    UInt32 file_type = 0;
    UInt32 ft_size = sizeof(file_type);
    status = AudioFileGetProperty(
        audio_file, kAudioFilePropertyFileFormat, &ft_size, &file_type);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to query file format: OSStatus " + osstatus_to_string(status));
    }

    // Get native format
    AudioStreamBasicDescription native_fmt = {};
    prop_size = sizeof(native_fmt);
    status = ExtAudioFileGetProperty(
        ext_file.get(), kExtAudioFileProperty_FileDataFormat, &prop_size, &native_fmt);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to get file format: OSStatus " + osstatus_to_string(status));
    }

    // Get frame count
    SInt64 total_frames = 0;
    prop_size = sizeof(total_frames);
    status = ExtAudioFileGetProperty(
        ext_file.get(), kExtAudioFileProperty_FileLengthFrames, &prop_size, &total_frames);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to get frame count: OSStatus " + osstatus_to_string(status));
    }

    AudioFileInfo info;
    info.frames = static_cast<int64_t>(total_frames);
    info.sample_rate = static_cast<int>(native_fmt.mSampleRate);
    info.channels = static_cast<int>(native_fmt.mChannelsPerFrame);
    info.duration = (native_fmt.mSampleRate > 0)
                        ? static_cast<double>(total_frames) / native_fmt.mSampleRate
                        : 0.0;
    info.subtype = format_to_subtype(native_fmt);
    info.container = file_type_to_container(file_type);

    return info;
}

// ---------------------------------------------------------------------------
// load_audio
// ---------------------------------------------------------------------------

std::pair<mlx::core::array, int> backend_load_audio(
    const std::string& path,
    std::optional<int> sr,
    double offset,
    std::optional<double> duration,
    bool mono,
    const std::string& layout,
    const std::string& dtype,
    const std::string& resample_quality) {

    // --- Validate inputs ---
    if (dtype != "float32" && dtype != "float16") {
        throw value_error("Unsupported dtype '" + dtype + "'. Must be 'float32' or 'float16'.");
    }
    if (layout != "channels_last" && layout != "channels_first") {
        throw value_error(
            "Invalid layout '" + layout +
            "'. Must be 'channels_last' or 'channels_first'.");
    }
    int quality_val = parse_resample_quality(resample_quality);

    if (offset < 0.0) {
        throw value_error("offset must be >= 0, got " + std::to_string(offset));
    }
    if (duration.has_value() && duration.value() <= 0.0) {
        throw value_error("duration must be > 0, got " + std::to_string(duration.value()));
    }
    if (sr.has_value() && sr.value() <= 0) {
        throw value_error("sr must be > 0");
    }

    check_file_exists(path);

    // --- WAV fast path via direct fread (bypasses AudioToolbox) ---
    // Used for all native-SR loads (full or partial). Direct fread avoids
    // AudioToolbox open/init overhead and is competitive or faster for all
    // WAV bit depths. Falls through to AudioToolbox only when resampling
    // is needed.
    if (is_wav_path(path)) {
        bool native_sr = !sr.has_value();
        WavInfo wav;
        if (parse_wav_header(path, wav)) {
            if (native_sr || sr.value() == wav.sample_rate) {
                return load_wav_fast(wav, path, offset, duration, mono, layout, dtype);
            }
        }
        // parse failed or needs resampling — fall through to AudioToolbox
    }

    // --- MP3 fast path via minimp3 (native SR only) ---
    if (is_mp3_path(path)) {
        // Open once — used for both SR check and decode
        ScopedMp3Decoder dec;
        dec.open(path);
        bool needs_resample = sr.has_value() && sr.value() != dec.sample_rate();
        if (!needs_resample) {
            return load_mp3_fast(dec, offset, duration, mono, layout, dtype);
        }
        // else: fall through to AudioToolbox for resampling
    }

    // --- Open file ---
    auto url = make_url(path);
    ScopedExtAudioFile ext_file;
    OSStatus status = ExtAudioFileOpenURL(url.get(), ext_file.ptr());
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to open audio file '" + path +
            "': OSStatus " + osstatus_to_string(status));
    }

    // Get native format
    AudioStreamBasicDescription native_fmt = {};
    UInt32 prop_size = sizeof(native_fmt);
    status = ExtAudioFileGetProperty(
        ext_file.get(), kExtAudioFileProperty_FileDataFormat, &prop_size, &native_fmt);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to get file format: OSStatus " + osstatus_to_string(status));
    }

    int native_sr = static_cast<int>(native_fmt.mSampleRate);
    int channels = static_cast<int>(native_fmt.mChannelsPerFrame);
    int out_sr = sr.value_or(native_sr);

    // Set client format (float32, interleaved, target SR)
    auto client_fmt = make_client_format(out_sr, channels);
    status = ExtAudioFileSetProperty(
        ext_file.get(), kExtAudioFileProperty_ClientDataFormat,
        sizeof(client_fmt), &client_fmt);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to set client format: OSStatus " + osstatus_to_string(status));
    }

    // Set resample quality if non-default and resampling is active
    if (quality_val >= 0 && out_sr != native_sr) {
        AudioConverterRef converter = nullptr;
        UInt32 conv_size = sizeof(converter);
        status = ExtAudioFileGetProperty(
            ext_file.get(), kExtAudioFileProperty_AudioConverter,
            &conv_size, &converter);
        if (status == noErr && converter != nullptr) {
            UInt32 quality = static_cast<UInt32>(quality_val);
            AudioConverterSetProperty(
                converter, kAudioConverterSampleRateConverterQuality,
                sizeof(quality), &quality);
            // Resync the ExtAudioFile with the updated converter settings
            CFArrayRef null_config = nullptr;
            ExtAudioFileSetProperty(
                ext_file.get(), kExtAudioFileProperty_ConverterConfig,
                sizeof(null_config), &null_config);
        }
    }

    // Get total frame count at native SR
    SInt64 native_frames = 0;
    prop_size = sizeof(native_frames);
    status = ExtAudioFileGetProperty(
        ext_file.get(), kExtAudioFileProperty_FileLengthFrames, &prop_size, &native_frames);
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to get frame count: OSStatus " + osstatus_to_string(status));
    }

    // Compute total frames at output SR
    int64_t total_frames_out;
    if (out_sr == native_sr) {
        total_frames_out = static_cast<int64_t>(native_frames);
    } else {
        total_frames_out = static_cast<int64_t>(
            std::ceil(static_cast<double>(native_frames) * out_sr / native_fmt.mSampleRate));
    }

    // --- Compute start_frame and frames_to_read ---
    int64_t start_frame = static_cast<int64_t>(std::floor(offset * out_sr));
    int64_t frames_to_read;

    auto target_dtype = (dtype == "float16") ? mlx::core::float16 : mlx::core::float32;

    if (start_frame >= total_frames_out) {
        // Offset beyond end: return empty tensor
        int out_channels = mono ? 1 : channels;
        mlx::core::Shape shape;
        if (layout == "channels_last") {
            shape = {0, static_cast<int32_t>(out_channels)};
        } else {
            shape = {static_cast<int32_t>(out_channels), 0};
        }
        return {mlx::core::array(
            std::initializer_list<int>{}, shape, target_dtype), out_sr};
    }

    if (duration.has_value()) {
        frames_to_read = static_cast<int64_t>(std::floor(duration.value() * out_sr));
        if (start_frame + frames_to_read > total_frames_out) {
            frames_to_read = total_frames_out - start_frame;
        }
    } else {
        frames_to_read = total_frames_out - start_frame;
    }

    // --- Seek ---
    if (start_frame > 0) {
        status = ExtAudioFileSeek(ext_file.get(), start_frame);
        if (status != noErr) {
            throw std::runtime_error(
                "Failed to seek: OSStatus " + osstatus_to_string(status));
        }
    }

    // --- Allocate decode buffer and read ---
    size_t buffer_bytes = static_cast<size_t>(frames_to_read) * channels * sizeof(float);
    float* buffer = static_cast<float*>(aligned_alloc_64(
        buffer_bytes > 0 ? buffer_bytes : 64));

    UInt32 frames_remaining = static_cast<UInt32>(frames_to_read);
    UInt32 total_read = 0;
    float* write_ptr = buffer;

    while (frames_remaining > 0) {
        UInt32 read_count = std::min(frames_remaining, kReadChunkFrames);

        AudioBufferList abl;
        abl.mNumberBuffers = 1;
        abl.mBuffers[0].mNumberChannels = static_cast<UInt32>(channels);
        abl.mBuffers[0].mDataByteSize =
            read_count * static_cast<UInt32>(channels) * sizeof(float);
        abl.mBuffers[0].mData = write_ptr;
        status = ExtAudioFileRead(ext_file.get(), &read_count, &abl);
        if (status != noErr) {
            std::free(buffer);
            throw std::runtime_error(
                "Failed to read audio: OSStatus " + osstatus_to_string(status));
        }
        if (read_count == 0) break;  // EOF

        total_read += read_count;
        frames_remaining -= read_count;
        write_ptr += static_cast<size_t>(read_count) * channels;
    }

    int64_t actual_frames = static_cast<int64_t>(total_read);

    // --- Transform 1: mono mixdown ---
    int out_channels = channels;
    if (mono && channels > 1) {
        size_t mono_bytes = static_cast<size_t>(actual_frames) * sizeof(float);
        float* mono_buf = static_cast<float*>(aligned_alloc_64(
            mono_bytes > 0 ? mono_bytes : 64));

        if (channels == 2) {
            // Stereo fast path: (L + R) / 2 using vDSP
            vDSP_vadd(
                buffer, 2,           // L channel, stride 2
                buffer + 1, 2,       // R channel, stride 2
                mono_buf, 1,         // output, stride 1
                static_cast<vDSP_Length>(actual_frames));
            float half = 0.5f;
            vDSP_vsmul(
                mono_buf, 1, &half,
                mono_buf, 1,
                static_cast<vDSP_Length>(actual_frames));
        } else {
            // General case: average all channels
            // Start with channel 0 (strided copy)
            strided_copy(buffer, channels, mono_buf, 1,
                         static_cast<int>(actual_frames));
            // Add remaining channels
            for (int c = 1; c < channels; ++c) {
                vDSP_vadd(
                    mono_buf, 1,
                    buffer + c, channels,
                    mono_buf, 1,
                    static_cast<vDSP_Length>(actual_frames));
            }
            float scale = 1.0f / static_cast<float>(channels);
            vDSP_vsmul(
                mono_buf, 1, &scale,
                mono_buf, 1,
                static_cast<vDSP_Length>(actual_frames));
        }

        std::free(buffer);
        buffer = mono_buf;
        out_channels = 1;
    }

    // --- Transform 2: channels_first deinterleave ---
    if (layout == "channels_first" && out_channels > 1) {
        size_t planar_bytes =
            static_cast<size_t>(actual_frames) * out_channels * sizeof(float);
        float* planar_buf = static_cast<float*>(aligned_alloc_64(planar_bytes));

        for (int c = 0; c < out_channels; ++c) {
            strided_copy(
                buffer + c,                        // source: interleaved
                out_channels,                      // source stride
                planar_buf + c * actual_frames,    // dest: planar
                1,                                 // dest stride
                static_cast<int>(actual_frames));
        }

        std::free(buffer);
        buffer = planar_buf;
    }

    // --- Wrap in mlx::core::array ---
    mlx::core::Shape shape;
    if (layout == "channels_first" && out_channels > 1) {
        shape = {static_cast<int32_t>(out_channels),
                 static_cast<int32_t>(actual_frames)};
    } else if (out_channels == 1 && layout == "channels_first") {
        shape = {1, static_cast<int32_t>(actual_frames)};
    } else {
        shape = {static_cast<int32_t>(actual_frames),
                 static_cast<int32_t>(out_channels)};
    }

    auto arr = mlx::core::array(
        static_cast<void*>(buffer),
        std::move(shape),
        mlx::core::float32,
        aligned_free);

    if (dtype == "float16") {
        arr = mlx::core::astype(arr, mlx::core::float16);
    }

    return {std::move(arr), out_sr};
}

// ---------------------------------------------------------------------------
// save_audio
// ---------------------------------------------------------------------------

void backend_save_audio(
    const std::string& path,
    mlx::core::array audio,
    int sr,
    const std::string& layout,
    const std::string& encoding,
    const std::string& bitrate,
    bool clip) {

    // --- Validate inputs ---
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
    if (audio.dtype() == mlx::core::float16) {
        audio = mlx::core::astype(audio, mlx::core::float32);
    }
    if (audio.ndim() == 1) {
        audio = mlx::core::reshape(audio, {audio.shape(0), 1});
    } else if (audio.ndim() != 2) {
        throw value_error(
            "audio must be 1D or 2D, got ndim=" + std::to_string(audio.ndim()));
    }

    // Determine output format from file extension
    auto out_fmt = parse_output_format(path);
    AudioFileTypeID output_file_type = out_fmt.file_type;

    // If .m4a with encoding="alac", override kind to ALAC
    if (out_fmt.kind == OutputFormat::AAC && encoding == "alac") {
        out_fmt.kind = OutputFormat::ALAC;
    }
    if (out_fmt.file_type == kAudioFileAIFFType && encoding != "pcm16") {
        // Float AIFF is represented as AIFC on Apple APIs.
        output_file_type = kAudioFileAIFCType;
    }

    // Validate encoding for the chosen format
    switch (out_fmt.kind) {
        case OutputFormat::PCM:
            if (bitrate != "auto") {
                throw value_error(
                    "bitrate is only supported for encoded formats, not WAV/AIFF/CAF");
            }
            if (encoding != "auto" && encoding != "float32" && encoding != "pcm16") {
                throw value_error(
                    "Unsupported encoding '" + encoding +
                    "' for " + path.substr(path.rfind('.')) +
                    ". Use 'auto', 'float32', or 'pcm16'.");
            }
            break;
        case OutputFormat::AAC:
            if (encoding != "auto" && encoding != "float32" && encoding != "pcm16") {
                throw value_error(
                    "Unsupported encoding '" + encoding +
                    "' for .m4a on macOS backend. Use 'auto', 'float32', or 'pcm16'.");
            }
            parse_bitrate(bitrate);  // throws on invalid
            break;
        case OutputFormat::FLAC:
            if (bitrate != "auto") {
                throw value_error("bitrate is not supported for .flac on macOS backend");
            }
            if (encoding != "float32" && encoding != "auto" && encoding != "pcm16") {
                throw value_error(
                    "Unsupported encoding '" + encoding +
                    "' for FLAC. Use 'auto', 'float32', or 'pcm16'.");
            }
            break;
        case OutputFormat::ALAC:
            if (bitrate != "auto") {
                throw value_error(
                    "bitrate is not supported when encoding='alac' for .m4a");
            }
            break;
        case OutputFormat::MP3:
            if (encoding != "auto" && encoding != "float32" && encoding != "pcm16") {
                throw value_error(
                    "Unsupported encoding '" + encoding +
                    "' for .mp3. Use 'auto', 'float32', or 'pcm16'.");
            }
            parse_mp3_bitrate_kbps(bitrate);  // throws on invalid
            break;
    }

    // Determine frames and channels from shape + layout
    int frames, channels;
    if (layout == "channels_last") {
        frames = audio.shape(0);
        channels = audio.shape(1);
    } else {
        channels = audio.shape(0);
        frames = audio.shape(1);
    }

    // Materialize the array on CPU
    mlx::core::eval(audio);

    const float* data = audio.data<float>();

    // Temporary buffer for interleaving (channels_first with >1 channel)
    std::unique_ptr<float, decltype(&std::free)> interleaved(nullptr, std::free);
    const float* write_data = data;

    if (layout == "channels_first" && channels > 1) {
        size_t buf_bytes = static_cast<size_t>(frames) * channels * sizeof(float);
        interleaved.reset(static_cast<float*>(aligned_alloc_64(buf_bytes)));

        for (int c = 0; c < channels; ++c) {
            strided_copy(
                data + c * frames,       // source: planar
                1,                       // source stride
                interleaved.get() + c,   // dest: interleaved
                channels,                // dest stride
                frames);
        }

        write_data = interleaved.get();
    }

    // If we already have a temp buffer (interleaved), clip in-place
    if (clip && interleaved) {
        size_t total = static_cast<size_t>(frames) * channels;
        float lo = -1.0f, hi = 1.0f;
        vDSP_vclip(interleaved.get(), 1, &lo, &hi, interleaved.get(), 1,
                    static_cast<vDSP_Length>(total));
    }

    // MP3: use vendored LAME encoder (AudioToolbox has no MP3 encoder)
    if (out_fmt.kind == OutputFormat::MP3) {
        save_mp3_lame(path, write_data, frames, channels, sr, bitrate);
        return;
    }

    // --- Set up file format ---
    auto url = make_url(path);
    AudioStreamBasicDescription file_fmt = {};
    bool needs_client_format = false;

    switch (out_fmt.kind) {
        case OutputFormat::PCM: {
            file_fmt.mSampleRate = static_cast<Float64>(sr);
            file_fmt.mFormatID = kAudioFormatLinearPCM;
            file_fmt.mChannelsPerFrame = static_cast<UInt32>(channels);
            file_fmt.mFramesPerPacket = 1;
            const bool is_aiff_family =
                (output_file_type == kAudioFileAIFFType || output_file_type == kAudioFileAIFCType);

            if (encoding == "pcm16") {
                file_fmt.mFormatFlags =
                    kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
                if (is_aiff_family) {
                    file_fmt.mFormatFlags |= kAudioFormatFlagIsBigEndian;
                }
                file_fmt.mBitsPerChannel = 16;
                file_fmt.mBytesPerFrame = static_cast<UInt32>(channels * 2);
                needs_client_format = true;
            } else {
                file_fmt.mFormatFlags =
                    kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
                if (is_aiff_family) {
                    file_fmt.mFormatFlags |= kAudioFormatFlagIsBigEndian;
                }
                file_fmt.mBitsPerChannel = 32;
                file_fmt.mBytesPerFrame = static_cast<UInt32>(channels * sizeof(float));
            }
            file_fmt.mBytesPerPacket = file_fmt.mBytesPerFrame;
            break;
        }
        case OutputFormat::AAC:
            file_fmt = make_aac_format(sr, channels);
            needs_client_format = true;
            break;
        case OutputFormat::FLAC:
            file_fmt = make_flac_format(sr, channels);
            needs_client_format = true;
            break;
        case OutputFormat::ALAC:
            file_fmt = make_alac_format(sr, channels);
            needs_client_format = true;
            break;
        case OutputFormat::MP3:
            // Unreachable: MP3 is handled by early return above
            break;
    }

    ScopedExtAudioFile ext_file;
    OSStatus status = ExtAudioFileCreateWithURL(
        url.get(),
        output_file_type,
        &file_fmt,
        nullptr,
        kAudioFileFlags_EraseFile,
        ext_file.ptr());
    if (status != noErr) {
        throw std::runtime_error(
            "Failed to create audio file '" + path +
            "': OSStatus " + osstatus_to_string(status));
    }

    // Set client format so ExtAudioFile converts from float32 input
    if (needs_client_format) {
        auto client_fmt = make_client_format(sr, channels);
        status = ExtAudioFileSetProperty(
            ext_file.get(), kExtAudioFileProperty_ClientDataFormat,
            sizeof(client_fmt), &client_fmt);
        if (status != noErr) {
            throw std::runtime_error(
                "Failed to set client format: OSStatus " + osstatus_to_string(status));
        }
    }

    // Set AAC bitrate if specified
    if (out_fmt.kind == OutputFormat::AAC) {
        UInt32 br = parse_bitrate(bitrate);
        if (br > 0) {
            AudioConverterRef converter = nullptr;
            UInt32 conv_size = sizeof(converter);
            status = ExtAudioFileGetProperty(
                ext_file.get(), kExtAudioFileProperty_AudioConverter,
                &conv_size, &converter);
            if (status == noErr && converter != nullptr) {
                AudioConverterSetProperty(
                    converter, kAudioConverterEncodeBitRate,
                    sizeof(br), &br);
            }
        }
    }

    // --- Write audio data ---
    // Determine if we need chunked clipping (channels_last + clip, no temp buffer yet)
    bool need_chunk_clip = clip && !interleaved;
    // Encoded formats must always use chunked writes (encoder buffer limits)
    bool use_chunked = need_chunk_clip || (out_fmt.kind != OutputFormat::PCM);

    if (!use_chunked) {
        // Fast path: data is ready — write in one call.
        AudioBufferList abl;
        abl.mNumberBuffers = 1;
        abl.mBuffers[0].mNumberChannels = static_cast<UInt32>(channels);
        abl.mBuffers[0].mDataByteSize =
            static_cast<UInt32>(frames) * static_cast<UInt32>(channels) * sizeof(float);
        abl.mBuffers[0].mData = const_cast<float*>(write_data);

        status = ExtAudioFileWrite(ext_file.get(), static_cast<UInt32>(frames), &abl);
        if (status != noErr) {
            throw std::runtime_error(
                "Failed to write audio: OSStatus " + osstatus_to_string(status));
        }
    } else {
        // Chunked write path
        constexpr int kChunkFrames = 131072;
        size_t chunk_samples = static_cast<size_t>(kChunkFrames) * channels;
        std::unique_ptr<float, decltype(&std::free)> clip_buf(nullptr, std::free);
        if (need_chunk_clip) {
            clip_buf.reset(static_cast<float*>(
                aligned_alloc_64(chunk_samples * sizeof(float))));
        }

        int frames_written = 0;
        while (frames_written < frames) {
            int chunk = std::min(kChunkFrames, frames - frames_written);
            const float* src = write_data +
                static_cast<size_t>(frames_written) * channels;

            AudioBufferList abl;
            abl.mNumberBuffers = 1;
            abl.mBuffers[0].mNumberChannels = static_cast<UInt32>(channels);
            abl.mBuffers[0].mDataByteSize =
                static_cast<UInt32>(chunk) * static_cast<UInt32>(channels) * sizeof(float);

            if (need_chunk_clip) {
                size_t n = static_cast<size_t>(chunk) * channels;
                float lo = -1.0f, hi = 1.0f;
                vDSP_vclip(src, 1, &lo, &hi, clip_buf.get(), 1,
                            static_cast<vDSP_Length>(n));
                abl.mBuffers[0].mData = clip_buf.get();
            } else {
                abl.mBuffers[0].mData = const_cast<float*>(src);
            }

            status = ExtAudioFileWrite(ext_file.get(), static_cast<UInt32>(chunk), &abl);
            if (status != noErr) {
                throw std::runtime_error(
                    "Failed to write audio: OSStatus " + osstatus_to_string(status));
            }

            frames_written += chunk;
        }
    }

    // Explicitly dispose the file handle so the filesystem flush is
    // deterministic (visible in profiling rather than deferred to dtor).
    ext_file.reset();
}

}  // namespace mlx_audio
