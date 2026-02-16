#pragma once

#if defined(__APPLE__)
#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace mlx_audio {

#if defined(__APPLE__)
/// RAII wrapper for ExtAudioFileRef.
class ScopedExtAudioFile {
  public:
    explicit ScopedExtAudioFile(ExtAudioFileRef ref = nullptr) : ref_(ref) {}
    ~ScopedExtAudioFile() {
        if (ref_) ExtAudioFileDispose(ref_);
    }

    // Non-copyable
    ScopedExtAudioFile(const ScopedExtAudioFile&) = delete;
    ScopedExtAudioFile& operator=(const ScopedExtAudioFile&) = delete;

    // Movable
    ScopedExtAudioFile(ScopedExtAudioFile&& other) noexcept : ref_(other.ref_) {
        other.ref_ = nullptr;
    }
    ScopedExtAudioFile& operator=(ScopedExtAudioFile&& other) noexcept {
        if (this != &other) {
            if (ref_) ExtAudioFileDispose(ref_);
            ref_ = other.ref_;
            other.ref_ = nullptr;
        }
        return *this;
    }

    ExtAudioFileRef get() const { return ref_; }
    ExtAudioFileRef* ptr() { return &ref_; }

    /// Explicitly dispose the file handle (makes flush deterministic in profiling).
    void reset() {
        if (ref_) {
            ExtAudioFileDispose(ref_);
            ref_ = nullptr;
        }
    }

  private:
    ExtAudioFileRef ref_;
};

/// RAII wrapper for CFURLRef.
class ScopedCFURL {
  public:
    explicit ScopedCFURL(CFURLRef ref = nullptr) : ref_(ref) {}
    ~ScopedCFURL() {
        if (ref_) CFRelease(ref_);
    }

    // Non-copyable
    ScopedCFURL(const ScopedCFURL&) = delete;
    ScopedCFURL& operator=(const ScopedCFURL&) = delete;

    // Movable
    ScopedCFURL(ScopedCFURL&& other) noexcept : ref_(other.ref_) {
        other.ref_ = nullptr;
    }
    ScopedCFURL& operator=(ScopedCFURL&& other) noexcept {
        if (this != &other) {
            if (ref_) CFRelease(ref_);
            ref_ = other.ref_;
            other.ref_ = nullptr;
        }
        return *this;
    }

    CFURLRef get() const { return ref_; }

  private:
    CFURLRef ref_;
};
#endif

}  // namespace mlx_audio
