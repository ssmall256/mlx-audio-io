#pragma once

// Aligned allocation helpers with no other project dependencies, so low-level
// headers such as audio_buffer.h can use them without pulling in audio_io.h
// (which includes audio_buffer.h itself).

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace mlx_audio {
namespace internal {

/// Allocate 64-byte aligned memory.
inline void* aligned_alloc_64(size_t bytes) {
    void* ptr = nullptr;
    if (bytes == 0) return nullptr;
    int rc = posix_memalign(&ptr, 64, bytes);
    if (rc != 0 || ptr == nullptr) {
        throw std::runtime_error("Failed to allocate " + std::to_string(bytes) +
                                 " bytes of aligned memory");
    }
    return ptr;
}

inline void aligned_free(void* ptr) {
    std::free(ptr);
}

}  // namespace internal
}  // namespace mlx_audio
