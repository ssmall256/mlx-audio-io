#pragma once

#include <cstring>
#include <string>

#if defined(__APPLE__)
#include <MacTypes.h>
#endif

namespace mlx_audio {

/// Convert an OSStatus code to a human-readable string.
/// If the code is a valid FourCC, return it as 'abcd' (numeric).
/// Otherwise just return the numeric value.
#if defined(__APPLE__)
inline std::string osstatus_to_string(OSStatus status) {
    // Try FourCC interpretation
    char fourcc[5] = {};
    uint32_t be = __builtin_bswap32(static_cast<uint32_t>(status));
    std::memcpy(fourcc, &be, 4);

    bool is_printable = true;
    for (int i = 0; i < 4; ++i) {
        if (fourcc[i] < 0x20 || fourcc[i] > 0x7E) {
            is_printable = false;
            break;
        }
    }

    if (is_printable) {
        return "'" + std::string(fourcc) + "' (" + std::to_string(status) + ")";
    }
    return std::to_string(status);
}
#else
inline std::string osstatus_to_string(int status) {
    return std::to_string(status);
}
#endif

}  // namespace mlx_audio
