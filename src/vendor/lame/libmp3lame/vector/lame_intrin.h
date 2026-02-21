/*
 * Stub header for vendored LAME build.
 * The original provides x86 SSE intrinsics which are not needed on ARM64.
 * All SIMD-optimized function declarations are guarded by HAVE_XMMINTRIN_H
 * which is not defined in our config.h.
 */

#ifndef LAME_INTRIN_H
#define LAME_INTRIN_H

/* No SIMD intrinsics — pure C fallback is used */

#endif /* LAME_INTRIN_H */
