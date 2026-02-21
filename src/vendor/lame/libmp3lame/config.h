/*
 * Minimal config.h for vendored LAME in mlx-audio-io.
 * Only defines what the LAME library sources need to compile
 * on macOS ARM64 (and Linux x86_64/ARM64 as a fallback).
 */

#ifndef LAME_CONFIG_H
#define LAME_CONFIG_H

#define STDC_HEADERS 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_ERRNO_H 1
#define HAVE_FCNTL_H 1
#define HAVE_STRCHR 1
#define HAVE_MEMCPY 1
#define HAVE_LIMITS_H 1

/* IEEE 754 float representation — true on ARM64 and x86_64 */
#define HAVE_IEEE754_FLOAT64 1
#define FLOAT8 float
#define REAL_IS_FLOAT 1
#define IEEE754_FLOAT32_REFERENCE 1

/* LAME uses ieee754_float32_t throughout; typedef it to float */
typedef float ieee754_float32_t;

/* We are building the library, not the frontend */
#define LAME_LIBRARY_BUILD 1

/* Disable mpglib decoder — we use minimp3 for decoding */
/* #undef HAVE_MPGLIB */

/* Disable x86 assembly optimizations */
/* #undef HAVE_NASM */
/* #undef HAVE_XMMINTRIN_H */
/* #undef MMX_choose_table */

#endif /* LAME_CONFIG_H */
