cmake_minimum_required(VERSION 3.24)
project(mlx_audio_io LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    # FORCE is load-bearing. scikit-build-core pre-seeds both of these as empty
    # cache entries, and `set(... CACHE ...)` will not overwrite an entry that
    # already exists. Without FORCE the guard fires, the set is silently a
    # no-op, and the value stays empty -- which is how deployment_target came
    # to be recorded as "" in every build, disabling the runtime macOS
    # version check that reads it.
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target" FORCE)
    endif()
    if(NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "macOS architectures" FORCE)
    endif()
endif()

set(LINUX_LIBAV_LINK_TARGETS)
if(UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(
        LIBAV
        REQUIRED
        IMPORTED_TARGET
        libavformat
        libavcodec
        libavutil
        libswresample
    )
    list(APPEND LINUX_LIBAV_LINK_TARGETS PkgConfig::LIBAV)
endif()

option(MLX_AUDIO_IO_ENABLE_SOXR "Enable libsoxr-backed resampling modes" ON)
set(MLX_AUDIO_IO_HAS_SOXR OFF)
if(MLX_AUDIO_IO_ENABLE_SOXR)
    find_path(SOXR_INCLUDE_DIR soxr.h)
    find_library(SOXR_LIBRARY NAMES soxr libsoxr)
    if(SOXR_INCLUDE_DIR AND SOXR_LIBRARY)
        set(MLX_AUDIO_IO_HAS_SOXR ON)
        message(STATUS "Found libsoxr: ${SOXR_LIBRARY}")
    else()
        message(STATUS "libsoxr not found; soxr_* resample quality modes disabled")
    endif()
endif()

# Find Python and nanobind
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Find MLX
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE MLX_DIR
)
find_package(MLX CONFIG REQUIRED PATHS "${MLX_DIR}")

if(APPLE AND TARGET mlx)
    get_target_property(MLX_IMPORTED_LINK_LIBS mlx INTERFACE_LINK_LIBRARIES)
    if(MLX_IMPORTED_LINK_LIBS AND MLX_IMPORTED_LINK_LIBS MATCHES "Xcode-latest\\.app")
        if(NOT CMAKE_OSX_SYSROOT)
            execute_process(
                COMMAND xcrun --sdk macosx --show-sdk-path
                OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif()
        if(CMAKE_OSX_SYSROOT)
            string(
                REGEX REPLACE
                    "/Applications/Xcode-latest\\.app/Contents/Developer/Platforms/MacOSX\\.platform/Developer/SDKs/MacOSX\\.sdk"
                    "${CMAKE_OSX_SYSROOT}"
                    MLX_IMPORTED_LINK_LIBS
                    "${MLX_IMPORTED_LINK_LIBS}"
            )
            set_target_properties(
                mlx
                PROPERTIES INTERFACE_LINK_LIBRARIES "${MLX_IMPORTED_LINK_LIBS}"
            )
        endif()
    endif()
endif()

set(BACKEND_SOURCES)
set(LAME_SOURCES)
if(APPLE)
    list(APPEND BACKEND_SOURCES src/cpp/audio_backend_apple.cpp)
    set(LAME_SOURCES
        src/vendor/lame/libmp3lame/VbrTag.c
        src/vendor/lame/libmp3lame/bitstream.c
        src/vendor/lame/libmp3lame/encoder.c
        src/vendor/lame/libmp3lame/fft.c
        src/vendor/lame/libmp3lame/gain_analysis.c
        src/vendor/lame/libmp3lame/id3tag.c
        src/vendor/lame/libmp3lame/lame.c
        src/vendor/lame/libmp3lame/mpglib_interface.c
        src/vendor/lame/libmp3lame/newmdct.c
        src/vendor/lame/libmp3lame/presets.c
        src/vendor/lame/libmp3lame/psymodel.c
        src/vendor/lame/libmp3lame/quantize.c
        src/vendor/lame/libmp3lame/quantize_pvt.c
        src/vendor/lame/libmp3lame/reservoir.c
        src/vendor/lame/libmp3lame/set_get.c
        src/vendor/lame/libmp3lame/tables.c
        src/vendor/lame/libmp3lame/takehiro.c
        src/vendor/lame/libmp3lame/util.c
        src/vendor/lame/libmp3lame/vbrquantize.c
        src/vendor/lame/libmp3lame/version.c
    )
    # LAME is C code: compile as C, enable config.h, suppress warnings
    set_source_files_properties(${LAME_SOURCES} PROPERTIES
        LANGUAGE C
        COMPILE_DEFINITIONS "HAVE_CONFIG_H"
        COMPILE_OPTIONS "-w"
    )
elseif(UNIX)
    list(APPEND BACKEND_SOURCES src/cpp/audio_backend_linux.cpp)
else()
    message(FATAL_ERROR "Unsupported platform for mlx-audio-io")
endif()

# Build the _core extension module
nanobind_add_module(
    _core
    NB_STATIC
    STABLE_ABI
    LTO
    NOMINSIZE
    NB_DOMAIN mlx
    src/cpp/bindings.cpp
    src/cpp/audio_io.cpp
    src/cpp/audio_stream.cpp
    src/cpp/mp3_decoder.cpp
    src/cpp/mp3_encoder.cpp
    src/cpp/linux_libav_probe.cpp
    src/cpp/resample_stream.cpp
    ${BACKEND_SOURCES}
    ${LAME_SOURCES}
)

target_include_directories(_core PRIVATE
    src/cpp
    src/vendor/minimp3
    src/vendor/lame
    src/vendor/lame/libmp3lame
)
target_link_libraries(_core PRIVATE mlx)
if(APPLE)
    target_link_libraries(_core PRIVATE "-framework AudioToolbox" "-framework Accelerate")
endif()
if(UNIX AND NOT APPLE)
    target_compile_definitions(_core PRIVATE MLX_AUDIO_IO_ENABLE_LIBAV=1)
    target_link_libraries(_core PRIVATE ${LINUX_LIBAV_LINK_TARGETS})
endif()
if(MLX_AUDIO_IO_HAS_SOXR)
    target_compile_definitions(_core PRIVATE MLX_AUDIO_IO_ENABLE_SOXR=1)
    target_include_directories(_core PRIVATE ${SOXR_INCLUDE_DIR})
    target_link_libraries(_core PRIVATE ${SOXR_LIBRARY})
endif()

# Install into the Python package directory
install(TARGETS _core LIBRARY DESTINATION mlx_audio_io)

set(MLX_AUDIO_IO_DEPLOYMENT_TARGET "")
if(APPLE)
    set(MLX_AUDIO_IO_DEPLOYMENT_TARGET "${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

set(MLX_AUDIO_IO_WHEEL_VERSION "")
if(DEFINED SKBUILD_PROJECT_VERSION AND NOT "${SKBUILD_PROJECT_VERSION}" STREQUAL "")
    set(MLX_AUDIO_IO_WHEEL_VERSION "${SKBUILD_PROJECT_VERSION}")
elseif(DEFINED PROJECT_VERSION AND NOT "${PROJECT_VERSION}" STREQUAL "")
    set(MLX_AUDIO_IO_WHEEL_VERSION "${PROJECT_VERSION}")
endif()

if("${MLX_AUDIO_IO_WHEEL_VERSION}" STREQUAL "")
    file(STRINGS "${CMAKE_SOURCE_DIR}/pyproject.toml" _pyproject_version_line REGEX "^version = \".*\"")
    if(_pyproject_version_line)
        list(GET _pyproject_version_line 0 _pyproject_version_first)
        string(REGEX REPLACE "^version = \"([^\"]+)\".*" "\\1" MLX_AUDIO_IO_WHEEL_VERSION "${_pyproject_version_first}")
    endif()
endif()

if("${MLX_AUDIO_IO_WHEEL_VERSION}" STREQUAL "")
    set(MLX_AUDIO_IO_WHEEL_VERSION "unknown")
endif()

set(MLX_AUDIO_IO_BUILD_MLX_VERSION "")
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.metadata as m; print(m.version('mlx'))"
    OUTPUT_VARIABLE MLX_AUDIO_IO_BUILD_MLX_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if("${MLX_AUDIO_IO_BUILD_MLX_VERSION}" STREQUAL "")
    set(MLX_AUDIO_IO_BUILD_MLX_VERSION "unknown")
endif()

# MLX versions this binary is declared compatible with, as a comma-separated
# list. MLX has no stable C++ ABI and no soname versioning, so this must only
# ever list versions the build was actually compiled and tested against --
# never an open range. Override at configure time once a version is verified:
#   -DMLX_AUDIO_IO_COMPATIBLE_MLX_VERSIONS="0.31.2,0.31.3"
if(NOT DEFINED MLX_AUDIO_IO_COMPATIBLE_MLX_VERSIONS OR "${MLX_AUDIO_IO_COMPATIBLE_MLX_VERSIONS}" STREQUAL "")
    set(MLX_AUDIO_IO_COMPATIBLE_MLX_VERSIONS "${MLX_AUDIO_IO_BUILD_MLX_VERSION}")
endif()

# nanobind must match the version MLX itself was built with. The extension
# shares nanobind's type registry with mlx.core through NB_DOMAIN, so
# mlx::core::array crosses the boundary using MLX's registered caster. A
# mismatch compiles and links cleanly and then fails on the first call with
# "Unable to convert function return value to a Python type" -- an error that
# says nothing about nanobind. Known pairings: MLX 0.31.x -> nanobind 2.12.0,
# MLX 0.32.x -> nanobind 2.15.0 (see MLX's own CMakeLists FetchContent GIT_TAG).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import importlib.metadata as m; print(m.version('nanobind'))"
    OUTPUT_VARIABLE MLX_AUDIO_IO_NANOBIND_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if("${MLX_AUDIO_IO_NANOBIND_VERSION}" STREQUAL "")
    set(MLX_AUDIO_IO_NANOBIND_VERSION "unknown")
endif()
message(STATUS "mlx-audio-io: building against MLX ${MLX_AUDIO_IO_BUILD_MLX_VERSION} with nanobind ${MLX_AUDIO_IO_NANOBIND_VERSION}")

# A mismatch here does not degrade -- it produces a binary that imports and
# then fails every call. Failing the build is the only signal a user gets,
# because under `pip install` CMake output is captured and shown on failure
# only, so message(WARNING) is invisible to exactly the person who needs it.
set(MLX_AUDIO_IO_NANOBIND_EXPECTED "")
if(MLX_AUDIO_IO_BUILD_MLX_VERSION MATCHES "^0\\.31\\.")
    set(MLX_AUDIO_IO_NANOBIND_EXPECTED "2.12.")
elseif(MLX_AUDIO_IO_BUILD_MLX_VERSION MATCHES "^0\\.32\\.")
    set(MLX_AUDIO_IO_NANOBIND_EXPECTED "2.15.")
endif()
if(NOT "${MLX_AUDIO_IO_NANOBIND_EXPECTED}" STREQUAL ""
   AND NOT MLX_AUDIO_IO_NANOBIND_VERSION MATCHES "^${MLX_AUDIO_IO_NANOBIND_EXPECTED}")
    if(MLX_AUDIO_IO_ALLOW_NANOBIND_MISMATCH)
        message(WARNING
            "MLX ${MLX_AUDIO_IO_BUILD_MLX_VERSION} expects nanobind "
            "${MLX_AUDIO_IO_NANOBIND_EXPECTED}x, found ${MLX_AUDIO_IO_NANOBIND_VERSION}. "
            "Continuing because MLX_AUDIO_IO_ALLOW_NANOBIND_MISMATCH is set; "
            "expect every call to fail with 'Unable to convert function return value'.")
    else()
        message(FATAL_ERROR
            "nanobind/MLX mismatch: MLX ${MLX_AUDIO_IO_BUILD_MLX_VERSION} is built with "
            "nanobind ${MLX_AUDIO_IO_NANOBIND_EXPECTED}x, but this build environment has "
            "nanobind ${MLX_AUDIO_IO_NANOBIND_VERSION}.\n"
            "The extension shares nanobind's type registry with mlx.core through "
            "NB_DOMAIN, so a mismatch links cleanly and then fails on every call with "
            "\"Unable to convert function return value to a Python type\".\n"
            "Fix: pip install \"nanobind==${MLX_AUDIO_IO_NANOBIND_EXPECTED}0\", or let the "
            "build backend pick it for you (it does this automatically unless you passed "
            "--no-build-isolation with nanobind already installed).\n"
            "To build anyway, configure with -DMLX_AUDIO_IO_ALLOW_NANOBIND_MISMATCH=ON.")
    endif()
endif()

configure_file(
    "${CMAKE_SOURCE_DIR}/python/mlx_audio_io/_build_info.json.in"
    "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json" DESTINATION mlx_audio_io)
