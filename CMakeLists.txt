cmake_minimum_required(VERSION 3.24)
project(mlx_audio_io LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target")
    endif()
    if(NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "macOS architectures")
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

configure_file(
    "${CMAKE_SOURCE_DIR}/python/mlx_audio_io/_build_info.json.in"
    "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json"
    @ONLY
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_build_info.json" DESTINATION mlx_audio_io)
