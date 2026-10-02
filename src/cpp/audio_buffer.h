#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "dlpack/dlpack.h"
#include "internal_utils.h"

namespace mlx_audio {

/// RAII wrapper for raw aligned audio memory.
struct AudioStorage {
    void* data = nullptr;
    size_t bytes = 0;
    bool owns = true;

    AudioStorage() = default;

    explicit AudioStorage(size_t size) : bytes(size), owns(true) {
        if (size > 0) {
            data = internal::aligned_alloc_64(size);
        }
    }

    AudioStorage(void* ptr, size_t size, bool take_ownership = true)
        : data(ptr), bytes(size), owns(take_ownership) {}

    ~AudioStorage() {
        if (owns && data) {
            std::free(data);
            data = nullptr;
        }
    }

    AudioStorage(const AudioStorage&) = delete;
    AudioStorage& operator=(const AudioStorage&) = delete;

    AudioStorage(AudioStorage&& other) noexcept
        : data(other.data), bytes(other.bytes), owns(other.owns) {
        other.data = nullptr;
        other.bytes = 0;
        other.owns = false;
    }

    AudioStorage& operator=(AudioStorage&& other) noexcept {
        if (this != &other) {
            if (owns && data) {
                std::free(data);
            }
            data = other.data;
            bytes = other.bytes;
            owns = other.owns;
            other.data = nullptr;
            other.bytes = 0;
            other.owns = false;
        }
        return *this;
    }
};

/// Context stored inside DLManagedTensor::manager_ctx.
struct DLPackContext {
    std::shared_ptr<AudioStorage> storage;
    int64_t shape[2];
    int64_t strides[2];
};

/// Independent multi-channel audio buffer supporting DLPack and Python Buffer Protocol.
class AudioBuffer {
public:
    std::shared_ptr<AudioStorage> storage;
    int64_t shape[2] = {0, 0};
    int64_t strides[2] = {0, 0};
    int ndim = 2;
    std::string dtype = "float32";

    AudioBuffer() = default;

    AudioBuffer(std::shared_ptr<AudioStorage> st,
                int64_t d0, int64_t d1,
                int64_t s0, int64_t s1,
                int n_dim = 2,
                std::string dt = "float32")
        : storage(std::move(st)),
          ndim(n_dim),
          dtype(std::move(dt)) {
        shape[0] = d0;
        shape[1] = d1;
        strides[0] = s0;
        strides[1] = s1;
    }

    int64_t frames() const { return (ndim == 1) ? shape[0] : shape[0]; }
    int channels() const { return (ndim == 1) ? 1 : static_cast<int>(shape[1]); }
    const void* data() const { return storage ? storage->data : nullptr; }
    void* data() { return storage ? storage->data : nullptr; }
    size_t itemsize() const { return (dtype == "float16") ? 2 : 4; }

    static AudioBuffer make_empty(int channels, const std::string& layout, const std::string& dtype) {
        AudioBuffer b;
        b.dtype = dtype;
        b.ndim = 2;
        if (layout == "channels_first") {
            b.shape[0] = channels;
            b.shape[1] = 0;
            b.strides[0] = 0;
            b.strides[1] = 1;
        } else {
            b.shape[0] = 0;
            b.shape[1] = channels;
            b.strides[0] = channels;
            b.strides[1] = 1;
        }
        b.storage = std::make_shared<AudioStorage>(0);
        return b;
    }
};

} // namespace mlx_audio
