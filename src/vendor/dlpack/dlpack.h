/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlpack.h
 * \brief The common header of DLPack.
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/*! brief The major version of DLPack */
#define DLPACK_MAJOR_VERSION 1
/*! brief The minor version of DLPack */
#define DLPACK_MINOR_VERSION 0

#ifndef DLPACK_DLL
#ifdef _WIN32
#ifdef DLPACK_EXPORTS
#define DLPACK_DLL __declspec(dllexport)
#else
#define DLPACK_DLL __declspec(dllimport)
#endif
#else
#define DLPACK_DLL
#endif
#endif

/*!
 * \brief The device type in DLDevice.
 */
typedef enum {
  /*! \brief CPU device */
  kDLCPU = 1,
  /*! \brief CUDA GPU device */
  kDLCUDA = 2,
  /*! \brief PINNED CUDA GPU device */
  kDLCUDAHost = 3,
  /*! \brief OpenCL devices */
  kDLOpenCL = 4,
  /*! \brief Vulkan reserved for GPU */
  kDLVulkan = 5,
  /*! \brief Metal for Apple GPU */
  kDLMetal = 8,
  /*! \brief VPI */
  kDLVPI = 9,
  /*! \brief ROCm GPUs for AMD GPUs */
  kDLROCm = 10,
  /*! \brief Pinned ROCm CPU memory */
  kDLROCmHost = 11,
  /*! \brief Reserved extension device */
  kDLExtDev = 12,
  /*! \brief CUDA managed/unified memory */
  kDLCUDAManaged = 13,
  /*! \brief Unified Shared Memory (USM) */
  kDLOneAPI = 14,
  /*! \brief GPU support for WebGPU */
  kDLWebGPU = 15,
  /*! \brief Qualcomm Hexagon DSP */
  kDLHexagon = 16,
  /*! \brief Microsoft Azure Maia */
  kDLMAIA = 17,
} DLDeviceType;

/*!
 * \brief A Device for Tensor and operator.
 */
typedef struct {
  /*! \brief The device type used in the device. */
  DLDeviceType device_type;
  /*! \brief The device index. */
  int32_t device_id;
} DLDevice;

/*!
 * \brief The type code in DLDataType.
 */
typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLBool = 6U,
} DLDataTypeCode;

/*!
 * \brief The data type the tensor can hold.
 */
typedef struct {
  /*! \brief Type code of base types. */
  uint8_t code;
  /*! \brief Number of bits, must be 8, 16, 32, or 64. */
  uint8_t bits;
  /*! \brief Number of lanes, usually 1. */
  uint16_t lanes;
} DLDataType;

/*!
 * \brief Plain C Tensor object, does not manage memory.
 */
typedef struct {
  /*! \brief The opaque data pointer points to the allocated data. */
  void* data;
  /*! \brief The device where the tensor resides. */
  DLDevice device;
  /*! \brief Number of dimensions */
  int32_t ndim;
  /*! \brief The data type of the pointer points to */
  DLDataType dtype;
  /*! \brief The shape of the tensor */
  int64_t* shape;
  /*! \brief Strides of the tensor (in number of elements, not bytes) */
  int64_t* strides;
  /*! \brief The offset in bytes to the beginning pointers to data */
  uint64_t byte_offset;
} DLTensor;

/*!
 * \brief C Tensor object, manages memory of DLTensor.
 */
typedef struct DLManagedTensor {
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
  /*! \brief Context to pass to deleter */
  void* manager_ctx;
  /*! \brief Destructor for DLManagedTensor */
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

/*!
 * \brief Versioned C Tensor object, manages memory of DLTensor.
 */
typedef struct DLManagedTensorVersioned {
  /*! \brief The API version the producer was compiled with. */
  uint64_t version;
  /*! \brief Context to pass to deleter */
  void* manager_ctx;
  /*! \brief Destructor for DLManagedTensorVersioned */
  void (*deleter)(struct DLManagedTensorVersioned* self);
  /*! \brief Flags */
  uint64_t flags;
  /*! \brief DLTensor which is being memory managed */
  DLTensor dl_tensor;
} DLManagedTensorVersioned;

#ifdef __cplusplus
}
#endif

#endif  // DLPACK_DLPACK_H_
