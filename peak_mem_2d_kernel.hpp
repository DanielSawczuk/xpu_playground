#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>

namespace peak_memory {

// One 2D message reads 16 dwords x 8 rows = 512 bytes. Four independent
// messages give each eSIMD work-item 2 KiB of useful read traffic.
inline constexpr unsigned kSurfaceWidth = 1024;
inline constexpr unsigned kBlockWidth = 16;
inline constexpr unsigned kBlockHeight = 8;
inline constexpr unsigned kLoadsPerWorkItem = 4;
inline constexpr unsigned kRowsPerWorkItem =
    kBlockHeight * kLoadsPerWorkItem;
inline constexpr unsigned kTilesX = kSurfaceWidth / kBlockWidth;
inline constexpr std::size_t kLocalSize = 16;
inline constexpr std::size_t kBytesPerWorkItem =
    kBlockWidth * kBlockHeight * kLoadsPerWorkItem * sizeof(float);

static_assert(kSurfaceWidth % kBlockWidth == 0);
static_assert(kBytesPerWorkItem == 2048);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
SYCL_EXTERNAL SYCL_ESIMD_KERNEL
void peak_mem_2d_kernel(const float *input, float *sink,
                        unsigned surface_height);

} // namespace peak_memory
