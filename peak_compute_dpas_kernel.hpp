#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace peak_compute {

using bf16 = sycl::ext::oneapi::bfloat16;

// Xe2/BMG BF16 DPAS: C[M,N] += A[M,K] * B[K,N].
inline constexpr int kSystolicDepth = 8;
inline constexpr int kRepeatCount = 8;
inline constexpr int kM = kRepeatCount;
inline constexpr int kN = 16;
inline constexpr int kK = 16;
inline constexpr int kASize = kM * kK;
inline constexpr int kBSize = kK * kN;
inline constexpr int kCSize = kM * kN;
inline constexpr int kAccumulators = 8;
inline constexpr std::uint64_t kFlopsPerDpas = 2ull * kM * kN * kK;
inline constexpr std::uint64_t kFlopsPerRound =
    kAccumulators * kFlopsPerDpas;

// Block-2D surfaces require at least a 64-byte pitch.
inline constexpr unsigned kSurfaceWidth = 32;
inline constexpr unsigned kSurfaceHeight = 16;
inline constexpr unsigned kSurfaceBytes = kSurfaceWidth * sizeof(bf16);
inline constexpr std::size_t kLocalSize = 16;

static_assert(kFlopsPerDpas == 4096);

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
SYCL_EXTERNAL SYCL_ESIMD_KERNEL
void peak_compute_dpas_kernel(const bf16 *a_surface, const bf16 *b_surface,
                              float *sink, unsigned rounds);

} // namespace peak_compute
