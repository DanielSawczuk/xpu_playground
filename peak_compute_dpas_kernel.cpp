#include "peak_compute_dpas_kernel.hpp"

#include <sycl/ext/oneapi/free_function_queries.hpp>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

namespace peak_compute {

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
SYCL_EXTERNAL SYCL_ESIMD_KERNEL
void peak_compute_dpas_kernel(const bf16 *a_surface, const bf16 *b_surface,
                              float *sink, unsigned rounds) {
  constexpr esimd::properties cache_properties{
      esimd::cache_hint_L1<esimd::cache_hint::cached>,
      esimd::cache_hint_L2<esimd::cache_hint::cached>};

  // Bring both tiles toward the XMX pipeline before consuming them. A is
  // row-major MxK. B is row-major KxN in memory; the transformed load produces
  // the vertical/VNNI packing required by DPAS src1.
  esimd::prefetch_2d<bf16, kK, kM>(
      a_surface, kSurfaceBytes - 1, kSurfaceHeight - 1, kSurfaceBytes - 1, 0,
      0, cache_properties);
  esimd::prefetch_2d<bf16, kN, kK>(
      b_surface, kSurfaceBytes - 1, kSurfaceHeight - 1, kSurfaceBytes - 1, 0,
      0, cache_properties);

  esimd::simd<bf16, kASize> a = esimd::load_2d<bf16, kK, kM>(
      a_surface, kSurfaceBytes - 1, kSurfaceHeight - 1, kSurfaceBytes - 1, 0,
      0, cache_properties);
  esimd::simd<bf16, kBSize> b =
      esimd::load_2d<bf16, kN, kK, 1, false, true>(
          b_surface, kSurfaceBytes - 1, kSurfaceHeight - 1,
          kSurfaceBytes - 1, 0, 0, cache_properties);

  // Distinct initial values prevent common-subexpression folding. Eight
  // independent dependency chains cover DPAS latency at full issue rate.
  esimd::simd<float, kCSize> c0(0.0f);
  esimd::simd<float, kCSize> c1(1.0f);
  esimd::simd<float, kCSize> c2(2.0f);
  esimd::simd<float, kCSize> c3(3.0f);
  esimd::simd<float, kCSize> c4(4.0f);
  esimd::simd<float, kCSize> c5(5.0f);
  esimd::simd<float, kCSize> c6(6.0f);
  esimd::simd<float, kCSize> c7(7.0f);

#pragma unroll 1
  for (unsigned round = 0; round < rounds; ++round) {
    c0 = xmx::dpas<kSystolicDepth, kRepeatCount, float>(c0, b, a);
    c1 = xmx::dpas<kSystolicDepth, kRepeatCount, float>(c1, b, a);
    c2 = xmx::dpas<kSystolicDepth, kRepeatCount, float>(c2, b, a);
    c3 = xmx::dpas<kSystolicDepth, kRepeatCount, float>(c3, b, a);
    c4 = xmx::dpas<kSystolicDepth, kRepeatCount, float>(c4, b, a);
    c5 = xmx::dpas<kSystolicDepth, kRepeatCount, float>(c5, b, a);
    c6 = xmx::dpas<kSystolicDepth, kRepeatCount, float>(c6, b, a);
    c7 = xmx::dpas<kSystolicDepth, kRepeatCount, float>(c7, b, a);
  }

  const float checksum = c0[0] + c1[0] + c2[0] + c3[0] + c4[0] + c5[0] +
                         c6[0] + c7[0];
  const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  const std::uint32_t byte_offset =
      item.get_global_linear_id() * sizeof(float);
  esimd::scatter<float, 1>(
      sink, esimd::simd<std::uint32_t, 1>(byte_offset),
      esimd::simd<float, 1>(checksum));
}

} // namespace peak_compute
