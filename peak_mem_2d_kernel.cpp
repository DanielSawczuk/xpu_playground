#include "peak_mem_2d_kernel.hpp"

#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>

#include <cstdint>

namespace esimd = sycl::ext::intel::esimd;
namespace esimd_exp = sycl::ext::intel::experimental::esimd;

namespace peak_memory {

SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::nd_range_kernel<1>))
SYCL_EXTERNAL SYCL_ESIMD_KERNEL
void peak_mem_2d_kernel(const float *input, float *sink,
                        unsigned surface_height) {
  constexpr unsigned surface_width_bytes = kSurfaceWidth * sizeof(float);
  const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
  const unsigned gid = item.get_global_linear_id();
  const int x = static_cast<int>((gid % kTilesX) * kBlockWidth);
  const int y = static_cast<int>((gid / kTilesX) * kRowsPerWorkItem);

  // Streaming at L1 avoids polluting the nearest cache. Keeping L2 cached is
  // the normal read-through path; the surface is much larger than cache.
  constexpr esimd::properties load_properties{
      esimd::cache_hint_L1<esimd::cache_hint::streaming>,
      esimd::cache_hint_L2<esimd::cache_hint::cached>};

  auto v0 = esimd::load_2d<float, kBlockWidth, kBlockHeight>(
      input, surface_width_bytes - 1, surface_height - 1,
      surface_width_bytes - 1, x, y, load_properties);
  auto v1 = esimd::load_2d<float, kBlockWidth, kBlockHeight>(
      input, surface_width_bytes - 1, surface_height - 1,
      surface_width_bytes - 1, x, y + kBlockHeight, load_properties);
  auto v2 = esimd::load_2d<float, kBlockWidth, kBlockHeight>(
      input, surface_width_bytes - 1, surface_height - 1,
      surface_width_bytes - 1, x, y + 2 * kBlockHeight, load_properties);
  auto v3 = esimd::load_2d<float, kBlockWidth, kBlockHeight>(
      input, surface_width_bytes - 1, surface_height - 1,
      surface_width_bytes - 1, x, y + 3 * kBlockHeight, load_properties);

  // Preserve each complete block-2D operation and expose a cheap checksum.
  esimd_exp::wait(v0);
  esimd_exp::wait(v1);
  esimd_exp::wait(v2);
  esimd_exp::wait(v3);
  const float checksum = v0[0] + v1[0] + v2[0] + v3[0];
  esimd::scatter<float, 1>(
      sink, esimd::simd<std::uint32_t, 1>(gid * sizeof(float)),
      esimd::simd<float, 1>(checksum));
}

} // namespace peak_memory
