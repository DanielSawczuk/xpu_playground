#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>

namespace gemm {

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr std::uint32_t kTileN = 256;
inline constexpr std::uint32_t kPrefetchK = 32;
inline constexpr std::uint32_t kPrefetchDistance = 3;
inline constexpr std::uint32_t kLocalSize = 32;
inline constexpr std::uint32_t kDpasM = 8;
inline constexpr std::uint32_t kDpasN = 16;
inline constexpr std::uint32_t kDpasK = 16;

template <typename T, bool LargeTile> struct Kernel {
  static constexpr std::uint32_t tile_m = LargeTile ? 192 : 128;
  static constexpr std::uint32_t register_rows = LargeTile ? 24 : 16;
  const T *a;
  const T *b;
  T *c;
  std::uint32_t m;
  std::uint32_t n;
  std::uint32_t k;

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<256>};
  }

  ESIMD_INLINE void store_panel(esimd::simd<float, 128> value,
                                std::uint32_t y, std::uint32_t x) const {
    auto output = esimd::convert<T>(value);
#pragma unroll
    for (std::uint32_t row = 0; row < 8; ++row)
      esimd::block_store(
          c + static_cast<std::size_t>(y + row) * n + x,
          output.template select<16, 1>(row * 16));
  }

  // The workgroup collectively covers each A and B panel exactly once. The
  // panels are retained in L1 and subsequently reused by the 32 work-items;
  // unlike an SLM staging scheme, reuse adds no copies or barriers.
  ESIMD_INLINE void cooperative_prefetch(std::uint32_t k0, std::uint32_t lid,
                                         std::uint32_t tile_m,
                                         std::uint32_t tile_n) const {
    if (k0 >= k)
      return;
    constexpr esimd::properties props{
        esimd::cache_hint_L1<esimd::cache_hint::cached>,
        esimd::cache_hint_L2<esimd::cache_hint::cached>};
    const std::uint32_t a_width = k * sizeof(T);
    const std::uint32_t b_width = n * sizeof(T);
    if (lid < (LargeTile ? 24u : 16u))
      esimd::prefetch_2d<T, kPrefetchK, kDpasM>(
          a, a_width - 1, m - 1, a_width - 1, k0,
          tile_m + lid * kDpasM, props);
    if (lid < 16) {
      const std::uint32_t panel = lid;
      esimd::prefetch_2d<T, kDpasN, kPrefetchK>(
          b, b_width - 1, k - 1, b_width - 1,
          tile_n + panel * kDpasN, k0, props);
    }
  }

  SYCL_ESIMD_KERNEL void operator()(sycl::nd_item<1> item) const {
    const std::uint32_t lid = item.get_local_linear_id();
    const std::uint32_t group = item.get_group_linear_id();
    const std::uint32_t groups_n = n / kTileN;
    const std::uint32_t group_m = group / groups_n;
    const std::uint32_t linear_n = group % groups_n;
    const std::uint32_t group_n =
        (group_m & 1) ? groups_n - 1 - linear_n : linear_n;
    const std::uint32_t tile_m = group_m * Kernel::tile_m;
    const std::uint32_t tile_n = group_n * kTileN;

    // Large work-items use the fastest no-spill point found on BMG: 24x64.
    // Smaller matrices use 16x64 with 128 GRFs for higher occupancy.
    const std::uint32_t out_y = tile_m + (lid & 7) * register_rows;
    const std::uint32_t out_x = tile_n + (lid >> 3) * 64;
    const std::uint32_t a_width = k * sizeof(T);
    const std::uint32_t b_width = n * sizeof(T);
    constexpr esimd::properties load_props{
        esimd::cache_hint_L1<esimd::cache_hint::cached>,
        esimd::cache_hint_L2<esimd::cache_hint::cached>};

    esimd::simd<float, 128> c00(0), c01(0), c02(0), c03(0);
    esimd::simd<float, 128> c10(0), c11(0), c12(0), c13(0);
    esimd::simd<float, 128> c20(0), c21(0), c22(0), c23(0);

#pragma unroll
    for (std::uint32_t distance = 0; distance < kPrefetchDistance; ++distance)
      cooperative_prefetch(distance * kPrefetchK, lid, tile_m, tile_n);

#pragma unroll 1
    for (std::uint32_t k0 = 0; k0 < k; k0 += kDpasK) {
      if ((k0 % kPrefetchK) == 0)
        cooperative_prefetch(k0 + kPrefetchDistance * kPrefetchK, lid,
                             tile_m, tile_n);

      auto a0 = esimd::load_2d<T, 16, 8>(a, a_width - 1, m - 1, a_width - 1, k0, out_y, load_props);
      auto a1 = esimd::load_2d<T, 16, 8>(a, a_width - 1, m - 1, a_width - 1, k0, out_y + 8, load_props);
      auto a2 = esimd::load_2d<T, 16, 8>(a, a_width - 1, m - 1, a_width - 1, k0, out_y + 16, load_props);
      auto bv = esimd::load_2d<T, 16, 16, 1, false, true>(b, b_width - 1, k - 1, b_width - 1, out_x, k0, load_props);
      c00 = xmx::dpas<8, 8, float>(c00, bv, a0); c10 = xmx::dpas<8, 8, float>(c10, bv, a1);
      if constexpr (LargeTile) c20 = xmx::dpas<8, 8, float>(c20, bv, a2);
      bv = esimd::load_2d<T, 16, 16, 1, false, true>(b, b_width - 1, k - 1, b_width - 1, out_x + 16, k0, load_props);
      c01 = xmx::dpas<8, 8, float>(c01, bv, a0); c11 = xmx::dpas<8, 8, float>(c11, bv, a1);
      if constexpr (LargeTile) c21 = xmx::dpas<8, 8, float>(c21, bv, a2);
      bv = esimd::load_2d<T, 16, 16, 1, false, true>(b, b_width - 1, k - 1, b_width - 1, out_x + 32, k0, load_props);
      c02 = xmx::dpas<8, 8, float>(c02, bv, a0); c12 = xmx::dpas<8, 8, float>(c12, bv, a1);
      if constexpr (LargeTile) c22 = xmx::dpas<8, 8, float>(c22, bv, a2);
      bv = esimd::load_2d<T, 16, 16, 1, false, true>(b, b_width - 1, k - 1, b_width - 1, out_x + 48, k0, load_props);
      c03 = xmx::dpas<8, 8, float>(c03, bv, a0); c13 = xmx::dpas<8, 8, float>(c13, bv, a1);
      if constexpr (LargeTile) c23 = xmx::dpas<8, 8, float>(c23, bv, a2);
    }

    store_panel(c00, out_y, out_x); store_panel(c01, out_y, out_x + 16); store_panel(c02, out_y, out_x + 32); store_panel(c03, out_y, out_x + 48);
    store_panel(c10, out_y + 8, out_x); store_panel(c11, out_y + 8, out_x + 16); store_panel(c12, out_y + 8, out_x + 32); store_panel(c13, out_y + 8, out_x + 48);
    if constexpr (LargeTile) {
      store_panel(c20, out_y + 16, out_x); store_panel(c21, out_y + 16, out_x + 16); store_panel(c22, out_y + 16, out_x + 32); store_panel(c23, out_y + 16, out_x + 48);
    }
  }
};

} // namespace gemm
