#pragma once

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>

namespace gemv {

namespace esimd = sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr std::uint32_t kRows = 1;
inline constexpr std::uint32_t kColumns = 32;
inline constexpr std::uint32_t kLocalSize = 32;

template <typename T> struct Kernel {
  const T *matrix;
  const T *vector;
  T *output;
  std::uint32_t m;
  std::uint32_t k;

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<128>};
  }

  SYCL_ESIMD_FUNCTION void run() const {
    const auto item = sycl::ext::oneapi::this_work_item::get_nd_item<1>();
    const std::uint32_t row0 = item.get_global_linear_id() * kRows;
    const std::uint32_t width = k * sizeof(T);
    constexpr esimd::properties matrix_props{
        esimd::cache_hint_L1<esimd::cache_hint::uncached>,
        esimd::cache_hint_L2<esimd::cache_hint::cached>};
    constexpr esimd::properties vector_props{
        esimd::cache_hint_L1<esimd::cache_hint::cached>,
        esimd::cache_hint_L2<esimd::cache_hint::cached>};

    esimd::simd<float, kColumns> sums0(0);

#pragma unroll 4
    for (std::uint32_t column = 0; column < k; column += kColumns) {
      const auto x = esimd::convert<float>(
          esimd::block_load<T, kColumns>(vector + column, vector_props));
      auto block = esimd::convert<float>(
          esimd::load_2d<T, kColumns, kRows>(
              matrix, width - 1, m - 1, width - 1, column, row0,
              matrix_props));
      sums0 += block.template select<kColumns, 1>(0 * kColumns) * x;
    }

    esimd::simd<float, kRows> result;
    result[0] = esimd::reduce<float>(sums0, std::plus<>());
    esimd::block_store(output + row0, esimd::convert<T>(result));
  }

  SYCL_ESIMD_KERNEL void operator()(sycl::nd_item<1>) const { run(); }
};

} // namespace gemv
