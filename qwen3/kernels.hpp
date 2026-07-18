#pragma once

#include "../gemv_kernel.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <functional>

namespace qwen3::device {

namespace esimd = sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;

inline constexpr std::uint32_t kHeadDim = 128;
inline constexpr std::uint32_t kAttentionHeads = 32;
inline constexpr std::uint32_t kKeyValueHeads = 8;

struct Embedding {
  const bf16 *table;
  bf16 *output;
  std::uint32_t token;

  SYCL_ESIMD_FUNCTION void run() const {
    const std::uint32_t index =
        sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
    output[index] = table[static_cast<std::uint64_t>(token) * 4096 + index];
  }
};

struct RmsNorm {
  const bf16 *input;
  const bf16 *weight;
  bf16 *output;
  std::uint32_t width;
  float epsilon;

  SYCL_ESIMD_FUNCTION void run() const {
    float sum = 0.0f;
    for (std::uint32_t base = 0; base < width; base += 32) {
      const auto x = esimd::convert<float>(esimd::block_load<bf16, 32>(input + base));
      sum += esimd::reduce<float>(x * x, std::plus<>());
    }
    const float scale = esimd::rsqrt(sum / static_cast<float>(width) + epsilon);
    for (std::uint32_t base = 0; base < width; base += 32) {
      const auto x = esimd::convert<float>(esimd::block_load<bf16, 32>(input + base));
      const auto w = esimd::convert<float>(esimd::block_load<bf16, 32>(weight + base));
      esimd::block_store(output + base, esimd::convert<bf16>(x * w * scale));
    }
  }
};

struct QkvRopeCache {
  bf16 *query;
  bf16 *key;
  const bf16 *value;
  const bf16 *query_norm;
  const bf16 *key_norm;
  const bf16 *rope_cos;
  const bf16 *rope_sin;
  bf16 *key_cache;
  bf16 *value_cache;
  std::uint32_t position;
  std::uint32_t maximum_sequence;

  SYCL_ESIMD_FUNCTION void run() const {
    const std::uint32_t head =
        sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
    if (position >= maximum_sequence)
      return;
    bf16 *q = query + head * kHeadDim;
    float q_sum = 0.0f;
    for (std::uint32_t base = 0; base < kHeadDim; base += 32) {
      const auto values = esimd::convert<float>(esimd::block_load<bf16, 32>(q + base));
      q_sum += esimd::reduce<float>(values * values, std::plus<>());
    }
    const float q_scale = esimd::rsqrt(q_sum / kHeadDim + 1.0e-6f);
    for (std::uint32_t base = 0; base < kHeadDim; base += 32) {
      const auto values = esimd::convert<float>(esimd::block_load<bf16, 32>(q + base));
      const auto weights = esimd::convert<float>(
          esimd::block_load<bf16, 32>(query_norm + base));
      esimd::block_store(q + base, esimd::convert<bf16>(values * weights * q_scale));
    }

    if (head < kKeyValueHeads) {
      bf16 *k = key + head * kHeadDim;
      float k_sum = 0.0f;
      for (std::uint32_t base = 0; base < kHeadDim; base += 32) {
        const auto values = esimd::convert<float>(esimd::block_load<bf16, 32>(k + base));
        k_sum += esimd::reduce<float>(values * values, std::plus<>());
      }
      const float k_scale = esimd::rsqrt(k_sum / kHeadDim + 1.0e-6f);
      for (std::uint32_t base = 0; base < kHeadDim; base += 32) {
        const auto values = esimd::convert<float>(esimd::block_load<bf16, 32>(k + base));
        const auto weights = esimd::convert<float>(
            esimd::block_load<bf16, 32>(key_norm + base));
        esimd::block_store(k + base, esimd::convert<bf16>(values * weights * k_scale));
      }
    }

    // Qwen3's rotate_half pairs the first and second 64-value halves.
    for (std::uint32_t base = 0; base < kHeadDim / 2; base += 32) {
      const auto a = esimd::convert<float>(esimd::block_load<bf16, 32>(q + base));
      const auto b = esimd::convert<float>(
          esimd::block_load<bf16, 32>(q + kHeadDim / 2 + base));
      const auto ca = esimd::convert<float>(esimd::block_load<bf16, 32>(
          rope_cos + static_cast<std::uint64_t>(position) * kHeadDim + base));
      const auto sa = esimd::convert<float>(esimd::block_load<bf16, 32>(
          rope_sin + static_cast<std::uint64_t>(position) * kHeadDim + base));
      const auto cb = esimd::convert<float>(esimd::block_load<bf16, 32>(
          rope_cos + static_cast<std::uint64_t>(position) * kHeadDim +
          kHeadDim / 2 + base));
      const auto sb = esimd::convert<float>(esimd::block_load<bf16, 32>(
          rope_sin + static_cast<std::uint64_t>(position) * kHeadDim +
          kHeadDim / 2 + base));
      esimd::block_store(q + base, esimd::convert<bf16>(a * ca - b * sa));
      esimd::block_store(q + kHeadDim / 2 + base,
                         esimd::convert<bf16>(b * cb + a * sb));
    }

    if (head < kKeyValueHeads) {
      bf16 *k = key + head * kHeadDim;
      bf16 *cache_k = key_cache +
          (static_cast<std::uint64_t>(position) * kKeyValueHeads + head) * kHeadDim;
      bf16 *cache_v = value_cache +
          (static_cast<std::uint64_t>(position) * kKeyValueHeads + head) * kHeadDim;
      for (std::uint32_t base = 0; base < kHeadDim / 2; base += 32) {
        const auto a = esimd::convert<float>(esimd::block_load<bf16, 32>(k + base));
        const auto b = esimd::convert<float>(
            esimd::block_load<bf16, 32>(k + kHeadDim / 2 + base));
        const auto ca = esimd::convert<float>(esimd::block_load<bf16, 32>(
            rope_cos + static_cast<std::uint64_t>(position) * kHeadDim + base));
        const auto sa = esimd::convert<float>(esimd::block_load<bf16, 32>(
            rope_sin + static_cast<std::uint64_t>(position) * kHeadDim + base));
        const auto cb = esimd::convert<float>(esimd::block_load<bf16, 32>(
            rope_cos + static_cast<std::uint64_t>(position) * kHeadDim +
            kHeadDim / 2 + base));
        const auto sb = esimd::convert<float>(esimd::block_load<bf16, 32>(
            rope_sin + static_cast<std::uint64_t>(position) * kHeadDim +
            kHeadDim / 2 + base));
        esimd::block_store(cache_k + base, esimd::convert<bf16>(a * ca - b * sa));
        esimd::block_store(cache_k + kHeadDim / 2 + base,
                           esimd::convert<bf16>(b * cb + a * sb));
      }
      for (std::uint32_t base = 0; base < kHeadDim; base += 32)
        esimd::block_store(cache_v + base,
            esimd::block_load<bf16, 32>(value + head * kHeadDim + base));
    }
  }
};

struct Attention {
  const bf16 *query;
  const bf16 *key_cache;
  const bf16 *value_cache;
  bf16 *output;
  std::uint32_t position;

  SYCL_ESIMD_FUNCTION void run() const {
    const std::uint32_t head =
        sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id();
    const std::uint32_t kv_head = head / (kAttentionHeads / kKeyValueHeads);
    const auto q = esimd::convert<float>(
        esimd::block_load<bf16, kHeadDim>(query + head * kHeadDim));
    esimd::simd<float, kHeadDim> weighted(0.0f);
    float maximum = -INFINITY;
    float denominator = 0.0f;
    constexpr float scale = 0.08838834764831845f; // 1/sqrt(128)
    for (std::uint32_t token = 0; token <= position; ++token) {
      const std::uint64_t cache_offset =
          (static_cast<std::uint64_t>(token) * kKeyValueHeads + kv_head) * kHeadDim;
      const auto k = esimd::convert<float>(
          esimd::block_load<bf16, kHeadDim>(key_cache + cache_offset));
      const float score = esimd::reduce<float>(q * k, std::plus<>()) * scale;
      const float new_maximum = maximum > score ? maximum : score;
      const float old_scale = esimd::exp(maximum - new_maximum);
      const float new_scale = esimd::exp(score - new_maximum);
      const auto v = esimd::convert<float>(
          esimd::block_load<bf16, kHeadDim>(value_cache + cache_offset));
      weighted = weighted * old_scale + v * new_scale;
      denominator = denominator * old_scale + new_scale;
      maximum = new_maximum;
    }
    esimd::block_store(output + head * kHeadDim,
                       esimd::convert<bf16>(weighted / denominator));
  }
};

struct Add {
  bf16 *left;
  const bf16 *right;
  std::uint32_t width;
  SYCL_ESIMD_FUNCTION void run() const {
    const std::uint32_t base =
        sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id() * 32;
    if (base < width) {
      const auto a = esimd::convert<float>(esimd::block_load<bf16, 32>(left + base));
      const auto b = esimd::convert<float>(esimd::block_load<bf16, 32>(right + base));
      esimd::block_store(left + base, esimd::convert<bf16>(a + b));
    }
  }
};

struct SiluMultiply {
  bf16 *gate;
  const bf16 *up;
  std::uint32_t width;
  SYCL_ESIMD_FUNCTION void run() const {
    const std::uint32_t base =
        sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_linear_id() * 32;
    if (base < width) {
      const auto g = esimd::convert<float>(esimd::block_load<bf16, 32>(gate + base));
      const auto u = esimd::convert<float>(esimd::block_load<bf16, 32>(up + base));
      const auto activated = g / (1.0f + esimd::exp(-g));
      esimd::block_store(gate + base, esimd::convert<bf16>(activated * u));
    }
  }
};

struct Argmax {
  const bf16 *values;
  std::uint32_t *token;
  std::uint32_t width;
  SYCL_ESIMD_FUNCTION void run() const {
    float best = -INFINITY;
    std::uint32_t best_index = 0;
    for (std::uint32_t index = 0; index < width; ++index) {
      const float value = static_cast<float>(values[index]);
      if (value != value || value == INFINITY || value == -INFINITY) {
        *token = 0xffffffffU;
        return;
      }
      // Strict comparison intentionally selects the lowest ID on ties.
      if (value > best) {
        best = value;
        best_index = index;
      }
    }
    *token = best_index;
  }
};

} // namespace qwen3::device
