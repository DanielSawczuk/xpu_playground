#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

// The primitive is deliberately runtime-agnostic.  The two hosts only create
// interoperable engines/streams and provide USM pointers.
inline dnnl::matmul make_onednn_matmul(const dnnl::engine &engine,
                                       dnnl::memory::dim n) {
  const auto desc = dnnl::memory::desc({n, n}, dnnl::memory::data_type::f32,
                                       dnnl::memory::format_tag::ab);
  return dnnl::matmul(
      dnnl::matmul::primitive_desc(engine, desc, desc, desc));
}

inline dnnl::matmul make_onednn_matmul(const dnnl::engine &engine,
                                       const dnnl::memory::desc &desc) {
  return dnnl::matmul(
      dnnl::matmul::primitive_desc(engine, desc, desc, desc));
}

inline std::unordered_map<int, dnnl::memory> onednn_matmul_args(
    dnnl::memory &source, dnnl::memory &weights, dnnl::memory &destination) {
  return {{DNNL_ARG_SRC, source},
          {DNNL_ARG_WEIGHTS, weights},
          {DNNL_ARG_DST, destination}};
}

enum class MatmulType { F32, BF16, F16 };

inline MatmulType parse_matmul_type(const char *value) {
  const std::string type(value);
  if (type == "f32") return MatmulType::F32;
  if (type == "bf16") return MatmulType::BF16;
  if (type == "fp16" || type == "f16") return MatmulType::F16;
  throw std::invalid_argument("type must be f32, bf16, or fp16");
}

inline const char *matmul_type_name(MatmulType type) {
  if (type == MatmulType::BF16) return "BF16";
  if (type == MatmulType::F16) return "FP16";
  return "FP32";
}

inline size_t matmul_element_size(MatmulType type) {
  return type == MatmulType::F32 ? sizeof(float) : sizeof(uint16_t);
}

inline dnnl::memory::data_type dnnl_matmul_type(MatmulType type) {
  if (type == MatmulType::BF16) return dnnl::memory::data_type::bf16;
  if (type == MatmulType::F16) return dnnl::memory::data_type::f16;
  return dnnl::memory::data_type::f32;
}

inline dnnl::memory::desc onednn_matmul_memory_desc(
    dnnl::memory::dim n, MatmulType type) {
  return dnnl::memory::desc({n, n}, dnnl_matmul_type(type),
                            dnnl::memory::format_tag::ab);
}

inline float half_to_float(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1f;
  uint32_t mantissa = value & 0x3ff;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) bits = sign;
    else {
      exponent = 113;
      while ((mantissa & 0x400) == 0) { mantissa <<= 1; --exponent; }
      bits = sign | (exponent << 23) | ((mantissa & 0x3ff) << 13);
    }
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline float matmul_value(const void *data, size_t index, MatmulType type) {
  if (type == MatmulType::F32) return static_cast<const float *>(data)[index];
  const uint16_t value = static_cast<const uint16_t *>(data)[index];
  if (type == MatmulType::F16) return half_to_float(value);
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline void initialize_onednn_matmul(void *a, void *b, void *c, size_t count,
                                     MatmulType type) {
  if (type == MatmulType::F32) {
    auto *af = static_cast<float *>(a), *bf = static_cast<float *>(b);
    auto *cf = static_cast<float *>(c);
    for (size_t i = 0; i != count; ++i) af[i] = bf[i] = 1.0f, cf[i] = 0.0f;
  } else {
    const uint16_t one = type == MatmulType::BF16 ? 0x3f80 : 0x3c00;
    auto *a16 = static_cast<uint16_t *>(a), *b16 = static_cast<uint16_t *>(b);
    auto *c16 = static_cast<uint16_t *>(c);
    for (size_t i = 0; i != count; ++i) a16[i] = b16[i] = one, c16[i] = 0;
  }
}

inline void check_onednn_matmul(const void *c, int n, MatmulType type) {
  const float expected = static_cast<float>(n);
  for (int i = 0; i != n * n; ++i)
    if (std::abs(matmul_value(c, i, type) - expected) > expected * 0.01f)
      throw std::runtime_error("oneDNN matmul verification failed");
}
