#include "kernels.hpp"

template <typename KernelName, typename Function>
__attribute__((sycl_kernel)) void make_kernel(Function function) { function(); }

using qwen3::device::bf16;

class QwenEmbeddingLevelZeroEntry;
void qwen_embedding(const bf16 *table, bf16 *output, unsigned token) {
  make_kernel<QwenEmbeddingLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    qwen3::device::Embedding{table, output, token}.run();
  });
}

class QwenRmsNormLevelZeroEntry;
void qwen_rms_norm(const bf16 *input, const bf16 *weight, bf16 *output,
                   unsigned width, float epsilon) {
  make_kernel<QwenRmsNormLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    qwen3::device::RmsNorm{input, weight, output, width, epsilon}.run();
  });
}

class QwenGemvLevelZeroEntry;
void qwen_gemv(const bf16 *matrix, const bf16 *vector, bf16 *output,
               unsigned rows, unsigned columns) {
  make_kernel<QwenGemvLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    gemv::Kernel<gemv::bf16>{matrix, vector, output, rows, columns}.run();
  });
}

class QwenQkvRopeCacheLevelZeroEntry;
void qwen_qkv_rope_cache(bf16 *query, bf16 *key, const bf16 *value,
                         const bf16 *query_norm, const bf16 *key_norm,
                         const bf16 *rope_cos, const bf16 *rope_sin,
                         bf16 *key_cache, bf16 *value_cache,
                         unsigned position, unsigned maximum_sequence) {
  make_kernel<QwenQkvRopeCacheLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    qwen3::device::QkvRopeCache{query, key, value, query_norm, key_norm,
        rope_cos, rope_sin, key_cache, value_cache, position,
        maximum_sequence}.run();
  });
}

class QwenAttentionLevelZeroEntry;
void qwen_attention(const bf16 *query, const bf16 *key_cache,
                    const bf16 *value_cache, bf16 *output,
                    unsigned position) {
  make_kernel<QwenAttentionLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    qwen3::device::Attention{query, key_cache, value_cache, output, position}.run();
  });
}

class QwenAddLevelZeroEntry;
void qwen_add(bf16 *left, const bf16 *right, unsigned width) {
  make_kernel<QwenAddLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    qwen3::device::Add{left, right, width}.run();
  });
}

class QwenSiluMultiplyLevelZeroEntry;
void qwen_silu_multiply(bf16 *gate, const bf16 *up, unsigned width) {
  make_kernel<QwenSiluMultiplyLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    qwen3::device::SiluMultiply{gate, up, width}.run();
  });
}

class QwenArgmaxLevelZeroEntry;
void qwen_argmax(const bf16 *values, unsigned *token, unsigned width) {
  make_kernel<QwenArgmaxLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    qwen3::device::Argmax{values, token, width}.run();
  });
}

int main() { return 0; }
