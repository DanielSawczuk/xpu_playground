#include "../gemm_kernel.hpp"
#include "../gemv_kernel.hpp"

template <typename KernelName, typename Function>
__attribute__((sycl_kernel)) void make_kernel(Function function) {
  function();
}

#if defined(GEMV_BF16)
class GemvBf16LevelZeroEntry;
void instantiate_gemv_bf16(const gemv::bf16 *matrix, const gemv::bf16 *vector,
                           gemv::bf16 *output, unsigned m, unsigned k) {
  make_kernel<GemvBf16LevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    gemv::Kernel<gemv::bf16>{matrix, vector, output, m, k}.run();
  });
}
#elif defined(GEMV_FP16)
class GemvFp16LevelZeroEntry;
void instantiate_gemv_fp16(const sycl::half *matrix, const sycl::half *vector,
                           sycl::half *output, unsigned m, unsigned k) {
  make_kernel<GemvFp16LevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    gemv::Kernel<sycl::half>{matrix, vector, output, m, k}.run();
  });
}
#elif defined(GEMM_BF16_SMALL)
class GemmBf16SmallLevelZeroEntry;
void instantiate_gemm_bf16_small(const gemm::bf16 *a, const gemm::bf16 *b,
                                 gemm::bf16 *c, unsigned m, unsigned n,
                                 unsigned k) {
  make_kernel<GemmBf16SmallLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    gemm::Kernel<gemm::bf16, false>{a, b, c, m, n, k}.run();
  });
}
#elif defined(GEMM_BF16_LARGE)
class GemmBf16LargeLevelZeroEntry;
void instantiate_gemm_bf16_large(const gemm::bf16 *a, const gemm::bf16 *b,
                                 gemm::bf16 *c, unsigned m, unsigned n,
                                 unsigned k) {
  make_kernel<GemmBf16LargeLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    gemm::Kernel<gemm::bf16, true>{a, b, c, m, n, k}.run();
  });
}
#elif defined(GEMM_FP16_SMALL)
class GemmFp16SmallLevelZeroEntry;
void instantiate_gemm_fp16_small(const sycl::half *a, const sycl::half *b,
                                 sycl::half *c, unsigned m, unsigned n,
                                 unsigned k) {
  make_kernel<GemmFp16SmallLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    gemm::Kernel<sycl::half, false>{a, b, c, m, n, k}.run();
  });
}
#elif defined(GEMM_FP16_LARGE)
class GemmFp16LargeLevelZeroEntry;
void instantiate_gemm_fp16_large(const sycl::half *a, const sycl::half *b,
                                 sycl::half *c, unsigned m, unsigned n,
                                 unsigned k) {
  make_kernel<GemmFp16LargeLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    gemm::Kernel<sycl::half, true>{a, b, c, m, n, k}.run();
  });
}
#else
#error "select one GEM entry variant"
#endif

int main() { return 0; }
