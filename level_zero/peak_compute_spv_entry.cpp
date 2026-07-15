#include "../peak_compute_dpas_kernel.hpp"

namespace {

class PeakComputeLevelZeroEntry;

template <typename KernelName, typename Function>
__attribute__((sycl_kernel)) void make_kernel(Function function) {
  function();
}

} // namespace

// This function is never called by the Level Zero host. Its device-side call
// instantiates a dispatchable SYCL kernel whose decomposed captures form the
// Level Zero argument ABI.
void instantiate_peak_compute_level_zero(const peak_compute::bf16 *a_surface,
                                         const peak_compute::bf16 *b_surface,
                                         float *sink, unsigned rounds) {
  make_kernel<PeakComputeLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    peak_compute::peak_compute_dpas_kernel(a_surface, b_surface, sink, rounds);
  });
}

int main() { return 0; }
