#include "../peak_mem_2d_kernel.hpp"

namespace {

class PeakMemLevelZeroEntry;

template <typename KernelName, typename Function>
__attribute__((sycl_kernel)) void make_kernel(Function function) {
  function();
}

} // namespace

// This function is never called by the Level Zero host. Its device-side call
// instantiates a dispatchable SYCL kernel whose decomposed captures form the
// Level Zero argument ABI.
void instantiate_peak_mem_level_zero(const float *input, float *sink,
                                     unsigned surface_height) {
  make_kernel<PeakMemLevelZeroEntry>([=]() SYCL_ESIMD_KERNEL {
    peak_memory::peak_mem_2d_kernel(input, sink, surface_height);
  });
}

int main() { return 0; }
