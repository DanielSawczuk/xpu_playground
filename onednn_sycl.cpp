#include "onednn_matmul.hpp"

#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/sycl.hpp>

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) try {
  const int n = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
  const MatmulType type = argc > 3 ? parse_matmul_type(argv[3])
                                   : MatmulType::F32;
  if (n <= 0 || iterations <= 0)
    throw std::invalid_argument(
        "usage: onednn_sycl [size] [iterations] [f32|bf16|fp16]");
  sycl::queue queue(sycl::gpu_selector_v,
                    sycl::property::queue::in_order{});
  auto engine = dnnl::sycl_interop::make_engine(
      queue.get_device(), queue.get_context());
  auto stream = dnnl::sycl_interop::make_stream(engine, queue);

  const size_t count = static_cast<size_t>(n) * n;
  const size_t bytes = count * matmul_element_size(type);
  void *a = sycl::malloc_device(bytes, queue);
  void *b = sycl::malloc_device(bytes, queue);
  void *c = sycl::malloc_device(bytes, queue);
  if (!a || !b || !c) throw std::bad_alloc();
  std::vector<uint8_t> host_a(bytes), host_b(bytes), host_c(bytes);
  initialize_onednn_matmul(
      host_a.data(), host_b.data(), host_c.data(), count, type);
  queue.memcpy(a, host_a.data(), bytes);
  queue.memcpy(b, host_b.data(), bytes);
  queue.memcpy(c, host_c.data(), bytes).wait();

  const auto desc = onednn_matmul_memory_desc(n, type);
  auto source = dnnl::memory(desc, engine, a);
  auto weights = dnnl::memory(desc, engine, b);
  auto destination = dnnl::memory(desc, engine, c);
  auto primitive = make_onednn_matmul(engine, desc);
  const auto args = onednn_matmul_args(source, weights, destination);
  for (int i = 0; i != 10; ++i)
    dnnl::sycl_interop::execute(primitive, stream, args);
  stream.wait();
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i != iterations; ++i)
    dnnl::sycl_interop::execute(primitive, stream, args);
  stream.wait();
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  queue.memcpy(host_c.data(), c, bytes).wait();
  check_onednn_matmul(host_c.data(), n, type);
  std::cout << "oneDNN matmul passed on "
            << queue.get_device().get_info<sycl::info::device::name>()
            << " (SYCL interop)\n"
            << "Type: " << matmul_type_name(type) << ", size: " << n << "x"
            << n << ", iterations: " << iterations
            << ", average: " << seconds * 1e3 / iterations << " ms, "
            << 2.0 * n * n * n * iterations / seconds / 1e12 << " TFLOP/s\n";
  sycl::free(a, queue);
  sycl::free(b, queue);
  sycl::free(c, queue);
  return 0;
} catch (const std::exception &error) {
  std::cerr << "Error: " << error.what() << '\n';
  return 1;
}
