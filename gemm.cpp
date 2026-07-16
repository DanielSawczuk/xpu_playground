#include "gemm_kernel.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

struct Options {
  std::uint32_t m = 4096, n = 4096, k = 4096;
  unsigned iterations = 20, warmups = 5;
  bool fp16 = false;
};

[[noreturn]] void usage(const char *program, int status) {
  std::ostream &out = status ? std::cerr : std::cout;
  out << "Usage: " << program
      << " [--size N | --m M --n N --k K] [--type bf16|fp16]\n"
      << "       [--iterations N] [--warmups N]\n";
  std::exit(status);
}

template <typename T> T positive(std::string_view text, std::string_view opt) {
  T value{};
  auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !value)
    throw std::invalid_argument(std::string(opt) + " requires a positive integer");
  return value;
}

Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);
    if (arg == "-h" || arg == "--help") usage(argv[0], 0);
    if (++i == argc) throw std::invalid_argument(std::string(arg) + " requires a value");
    std::string_view value(argv[i]);
    if (arg == "--size") o.m = o.n = o.k = positive<std::uint32_t>(value, arg);
    else if (arg == "--m") o.m = positive<std::uint32_t>(value, arg);
    else if (arg == "--n") o.n = positive<std::uint32_t>(value, arg);
    else if (arg == "--k") o.k = positive<std::uint32_t>(value, arg);
    else if (arg == "--iterations") o.iterations = positive<unsigned>(value, arg);
    else if (arg == "--warmups") o.warmups = positive<unsigned>(value, arg);
    else if (arg == "--type" && value == "fp16") o.fp16 = true;
    else if (arg == "--type" && value == "bf16") o.fp16 = false;
    else if (arg == "--type") throw std::invalid_argument("--type must be bf16 or fp16");
    else throw std::invalid_argument("unknown option: " + std::string(arg));
  }
  return o;
}

std::uint32_t round_up(std::uint32_t x, std::uint32_t n) {
  const std::uint64_t result = (static_cast<std::uint64_t>(x) + n - 1) / n * n;
  if (result > UINT32_MAX) throw std::overflow_error("matrix dimension is too large");
  return result;
}

double seconds(const sycl::event &event) {
  auto begin = event.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - begin) * 1e-9;
}

template <typename T, bool LargeTile> int run(const Options &o, sycl::queue &queue) {
  using Kernel = gemm::Kernel<T, LargeTile>;
  const std::uint32_t m = round_up(o.m, Kernel::tile_m);
  const std::uint32_t n = round_up(o.n, gemm::kTileN);
  const std::uint32_t k = round_up(o.k, gemm::kPrefetchK);
  const std::size_t as = static_cast<std::size_t>(m) * k;
  const std::size_t bs = static_cast<std::size_t>(k) * n;
  const std::size_t cs = static_cast<std::size_t>(m) * n;
  std::vector<T> ha(as, T(0)), hb(bs, T(0));
  for (std::uint32_t row = 0; row < o.m; ++row)
    std::fill_n(ha.data() + static_cast<std::size_t>(row) * k, o.k, T(0.5f));
  for (std::uint32_t row = 0; row < o.k; ++row)
    std::fill_n(hb.data() + static_cast<std::size_t>(row) * n, o.n, T(0.25f));

  T *a = sycl::aligned_alloc_device<T>(64, as, queue);
  T *b = sycl::aligned_alloc_device<T>(64, bs, queue);
  T *c = sycl::aligned_alloc_device<T>(64, cs, queue);
  if (!a || !b || !c) throw std::bad_alloc();
  auto release = [&] { sycl::free(a, queue); sycl::free(b, queue); sycl::free(c, queue); };
  try {
    queue.memcpy(a, ha.data(), as * sizeof(T)).wait_and_throw();
    queue.memcpy(b, hb.data(), bs * sizeof(T)).wait_and_throw();
    queue.memset(c, 0, cs * sizeof(T)).wait_and_throw();
    std::vector<T>().swap(ha); std::vector<T>().swap(hb);

    const std::size_t groups = (m / Kernel::tile_m) * (n / gemm::kTileN);
    const sycl::nd_range<1> range(groups * gemm::kLocalSize, gemm::kLocalSize);
    Kernel kernel{a, b, c, m, n, k};
    auto launch = [&] { return queue.parallel_for(range, kernel); };
    for (unsigned i = 0; i < o.warmups; ++i) launch().wait_and_throw();
    const long double flops = 2.0L * o.m * o.n * o.k;
    std::vector<double> rates;
    for (unsigned i = 0; i < o.iterations; ++i) {
      auto event = launch(); event.wait_and_throw();
      rates.push_back(static_cast<double>(flops) / seconds(event) / 1e12);
    }
    std::vector<T> check(std::min<std::uint32_t>(o.n, 64));
    queue.memcpy(check.data(), c, check.size() * sizeof(T)).wait_and_throw();
    const float expected = o.k * 0.125f;
    const float tolerance = std::max(0.5f, expected * 0.01f);
    for (T value : check)
      if (std::abs(static_cast<float>(value) - expected) > tolerance)
        throw std::runtime_error("GEMM verification failed: expected " +
                                 std::to_string(expected) + ", got " +
                                 std::to_string(static_cast<float>(value)));
    std::sort(rates.begin(), rates.end());
    const auto mid = rates.size() / 2;
    const double median = rates.size() & 1 ? rates[mid] : (rates[mid - 1] + rates[mid]) / 2;
    std::cout << "Type: " << (std::is_same_v<T, sycl::half> ? "FP16" : "BF16")
              << ", shape: " << o.m << 'x' << o.n << 'x' << o.k;
    if (m != o.m || n != o.n || k != o.k)
      std::cout << " (padded to " << m << 'x' << n << 'x' << k << ')';
    std::cout << '\n' << std::fixed << std::setprecision(2)
              << "Median: " << median << " TFLOP/s\nBest:   " << rates.back()
              << " TFLOP/s\nVerification: PASS\n";
  } catch (...) { release(); throw; }
  release();
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    Options options = parse(argc, argv);
    sycl::queue queue(sycl::gpu_selector_v,
                      sycl::property_list{sycl::property::queue::enable_profiling{}});
    auto device = queue.get_device();
    if (!device.has(sycl::aspect::ext_intel_esimd) ||
        !device.has(sycl::aspect::ext_intel_matrix))
      throw std::runtime_error("selected GPU does not support eSIMD XMX");
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << '\n';
    const bool large_tile = options.m >= 2560 && options.n >= 2560;
    if (options.fp16)
      return large_tile ? run<sycl::half, true>(options, queue)
                        : run<sycl::half, false>(options, queue);
    return large_tile ? run<gemm::bf16, true>(options, queue)
                      : run<gemm::bf16, false>(options, queue);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
