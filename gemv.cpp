#include "gemv_kernel.hpp"

#include <sycl/sycl.hpp>

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
  std::uint32_t m = 16384, k = 16384;
  unsigned iterations = 50, warmups = 5;
  bool fp16 = false;
};

[[noreturn]] void usage(const char *program, int status) {
  std::ostream &out = status ? std::cerr : std::cout;
  out << "Usage: " << program
      << " [--size N | --m M --k K] [--type bf16|fp16]\n"
      << "       [--iterations N] [--warmups N]\n";
  std::exit(status);
}

template <typename T> T positive(std::string_view text, std::string_view opt) {
  T value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      !value)
    throw std::invalid_argument(std::string(opt) +
                                " requires a positive integer");
  return value;
}

Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "-h" || arg == "--help") usage(argv[0], 0);
    if (++i == argc)
      throw std::invalid_argument(std::string(arg) + " requires a value");
    const std::string_view value(argv[i]);
    if (arg == "--size") o.m = o.k = positive<std::uint32_t>(value, arg);
    else if (arg == "--m") o.m = positive<std::uint32_t>(value, arg);
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

std::uint32_t round_up(std::uint32_t value, std::uint32_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

double seconds(const sycl::event &event) {
  const auto start = event.get_profiling_info<
      sycl::info::event_profiling::command_start>();
  const auto end = event.get_profiling_info<
      sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1e-9;
}

template <typename T> int run(const Options &o, sycl::queue &queue) {
  const std::uint32_t m = round_up(o.m, gemv::kRows * gemv::kLocalSize);
  const std::uint32_t k = round_up(o.k, gemv::kColumns);
  const std::size_t matrix_elements = static_cast<std::size_t>(m) * k;
  T *matrix = sycl::malloc_device<T>(matrix_elements, queue);
  T *vector = sycl::malloc_device<T>(k, queue);
  T *output = sycl::malloc_device<T>(m, queue);
  if (!matrix || !vector || !output) throw std::bad_alloc();
  auto release = [&] {
    sycl::free(matrix, queue); sycl::free(vector, queue);
    sycl::free(output, queue);
  };
  try {
    queue.parallel_for(sycl::range<1>(matrix_elements), [=](sycl::id<1> id) {
      std::uint32_t bits = static_cast<std::uint32_t>(id[0]);
      bits ^= bits >> 16;
      bits *= 0x7feb352du;
      bits ^= bits >> 15;
      bits *= 0x846ca68bu;
      bits ^= bits >> 16;
      matrix[id] = T(0.25f + static_cast<float>(bits & 255u) / 1024.0f);
    });
    queue.parallel_for(sycl::range<1>(k), [=](sycl::id<1> id) {
      const std::uint32_t bits = static_cast<std::uint32_t>(id[0]) * 747796405u;
      vector[id] = T(0.5f + static_cast<float>((bits >> 16) & 127u) / 512.0f);
    }).wait_and_throw();

    const sycl::nd_range<1> range(m / gemv::kRows, gemv::kLocalSize);
    gemv::Kernel<T> kernel{matrix, vector, output, m, k};
    auto launch = [&] { return queue.parallel_for(range, kernel); };
    for (unsigned i = 0; i < o.warmups; ++i) launch().wait_and_throw();
    const long double operations = 2.0L * o.m * o.k;
    const long double matrix_bytes = 2.0L * o.m * o.k;
    std::vector<double> rates, bandwidths;
    for (unsigned i = 0; i < o.iterations; ++i) {
      auto event = launch();
      event.wait_and_throw();
      const double elapsed = seconds(event);
      rates.push_back(static_cast<double>(operations) / elapsed / 1e9);
      bandwidths.push_back(static_cast<double>(matrix_bytes) / elapsed / 1e9);
    }
    std::vector<T> check(std::min<std::uint32_t>(o.m, 64));
    queue.memcpy(check.data(), output, check.size() * sizeof(T)).wait_and_throw();
    std::vector<T> reference_matrix(static_cast<std::size_t>(check.size()) * k);
    std::vector<T> reference_vector(k);
    queue.memcpy(reference_matrix.data(), matrix,
                 reference_matrix.size() * sizeof(T));
    queue.memcpy(reference_vector.data(), vector, k * sizeof(T)).wait_and_throw();
    for (std::size_t row = 0; row < check.size(); ++row) {
      float expected = 0.0f;
      for (std::uint32_t column = 0; column < o.k; ++column)
        expected += static_cast<float>(reference_matrix[row * k + column]) *
                    static_cast<float>(reference_vector[column]);
      const float tolerance = std::max(1.0f, std::abs(expected) * 0.01f);
      if (std::abs(static_cast<float>(check[row]) - expected) > tolerance)
        throw std::runtime_error("GEMV verification failed");
    }
    std::sort(rates.begin(), rates.end());
    std::sort(bandwidths.begin(), bandwidths.end());
    const std::size_t mid = rates.size() / 2;
    const double median_rate = rates.size() & 1 ? rates[mid] : (rates[mid - 1] + rates[mid]) * 0.5;
    const double median_bw = bandwidths.size() & 1 ? bandwidths[mid] : (bandwidths[mid - 1] + bandwidths[mid]) * 0.5;
    std::cout << "Type: " << (std::is_same_v<T, sycl::half> ? "FP16" : "BF16")
              << ", shape: " << o.m << 'x' << o.k;
    if (m != o.m || k != o.k) std::cout << " (padded to " << m << 'x' << k << ')';
    std::cout << '\n' << std::fixed << std::setprecision(2)
              << "Median: " << median_rate << " GFLOP/s, " << median_bw << " GB/s\n"
              << "Best:   " << rates.back() << " GFLOP/s, " << bandwidths.back() << " GB/s\n"
              << "Verification: PASS\n";
  } catch (...) {
    release();
    throw;
  }
  release();
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse(argc, argv);
    sycl::queue queue(sycl::gpu_selector_v,
                      sycl::property_list{sycl::property::queue::enable_profiling{}});
    std::cout << "Device: "
              << queue.get_device().get_info<sycl::info::device::name>() << '\n';
    return options.fp16 ? run<sycl::half>(options, queue)
                        : run<gemv::bf16>(options, queue);
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
