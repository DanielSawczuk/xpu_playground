#include "peak_compute_dpas_kernel.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace {

using namespace peak_compute;

constexpr double kB70PeakTFlops = 183.0;

struct Options {
  std::size_t work_items = 4096;
  unsigned rounds = 8192;
  unsigned iterations = 1000;
};

[[noreturn]] void usage(const char *program, int status) {
  std::ostream &out = status == 0 ? std::cout : std::cerr;
  out << "Usage: " << program
      << " [--work-items N] [--rounds N] [--iterations N]\n"
      << "  --work-items N  eSIMD work-items, rounded to 16 (default: 4096)\n"
      << "  --rounds N      eight-DPAS batches per work-item (default: 8192)\n"
      << "  --iterations N  timed kernel launches (default: 1000)\n";
  std::exit(status);
}

template <typename T>
T parse_positive(std::string_view text, std::string_view option) {
  T value{};
  const char *begin = text.data();
  const char *end = begin + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr != end || value == 0)
    throw std::invalid_argument(std::string(option) +
                                " requires a positive integer");
  return value;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--help" || arg == "-h")
      usage(argv[0], 0);
    if (i + 1 == argc)
      throw std::invalid_argument(std::string(arg) + " requires a value");
    if (arg == "--work-items")
      options.work_items =
          parse_positive<std::size_t>(argv[++i], "--work-items");
    else if (arg == "--rounds")
      options.rounds = parse_positive<unsigned>(argv[++i], "--rounds");
    else if (arg == "--iterations")
      options.iterations =
          parse_positive<unsigned>(argv[++i], "--iterations");
    else
      throw std::invalid_argument("unknown option: " + std::string(arg));
  }
  options.work_items =
      (options.work_items + kLocalSize - 1) / kLocalSize * kLocalSize;
  return options;
}

sycl::event launch_peak_compute(sycl::queue &queue, const bf16 *a_surface,
                                const bf16 *b_surface, float *sink,
                                std::size_t work_items, unsigned rounds) {
  return queue.submit([&](sycl::handler &handler) {
    syclexp::nd_launch(
        handler,
        sycl::nd_range<1>{sycl::range<1>{work_items},
                          sycl::range<1>{kLocalSize}},
        syclexp::kernel_function<peak_compute_dpas_kernel>, a_surface,
        b_surface, sink, rounds);
  });
}

double event_seconds(const sycl::event &event) {
  const auto start = event.get_profiling_info<
      sycl::info::event_profiling::command_start>();
  const auto end = event.get_profiling_info<
      sycl::info::event_profiling::command_end>();
  return static_cast<double>(end - start) * 1.0e-9;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    auto async_handler = [](sycl::exception_list exceptions) {
      for (const std::exception_ptr &exception : exceptions)
        std::rethrow_exception(exception);
    };
    sycl::queue queue(sycl::gpu_selector_v, async_handler,
                      sycl::property_list{
                          sycl::property::queue::enable_profiling{}});
    const sycl::device device = queue.get_device();

    std::cout << "Device: "
              << device.get_info<sycl::info::device::name>() << '\n'
              << "Driver: "
              << device.get_info<sycl::info::device::driver_version>() << '\n'
              << "Compute units: "
              << device.get_info<sycl::info::device::max_compute_units>()
              << '\n';

    if (!device.has(sycl::aspect::ext_intel_esimd))
      throw std::runtime_error("selected GPU does not support eSIMD");
    if (!device.has(sycl::aspect::ext_intel_matrix))
      throw std::runtime_error("selected GPU does not support XMX/DPAS");
    if (!device.get_info<
            esimd::info::device::has_2d_block_io_support>())
      throw std::runtime_error(
          "selected GPU does not report block-2D I/O support");

    const std::size_t surface_elements = kSurfaceWidth * kSurfaceHeight;
    bf16 *a_surface = sycl::aligned_alloc_device<bf16>(
        64, surface_elements, queue);
    bf16 *b_surface = sycl::aligned_alloc_device<bf16>(
        64, surface_elements, queue);
    float *sink = sycl::malloc_device<float>(options.work_items, queue);
    if (!a_surface || !b_surface || !sink) {
      if (a_surface)
        sycl::free(a_surface, queue);
      if (b_surface)
        sycl::free(b_surface, queue);
      if (sink)
        sycl::free(sink, queue);
      throw std::bad_alloc();
    }

    queue.fill(a_surface, bf16(0.5f), surface_elements).wait_and_throw();
    queue.fill(b_surface, bf16(0.5f), surface_elements).wait_and_throw();
    queue.fill(sink, 0.0f, options.work_items).wait_and_throw();

    const long double total_flops =
        static_cast<long double>(options.work_items) * options.rounds *
        kFlopsPerRound;
    std::cout << "DPAS shape: " << kM << 'x' << kN << 'x' << kK << " BF16, "
              << kFlopsPerDpas << " FLOP/instruction\n"
              << "Launch: " << options.work_items << " work-items x "
              << options.rounds << " rounds x " << kAccumulators
              << " DPAS = " << std::fixed << std::setprecision(3)
              << static_cast<double>(total_flops / 1.0e12L)
              << " TFLOP/launch\n";

    constexpr unsigned warmups = 3;
    for (unsigned i = 0; i < warmups; ++i)
      launch_peak_compute(queue, a_surface, b_surface, sink,
                          options.work_items, options.rounds)
          .wait_and_throw();

    std::vector<double> tflops;
    tflops.reserve(options.iterations);
    for (unsigned i = 0; i < options.iterations; ++i) {
      sycl::event event = launch_peak_compute(
          queue, a_surface, b_surface, sink, options.work_items,
          options.rounds);
      event.wait_and_throw();
      tflops.push_back(static_cast<double>(total_flops) /
                       event_seconds(event) / 1.0e12);
    }

    float first = 0.0f;
    float last = 0.0f;
    queue.memcpy(&first, sink, sizeof(float)).wait_and_throw();
    queue.memcpy(&last, sink + options.work_items - 1, sizeof(float))
        .wait_and_throw();
    const float expected =
        28.0f + static_cast<float>(kAccumulators) * options.rounds * 4.0f;
    const float tolerance = std::max(1.0f, expected * 1.0e-5f);
    if (std::abs(first - expected) > tolerance ||
        std::abs(last - expected) > tolerance) {
      sycl::free(a_surface, queue);
      sycl::free(b_surface, queue);
      sycl::free(sink, queue);
      throw std::runtime_error("DPAS checksum verification failed");
    }

    std::sort(tflops.begin(), tflops.end());
    const double best = tflops.back();
    const std::size_t middle = tflops.size() / 2;
    const double median = tflops.size() % 2
                              ? tflops[middle]
                              : 0.5 * (tflops[middle - 1] +
                                       tflops[middle]);

    std::cout << std::fixed << std::setprecision(2)
              << "Median BF16 compute: " << median << " TFLOP/s ("
              << (100.0 * median / kB70PeakTFlops)
              << "% of 183.0 TFLOP/s)\n"
              << "Best BF16 compute:   " << best << " TFLOP/s ("
              << (100.0 * best / kB70PeakTFlops)
              << "% of 183.0 TFLOP/s)\n"
              << "Checksum: PASS\n";

    sycl::free(a_surface, queue);
    sycl::free(b_surface, queue);
    sycl::free(sink, queue);
    return 0;
  } catch (const sycl::exception &error) {
    std::cerr << "SYCL error: " << error.what() << '\n';
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
  }
  return 1;
}
