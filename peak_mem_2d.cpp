#include "peak_mem_2d_kernel.hpp"

#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace {

using namespace peak_memory;

constexpr std::size_t kMiB = 1024 * 1024;
constexpr double kB70PeakGBs = 608.0;

class InitializeInput;

struct Options {
  std::size_t size_mib = 1024;
  unsigned iterations = 100;
};

// Generate finite, irregular float data. A uniform fill can be represented by
// the GPU's lossless memory compression and would report bandwidth far above
// the physical DRAM limit.
inline float input_value(std::size_t index) {
  std::uint32_t bits = static_cast<std::uint32_t>(index);
  bits ^= bits >> 16;
  bits *= 0x7feb352du;
  bits ^= bits >> 15;
  bits *= 0x846ca68bu;
  bits ^= bits >> 16;
  return 1.0f + static_cast<float>(bits & 0xffffu) * (1.0f / 65536.0f);
}

float expected_checksum(std::size_t gid, unsigned surface_height) {
  const std::size_t x = (gid % kTilesX) * kBlockWidth;
  const std::size_t y = (gid / kTilesX) * kRowsPerWorkItem;
  if (y + 3 * kBlockHeight >= surface_height)
    throw std::logic_error("checksum coordinate is outside the surface");

  float result = 0.0f;
  for (unsigned load = 0; load < kLoadsPerWorkItem; ++load)
    result += input_value(x + (y + load * kBlockHeight) * kSurfaceWidth);
  return result;
}

[[noreturn]] void usage(const char *program, int status) {
  std::ostream &out = status == 0 ? std::cout : std::cerr;
  out << "Usage: " << program << " [--size-mib N] [--iterations N]\n"
      << "  --size-mib N    DRAM read surface size (default: 1024)\n"
      << "  --iterations N  Timed kernel launches (default: 100)\n";
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
    if (arg == "--size-mib")
      options.size_mib =
          parse_positive<std::size_t>(argv[++i], "--size-mib");
    else if (arg == "--iterations")
      options.iterations =
          parse_positive<unsigned>(argv[++i], "--iterations");
    else
      throw std::invalid_argument("unknown option: " + std::string(arg));
  }
  return options;
}

sycl::event launch_peak_read(sycl::queue &queue, const float *input,
                             float *sink, unsigned surface_height,
                             std::size_t work_items) {
  return queue.submit([&](sycl::handler &handler) {
    syclexp::nd_launch(
        handler,
        sycl::nd_range<1>{sycl::range<1>{work_items},
                          sycl::range<1>{kLocalSize}},
        syclexp::kernel_function<peak_mem_2d_kernel>, input, sink,
        surface_height);
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

extern "C" int run_peak_mem_2d(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.size_mib >
        std::numeric_limits<std::size_t>::max() / kMiB)
      throw std::invalid_argument("--size-mib is too large");

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
              << device.get_info<sycl::info::device::driver_version>() << '\n';

    if (!device.has(sycl::aspect::ext_intel_esimd))
      throw std::runtime_error("selected GPU does not support eSIMD");
    if (!device.get_info<
            esimd::info::device::has_2d_block_io_support>())
      throw std::runtime_error(
          "selected GPU does not report block-2D I/O support");

    const std::size_t bytes = options.size_mib * kMiB;
    constexpr std::size_t row_bytes = kSurfaceWidth * sizeof(float);
    if (bytes % (row_bytes * kRowsPerWorkItem) != 0)
      throw std::invalid_argument(
          "surface size must be a multiple of 128 KiB");
    if (bytes >
        device.get_info<sycl::info::device::max_mem_alloc_size>())
      throw std::runtime_error(
          "requested surface exceeds the device's maximum allocation size");

    const std::size_t elements = bytes / sizeof(float);
    const auto surface_height = static_cast<unsigned>(bytes / row_bytes);
    const std::size_t work_items = bytes / kBytesPerWorkItem;
    if (work_items % kLocalSize != 0)
      throw std::logic_error("global range is not divisible by local size");

    float *input = sycl::malloc_device<float>(elements, queue);
    float *sink = sycl::malloc_device<float>(work_items, queue);
    if (!input || !sink) {
      if (input)
        sycl::free(input, queue);
      if (sink)
        sycl::free(sink, queue);
      throw std::bad_alloc();
    }

    queue
        .parallel_for<InitializeInput>(
            sycl::range<1>{elements}, [=](sycl::id<1> id) {
              input[id] = input_value(id[0]);
            })
        .wait_and_throw();
    queue.fill(sink, 0.0f, work_items).wait_and_throw();

    std::cout << "Surface: " << options.size_mib << " MiB, "
              << kSurfaceWidth << " x " << surface_height << " floats\n"
              << "Traffic: " << (kBytesPerWorkItem / 1024)
              << " KiB/read work-item, " << work_items << " work-items\n";

    constexpr unsigned warmups = 3;
    for (unsigned i = 0; i < warmups; ++i)
      launch_peak_read(queue, input, sink, surface_height, work_items)
          .wait_and_throw();

    std::vector<double> bandwidths;
    bandwidths.reserve(options.iterations);
    for (unsigned i = 0; i < options.iterations; ++i) {
      sycl::event event =
          launch_peak_read(queue, input, sink, surface_height, work_items);
      event.wait_and_throw();
      bandwidths.push_back(static_cast<double>(bytes) /
                           event_seconds(event) / 1.0e9);
    }

    float first = 0.0f;
    float last = 0.0f;
    queue.memcpy(&first, sink, sizeof(float)).wait_and_throw();
    queue.memcpy(&last, sink + work_items - 1, sizeof(float)).wait_and_throw();
    const float expected_first = expected_checksum(0, surface_height);
    const float expected_last =
        expected_checksum(work_items - 1, surface_height);
    if (std::abs(first - expected_first) > 1.0e-5f ||
        std::abs(last - expected_last) > 1.0e-5f) {
      sycl::free(input, queue);
      sycl::free(sink, queue);
      throw std::runtime_error("checksum verification failed");
    }

    std::sort(bandwidths.begin(), bandwidths.end());
    const double best = bandwidths.back();
    const std::size_t middle = bandwidths.size() / 2;
    const double median = bandwidths.size() % 2
                              ? bandwidths[middle]
                              : 0.5 * (bandwidths[middle - 1] +
                                       bandwidths[middle]);

    std::cout << std::fixed << std::setprecision(2)
              << "Median read bandwidth: " << median << " GB/s ("
              << (100.0 * median / kB70PeakGBs) << "% of 608.0 GB/s)\n"
              << "Best read bandwidth:   " << best << " GB/s ("
              << (100.0 * best / kB70PeakGBs) << "% of 608.0 GB/s)\n"
              << "Checksum: PASS\n";

    sycl::free(input, queue);
    sycl::free(sink, queue);
    return 0;
  } catch (const sycl::exception &error) {
    std::cerr << "SYCL error: " << error.what() << '\n';
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
  }
  return 1;
}

#ifdef XPU_PLAYGROUND_STANDALONE_HOST
int main(int argc, char **argv) { return run_peak_mem_2d(argc, argv); }
#endif
