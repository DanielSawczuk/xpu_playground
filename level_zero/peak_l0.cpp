#include <level_zero/ze_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kLocalSize = 16;

constexpr uint32_t kMemSurfaceWidth = 1024;
constexpr uint32_t kMemBlockWidth = 16;
constexpr uint32_t kMemBlockHeight = 8;
constexpr uint32_t kMemLoadsPerWorkItem = 4;
constexpr uint32_t kMemRowsPerWorkItem =
    kMemBlockHeight * kMemLoadsPerWorkItem;
constexpr uint64_t kMemBytesPerWorkItem =
    uint64_t{kMemBlockWidth} * kMemBlockHeight * sizeof(float) *
    kMemLoadsPerWorkItem;
constexpr double kExpectedBandwidthGBs = 608.0;

constexpr uint32_t kDpasM = 8;
constexpr uint32_t kDpasN = 16;
constexpr uint32_t kDpasK = 16;
constexpr uint32_t kDpasAccumulators = 8;
constexpr uint64_t kFlopsPerDpas =
    uint64_t{2} * kDpasM * kDpasN * kDpasK;
constexpr double kExpectedTflops = 183.0;

[[noreturn]] void throw_ze(ze_result_t result, const char *expression,
                           const char *file, int line) {
  std::ostringstream out;
  out << expression << " failed at " << file << ':' << line
      << " (Level Zero result 0x" << std::hex
      << static_cast<uint32_t>(result) << ')';
  throw std::runtime_error(out.str());
}

#define ZE_CHECK(expression)                                                   \
  do {                                                                         \
    const ze_result_t ze_check_result = (expression);                           \
    if (ze_check_result != ZE_RESULT_SUCCESS)                                   \
      throw_ze(ze_check_result, #expression, __FILE__, __LINE__);               \
  } while (false)

template <typename T> T make_descriptor(ze_structure_type_t type) {
  T descriptor{};
  descriptor.stype = type;
  return descriptor;
}

std::vector<uint8_t> read_binary(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open SPIR-V module: " + path);

  const std::streamsize length = input.tellg();
  if (length <= 0)
    throw std::runtime_error("empty SPIR-V module: " + path);

  std::vector<uint8_t> bytes(static_cast<size_t>(length));
  input.seekg(0);
  if (!input.read(reinterpret_cast<char *>(bytes.data()), length))
    throw std::runtime_error("cannot read SPIR-V module: " + path);
  return bytes;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  if ((values.size() & 1U) != 0)
    return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

float input_value(uint64_t index) {
  uint32_t bits = static_cast<uint32_t>(index);
  bits ^= bits >> 16;
  bits *= 0x7feb352dU;
  bits ^= bits >> 15;
  bits *= 0x846ca68bU;
  bits ^= bits >> 16;
  return 1.0f + static_cast<float>(bits & 0xffffU) * (1.0f / 65536.0f);
}

bool nearly_equal(float actual, float expected) {
  return std::abs(actual - expected) <=
         2.0e-5f * std::max(1.0f, std::abs(expected));
}

class LevelZeroRuntime {
public:
  LevelZeroRuntime() {
    ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

    uint32_t driver_count = 0;
    ZE_CHECK(zeDriverGet(&driver_count, nullptr));
    if (driver_count == 0)
      throw std::runtime_error("no Level Zero drivers found");

    std::vector<ze_driver_handle_t> drivers(driver_count);
    ZE_CHECK(zeDriverGet(&driver_count, drivers.data()));

    for (ze_driver_handle_t candidate_driver : drivers) {
      uint32_t device_count = 0;
      ZE_CHECK(zeDeviceGet(candidate_driver, &device_count, nullptr));
      std::vector<ze_device_handle_t> devices(device_count);
      if (device_count != 0)
        ZE_CHECK(zeDeviceGet(candidate_driver, &device_count, devices.data()));

      for (ze_device_handle_t candidate_device : devices) {
        auto candidate_properties = make_descriptor<ze_device_properties_t>(
            ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2);
        ZE_CHECK(zeDeviceGetProperties(candidate_device, &candidate_properties));
        if (candidate_properties.type == ZE_DEVICE_TYPE_GPU &&
            candidate_properties.vendorId == 0x8086) {
          driver_ = candidate_driver;
          device_ = candidate_device;
          properties_ = candidate_properties;
          break;
        }
      }
      if (device_ != nullptr)
        break;
    }

    if (device_ == nullptr)
      throw std::runtime_error("no Intel Level Zero GPU found");

    auto context_desc =
        make_descriptor<ze_context_desc_t>(ZE_STRUCTURE_TYPE_CONTEXT_DESC);
    ZE_CHECK(zeContextCreate(driver_, &context_desc, &context_));

    uint32_t group_count = 0;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device_, &group_count,
                                                     nullptr));
    std::vector<ze_command_queue_group_properties_t> groups(group_count);
    for (auto &group : groups)
      group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device_, &group_count,
                                                     groups.data()));

    uint32_t compute_ordinal = std::numeric_limits<uint32_t>::max();
    for (uint32_t index = 0; index < group_count; ++index) {
      if ((groups[index].flags &
           ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0) {
        compute_ordinal = index;
        break;
      }
    }
    if (compute_ordinal == std::numeric_limits<uint32_t>::max())
      throw std::runtime_error("GPU exposes no compute command queue group");

    auto queue_desc = make_descriptor<ze_command_queue_desc_t>(
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC);
    queue_desc.ordinal = compute_ordinal;
    queue_desc.index = 0;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandListCreateImmediate(context_, device_, &queue_desc,
                                           &command_list_));

    auto pool_desc = make_descriptor<ze_event_pool_desc_t>(
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC);
    pool_desc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE |
                      ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    pool_desc.count = 1;
    ZE_CHECK(zeEventPoolCreate(context_, &pool_desc, 1, &device_, &event_pool_));

    auto event_desc =
        make_descriptor<ze_event_desc_t>(ZE_STRUCTURE_TYPE_EVENT_DESC);
    event_desc.index = 0;
    event_desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    event_desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    ZE_CHECK(zeEventCreate(event_pool_, &event_desc, &event_));
  }

  LevelZeroRuntime(const LevelZeroRuntime &) = delete;
  LevelZeroRuntime &operator=(const LevelZeroRuntime &) = delete;

  ~LevelZeroRuntime() {
    if (command_list_ != nullptr)
      zeCommandListHostSynchronize(command_list_, UINT64_MAX);
    if (event_ != nullptr)
      zeEventDestroy(event_);
    if (event_pool_ != nullptr)
      zeEventPoolDestroy(event_pool_);
    if (kernel_ != nullptr)
      zeKernelDestroy(kernel_);
    if (module_ != nullptr)
      zeModuleDestroy(module_);
    if (command_list_ != nullptr)
      zeCommandListDestroy(command_list_);
    for (void *allocation : allocations_)
      zeMemFree(context_, allocation);
    if (context_ != nullptr)
      zeContextDestroy(context_);
  }

  const ze_device_properties_t &properties() const { return properties_; }

  void load_kernel(const std::string &spv_path, uint32_t local_size,
                   const std::string &entry_name_fragment,
                   const std::string &fallback_entry_name,
                   bool double_grf = false) {
    const std::vector<uint8_t> spirv = read_binary(spv_path);

    auto module_desc =
        make_descriptor<ze_module_desc_t>(ZE_STRUCTURE_TYPE_MODULE_DESC);
    const bool native = spv_path.size() >= 4 &&
                        spv_path.compare(spv_path.size() - 4, 4, ".bin") == 0;
    module_desc.format = native ? ZE_MODULE_FORMAT_NATIVE
                                : ZE_MODULE_FORMAT_IL_SPIRV;
    module_desc.inputSize = spirv.size();
    module_desc.pInputModule = spirv.data();
    const std::string build_flags = native ? "" : double_grf
        ? "-vc-codegen -doubleGRF -disable-finalizer-msg"
        : "-vc-codegen -disable-finalizer-msg";
    module_desc.pBuildFlags = build_flags.empty() ? nullptr : build_flags.c_str();

    ze_module_build_log_handle_t build_log = nullptr;
    const ze_result_t result = zeModuleCreate(context_, device_, &module_desc,
                                               &module_, &build_log);
    std::string log_text;
    if (build_log != nullptr) {
      size_t log_size = 0;
      zeModuleBuildLogGetString(build_log, &log_size, nullptr);
      if (log_size > 1) {
        std::vector<char> text(log_size);
        zeModuleBuildLogGetString(build_log, &log_size, text.data());
        log_text.assign(text.data());
      }
      zeModuleBuildLogDestroy(build_log);
    }
    if (result != ZE_RESULT_SUCCESS) {
      std::ostringstream out;
      out << "zeModuleCreate failed (Level Zero result 0x" << std::hex
          << static_cast<uint32_t>(result) << ')';
      if (!log_text.empty())
        out << "\nBuild log:\n" << log_text;
      throw std::runtime_error(out.str());
    }

    uint32_t name_count = 0;
    ZE_CHECK(zeModuleGetKernelNames(module_, &name_count, nullptr));
    std::vector<const char *> names(name_count);
    ZE_CHECK(zeModuleGetKernelNames(module_, &name_count, names.data()));

    std::string selected_name;
    for (const char *name : names) {
      if (name != nullptr &&
          std::string(name).find(entry_name_fragment) != std::string::npos) {
        selected_name = name;
        break;
      }
    }
    // Some Level Zero drivers report zero names for a SPIR-V IL module even
    // though zeKernelCreate accepts its OpEntryPoint name. The free-function
    // wrapper name is stable for these declarations; --kernel can override it.
    if (selected_name.empty() && !fallback_entry_name.empty())
      selected_name = fallback_entry_name;
    if (selected_name.empty()) {
      std::ostringstream out;
      out << "no kernel containing '" << entry_name_fragment << "' in "
          << spv_path << "; module entries:";
      for (const char *name : names)
        out << "\n  " << (name == nullptr ? "<null>" : name);
      throw std::runtime_error(out.str());
    }

    auto kernel_desc =
        make_descriptor<ze_kernel_desc_t>(ZE_STRUCTURE_TYPE_KERNEL_DESC);
    kernel_desc.pKernelName = selected_name.c_str();
    ZE_CHECK(zeKernelCreate(module_, &kernel_desc, &kernel_));
    ZE_CHECK(zeKernelSetGroupSize(kernel_, local_size, 1, 1));
    kernel_name_ = std::move(selected_name);
  }

  void *allocate_device(size_t bytes) {
    auto desc = make_descriptor<ze_device_mem_alloc_desc_t>(
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC);
    void *pointer = nullptr;
    ZE_CHECK(zeMemAllocDevice(context_, &desc, bytes, 64, device_, &pointer));
    allocations_.push_back(pointer);
    return pointer;
  }

  void *allocate_host(size_t bytes) {
    auto desc = make_descriptor<ze_host_mem_alloc_desc_t>(
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC);
    void *pointer = nullptr;
    ZE_CHECK(zeMemAllocHost(context_, &desc, bytes, 64, &pointer));
    allocations_.push_back(pointer);
    return pointer;
  }

  template <typename T> void set_argument(uint32_t index, const T &value) {
    ZE_CHECK(zeKernelSetArgumentValue(kernel_, index, sizeof(T), &value));
  }

  void copy(void *destination, const void *source, size_t bytes) {
    ZE_CHECK(zeCommandListAppendMemoryCopy(command_list_, destination, source,
                                            bytes, nullptr, 0, nullptr));
  }

  void fill(void *destination, const void *pattern, size_t pattern_size,
            size_t bytes) {
    ZE_CHECK(zeCommandListAppendMemoryFill(command_list_, destination, pattern,
                                            pattern_size, bytes, nullptr, 0,
                                            nullptr));
  }

  void synchronize() {
    ZE_CHECK(zeCommandListHostSynchronize(command_list_, UINT64_MAX));
  }

  double launch(uint32_t group_count_x) {
    ZE_CHECK(zeEventHostReset(event_));
    const ze_group_count_t groups = {group_count_x, 1, 1};
    ZE_CHECK(zeCommandListAppendLaunchKernel(command_list_, kernel_, &groups,
                                             event_, 0, nullptr));
    ZE_CHECK(zeEventHostSynchronize(event_, UINT64_MAX));

    ze_kernel_timestamp_result_t timestamp = {};
    ZE_CHECK(zeEventQueryKernelTimestamp(event_, &timestamp));

    const uint32_t valid_bits = properties_.kernelTimestampValidBits;
    const uint64_t mask =
        valid_bits >= 64 ? UINT64_MAX : ((uint64_t{1} << valid_bits) - 1);
    const uint64_t cycles =
        (timestamp.global.kernelEnd - timestamp.global.kernelStart) & mask;
    return static_cast<double>(cycles) /
           static_cast<double>(properties_.timerResolution);
  }

  const std::string &kernel_name() const { return kernel_name_; }

private:
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_command_list_handle_t command_list_ = nullptr;
  ze_event_pool_handle_t event_pool_ = nullptr;
  ze_event_handle_t event_ = nullptr;
  ze_module_handle_t module_ = nullptr;
  ze_kernel_handle_t kernel_ = nullptr;
  ze_device_properties_t properties_{};
  std::vector<void *> allocations_;
  std::string kernel_name_;
};

struct Options {
  enum class Mode { Memory, Compute, Gemv, Gemm } mode = Mode::Memory;
  std::string spv_path;
  uint64_t size_mib = 1024;
  uint32_t work_items = 4096;
  uint32_t rounds = 8192;
  uint32_t iterations = 100;
  uint32_t warmups = 3;
  uint32_t m = 0, n = 0, k = 0;
  bool fp16 = false;
  std::string kernel_entry;
};

uint64_t parse_u64(const std::string &value, const char *option) {
  size_t consumed = 0;
  uint64_t number = 0;
  try {
    number = std::stoull(value, &consumed);
  } catch (...) {
    throw std::runtime_error(std::string("invalid value for ") + option +
                             ": " + value);
  }
  if (consumed != value.size() || number == 0)
    throw std::runtime_error(std::string("invalid value for ") + option +
                             ": " + value);
  return number;
}

uint32_t parse_u32(const std::string &value, const char *option) {
  const uint64_t number = parse_u64(value, option);
  if (number > UINT32_MAX)
    throw std::runtime_error(std::string(option) + " is too large");
  return static_cast<uint32_t>(number);
}

void print_usage(const char *program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " mem [--spv FILE] [--kernel NAME] [--size-mib N] [--iterations N]\n"
      << "  " << program
      << " compute [--spv FILE] [--kernel NAME] [--work-items N] [--rounds N]"
         " [--iterations N]\n"
      << "  " << program
      << " gemv [--spv FILE] [--kernel NAME] [--size N | --m M --k K]"
         " [--type bf16|fp16] [--iterations N] [--warmups N]\n"
      << "  " << program
      << " gemm [--spv FILE] [--kernel NAME] [--size N | --m M --n N --k K]"
         " [--type bf16|fp16] [--iterations N] [--warmups N]\n";
}

Options parse_options(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    throw std::runtime_error("missing benchmark mode");
  }

  Options options;
  const std::string mode = argv[1];
  if (mode == "mem") {
    options.mode = Options::Mode::Memory;
    options.spv_path = "level_zero/peak_mem_2d.spv";
    options.kernel_entry =
        "_ZTSN12_GLOBAL__N_121PeakMemLevelZeroEntryE";
  } else if (mode == "compute") {
    options.mode = Options::Mode::Compute;
    options.spv_path = "level_zero/peak_compute_dpas.spv";
    options.kernel_entry =
        "_ZTSN12_GLOBAL__N_125PeakComputeLevelZeroEntryE";
    options.iterations = 1000;
  } else if (mode == "gemv") {
    options.mode = Options::Mode::Gemv;
    options.m = options.k = 16384;
    options.iterations = 50;
    options.warmups = 5;
  } else if (mode == "gemm") {
    options.mode = Options::Mode::Gemm;
    options.m = options.n = options.k = 4096;
    options.iterations = 20;
    options.warmups = 5;
  } else if (mode == "-h" || mode == "--help") {
    print_usage(argv[0]);
    std::exit(0);
  } else {
    throw std::runtime_error("unknown benchmark mode: " + mode);
  }

  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "-h" || option == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc)
      throw std::runtime_error("missing value for " + option);
    const std::string value = argv[++index];

    if (option == "--spv") {
      options.spv_path = value;
    } else if (option == "--kernel") {
      options.kernel_entry = value;
    } else if (option == "--iterations") {
      const uint64_t parsed = parse_u64(value, "--iterations");
      if (parsed > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("--iterations is too large");
      options.iterations = static_cast<uint32_t>(parsed);
    } else if (option == "--warmups" &&
               (options.mode == Options::Mode::Gemv ||
                options.mode == Options::Mode::Gemm)) {
      options.warmups = parse_u32(value, "--warmups");
    } else if (option == "--type" &&
               (options.mode == Options::Mode::Gemv ||
                options.mode == Options::Mode::Gemm)) {
      if (value == "fp16") options.fp16 = true;
      else if (value == "bf16") options.fp16 = false;
      else throw std::runtime_error("--type must be bf16 or fp16");
    } else if (option == "--size" &&
               (options.mode == Options::Mode::Gemv ||
                options.mode == Options::Mode::Gemm)) {
      const uint32_t size = parse_u32(value, "--size");
      options.m = options.k = size;
      if (options.mode == Options::Mode::Gemm) options.n = size;
    } else if (option == "--m" &&
               (options.mode == Options::Mode::Gemv ||
                options.mode == Options::Mode::Gemm)) {
      options.m = parse_u32(value, "--m");
    } else if (option == "--n" && options.mode == Options::Mode::Gemm) {
      options.n = parse_u32(value, "--n");
    } else if (option == "--k" &&
               (options.mode == Options::Mode::Gemv ||
                options.mode == Options::Mode::Gemm)) {
      options.k = parse_u32(value, "--k");
    } else if (option == "--size-mib" &&
               options.mode == Options::Mode::Memory) {
      options.size_mib = parse_u64(value, "--size-mib");
    } else if (option == "--work-items" &&
               options.mode == Options::Mode::Compute) {
      const uint64_t parsed = parse_u64(value, "--work-items");
      if (parsed > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("--work-items is too large");
      options.work_items = static_cast<uint32_t>(parsed);
    } else if (option == "--rounds" &&
               options.mode == Options::Mode::Compute) {
      const uint64_t parsed = parse_u64(value, "--rounds");
      if (parsed > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("--rounds is too large");
      options.rounds = static_cast<uint32_t>(parsed);
    } else {
      throw std::runtime_error("invalid option for this mode: " + option);
    }
  }
  if (options.mode == Options::Mode::Gemv && options.spv_path.empty()) {
    options.spv_path = options.fp16 ? "level_zero/gemv_fp16.bin"
                                    : "level_zero/gemv_bf16.bin";
    if (options.kernel_entry.empty())
      options.kernel_entry = options.fp16 ? "_ZTS22GemvFp16LevelZeroEntry"
                                          : "_ZTS22GemvBf16LevelZeroEntry";
  }
  if (options.mode == Options::Mode::Gemm && options.spv_path.empty()) {
    const bool large = options.m >= 2560 && options.n >= 2560;
    const std::string type = options.fp16 ? "fp16" : "bf16";
    const std::string tile = large ? "large" : "small";
    options.spv_path = "level_zero/gemm_" + type + '_' + tile + ".bin";
    if (options.kernel_entry.empty() && options.fp16)
      options.kernel_entry = large ? "_ZTS27GemmFp16LargeLevelZeroEntry"
                                   : "_ZTS27GemmFp16SmallLevelZeroEntry";
    else if (options.kernel_entry.empty())
      options.kernel_entry = large ? "_ZTS27GemmBf16LargeLevelZeroEntry"
                                   : "_ZTS27GemmBf16SmallLevelZeroEntry";
  }
  return options;
}

void print_common(const LevelZeroRuntime &runtime) {
  const auto &properties = runtime.properties();
  std::cout << "Device: " << properties.name << '\n'
            << "API: Level Zero\n"
            << "Kernel entry: " << runtime.kernel_name() << '\n';
}

int run_memory(const Options &options) {
  LevelZeroRuntime runtime;
  runtime.load_kernel(options.spv_path, kLocalSize, "peak_mem_2d_kernel",
                      options.kernel_entry);

  const uint64_t requested_bytes = options.size_mib * 1024ULL * 1024ULL;
  const uint64_t surface_quantum = uint64_t{kMemSurfaceWidth} * sizeof(float) *
                                   kMemRowsPerWorkItem;
  const uint64_t total_bytes =
      requested_bytes - requested_bytes % surface_quantum;
  if (total_bytes < surface_quantum)
    throw std::runtime_error("--size-mib is too small");
  if (total_bytes / sizeof(float) >
      std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("input has more than UINT32_MAX float elements");

  const uint64_t work_items_64 = total_bytes / kMemBytesPerWorkItem;
  if (work_items_64 > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("work-item count exceeds Level Zero group limits");
  const uint32_t work_items = static_cast<uint32_t>(work_items_64);
  const uint32_t groups = work_items / kLocalSize;

  auto *input = static_cast<float *>(runtime.allocate_device(total_bytes));
  auto *sink = static_cast<float *>(
      runtime.allocate_device(uint64_t{work_items} * sizeof(float)));

  const size_t staging_bytes = static_cast<size_t>(
      std::min<uint64_t>(total_bytes, 32ULL * 1024ULL * 1024ULL));
  auto *staging = static_cast<float *>(runtime.allocate_host(staging_bytes));
  for (uint64_t offset = 0; offset < total_bytes; offset += staging_bytes) {
    const size_t chunk_bytes = static_cast<size_t>(
        std::min<uint64_t>(staging_bytes, total_bytes - offset));
    const uint64_t base_element = offset / sizeof(float);
    const size_t chunk_elements = chunk_bytes / sizeof(float);
    for (size_t index = 0; index < chunk_elements; ++index)
      staging[index] = input_value(base_element + index);
    runtime.copy(reinterpret_cast<uint8_t *>(input) + offset, staging,
                 chunk_bytes);
    runtime.synchronize();
  }
  const uint32_t zero = 0;
  runtime.fill(sink, &zero, sizeof(zero),
               uint64_t{work_items} * sizeof(float));
  runtime.synchronize();

  const uint32_t surface_height =
      static_cast<uint32_t>(total_bytes /
                            (kMemSurfaceWidth * sizeof(float)));
  runtime.set_argument(0, input);
  runtime.set_argument(1, sink);
  runtime.set_argument(2, surface_height);

  print_common(runtime);
  std::cout << "Module: " << options.spv_path << '\n'
            << "Input: " << (total_bytes / (1024.0 * 1024.0)) << " MiB\n"
            << "Work-items: " << work_items << '\n'
            << "Traffic/work-item: " << kMemBytesPerWorkItem << " B\n";

  for (int warmup = 0; warmup < 3; ++warmup)
    (void)runtime.launch(groups);

  std::vector<double> bandwidths;
  bandwidths.reserve(options.iterations);
  for (uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
    const double seconds = runtime.launch(groups);
    bandwidths.push_back(static_cast<double>(total_bytes) / seconds / 1.0e9);
  }

  runtime.copy(staging, sink, sizeof(float));
  runtime.copy(staging + 1, sink + work_items - 1, sizeof(float));
  runtime.synchronize();
  const float first = staging[0];
  const float last = staging[1];

  const auto expected_checksum = [&](uint32_t gid) {
    const uint64_t x =
        uint64_t{gid % (kMemSurfaceWidth / kMemBlockWidth)} * kMemBlockWidth;
    const uint64_t y =
        uint64_t{gid / (kMemSurfaceWidth / kMemBlockWidth)} *
        kMemRowsPerWorkItem;
    float sum = 0.0f;
    for (uint32_t load = 0; load < kMemLoadsPerWorkItem; ++load)
      sum += input_value(x + (y + uint64_t{load} * kMemBlockHeight) *
                                 kMemSurfaceWidth);
    return sum;
  };
  const bool valid = nearly_equal(first, expected_checksum(0)) &&
                     nearly_equal(last, expected_checksum(work_items - 1));

  const double measured_median = median(bandwidths);
  const double measured_best =
      *std::max_element(bandwidths.begin(), bandwidths.end());
  std::cout << std::fixed << std::setprecision(2)
            << "Median bandwidth: " << measured_median << " GB/s ("
            << 100.0 * measured_median / kExpectedBandwidthGBs
            << "% of 608 GB/s)\n"
            << "Best bandwidth:   " << measured_best << " GB/s ("
            << 100.0 * measured_best / kExpectedBandwidthGBs
            << "% of 608 GB/s)\n"
            << "Checksum: " << (valid ? "PASS" : "FAIL") << '\n';
  return valid ? 0 : 2;
}

int run_compute(const Options &options) {
  LevelZeroRuntime runtime;
  runtime.load_kernel(options.spv_path, kLocalSize, "peak_compute_dpas_kernel",
                      options.kernel_entry);

  const uint32_t work_items =
      (options.work_items + kLocalSize - 1) / kLocalSize * kLocalSize;
  const uint32_t groups = work_items / kLocalSize;
  constexpr size_t surface_elements = 32 * 16;
  constexpr size_t surface_bytes = surface_elements * sizeof(uint16_t);

  auto *a = static_cast<uint16_t *>(runtime.allocate_device(surface_bytes));
  auto *b = static_cast<uint16_t *>(runtime.allocate_device(surface_bytes));
  auto *sink = static_cast<float *>(
      runtime.allocate_device(uint64_t{work_items} * sizeof(float)));
  auto *host_result =
      static_cast<float *>(runtime.allocate_host(2 * sizeof(float)));

  const uint16_t bf16_half = 0x3f00;
  const uint32_t zero = 0;
  runtime.fill(a, &bf16_half, sizeof(bf16_half), surface_bytes);
  runtime.fill(b, &bf16_half, sizeof(bf16_half), surface_bytes);
  runtime.fill(sink, &zero, sizeof(zero),
               uint64_t{work_items} * sizeof(float));
  runtime.synchronize();

  runtime.set_argument(0, a);
  runtime.set_argument(1, b);
  runtime.set_argument(2, sink);
  runtime.set_argument(3, options.rounds);

  const uint64_t flops_per_work_item =
      uint64_t{options.rounds} * kDpasAccumulators * kFlopsPerDpas;
  const long double total_flops =
      static_cast<long double>(work_items) * flops_per_work_item;

  print_common(runtime);
  std::cout << "Module: " << options.spv_path << '\n'
            << "Work-items: " << work_items << '\n'
            << "Rounds: " << options.rounds << '\n'
            << "DPAS/work-item: "
            << uint64_t{options.rounds} * kDpasAccumulators << '\n';

  for (int warmup = 0; warmup < 3; ++warmup)
    (void)runtime.launch(groups);

  std::vector<double> tflops;
  tflops.reserve(options.iterations);
  for (uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
    const double seconds = runtime.launch(groups);
    tflops.push_back(static_cast<double>(total_flops / seconds / 1.0e12L));
  }

  runtime.copy(host_result, sink, sizeof(float));
  runtime.copy(host_result + 1, sink + work_items - 1, sizeof(float));
  runtime.synchronize();
  const float expected =
      28.0f + static_cast<float>(kDpasAccumulators * options.rounds) * 4.0f;
  const bool valid = nearly_equal(host_result[0], expected) &&
                     nearly_equal(host_result[1], expected);

  const double measured_median = median(tflops);
  const double measured_best = *std::max_element(tflops.begin(), tflops.end());
  std::cout << std::fixed << std::setprecision(2)
            << "Median throughput: " << measured_median << " TFLOP/s ("
            << 100.0 * measured_median / kExpectedTflops
            << "% of 183 TFLOP/s)\n"
            << "Best throughput:   " << measured_best << " TFLOP/s ("
            << 100.0 * measured_best / kExpectedTflops
            << "% of 183 TFLOP/s)\n"
            << "Checksum: " << (valid ? "PASS" : "FAIL") << '\n';
  return valid ? 0 : 2;
}

uint32_t round_up(uint32_t value, uint32_t multiple) {
  const uint64_t rounded = (uint64_t{value} + multiple - 1) / multiple * multiple;
  if (rounded > UINT32_MAX)
    throw std::runtime_error("matrix dimension is too large");
  return static_cast<uint32_t>(rounded);
}

uint16_t float_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t float_to_fp16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000U;
  int exponent = static_cast<int>((bits >> 23) & 0xffU) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000U) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0xfffU +
                                         ((mantissa >> 13) & 1U)) >> 13));
  }
  if (exponent >= 31)
    return static_cast<uint16_t>(sign | 0x7c00U);
  mantissa += 0xfffU + ((mantissa >> 13) & 1U);
  if (mantissa & 0x800000U) {
    mantissa = 0;
    if (++exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               (mantissa >> 13));
}

float fp16_to_float(uint16_t value) {
  const uint32_t sign = uint32_t{value & 0x8000U} << 16;
  uint32_t exponent = (value >> 10) & 0x1fU;
  uint32_t mantissa = value & 0x3ffU;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) bits = sign;
    else {
      int shift = 0;
      while ((mantissa & 0x400U) == 0) { mantissa <<= 1; ++shift; }
      mantissa &= 0x3ffU;
      bits = sign | (static_cast<uint32_t>(127 - 14 - shift) << 23) |
             (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

uint16_t encode16(float value, bool fp16) {
  return fp16 ? float_to_fp16(value) : float_to_bf16(value);
}

float decode16(uint16_t value, bool fp16) {
  return fp16 ? fp16_to_float(value) : bf16_to_float(value);
}

template <typename Generator>
void initialize_u16(LevelZeroRuntime &runtime, uint16_t *device, uint64_t count,
                    Generator generate) {
  const size_t capacity = static_cast<size_t>(
      std::min<uint64_t>(count, 16ULL * 1024ULL * 1024ULL));
  auto *staging = static_cast<uint16_t *>(
      runtime.allocate_host(capacity * sizeof(uint16_t)));
  for (uint64_t offset = 0; offset < count; offset += capacity) {
    const size_t chunk = static_cast<size_t>(
        std::min<uint64_t>(capacity, count - offset));
    for (size_t i = 0; i < chunk; ++i) staging[i] = generate(offset + i);
    runtime.copy(device + offset, staging, chunk * sizeof(uint16_t));
    runtime.synchronize();
  }
}

int run_gemv(const Options &options) {
  constexpr uint32_t local_size = 32, columns = 32;
  const uint32_t m = round_up(options.m, local_size);
  const uint32_t k = round_up(options.k, columns);
  const uint64_t matrix_count = uint64_t{m} * k;
  LevelZeroRuntime runtime;
  runtime.load_kernel(options.spv_path, local_size, "Gemv",
                      options.kernel_entry);
  auto *matrix = static_cast<uint16_t *>(
      runtime.allocate_device(matrix_count * sizeof(uint16_t)));
  auto *vector = static_cast<uint16_t *>(
      runtime.allocate_device(uint64_t{k} * sizeof(uint16_t)));
  auto *output = static_cast<uint16_t *>(
      runtime.allocate_device(uint64_t{m} * sizeof(uint16_t)));
  initialize_u16(runtime, matrix, matrix_count, [&](uint64_t index) {
    uint32_t bits = static_cast<uint32_t>(index);
    bits ^= bits >> 16; bits *= 0x7feb352dU;
    bits ^= bits >> 15; bits *= 0x846ca68bU; bits ^= bits >> 16;
    return encode16(0.25f + static_cast<float>(bits & 255U) / 1024.0f,
                    options.fp16);
  });
  initialize_u16(runtime, vector, k, [&](uint64_t index) {
    const uint32_t bits = static_cast<uint32_t>(index) * 747796405U;
    return encode16(0.5f + static_cast<float>((bits >> 16) & 127U) / 512.0f,
                    options.fp16);
  });
  const uint16_t zero = 0;
  runtime.fill(output, &zero, sizeof(zero), uint64_t{m} * sizeof(uint16_t));
  runtime.synchronize();
  runtime.set_argument(0, matrix); runtime.set_argument(1, vector);
  runtime.set_argument(2, output); runtime.set_argument(3, m);
  runtime.set_argument(4, k);
  print_common(runtime);
  std::cout << "Module: " << options.spv_path << '\n'
            << "Type: " << (options.fp16 ? "FP16" : "BF16")
            << ", shape: " << options.m << 'x' << options.k;
  if (m != options.m || k != options.k)
    std::cout << " (padded to " << m << 'x' << k << ')';
  std::cout << '\n';
  const uint32_t groups = m / local_size;
  for (uint32_t i = 0; i < options.warmups; ++i) (void)runtime.launch(groups);
  std::vector<double> rates, bandwidths;
  const long double operations = 2.0L * options.m * options.k;
  const long double bytes = 2.0L * options.m * options.k;
  for (uint32_t i = 0; i < options.iterations; ++i) {
    const double seconds = runtime.launch(groups);
    rates.push_back(static_cast<double>(operations / seconds / 1.0e9L));
    bandwidths.push_back(static_cast<double>(bytes / seconds / 1.0e9L));
  }
  const uint32_t checks = std::min<uint32_t>(options.m, 64);
  auto *host = static_cast<uint16_t *>(
      runtime.allocate_host(uint64_t{checks} * sizeof(uint16_t)));
  runtime.copy(host, output, uint64_t{checks} * sizeof(uint16_t));
  runtime.synchronize();
  bool valid = true;
  for (uint32_t row = 0; row < checks; ++row) {
    float expected = 0;
    for (uint32_t column = 0; column < options.k; ++column) {
      const uint64_t index = uint64_t{row} * k + column;
      uint32_t bits = static_cast<uint32_t>(index);
      bits ^= bits >> 16; bits *= 0x7feb352dU;
      bits ^= bits >> 15; bits *= 0x846ca68bU; bits ^= bits >> 16;
      const float a = decode16(encode16(
          0.25f + static_cast<float>(bits & 255U) / 1024.0f, options.fp16),
          options.fp16);
      const uint32_t vb = column * 747796405U;
      const float b = decode16(encode16(
          0.5f + static_cast<float>((vb >> 16) & 127U) / 512.0f,
          options.fp16), options.fp16);
      expected += a * b;
    }
    const float actual = decode16(host[row], options.fp16);
    if (std::abs(actual - expected) > std::max(1.0f, std::abs(expected) * 0.01f))
      valid = false;
  }
  std::cout << std::fixed << std::setprecision(2)
            << "Median: " << median(rates) << " GFLOP/s, "
            << median(bandwidths) << " GB/s\nBest:   "
            << *std::max_element(rates.begin(), rates.end()) << " GFLOP/s, "
            << *std::max_element(bandwidths.begin(), bandwidths.end())
            << " GB/s\nVerification: " << (valid ? "PASS" : "FAIL") << '\n';
  return valid ? 0 : 2;
}

int run_gemm(const Options &options) {
  constexpr uint32_t local_size = 32, tile_n = 256, prefetch_k = 32;
  const bool large = options.m >= 2560 && options.n >= 2560;
  const uint32_t tile_m = large ? 256 : 128;
  const uint32_t m = round_up(options.m, tile_m);
  const uint32_t n = round_up(options.n, tile_n);
  const uint32_t k = round_up(options.k, prefetch_k);
  const uint64_t as = uint64_t{m} * k, bs = uint64_t{k} * n;
  const uint64_t cs = uint64_t{m} * n;
  LevelZeroRuntime runtime;
  runtime.load_kernel(options.spv_path, local_size, "Gemm",
                      options.kernel_entry, true);
  auto *a = static_cast<uint16_t *>(runtime.allocate_device(as * 2));
  auto *b = static_cast<uint16_t *>(runtime.allocate_device(bs * 2));
  auto *c = static_cast<uint16_t *>(runtime.allocate_device(cs * 2));
  initialize_u16(runtime, a, as, [&](uint64_t index) {
    return encode16(index / k < options.m && index % k < options.k ? 0.5f : 0,
                    options.fp16);
  });
  initialize_u16(runtime, b, bs, [&](uint64_t index) {
    return encode16(index / n < options.k && index % n < options.n ? 0.25f : 0,
                    options.fp16);
  });
  const uint16_t zero = 0;
  runtime.fill(c, &zero, sizeof(zero), cs * 2); runtime.synchronize();
  runtime.set_argument(0, a); runtime.set_argument(1, b);
  runtime.set_argument(2, c); runtime.set_argument(3, m);
  runtime.set_argument(4, n); runtime.set_argument(5, k);
  print_common(runtime);
  std::cout << "Module: " << options.spv_path << '\n'
            << "Type: " << (options.fp16 ? "FP16" : "BF16")
            << ", shape: " << options.m << 'x' << options.n << 'x' << options.k;
  if (m != options.m || n != options.n || k != options.k)
    std::cout << " (padded to " << m << 'x' << n << 'x' << k << ')';
  std::cout << '\n';
  const uint32_t groups = (m / tile_m) * (n / tile_n);
  for (uint32_t i = 0; i < options.warmups; ++i) (void)runtime.launch(groups);
  std::vector<double> rates;
  const long double flops = 2.0L * options.m * options.n * options.k;
  for (uint32_t i = 0; i < options.iterations; ++i)
    rates.push_back(static_cast<double>(flops / runtime.launch(groups) / 1.0e12L));
  const uint32_t checks = std::min<uint32_t>(options.n, 64);
  auto *host = static_cast<uint16_t *>(runtime.allocate_host(checks * 2));
  runtime.copy(host, c, checks * 2); runtime.synchronize();
  const float expected = options.k * 0.125f;
  const float tolerance = std::max(0.5f, expected * 0.01f);
  bool valid = true;
  for (uint32_t i = 0; i < checks; ++i)
    valid &= std::abs(decode16(host[i], options.fp16) - expected) <= tolerance;
  std::cout << std::fixed << std::setprecision(2)
            << "Median: " << median(rates) << " TFLOP/s\nBest:   "
            << *std::max_element(rates.begin(), rates.end())
            << " TFLOP/s\nVerification: " << (valid ? "PASS" : "FAIL") << '\n';
  return valid ? 0 : 2;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    switch (options.mode) {
    case Options::Mode::Memory: return run_memory(options);
    case Options::Mode::Compute: return run_compute(options);
    case Options::Mode::Gemv: return run_gemv(options);
    case Options::Mode::Gemm: return run_gemm(options);
    }
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
  }
}
