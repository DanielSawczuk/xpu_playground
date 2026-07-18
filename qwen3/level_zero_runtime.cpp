#include "level_zero_runtime.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace qwen3 {
namespace {

[[noreturn]] void throw_ze(ze_result_t result, const char *expression,
                           const char *file, int line) {
  std::ostringstream out;
  out << expression << " failed at " << file << ':' << line
      << " (Level Zero result 0x" << std::hex
      << static_cast<std::uint32_t>(result) << ')';
  throw std::runtime_error(out.str());
}

#define ZE_CHECK(expression)                                                   \
  do {                                                                         \
    const ze_result_t check_result = (expression);                              \
    if (check_result != ZE_RESULT_SUCCESS)                                      \
      throw_ze(check_result, #expression, __FILE__, __LINE__);                  \
  } while (false)

template <typename T> T descriptor(ze_structure_type_t type) {
  T result{};
  result.stype = type;
  return result;
}

std::vector<std::uint8_t> read_binary(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open BMG kernel module " + path +
                             "; build it with 'make qwen3-kernels'");
  const auto length = file.tellg();
  if (length <= 0)
    throw std::runtime_error("empty BMG kernel module: " + path);
  std::vector<std::uint8_t> result(static_cast<std::size_t>(length));
  file.seekg(0);
  if (!file.read(reinterpret_cast<char *>(result.data()), length))
    throw std::runtime_error("cannot read BMG kernel module: " + path);
  return result;
}

} // namespace

LevelZeroRuntime::LevelZeroRuntime(const std::string &module_path) {
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  std::uint32_t driver_count = 0;
  ZE_CHECK(zeDriverGet(&driver_count, nullptr));
  if (driver_count == 0)
    throw std::runtime_error("no Level Zero driver is installed");
  std::vector<ze_driver_handle_t> drivers(driver_count);
  ZE_CHECK(zeDriverGet(&driver_count, drivers.data()));
  ze_device_properties_t selected{};
  for (ze_driver_handle_t driver : drivers) {
    std::uint32_t count = 0;
    ZE_CHECK(zeDeviceGet(driver, &count, nullptr));
    std::vector<ze_device_handle_t> devices(count);
    if (count) ZE_CHECK(zeDeviceGet(driver, &count, devices.data()));
    for (ze_device_handle_t candidate : devices) {
      auto properties = descriptor<ze_device_properties_t>(
          ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2);
      ZE_CHECK(zeDeviceGetProperties(candidate, &properties));
      if (properties.type == ZE_DEVICE_TYPE_GPU && properties.vendorId == 0x8086) {
        driver_ = driver;
        device_ = candidate;
        selected = properties;
        break;
      }
    }
    if (device_) break;
  }
  if (!device_)
    throw std::runtime_error("no Intel Level Zero GPU was found");
  device_name_ = selected.name;
  vendor_id_ = selected.vendorId;
  device_id_ = selected.deviceId;

  std::uint32_t memory_count = 0;
  ZE_CHECK(zeDeviceGetMemoryProperties(device_, &memory_count, nullptr));
  std::vector<ze_device_memory_properties_t> memories(memory_count);
  for (auto &memory : memories)
    memory.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  if (memory_count)
    ZE_CHECK(zeDeviceGetMemoryProperties(device_, &memory_count, memories.data()));
  for (const auto &memory : memories)
    device_memory_ += memory.totalSize;

  auto context_desc = descriptor<ze_context_desc_t>(ZE_STRUCTURE_TYPE_CONTEXT_DESC);
  ZE_CHECK(zeContextCreate(driver_, &context_desc, &context_));
  std::uint32_t group_count = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device_, &group_count, nullptr));
  std::vector<ze_command_queue_group_properties_t> groups(group_count);
  for (auto &group : groups)
    group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(device_, &group_count, groups.data()));
  std::uint32_t ordinal = std::numeric_limits<std::uint32_t>::max();
  for (std::uint32_t i = 0; i != group_count; ++i)
    if (groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ordinal = i;
      break;
    }
  if (ordinal == std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("selected GPU has no Level Zero compute queue");
  auto queue_desc = descriptor<ze_command_queue_desc_t>(
      ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC);
  queue_desc.ordinal = ordinal;
  queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ZE_CHECK(zeCommandListCreateImmediate(context_, device_, &queue_desc,
                                         &command_list_));

  const std::vector<std::uint8_t> binary = read_binary(module_path);
  auto module_desc = descriptor<ze_module_desc_t>(ZE_STRUCTURE_TYPE_MODULE_DESC);
  module_desc.format = ZE_MODULE_FORMAT_NATIVE;
  module_desc.inputSize = binary.size();
  module_desc.pInputModule = binary.data();
  ze_module_build_log_handle_t build_log = nullptr;
  const ze_result_t create_result =
      zeModuleCreate(context_, device_, &module_desc, &module_, &build_log);
  std::string log_text;
  if (build_log) {
    std::size_t size = 0;
    zeModuleBuildLogGetString(build_log, &size, nullptr);
    if (size) {
      log_text.resize(size);
      zeModuleBuildLogGetString(build_log, &size, log_text.data());
    }
    zeModuleBuildLogDestroy(build_log);
  }
  if (create_result != ZE_RESULT_SUCCESS)
    throw std::runtime_error("Level Zero could not load the BMG Qwen3 module (result " +
                             std::to_string(create_result) + ")" +
                             (log_text.empty() ? "" : ": " + log_text));
  std::uint32_t name_count = 0;
  ZE_CHECK(zeModuleGetKernelNames(module_, &name_count, nullptr));
  std::vector<const char *> names(name_count);
  ZE_CHECK(zeModuleGetKernelNames(module_, &name_count, names.data()));
  for (const char *name : names) {
    if (!name) continue;
    auto kernel_desc = descriptor<ze_kernel_desc_t>(ZE_STRUCTURE_TYPE_KERNEL_DESC);
    kernel_desc.pKernelName = name;
    ze_kernel_handle_t kernel = nullptr;
    ZE_CHECK(zeKernelCreate(module_, &kernel_desc, &kernel));
    const std::string kernel_name(name);
    const std::uint32_t local_size =
        (kernel_name.find("RmsNorm") != std::string::npos ||
         kernel_name.find("Argmax") != std::string::npos) ? 1 : 32;
    ZE_CHECK(zeKernelSetGroupSize(kernel, local_size, 1, 1));
    local_sizes_.emplace(kernel_name, local_size);
    kernels_.emplace(name, kernel);
  }
  if (kernels_.size() < 8)
    throw std::runtime_error("Qwen3 module does not expose all eight required kernels");
}

LevelZeroRuntime::~LevelZeroRuntime() {
  if (command_list_)
    zeCommandListHostSynchronize(command_list_, UINT64_MAX);
  for (void *allocation : allocations_)
    zeMemFree(context_, allocation);
  for (const auto &[name, kernel] : kernels_) {
    (void)name;
    zeKernelDestroy(kernel);
  }
  if (module_) zeModuleDestroy(module_);
  if (command_list_) zeCommandListDestroy(command_list_);
  if (context_) zeContextDestroy(context_);
}

void LevelZeroRuntime::require_bmg_and_memory(std::uint64_t required_bytes) const {
  std::string lower = device_name_;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  const bool bmg_pci_family = (device_id_ & 0xff00U) == 0xe200U;
  if (vendor_id_ != 0x8086 ||
      (!bmg_pci_family && lower.find("bmg") == std::string::npos &&
       lower.find("b70") == std::string::npos &&
       lower.find("arc(tm) b") == std::string::npos &&
       lower.find("arc b") == std::string::npos))
    throw std::runtime_error("unsupported GPU '" + device_name_ +
                             "': Qwen3 kernels are AOT-compiled for Intel BMG/B70");
  constexpr std::uint64_t reserve = 512ULL << 20;
  if (device_memory_ < required_bytes || device_memory_ - required_bytes < reserve)
    throw std::runtime_error("insufficient BMG memory: need " +
        std::to_string((required_bytes + (1ULL << 20) - 1) >> 20) +
        " MiB plus a 512 MiB driver reserve, device reports " +
        std::to_string(device_memory_ >> 20) + " MiB");
}

void *LevelZeroRuntime::allocate_device(std::uint64_t bytes,
                                        std::uint64_t alignment) {
  auto relaxed = descriptor<ze_relaxed_allocation_limits_exp_desc_t>(
      ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC);
  relaxed.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
  auto desc = descriptor<ze_device_mem_alloc_desc_t>(
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC);
  desc.pNext = &relaxed;
  void *pointer = nullptr;
  ZE_CHECK(zeMemAllocDevice(context_, &desc, static_cast<std::size_t>(bytes),
                            static_cast<std::size_t>(alignment), device_, &pointer));
  allocations_.push_back(pointer);
  return pointer;
}

void LevelZeroRuntime::copy_to_device(void *destination, const void *source,
                                      std::size_t bytes) {
  ZE_CHECK(zeCommandListAppendMemoryCopy(command_list_, destination, source,
                                          bytes, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListHostSynchronize(command_list_, UINT64_MAX));
}
void LevelZeroRuntime::copy_from_device(void *destination, const void *source,
                                        std::size_t bytes) {
  ZE_CHECK(zeCommandListAppendMemoryCopy(command_list_, destination, source,
                                          bytes, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListHostSynchronize(command_list_, UINT64_MAX));
}

void LevelZeroRuntime::launch(const std::string &entry_fragment,
                              std::uint32_t work_items,
                              const std::vector<KernelArgument> &arguments) {
  ze_kernel_handle_t selected = nullptr;
  std::uint32_t local_size = 0;
  for (const auto &[name, kernel] : kernels_)
    if (name.find(entry_fragment) != std::string::npos) {
      if (selected)
        throw std::runtime_error("ambiguous Qwen3 kernel entry: " + entry_fragment);
      selected = kernel;
      local_size = local_sizes_.at(name);
    }
  if (!selected)
    throw std::runtime_error("Qwen3 kernel is absent from module: " + entry_fragment);
  for (std::uint32_t i = 0; i != arguments.size(); ++i) {
    const ze_result_t argument_result = zeKernelSetArgumentValue(
        selected, i, arguments[i].size, arguments[i].value);
    if (argument_result != ZE_RESULT_SUCCESS)
      throw std::runtime_error("Level Zero rejected argument " +
          std::to_string(i) + " (" + std::to_string(arguments[i].size) +
          " bytes) for " + entry_fragment + " (result " +
          std::to_string(argument_result) + ")");
  }
  const ze_group_count_t groups{(work_items + local_size - 1) / local_size, 1, 1};
  ZE_CHECK(zeCommandListAppendLaunchKernel(command_list_, selected, &groups,
                                            nullptr, 0, nullptr));
  const ze_result_t result =
      zeCommandListHostSynchronize(command_list_, UINT64_MAX);
  if (result != ZE_RESULT_SUCCESS)
    throw_ze(result, entry_fragment.c_str(), __FILE__, __LINE__);
}

} // namespace qwen3
