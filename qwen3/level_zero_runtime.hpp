#pragma once

#include <level_zero/ze_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qwen3 {

struct KernelArgument {
  std::size_t size;
  const void *value;
};

class LevelZeroRuntime {
public:
  explicit LevelZeroRuntime(const std::string &module_path);
  ~LevelZeroRuntime();
  LevelZeroRuntime(const LevelZeroRuntime &) = delete;
  LevelZeroRuntime &operator=(const LevelZeroRuntime &) = delete;

  void require_bmg_and_memory(std::uint64_t required_bytes) const;
  void *allocate_device(std::uint64_t bytes, std::uint64_t alignment = 4096);
  void copy_to_device(void *destination, const void *source, std::size_t bytes);
  void copy_from_device(void *destination, const void *source, std::size_t bytes);
  void launch(const std::string &entry_fragment, std::uint32_t work_items,
              const std::vector<KernelArgument> &arguments);

  const std::string &device_name() const { return device_name_; }
  std::uint64_t device_memory() const { return device_memory_; }

private:
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t device_ = nullptr;
  ze_context_handle_t context_ = nullptr;
  ze_command_list_handle_t command_list_ = nullptr;
  ze_module_handle_t module_ = nullptr;
  std::map<std::string, ze_kernel_handle_t> kernels_;
  std::map<std::string, std::uint32_t> local_sizes_;
  std::vector<void *> allocations_;
  std::string device_name_;
  std::uint64_t device_memory_ = 0;
  std::uint32_t vendor_id_ = 0;
  std::uint32_t device_id_ = 0;
};

template <typename T> KernelArgument kernel_argument(const T &value) {
  return KernelArgument{sizeof(T), &value};
}

} // namespace qwen3
