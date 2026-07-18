#include "../onednn_matmul.hpp"

#include <oneapi/dnnl/dnnl_ze.hpp>
#include <level_zero/ze_api.h>

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define ZE_CHECK(call)                                                         \
  do {                                                                         \
    if ((call) != ZE_RESULT_SUCCESS) throw std::runtime_error(#call);           \
  } while (false)

int main(int argc, char **argv) try {
  const int n = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
  const MatmulType type = argc > 3 ? parse_matmul_type(argv[3])
                                   : MatmulType::F32;
  if (n <= 0 || iterations <= 0)
    throw std::invalid_argument(
        "usage: onednn_l0 [size] [iterations] [f32|bf16|fp16]");
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t driver_count = 0;
  ZE_CHECK(zeDriverGet(&driver_count, nullptr));
  std::vector<ze_driver_handle_t> drivers(driver_count);
  ZE_CHECK(zeDriverGet(&driver_count, drivers.data()));
  if (drivers.empty()) throw std::runtime_error("no Level Zero driver");

  uint32_t device_count = 0;
  ZE_CHECK(zeDeviceGet(drivers[0], &device_count, nullptr));
  std::vector<ze_device_handle_t> devices(device_count);
  ZE_CHECK(zeDeviceGet(drivers[0], &device_count, devices.data()));
  if (devices.empty()) throw std::runtime_error("no Level Zero device");

  ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ze_context_handle_t context = nullptr;
  ZE_CHECK(zeContextCreate(drivers[0], &context_desc, &context));

  uint32_t group_count = 0;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(devices[0], &group_count,
                                                   nullptr));
  std::vector<ze_command_queue_group_properties_t> groups(group_count);
  for (auto &group : groups)
    group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
  ZE_CHECK(zeDeviceGetCommandQueueGroupProperties(devices[0], &group_count,
                                                   groups.data()));
  uint32_t ordinal = 0;
  while (ordinal != group_count &&
         !(groups[ordinal].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE))
    ++ordinal;
  if (ordinal == group_count) throw std::runtime_error("no compute queue");

  ze_command_queue_desc_t queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  queue_desc.ordinal = ordinal;
  queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_list_handle_t list = nullptr;
  ZE_CHECK(zeCommandListCreateImmediate(context, devices[0], &queue_desc, &list));

  const size_t count = static_cast<size_t>(n) * n;
  const size_t bytes = count * matmul_element_size(type);
  ze_device_mem_alloc_desc_t device_desc = {
      ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  void *a = nullptr, *b = nullptr, *c = nullptr;
  ZE_CHECK(zeMemAllocDevice(context, &device_desc, bytes, 64, devices[0], &a));
  ZE_CHECK(zeMemAllocDevice(context, &device_desc, bytes, 64, devices[0], &b));
  ZE_CHECK(zeMemAllocDevice(context, &device_desc, bytes, 64, devices[0], &c));
  std::vector<uint8_t> host_a(bytes), host_b(bytes), host_c(bytes);
  initialize_onednn_matmul(
      host_a.data(), host_b.data(), host_c.data(), count, type);
  ZE_CHECK(zeCommandListAppendMemoryCopy(
      list, a, host_a.data(), bytes, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListAppendMemoryCopy(
      list, b, host_b.data(), bytes, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListAppendMemoryCopy(
      list, c, host_c.data(), bytes, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));

  auto engine = dnnl::ze_interop::make_engine(drivers[0], devices[0], context);
  auto stream = dnnl::ze_interop::make_stream(engine, list);
  const auto desc = onednn_matmul_memory_desc(n, type);
  auto source = dnnl::ze_interop::make_memory(desc, engine, a);
  auto weights = dnnl::ze_interop::make_memory(desc, engine, b);
  auto destination = dnnl::ze_interop::make_memory(desc, engine, c);
  auto primitive = make_onednn_matmul(engine, desc);
  const auto args = onednn_matmul_args(source, weights, destination);
  for (int i = 0; i != 10; ++i)
    dnnl::ze_interop::execute(primitive, stream, args);
  stream.wait();
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i != iterations; ++i)
    dnnl::ze_interop::execute(primitive, stream, args);
  stream.wait();
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  ZE_CHECK(zeCommandListAppendMemoryCopy(
      list, host_c.data(), c, bytes, nullptr, 0, nullptr));
  ZE_CHECK(zeCommandListHostSynchronize(list, UINT64_MAX));
  check_onednn_matmul(host_c.data(), n, type);
  std::cout << "oneDNN matmul passed (Level Zero interop)\n"
            << "Type: " << matmul_type_name(type) << ", size: " << n << "x"
            << n << ", iterations: " << iterations
            << ", average: " << seconds * 1e3 / iterations << " ms, "
            << 2.0 * n * n * n * iterations / seconds / 1e12 << " TFLOP/s\n";

  zeMemFree(context, c); zeMemFree(context, b); zeMemFree(context, a);
  zeCommandListDestroy(list); zeContextDestroy(context);
  return 0;
} catch (const std::exception &error) {
  std::cerr << "Error: " << error.what() << '\n';
  return 1;
}
