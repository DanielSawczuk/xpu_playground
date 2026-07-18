#include "../level_zero_runtime.hpp"
#include "../model.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using bf16 = std::uint16_t;

bf16 b(float value) { return qwen3::float_to_bf16(value); }
float f(bf16 value) { return qwen3::bf16_to_float(value); }

void close(float actual, float expected, float atol = 0.03f,
           float rtol = 0.02f) {
  if (!std::isfinite(actual) ||
      std::abs(actual - expected) > atol + rtol * std::abs(expected)) {
    std::cerr << "mismatch: actual=" << actual << " expected=" << expected << '\n';
    std::abort();
  }
}

template <typename T>
T *device_vector(qwen3::LevelZeroRuntime &runtime, const std::vector<T> &host) {
  auto *device = static_cast<T *>(runtime.allocate_device(host.size() * sizeof(T), 64));
  runtime.copy_to_device(device, host.data(), host.size() * sizeof(T));
  return device;
}

template <typename T>
std::vector<T> download(qwen3::LevelZeroRuntime &runtime, T *device,
                        std::size_t count) {
  std::vector<T> result(count);
  runtime.copy_from_device(result.data(), device, result.size() * sizeof(T));
  return result;
}

void embedding(qwen3::LevelZeroRuntime &runtime) {
  std::vector<bf16> table(qwen3::kHiddenSize);
  for (std::uint32_t i = 0; i != table.size(); ++i)
    table[i] = b((int(i % 29) - 14) / 16.0f);
  auto *d_table = device_vector(runtime, table);
  auto *output = static_cast<bf16 *>(runtime.allocate_device(table.size() * 2, 64));
  std::uint32_t token = 0;
  runtime.launch("QwenEmbedding", qwen3::kHiddenSize,
      {qwen3::kernel_argument(d_table), qwen3::kernel_argument(output),
       qwen3::kernel_argument(token)});
  assert(download(runtime, output, table.size()) == table);
}

void rms_norm(qwen3::LevelZeroRuntime &runtime) {
  for (std::uint32_t width : {128U, qwen3::kHiddenSize}) {
    std::vector<bf16> input(width), weight(width, b(0.75f));
    float squares = 0;
    for (std::uint32_t i = 0; i != width; ++i) {
      input[i] = b((int(i % 31) - 15) / 32.0f);
      squares += f(input[i]) * f(input[i]);
    }
    auto *d_input = device_vector(runtime, input);
    auto *d_weight = device_vector(runtime, weight);
    auto *output = static_cast<bf16 *>(runtime.allocate_device(width * 2, 64));
    float epsilon = 1.0e-6f;
    runtime.launch("QwenRmsNorm", 1,
        {qwen3::kernel_argument(d_input), qwen3::kernel_argument(d_weight),
         qwen3::kernel_argument(output), qwen3::kernel_argument(width),
         qwen3::kernel_argument(epsilon)});
    const auto host = download(runtime, output, width);
    const float scale = 1.0f / std::sqrt(squares / width + epsilon);
    for (std::uint32_t i = 0; i != width; ++i)
      close(f(host[i]), f(input[i]) * f(weight[i]) * scale);
  }
}

void gemv(qwen3::LevelZeroRuntime &runtime) {
  std::uint32_t rows = 32, columns = 32;
  std::vector<bf16> matrix(rows * columns), vector(columns);
  for (std::uint32_t k = 0; k != columns; ++k)
    vector[k] = b((int(k % 7) - 3) / 8.0f);
  for (std::uint32_t i = 0; i != matrix.size(); ++i)
    matrix[i] = b((int(i % 11) - 5) / 16.0f);
  auto *d_matrix = device_vector(runtime, matrix);
  auto *d_vector = device_vector(runtime, vector);
  auto *output = static_cast<bf16 *>(runtime.allocate_device(rows * 2, 64));
  runtime.launch("QwenGemv", rows,
      {qwen3::kernel_argument(d_matrix), qwen3::kernel_argument(d_vector),
       qwen3::kernel_argument(output), qwen3::kernel_argument(rows),
       qwen3::kernel_argument(columns)});
  const auto host = download(runtime, output, rows);
  for (std::uint32_t row = 0; row != rows; ++row) {
    float expected = 0;
    for (std::uint32_t k = 0; k != columns; ++k)
      expected += f(matrix[row * columns + k]) * f(vector[k]);
    close(f(host[row]), expected);
  }
}

void qkv_rope_cache_and_attention(qwen3::LevelZeroRuntime &runtime) {
  std::vector<bf16> query(qwen3::kAttentionHeads * qwen3::kHeadDim);
  std::vector<bf16> key(qwen3::kKeyValueHeads * qwen3::kHeadDim);
  std::vector<bf16> value(key.size());
  std::vector<bf16> norm(qwen3::kHeadDim, b(1.0f));
  std::vector<bf16> cosine(qwen3::kHeadDim, b(1.0f));
  std::vector<bf16> sine(qwen3::kHeadDim, b(0.0f));
  for (std::uint32_t i = 0; i != query.size(); ++i)
    query[i] = b((int(i % 17) - 8) / 16.0f);
  for (std::uint32_t i = 0; i != key.size(); ++i) {
    key[i] = b((int(i % 13) - 6) / 16.0f);
    value[i] = b((int(i % 19) - 9) / 16.0f);
  }
  auto *d_query = device_vector(runtime, query);
  auto *d_key = device_vector(runtime, key);
  auto *d_value = device_vector(runtime, value);
  auto *d_norm = device_vector(runtime, norm);
  auto *d_cos = device_vector(runtime, cosine);
  auto *d_sin = device_vector(runtime, sine);
  auto *key_cache = static_cast<bf16 *>(runtime.allocate_device(key.size() * 2, 64));
  auto *value_cache = static_cast<bf16 *>(runtime.allocate_device(value.size() * 2, 64));
  std::uint32_t position = 0, maximum = 1;
  runtime.launch("QwenQkvRopeCache", qwen3::kAttentionHeads,
      {qwen3::kernel_argument(d_query), qwen3::kernel_argument(d_key),
       qwen3::kernel_argument(d_value), qwen3::kernel_argument(d_norm),
       qwen3::kernel_argument(d_norm), qwen3::kernel_argument(d_cos),
       qwen3::kernel_argument(d_sin), qwen3::kernel_argument(key_cache),
       qwen3::kernel_argument(value_cache), qwen3::kernel_argument(position),
       qwen3::kernel_argument(maximum)});
  assert(download(runtime, value_cache, value.size()) == value);
  auto *attention = static_cast<bf16 *>(
      runtime.allocate_device(qwen3::kHiddenSize * 2, 64));
  runtime.launch("QwenAttention", qwen3::kAttentionHeads,
      {qwen3::kernel_argument(d_query), qwen3::kernel_argument(key_cache),
       qwen3::kernel_argument(value_cache), qwen3::kernel_argument(attention),
       qwen3::kernel_argument(position)});
  const auto host_attention = download(runtime, attention, qwen3::kHiddenSize);
  for (std::uint32_t head = 0; head != qwen3::kAttentionHeads; ++head)
    for (std::uint32_t i = 0; i != qwen3::kHeadDim; ++i)
      close(f(host_attention[head * qwen3::kHeadDim + i]),
            f(value[(head / 4) * qwen3::kHeadDim + i]));

  // The insertion kernel must also address the final production cache slot.
  const std::uint64_t cache_elements = std::uint64_t(qwen3::kMaximumContext) *
      qwen3::kKeyValueHeads * qwen3::kHeadDim;
  auto *last_keys = static_cast<bf16 *>(runtime.allocate_device(cache_elements * 2));
  auto *last_values = static_cast<bf16 *>(runtime.allocate_device(cache_elements * 2));
  const std::uint64_t rope_elements =
      std::uint64_t(qwen3::kMaximumContext) * qwen3::kHeadDim;
  auto *full_cos = static_cast<bf16 *>(runtime.allocate_device(rope_elements * 2));
  auto *full_sin = static_cast<bf16 *>(runtime.allocate_device(rope_elements * 2));
  const std::uint64_t rope_offset =
      std::uint64_t(qwen3::kMaximumContext - 1) * qwen3::kHeadDim;
  runtime.copy_to_device(full_cos + rope_offset, cosine.data(), cosine.size() * 2);
  runtime.copy_to_device(full_sin + rope_offset, sine.data(), sine.size() * 2);
  position = qwen3::kMaximumContext - 1;
  maximum = qwen3::kMaximumContext;
  runtime.launch("QwenQkvRopeCache", qwen3::kAttentionHeads,
      {qwen3::kernel_argument(d_query), qwen3::kernel_argument(d_key),
       qwen3::kernel_argument(d_value), qwen3::kernel_argument(d_norm),
       qwen3::kernel_argument(d_norm), qwen3::kernel_argument(full_cos),
       qwen3::kernel_argument(full_sin), qwen3::kernel_argument(last_keys),
       qwen3::kernel_argument(last_values), qwen3::kernel_argument(position),
       qwen3::kernel_argument(maximum)});
  const std::uint64_t last_offset = cache_elements - value.size();
  assert(download(runtime, last_values + last_offset, value.size()) == value);
}

void elementwise_and_argmax(qwen3::LevelZeroRuntime &runtime) {
  std::uint32_t width = qwen3::kIntermediateSize;
  std::vector<bf16> left(width), right(width);
  for (std::uint32_t i = 0; i != width; ++i) {
    left[i] = b((int(i % 11) - 5) / 8.0f);
    right[i] = b((int(i % 7) - 3) / 8.0f);
  }
  auto *d_left = device_vector(runtime, left);
  auto *d_right = device_vector(runtime, right);
  runtime.launch("QwenAdd", width / 32,
      {qwen3::kernel_argument(d_left), qwen3::kernel_argument(d_right),
       qwen3::kernel_argument(width)});
  auto host = download(runtime, d_left, width);
  for (std::uint32_t i = 0; i != width; ++i)
    close(f(host[i]), f(left[i]) + f(right[i]));
  runtime.copy_to_device(d_left, left.data(), left.size() * 2);
  runtime.launch("QwenSiluMultiply", width / 32,
      {qwen3::kernel_argument(d_left), qwen3::kernel_argument(d_right),
       qwen3::kernel_argument(width)});
  host = download(runtime, d_left, width);
  for (std::uint32_t i = 0; i != width; ++i) {
    const float gate = f(left[i]);
    close(f(host[i]), gate / (1.0f + std::exp(-gate)) * f(right[i]));
  }

  width = qwen3::kVocabularySize;
  std::vector<bf16> logits(width, b(-1.0f));
  logits[17] = logits[42] = b(3.0f);
  auto *d_logits = device_vector(runtime, logits);
  auto *d_token = static_cast<std::uint32_t *>(runtime.allocate_device(4, 64));
  runtime.launch("QwenArgmax", 1,
      {qwen3::kernel_argument(d_logits), qwen3::kernel_argument(d_token),
       qwen3::kernel_argument(width)});
  assert(download(runtime, d_token, 1)[0] == 17);
  logits[3] = b(std::numeric_limits<float>::quiet_NaN());
  runtime.copy_to_device(d_logits, logits.data(), logits.size() * 2);
  runtime.launch("QwenArgmax", 1,
      {qwen3::kernel_argument(d_logits), qwen3::kernel_argument(d_token),
       qwen3::kernel_argument(width)});
  assert(download(runtime, d_token, 1)[0] == 0xffffffffU);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string module = argc == 2 ? argv[1] : "qwen3_kernels.bin";
    qwen3::LevelZeroRuntime runtime(module);
    runtime.require_bmg_and_memory(256ULL << 20);
    embedding(runtime);
    rms_norm(runtime);
    gemv(runtime);
    qkv_rope_cache_and_attention(runtime);
    elementwise_and_argmax(runtime);
    std::cout << "Qwen3 BMG Level Zero kernel dispatch: PASS on "
              << runtime.device_name() << '\n';
  } catch (const std::exception &error) {
    std::cerr << "Qwen3 BMG dispatch test: " << error.what() << '\n';
    return 1;
  }
}
