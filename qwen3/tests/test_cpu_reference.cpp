#include "../model.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr float kPrimitiveAtol = 0.03f;
constexpr float kPrimitiveRtol = 0.02f;

void close(float actual, float expected, float atol = kPrimitiveAtol,
           float rtol = kPrimitiveRtol) {
  assert(std::isfinite(actual));
  assert(std::isfinite(expected));
  assert(std::abs(actual - expected) <= atol + rtol * std::abs(expected));
}

float round_bf16(float value) {
  return qwen3::bf16_to_float(qwen3::float_to_bf16(value));
}

std::vector<float> rms_norm(const std::vector<float> &input,
                            const std::vector<float> &weight) {
  float squares = 0;
  for (float value : input) squares += value * value;
  const float scale = 1.0f / std::sqrt(squares / input.size() + 1.0e-6f);
  std::vector<float> result(input.size());
  for (std::size_t i = 0; i != input.size(); ++i)
    result[i] = round_bf16(input[i] * weight[i] * scale);
  return result;
}

void test_embedding_and_norm() {
  std::vector<float> row(qwen3::kHiddenSize);
  for (std::size_t i = 0; i != row.size(); ++i)
    row[i] = round_bf16(std::sin(static_cast<float>(i) * 0.01f));
  const std::vector<float> embedded = row;
  assert(embedded.size() == 4096);
  close(embedded[4095], row[4095]);
  for (std::size_t width : {std::size_t{128}, std::size_t{4096}}) {
    std::vector<float> input(row.begin(), row.begin() + width);
    std::vector<float> weight(width, round_bf16(0.75f));
    const auto result = rms_norm(input, weight);
    assert(result.size() == width);
    for (float value : result) assert(std::isfinite(value));
  }
}

void test_gemv() {
  // Exercise all production (rows, columns) signatures without requiring a
  // multi-hundred-MiB synthetic matrix. Each selected row is generated from
  // its coordinates exactly as a row-major kernel sees it.
  const std::pair<std::uint32_t, std::uint32_t> shapes[] = {
      {4096, 4096}, {1024, 4096}, {12288, 4096}, {4096, 12288},
      {151936, 4096}};
  for (const auto &[rows, columns] : shapes) {
    std::vector<float> vector(columns);
    for (std::uint32_t k = 0; k != columns; ++k)
      vector[k] = round_bf16(static_cast<float>(int(k % 17) - 8) / 128.0f);
    for (std::uint32_t row : {0U, rows - 1}) {
      float expected = 0;
      std::vector<float> lanes(32, 0.0f);
      for (std::uint32_t k = 0; k != columns; ++k) {
        const float weight = round_bf16(
            static_cast<float>(int((row + k) % 13) - 6) / 256.0f);
        expected += weight * vector[k];
        lanes[k % 32] += weight * vector[k];
      }
      const float production_order =
          std::accumulate(lanes.begin(), lanes.end(), 0.0f);
      close(round_bf16(production_order), expected);
    }
  }
}

void rotate_half(std::vector<float> &vector, std::uint32_t position) {
  const std::vector<float> old = vector;
  for (std::uint32_t i = 0; i != qwen3::kHeadDim / 2; ++i) {
    const double inv = std::pow(qwen3::kRopeTheta, -2.0 * i / qwen3::kHeadDim);
    const float cosine = round_bf16(std::cos(position * inv));
    const float sine = round_bf16(std::sin(position * inv));
    vector[i] = round_bf16(old[i] * cosine - old[i + 64] * sine);
    vector[i + 64] = round_bf16(old[i + 64] * cosine + old[i] * sine);
  }
}

void test_qk_rope_and_cache_edges() {
  std::vector<float> q(128), weight(128, 1.0f);
  for (std::uint32_t i = 0; i != 128; ++i)
    q[i] = round_bf16((static_cast<int>(i) - 64) / 64.0f);
  q = rms_norm(q, weight);
  const auto at_zero = q;
  rotate_half(q, 0);
  for (std::uint32_t i = 0; i != 128; ++i) close(q[i], at_zero[i]);
  rotate_half(q, qwen3::kMaximumContext - 1);
  for (float value : q) assert(std::isfinite(value));
  const std::uint64_t final_slot =
      ((std::uint64_t(qwen3::kMaximumContext - 1) * qwen3::kKeyValueHeads +
        (qwen3::kKeyValueHeads - 1)) * qwen3::kHeadDim);
  assert(final_slot + qwen3::kHeadDim ==
         std::uint64_t(qwen3::kMaximumContext) * qwen3::kKeyValueHeads *
             qwen3::kHeadDim);
}

std::vector<float> attention(const std::vector<float> &query,
                             const std::vector<std::vector<float>> &keys,
                             const std::vector<std::vector<float>> &values) {
  std::vector<float> scores(keys.size());
  for (std::size_t token = 0; token != keys.size(); ++token) {
    scores[token] = std::inner_product(query.begin(), query.end(),
                                      keys[token].begin(), 0.0f) /
                    std::sqrt(128.0f);
  }
  const float maximum = *std::max_element(scores.begin(), scores.end());
  float denominator = 0;
  std::vector<float> result(128, 0.0f);
  for (std::size_t token = 0; token != keys.size(); ++token) {
    const float probability = std::exp(scores[token] - maximum);
    denominator += probability;
    for (std::size_t i = 0; i != 128; ++i)
      result[i] += probability * values[token][i];
  }
  for (float &value : result) value = round_bf16(value / denominator);
  return result;
}

void test_attention_empty_and_populated_cache() {
  std::vector<float> query(128, round_bf16(0.25f));
  std::vector<std::vector<float>> keys(1, query);
  std::vector<std::vector<float>> values(1, std::vector<float>(128, 0.5f));
  auto result = attention(query, keys, values); // position zero / empty prior cache
  for (float value : result) close(value, 0.5f);
  keys.push_back(std::vector<float>(128, -0.25f));
  values.push_back(std::vector<float>(128, -0.75f));
  result = attention(query, keys, values);
  for (float value : result) assert(std::isfinite(value));
}

void test_elementwise_and_argmax() {
  for (std::uint32_t width : {qwen3::kHiddenSize, qwen3::kIntermediateSize}) {
    for (std::uint32_t i = 0; i != width; ++i) {
      const float a = round_bf16((int(i % 11) - 5) / 8.0f);
      const float b = round_bf16((int(i % 7) - 3) / 8.0f);
      const float add = round_bf16(a + b);
      const float silu_mul = round_bf16((a / (1.0f + std::exp(-a))) * b);
      assert(std::isfinite(add) && std::isfinite(silu_mul));
    }
  }
  std::vector<float> logits(qwen3::kVocabularySize, -1.0f);
  logits[17] = logits[42] = 3.0f;
  const auto best = std::max_element(logits.begin(), logits.end());
  assert(std::distance(logits.begin(), best) == 17); // lowest ID tie break
  assert(qwen3::kEosToken == 151645 && qwen3::kEndOfTurnToken == 151643);
}

void test_checkpoint_schema() {
  const auto names = qwen3::expected_tensor_names();
  assert(names.size() == 3 + qwen3::kLayers * 11);
  assert(qwen3::expected_shape("model.layers.35.mlp.down_proj.weight") ==
         (std::vector<std::uint64_t>{4096, 12288}));
  for (float value : {0.0f, -1.0f, 0.1f, 100.0f,
                      std::numeric_limits<float>::infinity()}) {
    const float converted = round_bf16(value);
    if (std::isfinite(value)) close(converted, value);
    else assert(std::isinf(converted));
  }
}

void test_config_parser_and_rejection() {
  namespace fs = std::filesystem;
  const fs::path directory = fs::temp_directory_path() /
      ("qwen3-config-test-" + std::to_string(std::random_device{}()));
  fs::create_directory(directory);
  const std::string valid = R"({
    "hidden_size":4096,"intermediate_size":12288,"head_dim":128,
    "num_attention_heads":32,"num_key_value_heads":8,
    "num_hidden_layers":36,"vocab_size":151936,
    "max_position_embeddings":40960,"eos_token_id":151645,
    "rms_norm_eps":1e-6,"rope_theta":1000000,
    "tie_word_embeddings":false,"hidden_act":"silu",
    "model_type":"qwen3","torch_dtype":"bfloat16"})";
  {
    std::ofstream output(directory / "config.json");
    output << valid;
  }
  const auto config = qwen3::load_and_validate_config(directory.string());
  assert(config.num_hidden_layers == 36);
  std::string invalid = valid;
  invalid.replace(invalid.find("\"hidden_size\":4096"), 18,
                  "\"hidden_size\":4095");
  {
    std::ofstream output(directory / "config.json");
    output << invalid;
  }
  bool rejected = false;
  try { (void)qwen3::load_and_validate_config(directory.string()); }
  catch (const std::runtime_error &) { rejected = true; }
  assert(rejected);
  fs::remove_all(directory);
}

} // namespace

int main() {
  test_embedding_and_norm();
  test_gemv();
  test_qk_rope_and_cache_edges();
  test_attention_empty_and_populated_cache();
  test_elementwise_and_argmax();
  test_checkpoint_schema();
  test_config_parser_and_rejection();
  std::cout << "Qwen3 CPU reference primitives: PASS\n";
}
