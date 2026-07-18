#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qwen3 {

inline constexpr std::uint32_t kHiddenSize = 4096;
inline constexpr std::uint32_t kIntermediateSize = 12288;
inline constexpr std::uint32_t kHeadDim = 128;
inline constexpr std::uint32_t kAttentionHeads = 32;
inline constexpr std::uint32_t kKeyValueHeads = 8;
inline constexpr std::uint32_t kLayers = 36;
inline constexpr std::uint32_t kVocabularySize = 151936;
inline constexpr std::uint32_t kMaximumContext = 40960;
inline constexpr std::uint32_t kEosToken = 151645;
inline constexpr std::uint32_t kEndOfTurnToken = 151643;
inline constexpr float kRmsNormEpsilon = 1.0e-6f;
inline constexpr double kRopeTheta = 1000000.0;

struct Config {
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t num_attention_heads = 0;
  std::uint32_t num_key_value_heads = 0;
  std::uint32_t num_hidden_layers = 0;
  std::uint32_t vocab_size = 0;
  std::uint32_t max_position_embeddings = 0;
  std::uint32_t eos_token_id = 0;
  float rms_norm_eps = 0;
  double rope_theta = 0;
  bool tie_word_embeddings = false;
  std::string hidden_act;
  std::string model_type;
  std::string torch_dtype;
};

struct TensorInfo {
  std::string name;
  std::string shard;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t file_offset = 0;
  std::uint64_t byte_size = 0;
  std::uint64_t arena_offset = 0;
};

struct Checkpoint {
  Config config;
  std::string model_dir;
  std::map<std::string, TensorInfo> tensors;
  std::uint64_t arena_size = 0;
};

Config load_and_validate_config(const std::string &model_dir);
Checkpoint load_and_validate_checkpoint(const std::string &model_dir);

std::vector<std::string> expected_tensor_names();
std::vector<std::uint64_t> expected_shape(const std::string &name);

// Safetensors is little-endian. This helper is also used by tests and the host
// staging path to produce architecture-independent BF16 tables.
std::uint16_t float_to_bf16(float value);
float bf16_to_float(std::uint16_t value);

} // namespace qwen3
