#include "level_zero_runtime.hpp"
#include "model.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string model_dir;
  std::string input_ids;
  std::uint32_t max_new_tokens = 0;
  std::uint32_t max_sequence = qwen3::kMaximumContext;
  bool has_max_new_tokens = false;
};

[[noreturn]] void usage_error(const std::string &message) {
  throw std::runtime_error(message +
      "\nUsage: qwen3_l0 --model-dir DIR --input-ids FILE|- "
      "--max-new-tokens N [--max-seq-len N]");
}

std::uint32_t parse_u32(const std::string &text, const std::string &option,
                        bool allow_zero) {
  std::size_t consumed = 0;
  unsigned long long value = 0;
  try { value = std::stoull(text, &consumed); }
  catch (...) { usage_error("invalid value for " + option + ": " + text); }
  if (consumed != text.size() || value > std::numeric_limits<std::uint32_t>::max() ||
      (!allow_zero && value == 0))
    usage_error("invalid value for " + option + ": " + text);
  return static_cast<std::uint32_t>(value);
}

Options parse_options(int argc, char **argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "-h" || option == "--help") {
      std::cout << "Usage: " << argv[0]
                << " --model-dir DIR --input-ids FILE|- --max-new-tokens N "
                   "[--max-seq-len N]\n";
      std::exit(0);
    }
    if (index + 1 == argc)
      usage_error("missing value for " + option);
    const std::string value = argv[++index];
    if (option == "--model-dir") result.model_dir = value;
    else if (option == "--input-ids") result.input_ids = value;
    else if (option == "--max-new-tokens") {
      result.max_new_tokens = parse_u32(value, option, true);
      result.has_max_new_tokens = true;
    }
    else if (option == "--max-seq-len")
      result.max_sequence = parse_u32(value, option, false);
    else usage_error("unknown option: " + option);
  }
  if (result.model_dir.empty()) usage_error("--model-dir is required");
  if (result.input_ids.empty()) usage_error("--input-ids is required");
  if (!result.has_max_new_tokens) usage_error("--max-new-tokens is required");
  if (result.max_sequence > qwen3::kMaximumContext)
    usage_error("--max-seq-len cannot exceed 40960");
  return result;
}

std::vector<std::uint32_t> read_ids(const Options &options) {
  std::ifstream file;
  std::istream *input = &std::cin;
  if (options.input_ids != "-") {
    file.open(options.input_ids);
    if (!file)
      throw std::runtime_error("cannot open input token file: " + options.input_ids);
    input = &file;
  }
  std::vector<std::uint32_t> ids;
  std::string token;
  while (*input >> token) {
    std::size_t consumed = 0;
    unsigned long long id;
    try { id = std::stoull(token, &consumed); }
    catch (...) { throw std::runtime_error("invalid input token ID: " + token); }
    if (consumed != token.size() || id >= qwen3::kVocabularySize)
      throw std::runtime_error("input token ID is outside [0, 151935]: " + token);
    ids.push_back(static_cast<std::uint32_t>(id));
  }
  if (!input->eof())
    throw std::runtime_error("failed while reading input token IDs");
  if (ids.empty())
    throw std::runtime_error("at least one input token ID is required");
  if (ids.size() > options.max_sequence ||
      options.max_new_tokens > options.max_sequence - ids.size())
    throw std::runtime_error("prompt plus requested generation exceeds --max-seq-len");
  return ids;
}

std::string executable_directory(const char *program) {
  const std::string path(program);
  const auto slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string layer_name(std::uint32_t layer, const char *suffix) {
  return "model.layers." + std::to_string(layer) + "." + suffix;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment = 4096) {
  return (value + alignment - 1) & ~(alignment - 1);
}

class Engine {
public:
  Engine(const qwen3::Checkpoint &checkpoint, std::uint32_t maximum_sequence,
         const std::string &module_path)
      : checkpoint_(checkpoint), maximum_sequence_(maximum_sequence),
        runtime_(module_path) {
    constexpr std::uint64_t bf16_size = 2;
    cache_layer_bytes_ = static_cast<std::uint64_t>(maximum_sequence_) *
        qwen3::kKeyValueHeads * qwen3::kHeadDim * bf16_size;
    const std::uint64_t cache_bytes = cache_layer_bytes_ * qwen3::kLayers;
    const std::uint64_t rope_bytes = static_cast<std::uint64_t>(maximum_sequence_) *
        qwen3::kHeadDim * bf16_size;
    activation_bytes_ = 0;
    x_ = reserve(qwen3::kHiddenSize * bf16_size);
    norm_ = reserve(qwen3::kHiddenSize * bf16_size);
    query_ = reserve(qwen3::kAttentionHeads * qwen3::kHeadDim * bf16_size);
    key_ = reserve(qwen3::kKeyValueHeads * qwen3::kHeadDim * bf16_size);
    value_ = reserve(qwen3::kKeyValueHeads * qwen3::kHeadDim * bf16_size);
    attention_ = reserve(qwen3::kHiddenSize * bf16_size);
    temporary_ = reserve(qwen3::kHiddenSize * bf16_size);
    gate_ = reserve(qwen3::kIntermediateSize * bf16_size);
    up_ = reserve(qwen3::kIntermediateSize * bf16_size);
    logits_ = reserve(qwen3::kVocabularySize * bf16_size);
    token_ = reserve(sizeof(std::uint32_t));
    activation_bytes_ = align_up(activation_bytes_);
    const std::uint64_t required = checkpoint_.arena_size + cache_bytes * 2 +
        rope_bytes * 2 + activation_bytes_;
    runtime_.require_bmg_and_memory(required);
    std::cerr << "Qwen3-8B: " << runtime_.device_name() << ", "
              << (runtime_.device_memory() >> 20) << " MiB; allocating "
              << ((required + (1 << 20) - 1) >> 20) << " MiB\n";
    weights_ = runtime_.allocate_device(checkpoint_.arena_size);
    key_cache_ = runtime_.allocate_device(cache_bytes);
    value_cache_ = runtime_.allocate_device(cache_bytes);
    rope_cos_ = runtime_.allocate_device(rope_bytes);
    rope_sin_ = runtime_.allocate_device(rope_bytes);
    activations_ = runtime_.allocate_device(activation_bytes_);
    upload_weights();
    upload_rope();
  }

  std::uint32_t decode(std::uint32_t input_token, std::uint32_t position) {
    void *embedding = tensor("model.embed_tokens.weight");
    void *x = activation(x_);
    launch("QwenEmbedding", qwen3::kHiddenSize,
           {qwen3::kernel_argument(embedding), qwen3::kernel_argument(x),
            qwen3::kernel_argument(input_token)});
    for (std::uint32_t layer = 0; layer != qwen3::kLayers; ++layer) {
      void *norm = activation(norm_);
      rms(x, tensor(layer_name(layer, "input_layernorm.weight")), norm,
          qwen3::kHiddenSize);
      gemv(tensor(layer_name(layer, "self_attn.q_proj.weight")), norm,
           activation(query_), qwen3::kHiddenSize, qwen3::kHiddenSize);
      gemv(tensor(layer_name(layer, "self_attn.k_proj.weight")), norm,
           activation(key_), qwen3::kKeyValueHeads * qwen3::kHeadDim,
           qwen3::kHiddenSize);
      gemv(tensor(layer_name(layer, "self_attn.v_proj.weight")), norm,
           activation(value_), qwen3::kKeyValueHeads * qwen3::kHeadDim,
           qwen3::kHiddenSize);
      void *layer_keys = byte_offset(key_cache_, cache_layer_bytes_ * layer);
      void *layer_values = byte_offset(value_cache_, cache_layer_bytes_ * layer);
      void *query = activation(query_), *key = activation(key_);
      void *value = activation(value_);
      void *q_norm = tensor(layer_name(layer, "self_attn.q_norm.weight"));
      void *k_norm = tensor(layer_name(layer, "self_attn.k_norm.weight"));
      launch("QwenQkvRopeCache", qwen3::kAttentionHeads,
          {qwen3::kernel_argument(query), qwen3::kernel_argument(key),
           qwen3::kernel_argument(value), qwen3::kernel_argument(q_norm),
           qwen3::kernel_argument(k_norm), qwen3::kernel_argument(rope_cos_),
           qwen3::kernel_argument(rope_sin_), qwen3::kernel_argument(layer_keys),
           qwen3::kernel_argument(layer_values), qwen3::kernel_argument(position),
           qwen3::kernel_argument(maximum_sequence_)});
      void *attention = activation(attention_);
      launch("QwenAttention", qwen3::kAttentionHeads,
          {qwen3::kernel_argument(query), qwen3::kernel_argument(layer_keys),
           qwen3::kernel_argument(layer_values), qwen3::kernel_argument(attention),
           qwen3::kernel_argument(position)});
      gemv(tensor(layer_name(layer, "self_attn.o_proj.weight")), attention,
           activation(temporary_), qwen3::kHiddenSize, qwen3::kHiddenSize);
      add(x, activation(temporary_), qwen3::kHiddenSize);
      rms(x, tensor(layer_name(layer, "post_attention_layernorm.weight")), norm,
          qwen3::kHiddenSize);
      gemv(tensor(layer_name(layer, "mlp.gate_proj.weight")), norm,
           activation(gate_), qwen3::kIntermediateSize, qwen3::kHiddenSize);
      gemv(tensor(layer_name(layer, "mlp.up_proj.weight")), norm,
           activation(up_), qwen3::kIntermediateSize, qwen3::kHiddenSize);
      void *gate = activation(gate_), *up = activation(up_);
      std::uint32_t width = qwen3::kIntermediateSize;
      launch("QwenSiluMultiply", width / 32,
          {qwen3::kernel_argument(gate), qwen3::kernel_argument(up),
           qwen3::kernel_argument(width)});
      gemv(tensor(layer_name(layer, "mlp.down_proj.weight")), gate,
           activation(temporary_), qwen3::kHiddenSize, qwen3::kIntermediateSize);
      add(x, activation(temporary_), qwen3::kHiddenSize);
    }
    rms(x, tensor("model.norm.weight"), activation(norm_), qwen3::kHiddenSize);
    gemv(tensor("lm_head.weight"), activation(norm_), activation(logits_),
         qwen3::kVocabularySize, qwen3::kHiddenSize);
    void *logits = activation(logits_), *device_token = activation(token_);
    std::uint32_t vocabulary = qwen3::kVocabularySize;
    launch("QwenArgmax", 1,
        {qwen3::kernel_argument(logits), qwen3::kernel_argument(device_token),
         qwen3::kernel_argument(vocabulary)});
    std::uint32_t result = 0;
    runtime_.copy_from_device(&result, device_token, sizeof(result));
    if (result == std::numeric_limits<std::uint32_t>::max())
      throw std::runtime_error("model produced a NaN or infinity in vocabulary logits");
    return result;
  }

private:
  std::uint64_t reserve(std::uint64_t bytes) {
    activation_bytes_ = align_up(activation_bytes_, 64);
    const std::uint64_t result = activation_bytes_;
    activation_bytes_ += bytes;
    return result;
  }
  static void *byte_offset(void *pointer, std::uint64_t offset) {
    return static_cast<unsigned char *>(pointer) + offset;
  }
  void *activation(std::uint64_t offset) { return byte_offset(activations_, offset); }
  void *tensor(const std::string &name) {
    return byte_offset(weights_, checkpoint_.tensors.at(name).arena_offset);
  }
  void launch(const std::string &name, std::uint32_t work_items,
              const std::vector<qwen3::KernelArgument> &arguments) {
    runtime_.launch(name, work_items, arguments);
  }
  void rms(void *input, void *weight, void *output, std::uint32_t width) {
    float epsilon = qwen3::kRmsNormEpsilon;
    launch("QwenRmsNorm", 1,
        {qwen3::kernel_argument(input), qwen3::kernel_argument(weight),
         qwen3::kernel_argument(output), qwen3::kernel_argument(width),
         qwen3::kernel_argument(epsilon)});
  }
  void gemv(void *matrix, void *vector, void *output, std::uint32_t rows,
            std::uint32_t columns) {
    launch("QwenGemv", rows,
        {qwen3::kernel_argument(matrix), qwen3::kernel_argument(vector),
         qwen3::kernel_argument(output), qwen3::kernel_argument(rows),
         qwen3::kernel_argument(columns)});
  }
  void add(void *left, void *right, std::uint32_t width) {
    launch("QwenAdd", width / 32,
        {qwen3::kernel_argument(left), qwen3::kernel_argument(right),
         qwen3::kernel_argument(width)});
  }
  void upload_weights() {
    constexpr std::size_t chunk_size = 64 << 20;
    std::vector<char> chunk(chunk_size);
    std::string active_shard;
    std::ifstream shard;
    std::uint64_t uploaded = 0;
    for (const auto &[name, tensor_info] : checkpoint_.tensors) {
      (void)name;
      if (active_shard != tensor_info.shard) {
        shard.close();
        active_shard = tensor_info.shard;
        shard.open(checkpoint_.model_dir + "/" + active_shard, std::ios::binary);
        if (!shard)
          throw std::runtime_error("cannot reopen checkpoint shard " + active_shard);
      }
      shard.clear();
      shard.seekg(static_cast<std::streamoff>(tensor_info.file_offset));
      std::uint64_t remaining = tensor_info.byte_size;
      std::uint64_t destination_offset = tensor_info.arena_offset;
      while (remaining) {
        const std::size_t amount = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, chunk.size()));
        if (!shard.read(chunk.data(), amount))
          throw std::runtime_error("short read while streaming tensor " + tensor_info.name);
        runtime_.copy_to_device(byte_offset(weights_, destination_offset),
                                chunk.data(), amount);
        remaining -= amount;
        destination_offset += amount;
        uploaded += amount;
      }
    }
    std::cerr << "Qwen3-8B: streamed " << (uploaded >> 20)
              << " MiB of BF16 checkpoint weights\n";
  }
  void upload_rope() {
    const std::uint64_t elements =
        static_cast<std::uint64_t>(maximum_sequence_) * qwen3::kHeadDim;
    std::vector<std::uint16_t> cosine(elements), sine(elements);
    for (std::uint32_t position = 0; position != maximum_sequence_; ++position) {
      for (std::uint32_t dimension = 0; dimension != qwen3::kHeadDim; ++dimension) {
        const std::uint32_t frequency_index = dimension % (qwen3::kHeadDim / 2);
        const double inverse_frequency = std::pow(
            qwen3::kRopeTheta,
            -2.0 * static_cast<double>(frequency_index) / qwen3::kHeadDim);
        const double angle = static_cast<double>(position) * inverse_frequency;
        const std::uint64_t index =
            static_cast<std::uint64_t>(position) * qwen3::kHeadDim + dimension;
        cosine[index] = qwen3::float_to_bf16(static_cast<float>(std::cos(angle)));
        sine[index] = qwen3::float_to_bf16(static_cast<float>(std::sin(angle)));
      }
    }
    runtime_.copy_to_device(rope_cos_, cosine.data(), cosine.size() * 2);
    runtime_.copy_to_device(rope_sin_, sine.data(), sine.size() * 2);
  }

  const qwen3::Checkpoint &checkpoint_;
  std::uint32_t maximum_sequence_;
  qwen3::LevelZeroRuntime runtime_;
  void *weights_ = nullptr, *key_cache_ = nullptr, *value_cache_ = nullptr;
  void *rope_cos_ = nullptr, *rope_sin_ = nullptr, *activations_ = nullptr;
  std::uint64_t cache_layer_bytes_ = 0, activation_bytes_ = 0;
  std::uint64_t x_ = 0, norm_ = 0, query_ = 0, key_ = 0, value_ = 0;
  std::uint64_t attention_ = 0, temporary_ = 0, gate_ = 0, up_ = 0;
  std::uint64_t logits_ = 0, token_ = 0;
};

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::vector<std::uint32_t> prompt = read_ids(options);
    const qwen3::Checkpoint checkpoint =
        qwen3::load_and_validate_checkpoint(options.model_dir);
    const std::string module_path =
        executable_directory(argv[0]) + "/qwen3_kernels.bin";
    Engine engine(checkpoint, options.max_sequence, module_path);
    std::uint32_t next = 0;
    for (std::uint32_t position = 0; position != prompt.size(); ++position)
      next = engine.decode(prompt[position], position);
    std::vector<std::uint32_t> generated;
    for (std::uint32_t i = 0; i != options.max_new_tokens; ++i) {
      generated.push_back(next);
      if (next == qwen3::kEosToken || next == qwen3::kEndOfTurnToken)
        break;
      if (i + 1 != options.max_new_tokens)
        next = engine.decode(next, static_cast<std::uint32_t>(prompt.size() + i));
    }
    std::cout << "{\"generated_ids\":[";
    for (std::size_t i = 0; i != generated.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << generated[i];
    }
    std::cout << "]}\n";
  } catch (const std::exception &error) {
    std::cerr << "qwen3_l0: " << error.what() << '\n';
    return 1;
  }
}
