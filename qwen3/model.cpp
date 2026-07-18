#include "model.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qwen3 {
namespace {

struct Json {
  enum class Kind { null, boolean, number, string, array, object } kind;
  bool boolean = false;
  double number = 0;
  std::string string;
  std::vector<Json> array;
  std::map<std::string, Json> object;

  explicit Json(Kind kind_value, bool boolean_value = false)
      : kind(kind_value), boolean(boolean_value) {}

  const Json &at(const std::string &key) const {
    const auto found = object.find(key);
    if (kind != Kind::object || found == object.end())
      throw std::runtime_error("missing JSON field '" + key + "'");
    return found->second;
  }
  const Json *find(const std::string &key) const {
    if (kind != Kind::object)
      return nullptr;
    const auto found = object.find(key);
    return found == object.end() ? nullptr : &found->second;
  }
};

class JsonParser {
public:
  explicit JsonParser(std::string input) : input_(std::move(input)) {}

  Json parse() {
    Json value = parse_value();
    whitespace();
    if (position_ != input_.size())
      fail("trailing content");
    return value;
  }

private:
  [[noreturn]] void fail(const std::string &message) const {
    throw std::runtime_error("invalid JSON at byte " +
                             std::to_string(position_) + ": " + message);
  }
  void whitespace() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\n' ||
            input_[position_] == '\r' || input_[position_] == '\t'))
      ++position_;
  }
  char take() {
    if (position_ == input_.size())
      fail("unexpected end of input");
    return input_[position_++];
  }
  Json parse_value() {
    whitespace();
    if (position_ == input_.size())
      fail("expected value");
    switch (input_[position_]) {
    case '{': return parse_object();
    case '[': return parse_array();
    case '"': {
      Json value{Json::Kind::string};
      value.string = parse_string();
      return value;
    }
    case 't': return literal("true", Json{Json::Kind::boolean, true});
    case 'f': return literal("false", Json{Json::Kind::boolean, false});
    case 'n': return literal("null", Json{Json::Kind::null});
    default: return parse_number();
    }
  }
  Json literal(const char *word, Json value) {
    const std::size_t length = std::strlen(word);
    if (input_.compare(position_, length, word) != 0)
      fail(std::string("expected ") + word);
    position_ += length;
    return value;
  }
  std::string parse_string() {
    if (take() != '"')
      fail("expected string");
    std::string result;
    while (true) {
      const char c = take();
      if (c == '"')
        return result;
      if (static_cast<unsigned char>(c) < 0x20)
        fail("control byte in string");
      if (c != '\\') {
        result += c;
        continue;
      }
      const char escaped = take();
      switch (escaped) {
      case '"': result += '"'; break;
      case '\\': result += '\\'; break;
      case '/': result += '/'; break;
      case 'b': result += '\b'; break;
      case 'f': result += '\f'; break;
      case 'n': result += '\n'; break;
      case 'r': result += '\r'; break;
      case 't': result += '\t'; break;
      case 'u': {
        // Names used by Qwen checkpoints are ASCII. Preserve other JSON
        // strings as UTF-8 so diagnostics/config metadata remain usable.
        unsigned code = 0;
        for (int i = 0; i != 4; ++i) {
          const char h = take();
          code <<= 4;
          if (h >= '0' && h <= '9') code += h - '0';
          else if (h >= 'a' && h <= 'f') code += h - 'a' + 10;
          else if (h >= 'A' && h <= 'F') code += h - 'A' + 10;
          else fail("invalid Unicode escape");
        }
        if (code < 0x80) result += static_cast<char>(code);
        else if (code < 0x800) {
          result += static_cast<char>(0xc0 | (code >> 6));
          result += static_cast<char>(0x80 | (code & 0x3f));
        } else {
          result += static_cast<char>(0xe0 | (code >> 12));
          result += static_cast<char>(0x80 | ((code >> 6) & 0x3f));
          result += static_cast<char>(0x80 | (code & 0x3f));
        }
        break;
      }
      default: fail("invalid string escape");
      }
    }
  }
  Json parse_number() {
    const std::size_t begin = position_;
    if (input_[position_] == '-') ++position_;
    while (position_ < input_.size() && input_[position_] >= '0' &&
           input_[position_] <= '9') ++position_;
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') ++position_;
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) ++position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') ++position_;
    }
    if (begin == position_)
      fail("expected number");
    const std::string text = input_.substr(begin, position_ - begin);
    char *end = nullptr;
    errno = 0;
    const double number = std::strtod(text.c_str(), &end);
    if (errno != 0 || end != text.c_str() + text.size())
      fail("invalid number");
    Json value{Json::Kind::number};
    value.number = number;
    return value;
  }
  Json parse_array() {
    take();
    Json value{Json::Kind::array};
    whitespace();
    if (position_ < input_.size() && input_[position_] == ']') {
      ++position_; return value;
    }
    while (true) {
      value.array.push_back(parse_value());
      whitespace();
      const char separator = take();
      if (separator == ']') return value;
      if (separator != ',') fail("expected ',' or ']'");
    }
  }
  Json parse_object() {
    take();
    Json value{Json::Kind::object};
    whitespace();
    if (position_ < input_.size() && input_[position_] == '}') {
      ++position_; return value;
    }
    while (true) {
      whitespace();
      if (position_ == input_.size() || input_[position_] != '"')
        fail("expected object key");
      std::string key = parse_string();
      whitespace();
      if (take() != ':') fail("expected ':'");
      if (!value.object.emplace(std::move(key), parse_value()).second)
        fail("duplicate object key");
      whitespace();
      const char separator = take();
      if (separator == '}') return value;
      if (separator != ',') fail("expected ',' or '}'");
    }
  }

  std::string input_;
  std::size_t position_ = 0;
};

std::string read_file(const std::string &path, std::uint64_t limit = 1 << 24) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open " + path);
  const auto length = input.tellg();
  if (length < 0 || static_cast<std::uint64_t>(length) > limit)
    throw std::runtime_error("invalid or unexpectedly large JSON file: " + path);
  std::string contents(static_cast<std::size_t>(length), '\0');
  input.seekg(0);
  if (!input.read(contents.data(), length))
    throw std::runtime_error("cannot read " + path);
  return contents;
}

std::uint32_t unsigned_field(const Json &root, const std::string &name) {
  const Json &value = root.at(name);
  if (value.kind != Json::Kind::number || value.number < 0 ||
      value.number > std::numeric_limits<std::uint32_t>::max() ||
      value.number != static_cast<std::uint32_t>(value.number))
    throw std::runtime_error("config field '" + name + "' must be an unsigned integer");
  return static_cast<std::uint32_t>(value.number);
}
double number_field(const Json &root, const std::string &name) {
  const Json &value = root.at(name);
  if (value.kind != Json::Kind::number)
    throw std::runtime_error("config field '" + name + "' must be numeric");
  return value.number;
}
std::string string_field(const Json &root, const std::string &name) {
  const Json &value = root.at(name);
  if (value.kind != Json::Kind::string)
    throw std::runtime_error("config field '" + name + "' must be a string");
  return value.string;
}
bool bool_field(const Json &root, const std::string &name) {
  const Json &value = root.at(name);
  if (value.kind != Json::Kind::boolean)
    throw std::runtime_error("config field '" + name + "' must be Boolean");
  return value.boolean;
}

std::uint64_t checked_product(const std::vector<std::uint64_t> &shape) {
  std::uint64_t value = 1;
  for (std::uint64_t dimension : shape) {
    if (dimension == 0 || value > std::numeric_limits<std::uint64_t>::max() / dimension)
      throw std::runtime_error("invalid tensor shape");
    value *= dimension;
  }
  return value;
}

std::uint64_t file_size(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file)
    throw std::runtime_error("cannot open checkpoint shard " + path);
  const auto size = file.tellg();
  if (size < 0)
    throw std::runtime_error("cannot determine shard size " + path);
  return static_cast<std::uint64_t>(size);
}

std::uint64_t read_u64_le(std::ifstream &file, const std::string &path) {
  unsigned char bytes[8];
  if (!file.read(reinterpret_cast<char *>(bytes), sizeof(bytes)))
    throw std::runtime_error("truncated safetensors header in " + path);
  std::uint64_t result = 0;
  for (unsigned i = 0; i != 8; ++i)
    result |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
  return result;
}

std::map<std::string, TensorInfo>
read_shard_headers(const std::string &model_dir,
                   const std::map<std::string, std::string> &weight_map) {
  std::map<std::string, std::set<std::string>> by_shard;
  for (const auto &[name, shard] : weight_map)
    by_shard[shard].insert(name);
  std::map<std::string, TensorInfo> result;
  for (const auto &[shard, indexed_names] : by_shard) {
    const std::string path = model_dir + "/" + shard;
    const std::uint64_t size = file_size(path);
    std::ifstream file(path, std::ios::binary);
    const std::uint64_t header_length = read_u64_le(file, path);
    if (header_length == 0 || header_length > (1ULL << 30) ||
        header_length > size - std::min<std::uint64_t>(size, 8))
      throw std::runtime_error("invalid safetensors header length in " + path);
    std::string header(static_cast<std::size_t>(header_length), '\0');
    if (!file.read(header.data(), static_cast<std::streamsize>(header_length)))
      throw std::runtime_error("truncated safetensors header in " + path);
    const Json root = JsonParser(std::move(header)).parse();
    if (root.kind != Json::Kind::object)
      throw std::runtime_error("safetensors header is not an object: " + path);
    for (const auto &[header_name, entry] : root.object) {
      (void)entry;
      if (header_name != "__metadata__" && !indexed_names.count(header_name))
        throw std::runtime_error("shard contains tensor absent from index: " +
                                 header_name);
    }
    std::vector<std::pair<std::pair<std::uint64_t, std::uint64_t>, std::string>>
        ranges;
    for (const std::string &name : indexed_names) {
      const Json *entry = root.find(name);
      if (!entry)
        throw std::runtime_error("index names tensor absent from shard: " + name);
      TensorInfo tensor;
      tensor.name = name;
      tensor.shard = shard;
      tensor.dtype = string_field(*entry, "dtype");
      const Json &shape = entry->at("shape");
      const Json &offsets = entry->at("data_offsets");
      if (shape.kind != Json::Kind::array || offsets.kind != Json::Kind::array ||
          offsets.array.size() != 2)
        throw std::runtime_error("invalid safetensors metadata for " + name);
      for (const Json &dimension : shape.array) {
        if (dimension.kind != Json::Kind::number || dimension.number <= 0 ||
            dimension.number != static_cast<std::uint64_t>(dimension.number))
          throw std::runtime_error("invalid shape for tensor " + name);
        tensor.shape.push_back(static_cast<std::uint64_t>(dimension.number));
      }
      const Json &begin = offsets.array[0], &end = offsets.array[1];
      if (begin.kind != Json::Kind::number || end.kind != Json::Kind::number ||
          begin.number < 0 || end.number < begin.number ||
          begin.number != static_cast<std::uint64_t>(begin.number) ||
          end.number != static_cast<std::uint64_t>(end.number))
        throw std::runtime_error("invalid data offsets for tensor " + name);
      const std::uint64_t relative_begin = static_cast<std::uint64_t>(begin.number);
      const std::uint64_t relative_end = static_cast<std::uint64_t>(end.number);
      tensor.file_offset = 8 + header_length + relative_begin;
      tensor.byte_size = relative_end - relative_begin;
      if (tensor.file_offset > size || tensor.byte_size > size - tensor.file_offset)
        throw std::runtime_error("tensor extends beyond shard: " + name);
      if (tensor.dtype != "BF16")
        throw std::runtime_error("tensor " + name + " has dtype " + tensor.dtype +
                                 "; Qwen3 inference requires BF16");
      if (checked_product(tensor.shape) * 2 != tensor.byte_size)
        throw std::runtime_error("byte size does not match BF16 shape for " + name);
      ranges.push_back({{tensor.file_offset, tensor.file_offset + tensor.byte_size},
                        name});
      result.emplace(name, std::move(tensor));
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i != ranges.size(); ++i)
      if (ranges[i].first.first < ranges[i - 1].first.second)
        throw std::runtime_error("overlapping safetensors payloads: " +
                                 ranges[i - 1].second + " and " + ranges[i].second);
  }
  return result;
}

void require_equal(const std::string &name, std::uint64_t actual,
                   std::uint64_t expected) {
  if (actual != expected)
    throw std::runtime_error("unsupported Qwen3-8B config: " + name + " is " +
                             std::to_string(actual) + ", expected " +
                             std::to_string(expected));
}

} // namespace

Config load_and_validate_config(const std::string &model_dir) {
  const Json root = JsonParser(read_file(model_dir + "/config.json")).parse();
  Config config;
  config.hidden_size = unsigned_field(root, "hidden_size");
  config.intermediate_size = unsigned_field(root, "intermediate_size");
  config.head_dim = unsigned_field(root, "head_dim");
  config.num_attention_heads = unsigned_field(root, "num_attention_heads");
  config.num_key_value_heads = unsigned_field(root, "num_key_value_heads");
  config.num_hidden_layers = unsigned_field(root, "num_hidden_layers");
  config.vocab_size = unsigned_field(root, "vocab_size");
  config.max_position_embeddings = unsigned_field(root, "max_position_embeddings");
  config.eos_token_id = unsigned_field(root, "eos_token_id");
  config.rms_norm_eps = static_cast<float>(number_field(root, "rms_norm_eps"));
  config.rope_theta = number_field(root, "rope_theta");
  config.tie_word_embeddings = bool_field(root, "tie_word_embeddings");
  config.hidden_act = string_field(root, "hidden_act");
  config.model_type = string_field(root, "model_type");
  config.torch_dtype = string_field(root, "torch_dtype");

  require_equal("hidden_size", config.hidden_size, kHiddenSize);
  require_equal("intermediate_size", config.intermediate_size, kIntermediateSize);
  require_equal("head_dim", config.head_dim, kHeadDim);
  require_equal("num_attention_heads", config.num_attention_heads, kAttentionHeads);
  require_equal("num_key_value_heads", config.num_key_value_heads, kKeyValueHeads);
  require_equal("num_hidden_layers", config.num_hidden_layers, kLayers);
  require_equal("vocab_size", config.vocab_size, kVocabularySize);
  if (config.max_position_embeddings < kMaximumContext)
    throw std::runtime_error("checkpoint context is shorter than 40960 tokens");
  if (config.eos_token_id != kEosToken || config.rms_norm_eps != kRmsNormEpsilon ||
      config.rope_theta != kRopeTheta || config.tie_word_embeddings ||
      config.hidden_act != "silu" || config.model_type != "qwen3" ||
      config.torch_dtype != "bfloat16")
    throw std::runtime_error("checkpoint is not the supported BF16 Qwen3-8B architecture");
  return config;
}

std::vector<std::string> expected_tensor_names() {
  std::vector<std::string> names{"model.embed_tokens.weight", "model.norm.weight",
                                 "lm_head.weight"};
  const char *suffixes[] = {
      "input_layernorm.weight", "post_attention_layernorm.weight",
      "self_attn.q_proj.weight", "self_attn.k_proj.weight",
      "self_attn.v_proj.weight", "self_attn.o_proj.weight",
      "self_attn.q_norm.weight", "self_attn.k_norm.weight",
      "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"};
  for (std::uint32_t layer = 0; layer != kLayers; ++layer)
    for (const char *suffix : suffixes)
      names.push_back("model.layers." + std::to_string(layer) + "." + suffix);
  return names;
}

std::vector<std::uint64_t> expected_shape(const std::string &name) {
  if (name == "model.embed_tokens.weight" || name == "lm_head.weight")
    return {kVocabularySize, kHiddenSize};
  if (name == "model.norm.weight") return {kHiddenSize};
  const auto suffix_at = name.find('.', std::string("model.layers.").size());
  const std::string suffix = suffix_at == std::string::npos ? "" : name.substr(suffix_at + 1);
  if (suffix == "input_layernorm.weight" || suffix == "post_attention_layernorm.weight")
    return {kHiddenSize};
  if (suffix == "self_attn.q_proj.weight" || suffix == "self_attn.o_proj.weight")
    return {kHiddenSize, kHiddenSize};
  if (suffix == "self_attn.k_proj.weight" || suffix == "self_attn.v_proj.weight")
    return {kKeyValueHeads * kHeadDim, kHiddenSize};
  if (suffix == "self_attn.q_norm.weight" || suffix == "self_attn.k_norm.weight")
    return {kHeadDim};
  if (suffix == "mlp.gate_proj.weight" || suffix == "mlp.up_proj.weight")
    return {kIntermediateSize, kHiddenSize};
  if (suffix == "mlp.down_proj.weight")
    return {kHiddenSize, kIntermediateSize};
  throw std::runtime_error("unknown Qwen3 tensor name: " + name);
}

Checkpoint load_and_validate_checkpoint(const std::string &model_dir) {
  Checkpoint checkpoint;
  checkpoint.model_dir = model_dir;
  checkpoint.config = load_and_validate_config(model_dir);
  const Json index = JsonParser(read_file(model_dir + "/model.safetensors.index.json")).parse();
  const Json &metadata = index.at("metadata");
  const Json &declared_size_json = metadata.at("total_size");
  if (declared_size_json.kind != Json::Kind::number ||
      declared_size_json.number < 0 ||
      declared_size_json.number !=
          static_cast<std::uint64_t>(declared_size_json.number))
    throw std::runtime_error("safetensors index total_size must be an unsigned integer");
  const std::uint64_t declared_size =
      static_cast<std::uint64_t>(declared_size_json.number);
  const Json &map = index.at("weight_map");
  if (map.kind != Json::Kind::object)
    throw std::runtime_error("safetensors index weight_map is not an object");
  std::map<std::string, std::string> weight_map;
  for (const auto &[name, shard] : map.object) {
    if (shard.kind != Json::Kind::string || shard.string.empty() ||
        shard.string.find('/') != std::string::npos)
      throw std::runtime_error("invalid shard name for tensor " + name);
    weight_map.emplace(name, shard.string);
  }
  const std::vector<std::string> expected = expected_tensor_names();
  const std::set<std::string> expected_set(expected.begin(), expected.end());
  for (const auto &[name, shard] : weight_map)
    if (!expected_set.count(name))
      throw std::runtime_error("unexpected tensor in checkpoint: " + name);
  for (const std::string &name : expected)
    if (!weight_map.count(name))
      throw std::runtime_error("checkpoint is missing tensor: " + name);
  checkpoint.tensors = read_shard_headers(model_dir, weight_map);
  constexpr std::uint64_t alignment = 4096;
  std::uint64_t payload_size = 0;
  for (const std::string &name : expected) {
    TensorInfo &tensor = checkpoint.tensors.at(name);
    if (tensor.shape != expected_shape(name))
      throw std::runtime_error("wrong shape for tensor " + name);
    checkpoint.arena_size = (checkpoint.arena_size + alignment - 1) & ~(alignment - 1);
    tensor.arena_offset = checkpoint.arena_size;
    checkpoint.arena_size += tensor.byte_size;
    payload_size += tensor.byte_size;
  }
  if (payload_size != declared_size)
    throw std::runtime_error("safetensors index total_size does not match tensor payloads");
  return checkpoint;
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16);
}
float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

} // namespace qwen3
