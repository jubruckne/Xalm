#include "model.h"

#include "json.hpp"
#include <algorithm>
#include <cfloat>
#include "fmt/format.h"
#include <iostream>
#include "types.h"
#include <string>

using json = nlohmann::json;

void Config::from_yalm(YALMData& yalm, const int context) {
  dim = std::stoi(yalm.metadata.at("dim").get<std::string>());
  hidden_dim = std::stoi(yalm.metadata.at("hidden_dim").get<std::string>());
  head_dim = std::stoi(yalm.metadata.at("head_dim").get<std::string>());
  n_layers = std::stoi(yalm.metadata.at("n_layers").get<std::string>());
  n_heads = std::stoi(yalm.metadata.at("n_heads").get<std::string>());
  n_kv_heads = std::stoi(yalm.metadata.at("n_kv_heads").get<std::string>());
  vocab_size = std::stoi(yalm.metadata.at("vocab_size").get<std::string>());

  // for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly specified
  max_seq_len = std::min(std::stoi(yalm.metadata.at("max_seq_len").get<std::string>()), 4096);
  if (context) {
    max_seq_len = context;
  }

  rope_theta = std::stof(yalm.metadata.at("rope_theta").get<std::string>());
  rotary_dim = std::stoi(yalm.metadata.at("rotary_dim").get<std::string>());

  norm_eps = std::stof(yalm.metadata.value("norm_eps", "1e-5"));

  const std::string act_str = yalm.metadata.value("act_type", "gelu");
  if (act_str == "gelu") {
    act = ActivationType::GELU;
  } else if (act_str == "silu") {
    act = ActivationType::SILU;
  } else {
    std::cerr << "unsupported act_type, defaulting to gelu" << std::endl;
    act = ActivationType::GELU;
  }

  const std::string norm_type_str = yalm.metadata.value("norm_type", "rmsnorm");
  if (norm_type_str == "rmsnorm") {
    norm_type = LayerNormType::RMSNorm;
  } else {
    std::cerr << "unsupported norm_type, defaulting to rmsnorm" << std::endl;
    norm_type = LayerNormType::RMSNorm;
  }

  qkv_clip = yalm.metadata.contains("qkv_clip") ? std::stof(yalm.metadata.at("qkv_clip").get<std::string>()) : FLT_MAX;

  tie_word_embeddings = yalm.metadata.at("tie_word_embeddings").get<std::string>() == "True";
}

size_t Config::active_bytes(const size_t pos) const {
  const size_t weight_size = sizeof(float);

  size_t bytes_per_block = 0;
  bytes_per_block += 2 * dim * sizeof(float); // rms_att_weight, rms_ffn_weight
  bytes_per_block += n_heads * head_dim * dim * weight_size; // wq
  bytes_per_block += 2 * n_kv_heads * head_dim * dim * weight_size; // wk, wv
  bytes_per_block += n_heads * head_dim * dim * weight_size; // wo
  bytes_per_block += 3 * dim * hidden_dim * weight_size; // w1, w2, w3
  const size_t kv_len = std::min(static_cast<size_t>(max_seq_len), pos + 1);
  size_t kv_entry_size = sizeof(float16_t);
  bytes_per_block += 2 * kv_len * n_kv_heads * head_dim * kv_entry_size; // key_cache, value_cache

  size_t bytes = 0;
  bytes += dim * weight_size; // 1 row of token_embedding_table
  bytes += n_layers * bytes_per_block; // blocks
  bytes += dim * sizeof(float); // rms_final_weight
  bytes += vocab_size * dim * sizeof(float); // wcls

  return bytes;
}

const Tensor* check_tensor(const Tensor* tensor, const std::vector<int> &shape) {
  if (tensor == nullptr) {
    std::cerr << "FATAL: missing tensor" << std::endl;
    assert(false);
    return nullptr;
  }

  if (tensor->shape != shape) {
    throw std::invalid_argument(std::format("FATAL: tensor mismatch for {}. Expected {}, got {}", tensor->name, shape, tensor->shape));
  }

  return tensor;
}

const Tensor* get_tensor(const YALMData& yalm, const std::string& key) {
	const auto it = yalm.tensors.find(key);
	if (it == yalm.tensors.end()) {
		std::cerr << "FATAL: missing tensor: " << key << std::endl;
		assert(false);
		return nullptr;
	}
	const Tensor& tensor = it->second;
	return &tensor;
}

const Tensor* get_tensor(const YALMData& yalm, const std::string& key, const std::vector<int> &shape) {
  const auto it = yalm.tensors.find(key);
  if (it == yalm.tensors.end()) {
    std::cerr << "FATAL: missing tensor: " << key << std::endl;
    assert(false);
    return nullptr;
  }
  const Tensor& tensor = it->second;

  if (tensor.shape != shape) {
    throw std::invalid_argument(std::format("FATAL: tensor mismatch for {}. Expected {}, got {}", tensor.name, shape, tensor.shape));
  }

  return &tensor;
}

Block::Block(
  const int layer_i,
  const std::shared_ptr<Config> config,
  const Tensor* rms_att_weight,
  const Tensor* rms_ffn_weight,
  const Tensor* wq,
  const Tensor* wk,
  const Tensor* wv,
  const Tensor* wo,
  const Tensor* w1,
  const Tensor* w2,
  const Tensor* w3
) {
  _layer_i = layer_i;
  _config = config;

  _rms_att_weight = check_tensor(rms_att_weight, {config->dim});
  _rms_ffn_weight = check_tensor(rms_ffn_weight, {config->dim});

  _wq = check_tensor(wq, {config->n_heads * config->head_dim, config->dim});
  _wk = check_tensor(wk, {config->n_kv_heads * config->head_dim, config->dim});
  _wv = check_tensor(wv, {config->n_kv_heads * config->head_dim, config->dim});

  _wo = check_tensor(wo, {config->dim, config->n_heads * config->head_dim});

  _w1 = check_tensor(w1, {config->hidden_dim, config->dim});
  _w2 = check_tensor(w2, {config->dim, config->hidden_dim});
  _w3 = check_tensor(w3, {config->hidden_dim, config->dim});

  _key_cache = new float16_t[config->max_seq_len * config->n_kv_heads * config->head_dim]();
  _value_cache = new float16_t[config->max_seq_len * config->n_kv_heads * config->head_dim]();
}

Block::~Block() {
  if (_device == Device::CPU) {
    delete[] _key_cache;
    delete[] _value_cache;
  } else {
    std::cerr << "FATAL: unsupported device " << std::endl;
  }
}

void Block::block(
  const InferenceState& s,  // inference state
  const int pos,            // index of the current token in the sequence
  const int kv_sink,        // number of sink tokens currently in the KV cache
  const int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  const int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  /*
  if (_device == Device::CUDA) {
    switch (_config->weight_dtype) {
      case DType::F32: {
        _block_cuda<float>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case DType::F16: {
        _block_cuda<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for cuda");
      }
    }
  } else { */
      _block_cpu(s, pos, kv_sink, kv_pos, kv_len);
  // }

}

InferenceState::InferenceState(const std::shared_ptr<Config> config): 
  _config(config) {
  assert(config);
  _x = new float[config->dim]();
  _xb = new float[config->dim]();
  _xb2 = new float[config->dim]();
  _hb = new float[config->hidden_dim]();
  _hb2 = new float[config->hidden_dim]();
  _q = new float[config->n_heads * config->head_dim]();
  _k = new float[config->n_kv_heads * config->head_dim]();
  _v = new float[config->n_kv_heads * config->head_dim]();
  _att = new float[config->n_heads * config->max_seq_len]();
  _logits = new float[config->vocab_size]();
}

InferenceState::~InferenceState() {
  if (_device == Device::CPU) {
    delete[] _x;
    delete[] _xb;
    delete[] _xb2;
    delete[] _hb;
    delete[] _hb2;
    delete[] _q;
    delete[] _k;
    delete[] _v;
    delete[] _att;
    delete[] _logits;
  } else {
    std::cerr << "FATAL: unsupported device " << std::endl;
  }
}

Model::Model(YALMData& yalm, const int context) {
  config = std::make_shared<Config>();
  config->from_yalm(yalm, context);
  printf("loading model...\n");

  token_embedding_table = get_tensor(yalm, "embed.weight", {config->vocab_size, config->dim});

  for (int i = 0; i < config->n_layers; ++i) {
    blocks.emplace_back(std::make_shared<Block>(
      i,
      config,
      get_tensor(yalm, fmt::format("l.{}.attn.norm.weight", i)),
      get_tensor(yalm, fmt::format("l.{}.mlp.norm.weight", i)),
      get_tensor(yalm, fmt::format("l.{}.attn.q.weight", i)),
      get_tensor(yalm, fmt::format("l.{}.attn.k.weight", i)),
      get_tensor(yalm, fmt::format("l.{}.attn.v.weight", i)),
      get_tensor(yalm, fmt::format("l.{}.attn.down.weight", i)),
      get_tensor(yalm, fmt::format("l.{}.mlp.gate.weight", i)),
      get_tensor(yalm, fmt::format("l.{}.mlp.down.weight", i)),
      get_tensor(yalm, fmt::format("l.{}.mlp.up.weight", i))
    ));
  }

  rms_final_weight = get_tensor(yalm, "output.norm.weight", {config->dim});

  if (config->tie_word_embeddings) {
    wcls = token_embedding_table;
  } else {
    wcls = get_tensor(yalm, "output.weight", {config->vocab_size, config->dim});
  }
}

void Model::forward(const InferenceState& s, const int token, const int pos, const InferenceMode mode) const {
  if (s.device() != _device) {
    std::cerr << "FATAL: inference state device mismatch" << std::endl;
    assert(false);
    return;
  }
  //if (_device == Device::CUDA) {
  //  _forward_cuda(s, token, pos, mode);
  //} else {
    _forward_cpu(s, token, pos, mode);
  //}
}
