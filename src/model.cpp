#include "model.h"

#include <algorithm>
#include <string>
#include "fmt/format.h"
#include "json.hpp"
#include "types.h"
#include "xalm.h"

using json = nlohmann::json;

[[nodiscard]] size_t Model::active_bytes(const size_t pos) const {
	size_t bytes = 0;
	bytes += config.dim * token_embedding_table.type.bit_size / 8; // 1 row of token_embedding_table
	bytes += config.dim * rms_final_weight.type.bit_size / 8; // rms_final_weight
	bytes += config.vocab_size * config.dim * wcls.type.bit_size / 8; // wcls

	for (int l = 0; l < config.n_layers; ++l) {
		bytes += config.dim * blocks[l].rms_att_weight.type.bit_size / 8;
		bytes += config.dim * blocks[l].rms_ffn_weight.type.bit_size / 8;
		bytes += config.n_heads * config.head_dim * config.dim * blocks[l].wq.type.bit_size / 8;
		bytes += config.n_kv_heads * config.head_dim * config.dim * blocks[l].wk.type.bit_size / 8;
		bytes += config.n_kv_heads * config.head_dim * config.dim * blocks[l].wv.type.bit_size / 8;
		bytes += config.n_heads * config.head_dim * config.dim * blocks[l].wo.type.bit_size / 8;
		bytes += config.dim * config.hidden_dim * blocks[l].w1.type.bit_size / 8;
		bytes += config.dim * config.hidden_dim * blocks[l].w2.type.bit_size / 8;
		bytes += config.dim * config.hidden_dim * blocks[l].w3.type.bit_size / 8;

		const size_t kv_len = std::min(static_cast<size_t>(config.max_seq_len), pos + 1);
		size_t kv_entry_size = sizeof(float16_t);
		bytes += 2 * kv_len * config.n_kv_heads * config.head_dim * kv_entry_size; // key_cache, value_cache
	}

	return bytes;
}

void Block::block(const Config& config,
	              const InferenceState &s, // inference state
				  const int pos, // index of the current token in the sequence
				  const int kv_sink, // number of sink tokens currently in the KV cache
				  const int kv_pos, // index of the current token in the kv cache, must be in [0..kv_len) since kv cache
									// is a ring buffer
				  const int kv_len // number of tokens in the kv cache that we will attend over
) const {
	_block_cpu(config, s, pos, kv_sink, kv_pos, kv_len);
}

Model Model::from_xalm(Xalm::file_info &xalm, const int context) {
	auto config = Config::from_xalm(xalm, context);
	printf("loading model...\n");

	auto load_tensor = [&xalm](const std::string& name, const std::vector<int>& expected_shape = {}) -> Tensor {
		const auto ti = xalm.tensors.at(name);

		if (!expected_shape.empty()) {
			if (expected_shape.size() != ti.shape.size()) {
				throw std::invalid_argument(
						std::format("shape mismatch for {}: {} vs {} expected!", name, ti.shape, expected_shape));
			}
			for (int i = 0; i < expected_shape.size(); i++) {
				if (expected_shape[i] != ti.shape[i]) {
					throw std::invalid_argument(
							std::format("shape mismatch for {}: {} vs {} expected!", name, ti.shape, expected_shape));
				}
			}
		}

		auto tensor = Tensor::zeroes(ti.type, ti.shape, ti.name);
		xalm.load_tensor_data(ti, tensor.get_buffer()->span());
		return tensor;
	};

	auto token_embedding_table = load_tensor("embed.weight", {config.vocab_size, config.dim});
	std::vector<Block> blocks;
	blocks.reserve(config.n_layers);

	for (int i = 0; i < config.n_layers; ++i) {
		blocks.emplace_back(
			i,
			load_tensor(fmt::format("l.{}.attn.norm.weight", i),{config.dim}),
			load_tensor(fmt::format("l.{}.mlp.norm.weight", i),{config.dim}),
			load_tensor(fmt::format("l.{}.attn.q.weight", i),{config.n_heads * config.head_dim, config.dim}),
			load_tensor(fmt::format("l.{}.attn.k.weight", i),{config.n_kv_heads * config.head_dim, config.dim}),
			load_tensor(fmt::format("l.{}.attn.v.weight", i),{config.n_kv_heads * config.head_dim, config.dim}),
			load_tensor(fmt::format("l.{}.attn.down.weight", i),{config.dim, config.n_heads * config.head_dim}),
			load_tensor(fmt::format("l.{}.mlp.gate.weight", i),{config.hidden_dim, config.dim}),
			load_tensor(fmt::format("l.{}.mlp.down.weight", i),{config.dim, config.hidden_dim}),
			load_tensor(fmt::format("l.{}.mlp.up.weight", i),{config.hidden_dim, config.dim}),
			new float16_t[config.max_seq_len * config.n_kv_heads * config.head_dim],
			new float16_t[config.max_seq_len * config.n_kv_heads * config.head_dim]
		);
	}

	auto rms_final_weight = load_tensor("output.norm.weight",{config.dim});
	auto wcls = config.tie_word_embeddings
		? load_tensor("embed.weight",{config.vocab_size, config.dim})
		: load_tensor("output.weight",{config.vocab_size, config.dim});

	return Model{config, token_embedding_table, blocks, rms_final_weight, wcls};
}

void Model::forward(const InferenceState &s, const int token, const int pos, const InferenceMode mode) const {
	_forward_cpu(s, token, pos, mode);
}
