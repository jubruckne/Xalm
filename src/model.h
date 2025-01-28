#pragma once

#include "tensor.h"
#include "types.h"
#include "xalm.h"

#include <cfloat>
#include <vector>

constexpr int KV_SINKS = 2;

enum class ActivationType {
	GELU,
	SILU,
};

enum class LayerNormType {
	RMSNorm,
};

enum class Device {
	CPU,
};

struct Config {
	int dim; // transformer input & output dimension
	int hidden_dim; // dimension of hidden layer in feedforward network
	int head_dim; // dimension of each attention head, usually dim / n_heads
	int n_layers; // number of layers
	int n_heads; // number of attention query heads
	int n_kv_heads; // number of key and value heads; can be < n_heads (1 is MultiQueryAttention, >1 is
					// GroupedQueryAttention)
	int vocab_size; // vocabulary size
	int max_seq_len; // max sequence length
	float rope_theta; // RoPE theta
	int rotary_dim; // dimension of rotary position encoding (elements after that don't get rotated)
	float norm_eps; // epsilon for layer normalization
	ActivationType act; // activation function
	LayerNormType norm_type; // norm type
	float qkv_clip; // clip qkv values to [-clip, clip]
	bool tie_word_embeddings; // tie input and output embeddings

	// If nonzero `context` is supplied, max sequence length is limited to `context`.
	[[nodiscard]] static Config from_xalm(Xalm::file_info& xalm, const int context = 0) {
		Config config{};
		config.dim = std::stoi(xalm.metadata.at("dim").get<std::string>());
		config.hidden_dim = std::stoi(xalm.metadata.at("hidden_dim").get<std::string>());
		config.head_dim = std::stoi(xalm.metadata.at("head_dim").get<std::string>());
		config.n_layers = std::stoi(xalm.metadata.at("n_layers").get<std::string>());
		config.n_heads = std::stoi(xalm.metadata.at("n_heads").get<std::string>());
		config.n_kv_heads = std::stoi(xalm.metadata.at("n_kv_heads").get<std::string>());
		config.vocab_size = std::stoi(xalm.metadata.at("vocab_size").get<std::string>());

		// for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly
		// specified
		config.max_seq_len = std::min(std::stoi(xalm.metadata.at("max_seq_len").get<std::string>()), 4096);
		if (context) {
			config.max_seq_len = context;
		}

		config.rope_theta = std::stof(xalm.metadata.at("rope_theta").get<std::string>());
		config.rotary_dim = std::stoi(xalm.metadata.at("rotary_dim").get<std::string>());

		config.norm_eps = std::stof(xalm.metadata.value("norm_eps", "1e-5"));

		const std::string act_str = xalm.metadata.value("act_type", "gelu");
		if (act_str == "gelu") {
			config.act = ActivationType::GELU;
		} else if (act_str == "silu") {
			config.act = ActivationType::SILU;
		} else {
			std::cerr << "unsupported act_type, defaulting to gelu" << std::endl;
			config.act = ActivationType::GELU;
		}

		const std::string norm_type_str = xalm.metadata.value("norm_type", "rmsnorm");
		if (norm_type_str == "rmsnorm") {
			config.norm_type = LayerNormType::RMSNorm;
		} else {
			std::cerr << "unsupported norm_type, defaulting to rmsnorm" << std::endl;
			config.norm_type = LayerNormType::RMSNorm;
		}

		config.qkv_clip =
				xalm.metadata.contains("qkv_clip") ? std::stof(xalm.metadata.at("qkv_clip").get<std::string>()) : FLT_MAX;

		config.tie_word_embeddings = xalm.metadata.at("tie_word_embeddings").get<std::string>() == "True";

		return config;
	}
};

// Buffer for all state used during a forward pass.
// Members are reused across subsequent blocks and passes.
// This lets us avoid allocations during inference.
struct InferenceState {
	explicit InferenceState(const Config& config): head_dim(config.head_dim), max_seq_len(config.max_seq_len) {
		_x = new float[config.dim]();
		_xb = new float[config.dim]();
		_xb2 = new float[config.dim]();
		_hb = new float[config.hidden_dim]();
		_hb2 = new float[config.hidden_dim]();
		_q = new float[config.n_heads * config.head_dim]();
		_k = new float[config.n_kv_heads * config.head_dim]();
		_v = new float[config.n_kv_heads * config.head_dim]();
		_att = new float[config.n_heads * config.max_seq_len]();
		_logits = new float[config.vocab_size]();
	}

	~InferenceState() {
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
	}

	// current activations
	float* x() const { return _x; }
	float* xb() const { return _xb; }
	float* xb(int head) const { return _xb + head_dim * head; }
	// TODO: do we need xb2?
	float* xb2() const { return _xb2; }
	float* xb2(int head) const { return _xb2 + head_dim * head; }
	float* hb() const { return _hb; }
	float* hb2() const { return _hb2; }
	float* q() const { return _q; }
	float* q(int head) const { return _q + head_dim * head; }
	float* k() const { return _k; }
	float* v() const { return _v; }
	float* att() const { return _att; }
	float* att(int head) const { return _att + max_seq_len * head; }
	// LM head
	float* logits() const { return _logits; }

private:
	int head_dim;
	int max_seq_len;
	// current activations
	float* _x = nullptr; // (dim,) - latest activation
	float* _xb = nullptr; // (dim,) - activation inside a residual branch
	// TODO: do we need xb2?
	float* _xb2 = nullptr; // (dim,) - activation inside a residual branch (second slot)
	float* _hb = nullptr; // (hidden_dim,) - buffer for hidden dimension in feedforward network
	float* _hb2 = nullptr; // (hidden_dim,) - buffer for hidden dimension in feedforward network (second slot)
	float* _q = nullptr; // (n_heads * head_dim,) - query vectors for latest timestamp
	float* _k = nullptr; // (n_kv_heads * head_dim,) - key vectors for latest timestamp
	float* _v = nullptr; // (n_kv_heads * head_dim,) - value vectors for latest timestamp
	float* _att = nullptr; // (n_heads, seq_len) - buffer for attention scores
	float* _logits = nullptr; // (vocab_size,) - final output logits
};

struct Block {
	// Move semantics
	Block(Block&& other) = default;
	Block& operator=(Block&& other) = default;

	Block(
		const int layer_i,
		Tensor& rms_att_weight, Tensor& rms_ffn_weight,
		Tensor& wq, Tensor& wk,Tensor& wv, Tensor& wo,
		Tensor& w1, Tensor& w2, Tensor& w3,
		float16_t* key_cache, float16_t* value_cache):
	layer_i(layer_i),
	rms_att_weight(std::move(rms_att_weight)),
	rms_ffn_weight(std::move(rms_ffn_weight)),
	wq(std::move(wq)),
	wk(std::move(wk)),
	wv(std::move(wv)),
	wo(std::move(wo)),
	w1(std::move(w1)),
	w2(std::move(w2)),
	w3(std::move(w3)),
	key_cache(key_cache),
	value_cache(value_cache) {}

	Block(
		const int layer_i,
		Tensor&& rms_att_weight, Tensor&& rms_ffn_weight,
		Tensor&& wq, Tensor&& wk,Tensor&& wv, Tensor&& wo,
		Tensor&& w1, Tensor&& w2, Tensor&& w3,
		float16_t* key_cache, float16_t* value_cache):
	layer_i(layer_i),
	rms_att_weight(std::move(rms_att_weight)),
	rms_ffn_weight(std::move(rms_ffn_weight)),
	wq(std::move(wq)),
	wk(std::move(wk)),
	wv(std::move(wv)),
	wo(std::move(wo)),
	w1(std::move(w1)),
	w2(std::move(w2)),
	w3(std::move(w3)),
	key_cache(key_cache),
	value_cache(value_cache) {}

	Block(const Block&) = delete;
	Block& operator=(const Block&) = delete;

	int layer_i;

	// weights for norms
	Tensor rms_att_weight; // (dim) rmsnorm weights
	Tensor rms_ffn_weight; // (dim)

	// weights for self-attention
	Tensor wq; // (n_heads * head_dim, dim)
	Tensor wk; // (n_kv_heads * head_dim, dim)
	Tensor wv; // (n_kv_heads * head_dim, dim)
	Tensor wo; // (dim, n_heads * head_dim)

	// weights for ffn
	Tensor w1; // (n_experts?, hidden_dim, dim)
	Tensor w2; // (n_experts?, dim, hidden_dim)
	Tensor w3; // (n_experts?, hidden_dim, dim) - GLU weights

	// kv cache
	float16_t* key_cache; // (seq_len, n_kv_heads * head_dim)
	float16_t* value_cache; // (seq_len, n_kv_heads * head_dim)

	// Compute forward pass for this block and update the inference state accordingly.
	// PRECONDITIONS:
	// - `s.x()` contains the input to the block. Output will also go here.
	// - Block KV cache is hydrated.
	void block(const Config& config,
		       const InferenceState& s, // inference state
			   int pos, // index of the current token in the sequence
			   int kv_sink, // number of sink tokens currently in the KV cache
			   int kv_pos, // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a
						   // ring buffer
			   int kv_len // number of tokens in the kv cache that we will attend over
	) const;

private:
	void _block_cpu(const Config& config,
					const InferenceState& s, // inference state
					int pos, // index of the current token in the sequence
					int kv_sink, // number of sink tokens currently in the KV cache
					int kv_pos, // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is
								// a ring buffer
					int kv_len // number of tokens in the kv cache that we will attend over
	) const;
};

enum class InferenceMode {
	HYDRATE_KV_CACHE, // only hydrate the KV cache and don't compute output logits
	OUTPUT_LOGITS // set InferenceState logits to logits for the next token
};

struct Model {
	static Model from_xalm(Xalm::file_info& xalm, int context);

	// Disallow copying
	Model(const Model&) = delete;
	Model& operator=(const Model&) = delete;

	Config config;

	Tensor token_embedding_table; // (vocab_size, dim)

	std::vector<Block> blocks;

	Tensor rms_final_weight; // (dim,)
	Tensor wcls; // (vocab_size, dim) classifier weights for the logits, on the last layer

	[[nodiscard]] size_t active_bytes(size_t pos) const;

	void forward(const InferenceState& s, int token, int pos, InferenceMode mode = InferenceMode::OUTPUT_LOGITS) const;

private:
	Model(const Config& config, Tensor& token_embedding_table, std::vector<Block>& blocks, Tensor& rms_final_weight, Tensor& wcls):
		config(config),
		token_embedding_table(std::move(token_embedding_table)),
		blocks(std::move(blocks)),
		rms_final_weight(std::move(rms_final_weight)),
		wcls(std::move(wcls)) {}

	void _forward_cpu(const InferenceState& s, int token, int pos, InferenceMode mode) const;
	void _copy_embedding(const InferenceState& s, int token) const;
};

////////////////////////////////////////
// Exposed for tests
////////////////////////////////////////
void attn(
		float* xout, // (dim,) - output vector
		float* atth, // (kv_len,) - scratch space to hold attention scores of the sequence
		float* qh, // (head_dim,) - query vector for this head
		float16_t*
				kh, // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
		float16_t*
				vh, // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
		int head_dim, // size of the "key-space"
		int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
		int kv_len // number of tokens of the sequence we will attend over
);

void mha_cpu(float* xout, // (n_heads, head_dim)
			 float* att, // (n_heads, max_seq_len)
			 float16_t* kb, // (max_seq_len, n_kv_heads, head_dim)
			 float16_t* vb, // (max_seq_len, n_kv_heads, head_dim)
			 float* q, // (n_heads, head_dim)
			 int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads);
void mha_cuda(float* xout, // (n_heads, head_dim)
			  float* att, // (n_heads, max_seq_len)
			  float16_t* kb, // (max_seq_len, n_kv_heads, head_dim)
			  float16_t* vb, // (max_seq_len, n_kv_heads, head_dim)
			  float* q, // (n_heads, head_dim)
			  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads);

void matmul(float* xout, const float* x, const Tensor& w, int n, int d) noexcept;
void matmul(const Tensor& xout, const Tensor& a, const Tensor& b) noexcept;
