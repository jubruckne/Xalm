#include "model.h"

#include <assert.h>
#include <cfloat>
#include <math.h>

#include "console.h"
#include "profiler.h"
#include "types.h"

[[maybe_unused]] static void matmul(float *__restrict__ xout, const float *__restrict__ x, const float *__restrict__ w,
									const int n, const int d) noexcept {
	// W (d,n) @ x (n,) -> xout (d,)
	profile(std::format("{}x{}", n, d));

	assert(n % 16 == 0);
	assert(d % 16 == 0);
	assert(reinterpret_cast<uintptr_t>(xout) % 16 == 0);
	assert(reinterpret_cast<uintptr_t>(x) % 16 == 0);
	assert(reinterpret_cast<uintptr_t>(w) % 16 == 0);

	for (int i = 0; i < d; i++) {
		float val = 0.0f;
#pragma omp simd
		for (int j = 0; j < n; j++) {
			val += w[i * n + j] * x[j];
		}
		xout[i] = val;
	}
}

[[maybe_unused]] static void matmul(float *xout, const float *x, const float16_t *w, const int n,
									const int d) noexcept {
	// W (d,n) @ x (n,) -> xout (d,)
	profile(std::format("{}x{}", n, d));

	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
		for (int j = 0; j < n; j++) {
			val += w[i * n + j] * x[j];
		}
		xout[i] = val;
	}
}

template<typename T>
constexpr T dynamic = -1;
template<typename T>
consteval bool is_dynamic(const T value) {
	return value == dynamic<T>;
}

template<typename>
consteval std::string_view type_name() {
	return "void";
}
template<>
consteval std::string_view type_name<float32_t>() {
	return "float32_t";
}
template<>
consteval std::string_view type_name<float16_t>() {
	return "float16_t";
}
template<>
consteval std::string_view type_name<bfloat16_t>() {
	return "bfloat16_t";
}
template<>
consteval std::string_view type_name<f8e4m3_t>() {
	return "f8e4m3_t";
}
template<>
consteval std::string_view type_name<f8e5m2_t>() {
	return "f8e5m2_t";
}
template<>
consteval std::string_view type_name<f8e3m4_t>() {
	return "f8e3m4_t";
}
template<>
consteval std::string_view type_name<f8e2m5_t>() {
	return "f8e2m5_t";
}

#include <arm_neon.h>

template<typename T>
concept RmsNormType = std::is_same_v<T, float32_t> || std::is_same_v<T, bfloat16_t>;

template<typename T>
concept NativeTypes = std::is_same_v<T, float32_t> || std::is_same_v<T, float16_t>;

template<typename T>
concept AllowedTypes = std::is_same_v<T, float32_t> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t> ||
					   std::is_same_v<T, f8e5m2_t> || std::is_same_v<T, f8e4m3_t>;

template<AllowedTypes T, int N = dynamic<int>, int D = dynamic<int>>
static void matmul(float *__restrict__ xout, const float *__restrict__ x, const T *__restrict__ w, const int n = N,
				   const int d = D) noexcept {
	// W (d,n) @ x (n,) -> xout (d,)
	profile(std::format("matmul(float*, float*, {}, {}, {})", type_name<T>(), n, d));

	assert(n % 16 == 0);
	assert(d % 16 == 0);
	assert(reinterpret_cast<uintptr_t>(xout) % 8 == 0);
	assert(reinterpret_cast<uintptr_t>(x) % 8 == 0);
	assert(reinterpret_cast<uintptr_t>(w) % 8 == 0);

	int i;
	int j;
#pragma omp parallel for private(i, j)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
#pragma omp simd
		for (j = 0; j < n; j++) {
			if constexpr (std::is_same<T, float32_t>() || std::is_same<T, float16_t>()) {
				val += w[i * n + j] * x[j];
			} else if constexpr (std::is_same<T, bfloat16_t>()) {
				val += bf16_to_f32(reinterpret_cast<const uint16_t *>(w)[i * n + j]) * x[j];
			} else {
				val += T::to_float(w[i * n + j]) * x[j];
			}
		}
		xout[i] = val;
	}
}

[[maybe_unused]] static void matmul(float *__restrict__ xout, const float *__restrict__ x,
									const f8e4m3_t *__restrict__ w, const int n, const int d) noexcept {
	// W (d,n) @ x (n,) -> xout (d,)
	profile(std::format("{}x{}", n, d));

	assert(n % 16 == 0);
	assert(d % 16 == 0);
	assert(reinterpret_cast<uintptr_t>(xout) % 8 == 0);
	assert(reinterpret_cast<uintptr_t>(x) % 8 == 0);
	assert(reinterpret_cast<uintptr_t>(w) % 8 == 0);

	int i;
	int j;
#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
#pragma omp simd
		for (j = 0; j < n; j++) {
			val += f8e4m3_t::to_float(w[i * n + j]) * x[j];
		}
		xout[i] = val;
	}
}

[[maybe_unused]] static void matmul(float *__restrict__ xout, const float *__restrict__ x,
									const f8e5m2_t *__restrict__ w, const int n, const int d) noexcept {
	// W (d,n) @ x (n,) -> xout (d,)
	profile(std::format("{}x{}", n, d));

	assert(n % 16 == 0);
	assert(d % 16 == 0);
	assert(reinterpret_cast<uintptr_t>(xout) % 8 == 0);
	assert(reinterpret_cast<uintptr_t>(x) % 8 == 0);
	assert(reinterpret_cast<uintptr_t>(w) % 8 == 0);

	int i;
	int j;
#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
#pragma omp simd
		for (j = 0; j < n; j++) {
			val += f8e5m2_t::to_float(w[i * n + j]) * x[j];
		}
		xout[i] = val;
	}
}

void matmul(float *__restrict__ xout, const float *__restrict__ x, const Tensor& w, const int n, const int d) noexcept {
	switch (w.type) {
		case Type::F32: {
			matmul<float32_t>(xout, x,w.get_buffer()->get<float32_t>(), n, d);
			return;
		}
		case Type::F16: {
			matmul<float16_t>(xout, x,w.get_buffer()->get<float16_t>(), n, d);
			return;
		}
		case Type::BF16: {
			matmul<bfloat16_t>(xout, x,w.get_buffer()->get<bfloat16_t>(), n, d);
			return;
		}
		case Type::F8_E4M3: {
			matmul<f8e4m3_t>(xout, x,w.get_buffer()->get<f8e4m3_t>(), n, d);
			return;
		}
		case Type::F8_E5M2: {
			matmul<f8e5m2_t>(xout, x, w.get_buffer()->get<f8e5m2_t>(), n, d);
			return;
		}
		default: {
			std::print("matmul: unsupported data type: {}\n", w.type.name().data());
			assert(false);
		}
	}
}

void matmul(const Tensor& xout, const Tensor& a, const Tensor& b) noexcept {
	const auto n = a.shape[0];
	const auto d = a.shape[1];
	matmul(xout.get_buffer()->get<float>(), a.get_buffer()->get<float>(), b, n, d);
}

static void rmsnorm(float* o, const float* x, const RmsNormType auto* weight, const int size, const float eps)  {
	profile();

	float rms = 0.0f;
	for (int i = 0; i < size; ++i) {
		rms += x[i] * x[i];
	}
	rms = sqrtf(rms / static_cast<float>(size) + eps);
	const float scale = 1.0f / rms;
	for (int i = 0; i < size; ++i) {
		o[i] = x[i] * scale * weight[i];
	}
}

static void rmsnorm(float* o, const float* x, const Tensor& w, const int size, const float eps) {
	switch (w.type) {
		case Type::F32: {
			rmsnorm(o, x, w.get_data<float32_t>(), size, eps);
			return;
		}
		case Type::BF16: {
			rmsnorm(o, x, w.get_data<bfloat16_t>(), size, eps);
			return;
		}
		default:
			throw std::runtime_error("rmsnorm: unsupported data type");
	}
}

[[maybe_unused]] static void layernorm(float *o, const float *x, const float *weight, const float *bias, const int size,
									   const float eps) {
	profile();

	float mean = 0.0f;
	for (int i = 0; i < size; ++i) {
		mean += x[i];
	}
	mean /= size;
	float var = 0.0f;
	for (int i = 0; i < size; ++i) {
		var += (x[i] - mean) * (x[i] - mean);
	}
	var /= size;
	const float scale = 1.0f / sqrtf(var + eps);
	if (bias) {
		for (int i = 0; i < size; ++i) {
			o[i] = (x[i] - mean) * scale * weight[i] + bias[i];
		}
	} else {
		for (int i = 0; i < size; ++i) {
			o[i] = (x[i] - mean) * scale * weight[i];
		}
	}
}

// Compute the softmax of an input vector `x` of length `size` and store it in `o`.
static void softmax(float *__restrict__ o, const float *__restrict__ x, const int size) {
	profile();

	float score_max = std::numeric_limits<float>::lowest();
	for (int i = 0; i < size; ++i) {
		if (x[i] > score_max) {
			score_max = x[i];
		}
	}
	float score_sum = 0.0f;
	for (int i = 0; i < size; ++i) {
		o[i] = expf(x[i] - score_max);
		score_sum += o[i];
	}
	for (int i = 0; i < size; ++i) {
		o[i] /= score_sum;
	}
}

inline float gelu(const float x) { return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x))); }

inline float silu(const float x) { return x / (1.0f + expf(-x)); }

inline float clip(const float x, const float v) { return x < -v ? -v : (x > v ? v : x); }

// TODO annotate me
static void rope(float *vec, const int d, const int head_dim, const int pos, const float theta, const int rotary_dim) {
	profile();

	for (int i = 0; i < d; i += 2) {
		const int j_head = i % head_dim;
		const float freq = j_head >= rotary_dim
								   ? 0.f
								   : 1.0f / powf(theta, static_cast<float>(j_head) / static_cast<float>(rotary_dim));
		const float val = pos * freq;
		const float fcr = cosf(val);
		const float fci = sinf(val);

		const float v0 = vec[i];
		const float v1 = vec[i + 1];
		vec[i] = v0 * fcr - v1 * fci;
		vec[i + 1] = v0 * fci + v1 * fcr;
	}
}

// Compute next value in a sequence for a single causal self-attention head.
void attn(float *xout, // (dim,) - output vector
		  float *atth, // (kv_len,) - scratch space to hold attention scores of the sequence
		  float *qh, // (head_dim,) - query vector for this head
		  float16_t *
				  kh, // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
		  float16_t *vh, // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV
						 // heads
		  const int head_dim, // size of the "key-space"
		  const int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is
								// GroupedQueryAttention)
		  const int kv_len // number of tokens of the sequence we will attend over
) {
	const int kv_stride = n_kv_heads * head_dim; // stride per token in this kv head
	auto const sqrt_head_dim = 1.0f / sqrtf(head_dim);
	// calculate attention scores as dot products of q and k
	for (int t = 0; t < kv_len; ++t) {
		float score = 0.0f;
		for (int i = 0; i < head_dim; ++i) {
			score += qh[i] * kh[t * kv_stride + i];
		}
		atth[t] = score * sqrt_head_dim;
	}

	// softmax the scores to get attention weights over [0..kv_len)
	softmax(atth, atth, kv_len);

	// mix values with attention weights
	for (int i = 0; i < head_dim; ++i) {
		float vi = 0.0f;
		for (int t = 0; t < kv_len; ++t) {
			vi += atth[t] * vh[t * kv_stride + i];
		}
		xout[i] = vi;
	}
}

// Compute forward pass for a single block and update the inference state accordingly.
// PRECONDITIONS:
// - `s.x()` contains the input to the block. Output will also go here.
// - Block KV cache is hydrated.
void Block::_block_cpu(const Config& config,
					   const InferenceState &s, // inference state
					   const int pos, // index of the current token in the sequence
					   const int kv_sink, // number of sink tokens currently in the KV cache
					   const int kv_pos, // index of the current token in the kv cache, must be in [0..kv_len) since kv
										 // cache is a ring buffer
					   const int kv_len // number of tokens in the kv cache that we will attend over
) const {
	profile();

	// attention pre-norm
	switch (config.norm_type) {
		case LayerNormType::RMSNorm: {
			rmsnorm(s.xb(), s.x(), rms_att_weight, config.dim, config.norm_eps);
			break;
		}
		default: {
			assert(false && "unsupported activation type");
		}
	}

	const int q_dim = config.n_heads * config.head_dim;
	const int kv_dim = config.n_kv_heads * config.head_dim;

	// qkv matmuls for this position
	matmul(s.q(), s.xb(), wq, config.dim, q_dim);
	matmul(s.k(), s.xb(), wk, config.dim, kv_dim);
	matmul(s.v(), s.xb(), wv, config.dim, kv_dim);

	// some models require clipping qkv values
	for (int i = 0; i < q_dim; ++i) {
		s.q()[i] = clip(s.q()[i], config.qkv_clip);
	}
	for (int i = 0; i < kv_dim; ++i) {
		s.k()[i] = clip(s.k()[i], config.qkv_clip);
		s.v()[i] = clip(s.v()[i], config.qkv_clip);
	}

	// key and value point to the kv cache
	float16_t *kb = key_cache;
	float16_t *vb = value_cache;

	{
		profile("rope");
		// RoPE relative positional encoding: complex-valued rotate q and k in each head
		rope(s.q(), q_dim, config.head_dim, pos, config.rope_theta, config.rotary_dim);
		rope(s.k(), kv_dim, config.head_dim, pos, config.rope_theta, config.rotary_dim);

		// update kv cache
		for (int i = 0; i < kv_dim; ++i) {
			kb[kv_pos * kv_dim + i] = s.k()[i];
			vb[kv_pos * kv_dim + i] = s.v()[i];
		}

		// Sink tokens remain untouched while the rest of the KV cache is incrementally
		// replaced in ring order, but sink i must always be positioned (max_seq_len - i)
		// away from current timestep. Hence, each forward pass, rotate sink tokens
		// forward by 1. See https://arxiv.org/abs/2309.17453 for more.

		for (int r = 0; r < kv_sink; r++) {
			for (int i = 0; i < kv_dim; ++i) {
				s.k()[i] = kb[r * kv_dim + i];
			}

			rope(s.k(), kv_dim, config.head_dim, 1, config.rope_theta, config.rotary_dim);

			for (int i = 0; i < kv_dim; i++) {
				kb[r * kv_dim + i] = s.k()[i];
			}
		}
	}

	// Multi-head attention. Iterate over all heads.
	const int q_per_kv_head =
			config.n_heads / config.n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
	int h;
#pragma omp parallel for private(h)
	for (h = 0; h < config.n_heads; h++) {
		int kv_head_offset = (h / q_per_kv_head) * config.head_dim;
		float16_t *kh = kb + kv_head_offset;
		float16_t *vh = vb + kv_head_offset;
		attn(s.xb2(h), s.att(h), s.q(h), kh, vh, config.head_dim, config.n_kv_heads, kv_len);
	}

	// final matmul to get output of the attention, using `hb` as temp storage
	matmul(s.hb(), s.xb2(), wo, q_dim, config.dim);

	// residual connection back into x
	for (int i = 0; i < config.dim; ++i) {
		s.x()[i] += s.hb()[i];
	}

	// ffn pre-norm
	switch (config.norm_type) {
		case LayerNormType::RMSNorm: {
			rmsnorm(s.xb(), s.x(), rms_ffn_weight, config.dim, config.norm_eps);
			break;
		}
		default: {
			assert(false && "unsupported norm type");
		}
	}

	{
		profile("mlp");
		// mix self.w2(F.silu(self.w1(x)) * self.w3(x))
		// Note this is a feedforward with a GLU, not a simple MLP.
		matmul(s.hb(), s.xb(), w1, config.dim, config.hidden_dim);
		matmul(s.hb2(), s.xb(), w3, config.dim, config.hidden_dim);
		{
			profile("mlp_act");
			switch (config.act) {
				case ActivationType::GELU: {
					for (int i = 0; i < config.hidden_dim; ++i) {
						s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
					}
					break;
				}
				case ActivationType::SILU: {
					for (int i = 0; i < config.hidden_dim; ++i) {
						s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
					}
					break;
				}
				default: {
					assert(false && "unsupported activation type");
				}
			}
		}

		matmul(s.xb2(), s.hb(), w2, config.hidden_dim, config.dim);
		// residual connection back into x
		for (int i = 0; i < config.dim; ++i) {
			s.x()[i] += s.xb2()[i];
		}
	}
}

void mha_cpu(float *xout, // (n_heads, head_dim)
			 float *att, // (n_heads, max_seq_len)
			 float16_t *kb, // (max_seq_len, n_kv_heads, head_dim)
			 float16_t *vb, // (max_seq_len, n_kv_heads, head_dim)
			 float *q, // (n_heads, head_dim)
			 const int head_dim, const int kv_len, const int max_seq_len, const int n_heads, const int n_kv_heads) {
	profile();

	// Multihead attention. Iterate over all heads.
	const int q_per_kv_head =
			n_heads / n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
	int h;
#pragma omp parallel for private(h)
	for (h = 0; h < n_heads; h++) {
		int kv_head_offset = (h / q_per_kv_head) * head_dim;
		float16_t *kh = kb + kv_head_offset;
		float16_t *vh = vb + kv_head_offset;
		attn(xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, kh, vh, head_dim, n_kv_heads, kv_len);
	}
}

void ffn_cpu(float *xout, const float *x, const float *w1, const float *w2, const float *w3, const int hidden_dim,
			 const int dim, const ActivationType act) {
	profile();

	auto *hb = new float[hidden_dim];
	auto *hb2 = new float[hidden_dim];
	// mix self.w2(F.silu(self.w1(x)) * self.w3(x))
	// Note this is a feedforward with a GLU, not a simple MLP.
	matmul(hb, x, w1, dim, hidden_dim);
	matmul(hb2, x, w3, dim, hidden_dim);
	switch (act) {
		case ActivationType::GELU: {
			for (int i = 0; i < hidden_dim; ++i) {
				hb[i] = gelu(hb[i]) * hb2[i];
			}
			break;
		}
		case ActivationType::SILU: {
			for (int i = 0; i < hidden_dim; ++i) {
				hb[i] = silu(hb[i]) * hb2[i];
			}
			break;
		}
		default: {
			assert(false && "unsupported activation type");
		}
	}

	matmul(xout, hb, w2, hidden_dim, dim);

	delete[] hb;
	delete[] hb2;
}

void Model::_copy_embedding(const InferenceState &s, const int token) const {
	profile();

	switch (token_embedding_table.type) {
		case Type::F32: {
			const auto *emb = token_embedding_table.get_buffer()->get<float>();
			for (int i = 0; i < config.dim; ++i) {
				s.x()[i] = emb[token * config.dim + i];
			}
			break;
		}
		case Type::F16: {
			const auto emb = token_embedding_table.get_buffer()->get<float16_t>();
			for (int i = 0; i < config.dim; ++i) {
				s.x()[i] = emb[token * config.dim + i];
			}
			break;
		}
		case Type::BF16: {
			const auto *emb = token_embedding_table.get_buffer()->get<uint16_t>();
			for (int i = 0; i < config.dim; ++i) {
				s.x()[i] = bf16_to_f32(emb[token * config.dim + i]);
			}
			break;
		}
		case Type::F8_E4M3: {
			const auto *emb = token_embedding_table.get_buffer()->get<f8e4m3_t>();
			for (int i = 0; i < config.dim; ++i) {
				s.x()[i] = f8e4m3_t::to_float(emb[token * config.dim + i]);
			}
			break;
		}
		default: {
			console::print("unsupported weight dtype: {}", token_embedding_table.type.name());
			assert(false && "unsupported weight dtype");
		}
	}
}

void Model::_forward_cpu(const InferenceState &s, const int token, const int pos, const InferenceMode mode) const {
	// copy the token embedding into `x`
	_copy_embedding(s, token);

	// When decoding past the context length, keep the first few tokens in the KV cache
	// untouched as "attention sinks" while replacing the rest in ring order.
	// See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
	const int kv_sink = pos >= config.max_seq_len ? KV_SINKS : 0;
	const int kv_pos = kv_sink + (pos - kv_sink) % (config.max_seq_len - kv_sink);
	const int kv_len = pos >= config.max_seq_len ? config.max_seq_len : pos + 1;

	// forward all layers in order
	for (const auto &b: blocks) {
		b.block(config, s, pos, kv_sink, kv_pos, kv_len);
	}

	if (mode == InferenceMode::HYDRATE_KV_CACHE) {
		// only hydrate the KV cache and don't compute output logits
		return;
	}

	// final layer norm
	switch (config.norm_type) {
		case LayerNormType::RMSNorm: {
			rmsnorm(s.x(), s.x(), rms_final_weight, config.dim, config.norm_eps);
			break;
		}
		default: {
			assert(false && "unsupported norm type");
		}
	}

	// classifier into logits
	matmul(s.logits(), s.x(), wcls, config.dim, config.vocab_size);
}
