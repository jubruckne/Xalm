#include "model.h"

#include <assert.h>
#include <cfloat>
#include <math.h>

#include "profiler.h"
#include "types.h"

#if DEBUG_MODEL
#include "fmt/format.h"
static std::map<std::string, DebugTensor> _debug_map;
std::map<std::string, DebugTensor>& debug_map_cpu() {
  return _debug_map;
}
template <typename T>
static std::vector<T> copy_debug_tensor(T* x, size_t size) {
  std::vector<T> out(size);
  for (size_t i = 0; i < size; i++) {
    out[i] = x[i];
  }
  return out;
}
template <typename T>
static void save_debug_tensor(const std::string& name, T* x, size_t size) {
  _debug_map[name] = DebugTensor(copy_debug_tensor<T>(x, size));
}
#endif


static void matmul(float* __restrict__ xout, const float* __restrict__ x, const float* __restrict__ w, const int n, const int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  profile(std::format("{}x{}", n,d));

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

static void matmul(float* xout, const float* x, const f16_t* w, const int n, const int d) {
	// W (d,n) @ x (n,) -> xout (d,)
  profile(std::format("{}x{}", n,d));

	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < d; i++) {
		float val = 0.0f;
		for (int j = 0; j < n; j++) {
			val += f16_t::to_float(w[i * n + j]) * x[j];
		}
		xout[i] = val;
	}
}

static void matmul(float* __restrict__ xout, const float* __restrict__ x, const f8e4m3_t* __restrict__ w, const int n, const int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  profile(std::format("{}x{}", n,d));

  assert(n % 16 == 0);
  assert(d % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(xout) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(x) % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(w) % 16 == 0);

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

void matmul(float* xout, const float* x, const Tensor* w, const int n, const int d) {
  switch (w->type) {
    case Type::F32: {
      matmul(xout, x, static_cast<const float*>(w->data), n, d);
      return;
    }
    case Type::F16: {
      matmul(xout, x, static_cast<const f16_t*>(w->data), n, d);
      return;
    }
    case Type::F8: {
      matmul(xout, x, static_cast<const f8e4m3_t*>(w->data), n, d);
      return;
    }
    default: {
      printf("matmul: unsupported data type: %s\n", w->type.to_string().data());
      assert(false);
    }
  }
}

void matmul(const Tensor& xout, const Tensor& a, const Tensor& b) {
	const auto n = a.shape[0];
	const auto d = a.shape[1];
	matmul(static_cast<float*>(xout.data), static_cast<const float*>(a.data), &b, n, d);
}

static void rmsnorm(float* o, const float* x, const float* weight, const int size, const float eps) {
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

[[maybe_unused]] static void layernorm(float* o, const float* x, const float* weight, const float* bias, const int size, const float eps) {
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
  float scale = 1.0f / sqrtf(var + eps);
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
static void softmax(float* o, float* x, int size) {
  profile();

  float score_max = -FLT_MAX;
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

inline float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

inline float silu(float x) {
  return x / (1.0f + expf(-x));
}

inline float clip(float x, float v) {
  return x < -v ? -v : (x > v ? v : x);
}

// TODO annotate me
static void rope(float* vec, int d, int head_dim, int pos, float theta, int rotary_dim) {
  profile();

  for (int i = 0; i < d; i += 2) {
    int j_head = i % head_dim;
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    float v0 = vec[i];
    float v1 = vec[i + 1];
    vec[i] = v0 * fcr - v1 * fci;
    vec[i + 1] = v0 * fci + v1 * fcr;
  }
}

// Compute next value in a sequence for a single causal self-attention head.
void attn(
  float* xout,    // (dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  f16_t* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  f16_t* vh,      // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
) {
  int kv_stride = n_kv_heads * head_dim; // stride per token in this kv head
  // calculate attention scores as dot products of q and k
  for (int t = 0; t < kv_len; ++t) {
    float score = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      score += qh[i] * f16_t::to_float(kh[t * kv_stride + i]);
    }
    score /= sqrtf(head_dim);
    atth[t] = score;
  }

  // softmax the scores to get attention weights over [0..kv_len)
  softmax(atth, atth, kv_len);

  // mix values with attention weights
  for (int i = 0; i < head_dim; ++i) {
    float vi = 0.0f;
    for (int t = 0; t < kv_len; ++t) {
      vi += atth[t] * f16_t::to_float(vh[t * kv_stride + i]);
    }
    xout[i] = vi;
  }
}

// Compute forward pass for a single block and update the inference state accordingly.
// PRECONDITIONS: 
// - `s.x()` contains the input to the block. Output will also go here.
// - Block KV cache is hydrated.
void Block::_block_cpu(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,        // number of sink tokens currently in the KV cache
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  profile();

  assert(_config);
  const Config& c = *_config;

  // attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      assert(rms_att_weight()->type == Type::F32);
      rmsnorm(s.xb(), s.x(), static_cast<const float *>(rms_att_weight()->data), c.dim, c.norm_eps);
      break;
    }
  }

  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  // qkv matmuls for this position
  matmul(s.q(), s.xb(), wq(), c.dim, q_dim);
  matmul(s.k(), s.xb(), wk(), c.dim, kv_dim);
  matmul(s.v(), s.xb(), wv(), c.dim, kv_dim);

  // some models require clipping qkv values
  for (int i = 0; i < q_dim; ++i) {
    s.q()[i] = clip(s.q()[i], c.qkv_clip);
  }
  for (int i = 0; i < kv_dim; ++i) {
    s.k()[i] = clip(s.k()[i], c.qkv_clip);
    s.v()[i] = clip(s.v()[i], c.qkv_clip);
  }

  // RoPE relative positional encoding: complex-valued rotate q and k in each head
  rope(s.q(), q_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  rope(s.k(), kv_dim, c.head_dim, pos, c.rope_theta, c.rotary_dim);
  
  // key and value point to the kv cache
  f16_t* kb = key_cache();
  f16_t* vb = value_cache();
  // update kv cache
  for (int i = 0; i < kv_dim; ++i) {
    kb[kv_pos * kv_dim + i] = f16_t::from_float(s.k()[i]);
    vb[kv_pos * kv_dim + i] = f16_t::from_float(s.v()[i]);
  }

  // Sink tokens remain untouched while the rest of the KV cache is incrementally 
  // replaced in ring order, but sink i must always be positioned (max_seq_len - i)
  // away from current timestep. Hence, each forward pass, rotate sink tokens 
  // forward by 1. See https://arxiv.org/abs/2309.17453 for more.
  for (int r = 0; r < kv_sink; r++) {
    for (int i = 0; i < kv_dim; ++i) {
      s.k()[i] = f16_t::to_float(kb[r * kv_dim + i]);
    }

    rope(s.k(), kv_dim, c.head_dim, 1, c.rope_theta, c.rotary_dim);

    for (int i = 0; i < kv_dim; i++) {
      kb[r * kv_dim + i] = f16_t::from_float(s.k()[i]);
    }
  }

  // Multi-head attention. Iterate over all heads.
  int q_per_kv_head = c.n_heads / c.n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < c.n_heads; h++) {
    int kv_head_offset = (h / q_per_kv_head) * c.head_dim;
    f16_t* kh = kb + kv_head_offset;
    f16_t* vh = vb + kv_head_offset;
    attn(s.xb2(h), s.att(h), s.q(h), kh, vh, c.head_dim, c.n_kv_heads, kv_len);
  }

  // final matmul to get output of the attention, using `hb` as temp storage
  matmul(s.hb(), s.xb2(), wo(), q_dim, c.dim);

  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.hb()[i];
  }
  
  // ffn pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      assert(rms_ffn_weight()->type == Type::F32);
      rmsnorm(s.xb(), s.x(), static_cast<const float *>(rms_ffn_weight()->data), c.dim, c.norm_eps);
      break;
    }
  }

  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  matmul(s.hb(), s.xb(), w1(), c.dim, c.hidden_dim);
  matmul(s.hb2(), s.xb(), w3(), c.dim, c.hidden_dim);
  switch (c.act) {
    case ActivationType::GELU: {
      for (int i = 0; i < c.hidden_dim; ++i) {
        s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
      }
      break;
    }
    case ActivationType::SILU: {
      for (int i = 0; i < c.hidden_dim; ++i) {
        s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
      }
      break;
    }
  }

  matmul(s.xb2(), s.hb(), w2(), c.hidden_dim, c.dim);
  // residual connection back into x
  for (int i = 0; i < c.dim; ++i) {
    s.x()[i] += s.xb2()[i];
  }
}

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  profile();

  // Multihead attention. Iterate over all heads.
  int q_per_kv_head = n_heads / n_kv_heads; // query heads per kv head (for MultiQueryAttention/GroupedQueryAttention)
  int h;
#pragma omp parallel for private(h)
  for (h = 0; h < n_heads; h++) {
    int kv_head_offset = (h / q_per_kv_head) * head_dim;
    f16_t* kh = kb + kv_head_offset;
    f16_t* vh = vb + kv_head_offset;
    attn(
      xout + head_dim * h, att + max_seq_len * h, q + head_dim * h, 
      kh, vh, head_dim, n_kv_heads, kv_len
    );
  }
}

void ffn_cpu(
  float* xout, float* x, 
  float* w1, float* w2, float* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  profile();

  auto* hb = new float[hidden_dim];
  auto* hb2 = new float[hidden_dim];
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
  }

  matmul(xout, hb, w2, hidden_dim, dim);
  
  delete[] hb;
  delete[] hb2;
}

void Model::_copy_embedding(const InferenceState& s, const int token) const {
  profile();

  const Config& c = *config;

  switch (token_embedding_table->type) {
    case Type::F32: {
      const auto* emb = static_cast<float*>(token_embedding_table->data);
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] = emb[token * c.dim + i];
      }
      break;
    }
    case Type::F16: {
      const auto* emb = static_cast<f16_t*>(token_embedding_table->data);
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] = f16_t::to_float(emb[token * c.dim + i]);
      }
      break;
    }
    case Type::F8: {
      const auto* emb = static_cast<f8e4m3_t*>(token_embedding_table->data);
      for (int i = 0; i < c.dim; ++i) {
        s.x()[i] = f8e4m3_t::to_float(emb[token * c.dim + i]);
      }
      break;
    }
    default: {
      assert(false && "unsupported weight dtype");
    }
  }
}

void Model::_forward_cpu(InferenceState& s, const int token, const int pos, const InferenceMode mode) const {
  const Config& c = *config;

  // copy the token embedding into `x`
  _copy_embedding(s, token);

  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

  // forward all layers in order
  for (const auto& b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    // only hydrate the KV cache and don't compute output logits
    return;
  }

  // final layer norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      assert(rms_final_weight->type == Type::F32);
      rmsnorm(s.x(), s.x(), static_cast<const float *>(rms_final_weight->data), c.dim, c.norm_eps);
      break;
    }
  }

  // classifier into logits
  matmul(s.logits(), s.x(), wcls, c.dim, c.vocab_size);
}
