#pragma once

#include "tensor.h"
#include "types.h"
#include <memory>
#include <vector>

#define DEBUG_MODEL 0

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
  int dim;                  // transformer input & output dimension
  int hidden_dim;           // dimension of hidden layer in feedforward network
  int head_dim;             // dimension of each attention head, usually dim / n_heads
  int n_layers;             // number of layers
  int n_heads;              // number of attention query heads
  int n_kv_heads;           // number of key and value heads; can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int vocab_size;           // vocabulary size
  int max_seq_len;          // max sequence length
  float rope_theta;         // RoPE theta
  int rotary_dim;           // dimension of rotary position encoding (elements after that don't get rotated)
  float norm_eps;           // epsilon for layer normalization
  ActivationType act;       // activation function
  LayerNormType norm_type;  // norm type
  float qkv_clip;           // clip qkv values to [-clip, clip]
  bool tie_word_embeddings; // tie input and output embeddings

  // Data type of the weights according to config, used
  // to safety check tensor dtype at initialization time.
  // Type weight_dtype = Type::Unknown;

  // If nonzero `context` is supplied, max sequence length is limited to `context`.
  void from_yalm(YALMData& yalm, int context = 0);
  [[nodiscard]] size_t active_bytes(size_t pos) const;
};

// Buffer for all state used during a forward pass.
// Members are reused across subsequent blocks and passes.
// This lets us avoid allocations during inference.
struct InferenceState {
  InferenceState(std::shared_ptr<Config> config);
  ~InferenceState();

  // current activations
  float* x() const { return _x; }
  float* xb() const { return _xb; }
  float* xb(int head) const { return _xb + _config->head_dim * head; }
  // TODO: do we need xb2?
  float* xb2() const { return _xb2; }
  float* xb2(int head) const { return _xb2 + _config->head_dim * head; }
  float* hb() const { return _hb; }
  float* hb2() const { return _hb2; }
  float* q() const { return _q; }
  float* q(int head) const { return _q + _config->head_dim * head; }
  float* k() const { return _k; }
  float* v() const { return _v; }
  float* att() const { return _att; }
  float* att(int head) const { return _att + _config->max_seq_len * head; }
  // LM head
  float* logits() const { return _logits; }

  // void cuda();
  Device device() const { return _device; }

private:
  std::shared_ptr<Config> _config;
  Device _device = Device::CPU;

  // current activations
  float* _x = nullptr;         // (dim,) - latest activation
  float* _xb = nullptr;        // (dim,) - activation inside a residual branch
  // TODO: do we need xb2?
  float* _xb2 = nullptr;       // (dim,) - activation inside a residual branch (second slot)
  float* _hb = nullptr;        // (hidden_dim,) - buffer for hidden dimension in feedforward network
  float* _hb2 = nullptr;       // (hidden_dim,) - buffer for hidden dimension in feedforward network (second slot)
  float* _q = nullptr;         // (n_heads * head_dim,) - query vectors for latest timestamp
  float* _k = nullptr;         // (n_kv_heads * head_dim,) - key vectors for latest timestamp
  float* _v = nullptr;         // (n_kv_heads * head_dim,) - value vectors for latest timestamp
  float* _att = nullptr;       // (n_heads, seq_len) - buffer for attention scores
  float* _logits = nullptr;    // (vocab_size,) - final output logits
};

struct Block {
  Block(
    int layer_i,
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
  );
  ~Block();
  [[nodiscard]] int layer_i() const { return _layer_i; }
  [[nodiscard]] const Tensor* rms_att_weight() const { return _rms_att_weight; }
  [[nodiscard]] const Tensor* rms_ffn_weight() const { return _rms_ffn_weight; }
  [[nodiscard]] const Tensor* wq() const { return (_wq); }
  [[nodiscard]] const Tensor* wk() const { return (_wk); }
  [[nodiscard]] const Tensor* wv() const { return (_wv); }
  [[nodiscard]] const Tensor* wo() const { return (_wo); }
  [[nodiscard]] const Tensor* w1() const { return (_w1); }
  [[nodiscard]] const Tensor* w2() const { return (_w2); }
  [[nodiscard]] const Tensor* w3() const { return (_w3); }
  [[nodiscard]] float16_t* key_cache() const { return _key_cache; }
  [[nodiscard]] float16_t* value_cache() const { return _value_cache; }


  // Compute forward pass for this block and update the inference state accordingly.
  // PRECONDITIONS: 
  // - `s.x()` contains the input to the block. Output will also go here.
  // - Block KV cache is hydrated.
  void block(
    const InferenceState& s,  // inference state
    int pos,            // index of the current token in the sequence
    int kv_sink,        // number of sink tokens currently in the KV cache
    int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
    int kv_len          // number of tokens in the kv cache that we will attend over
  ) const;

  // void cuda();

private:
  void _block_cpu(
    const InferenceState& s,  // inference state
    int pos,            // index of the current token in the sequence
    int kv_sink,        // number of sink tokens currently in the KV cache
    int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
    int kv_len          // number of tokens in the kv cache that we will attend over
  ) const;

  int _layer_i = 0;

  std::shared_ptr<Config> _config;
  Device _device = Device::CPU;

  // weights for norms
  const Tensor *_rms_att_weight = nullptr; // (dim) rmsnorm weights
  const Tensor *_rms_ffn_weight = nullptr; // (dim)

  // weights for self-attention
  const Tensor *_wq = nullptr; // (n_heads * head_dim, dim)
  const Tensor *_wk = nullptr; // (n_kv_heads * head_dim, dim)
  const Tensor *_wv = nullptr; // (n_kv_heads * head_dim, dim)
  const Tensor *_wo = nullptr; // (dim, n_heads * head_dim)
  
  // weights for ffn
  const Tensor *_w1 = nullptr; // (n_experts?, hidden_dim, dim)
  const Tensor *_w2 = nullptr; // (n_experts?, dim, hidden_dim)
  const Tensor *_w3 = nullptr; // (n_experts?, hidden_dim, dim) - GLU weights

  // kv cache
  float16_t* _key_cache = nullptr;   // (seq_len, n_kv_heads * head_dim)
  float16_t* _value_cache = nullptr; // (seq_len, n_kv_heads * head_dim)
};

enum class InferenceMode {
  HYDRATE_KV_CACHE, // only hydrate the KV cache and don't compute output logits
  OUTPUT_LOGITS // set InferenceState logits to logits for the next token
};

struct Model {
  std::shared_ptr<Config> config;

  std::vector<std::shared_ptr<Block>> blocks;
  
  // token embedding table
  const Tensor *token_embedding_table = nullptr; // (vocab_size, dim)
  // final norm
  const Tensor* rms_final_weight = nullptr; // (dim,)
  // classifier weights for the logits, on the last layer
  const Tensor *wcls = nullptr; // (vocab_size, dim)

  explicit Model(YALMData& yalm, int context = 0);
  
  void forward(const InferenceState& s, int token, int pos, InferenceMode mode = InferenceMode::OUTPUT_LOGITS) const;

private:
  void _forward_cpu(const InferenceState& s, int token, int pos, InferenceMode mode) const;
  void _forward_cuda(InferenceState& s, int token, int pos, InferenceMode mode);
  void _copy_embedding(const InferenceState& s, int token) const;

  Device _device = Device::CPU;
};

#if DEBUG_MODEL
struct DebugTensor {
  enum struct DataType {
    F32,
    F16,
  };

  DebugTensor() = default;
  DebugTensor(const std::vector<float>& data);
  DebugTensor(const std::vector<f16_t>& data);
  DebugTensor& operator=(const DebugTensor& other) = default;
  float max_err(const DebugTensor& other) const;

  std::vector<float> data_f32;
  std::vector<f16_t> data_f16;
  DataType data_type;
};
std::map<std::string, DebugTensor>& debug_map_cpu();
std::map<std::string, DebugTensor>& debug_map_cuda();
#endif

////////////////////////////////////////
// Exposed for tests
////////////////////////////////////////
void attn(
  float* xout,    // (dim,) - output vector
  float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
  float* qh,      // (head_dim,) - query vector for this head
  float16_t* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
  float16_t* vh,      // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
  int head_dim,   // size of the "key-space"
  int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
  int kv_len      // number of tokens of the sequence we will attend over
);

void mha_cpu(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  float16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  float16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
);
void mha_cuda(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  float16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  float16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
);

void matmul(float* xout, const float* x, const Tensor* w, int n, int d) noexcept;
void matmul(const Tensor& xout, const Tensor& a, const Tensor& b) noexcept;
