#include "sampler.h"

Sampler::Sampler(const std::shared_ptr<Config> config) noexcept {
  vocab_size = config->vocab_size;
}

float Sampler::sample_prob(const int index, const InferenceState& s) const {
  const float* logits = s.logits();
  // Find max value to moderate the logits later on for numerical stability
  float max_val = std::numeric_limits<float>::min();;
  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
    }
  }
  float sum = 0;
  for (int i = 0; i < vocab_size; ++i) {
    sum += expf(logits[i] - max_val);
  }
  return expf(logits[index] - max_val) / sum;
}

int Sampler::sample_argmax(const InferenceState& s) const {
  const float* logits = s.logits();
  int argmax = 0;
  float max_val = std::numeric_limits<float>::min();
  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] > max_val) {
      max_val = logits[i];
      argmax = i;
    }
  }
  return argmax;
}
