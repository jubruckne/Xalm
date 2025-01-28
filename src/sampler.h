#pragma once

#include "model.h"

struct Sampler {
  int vocab_size;

  explicit Sampler(const Config& config) noexcept: vocab_size(config.vocab_size) {}

  // Return the probability score corresponding to `logits[index]`.
  // This is equivalent to taking the softmax of the logits and returning
  // the value at index `index`.
  [[nodiscard]] float sample_prob(int index, const InferenceState& s) const;
  // Return the index of the maximum value in `logits`.
  [[nodiscard]] int sample_argmax(const InferenceState& s) const;
};