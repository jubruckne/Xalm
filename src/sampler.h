#pragma once

#include "model.h"

#include <memory>

struct Sampler {
  int vocab_size;

  explicit Sampler(const std::shared_ptr<Config> config) noexcept;

  // Return the probability score corresponding to `logits[index]`.
  // This is equivalent to taking the softmax of the logits and returning
  // the value at index `index`.
  float sample_prob(int index, const InferenceState& s) const;
  // Return the index of the maximum value in `logits`.
  int sample_argmax(const InferenceState& s) const;
};