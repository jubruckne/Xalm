#pragma once

#include "json.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include "types.h"

using json = nlohmann::json;

struct Tensor {
  std::string name;
  mutable Type dtype = Type::Unknown;
  std::array<int, 4> shape = {0, 0, 0, 0};
  void* data = nullptr;
  size_t size; // size in bytes (number of elements * element size)

  // Returns 0 if successful, other if failed
  int from_json(const std::string& name, const json& j, void* bytes_ptr, size_t bytes_size);
};

struct YALMData {
  void* data = nullptr;
  size_t size;

  json metadata;

  std::unordered_map<std::string, Tensor> tensors;

  // Initialize a YALMData object from a .yalm file which was created by `convert.py`.
  // Returns 0 if successful, other if failed
  int from_file(const std::string& filename);
};