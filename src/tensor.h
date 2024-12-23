#pragma once

#include "json.hpp"

#include <array>
#include <string>
#include <unordered_map>
#include "types.h"

using json = nlohmann::json;

class Tensor {
public:
  std::string name;
  Type type = Type::Unknown;
  std::array<int, 4> shape = {0, 0, 0, 0};
  void* data = nullptr;
  size_t size = 0;
  size_t linear_length = 0;

  Tensor();
  ~Tensor();

  static Tensor view(void* data, size_t size, Type type, const std::array<int, 4>& shape, const std::string& name = "");
  static Tensor zeroes(Type type, const std::array<int, 4>& shape, const std::string& name = "");
  static Tensor uniform(Type type, const std::array<int, 4>& shape, float min, float max, const std::string& name = "");

  int from_json(const std::string& name, const json& j, void* bytes_ptr, size_t bytes_size);

private:
  bool mem_owned = false;
  Tensor(std::string name, Type type, const std::array<int, 4>& shape, void* data, size_t size, bool mem_owned);
  static size_t calculate_size(Type type, const std::array<int, 4>& shape);
};

struct YALMData {
  void* data = nullptr;
  size_t size;

  json metadata;

  std::unordered_map<std::string, Tensor> tensors;

  int from_file(const std::string& filename);
};