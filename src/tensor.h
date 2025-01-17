#pragma once

#include "json.hpp"

#include <string>
#include <unordered_map>
#include "types.h"

using json = nlohmann::json;

class Tensor {
public:
  uint8_t rank;
  std::string name;
  Type type = Type::Unknown;
  std::vector<int> shape = {};
  void* data = nullptr;
  size_t size = 0;
  size_t linear_length = 0;

  template<class ...Indices> size_t flatten_indices(Indices... indices) const requires (std::integral<Indices> && ...);

  template <typename... Indices> requires (std::unsigned_integral<Indices> && ...)
  const float& operator[](Indices... indices) const;

  template <typename... Indices> requires (std::unsigned_integral<Indices> && ...)
  std::vector<float32_t> get_row(Indices... indices) const;

  [[nodiscard]] Tensor operator*(float32_t factor) const;
  [[nodiscard]] Tensor operator-(const Tensor& other) const;

  Tensor();
  ~Tensor();

  static Tensor view(void* data, size_t size, Type type, const std::vector<int> &shape, const std::string& name = "");
  static Tensor zeroes(Type type, const std::vector<int> &shape, const std::string& name = "");
  static Tensor uniform(Type type, const std::vector<int> &shape, float min, float max, const std::string& name = "");

  [[nodiscard]] std::string format(size_t show_rows = 8, size_t show_columns = 8) const;

  template<class target_type> [[nodiscard]] Tensor convert_to() const;
  [[nodiscard]] Tensor convert_to(Type target_type) const;

  int from_json(const std::string& name, const json& val, void* bytes_ptr, size_t bytes_size);

private:
  bool mem_owned = false;
  Tensor(std::string name, Type type, const std::vector<int> &shape, void* data, size_t size, bool mem_owned);
  static size_t calculate_size(Type type, const std::vector<int> &shape);
};

struct YALMData {
  void* data = nullptr;
  std::string filename;
  size_t size;

  json metadata;

  std::unordered_map<std::string, Tensor> tensors;

  [[nodiscard]] std::string format() const;

  int from_file(const std::string& filename);
};
