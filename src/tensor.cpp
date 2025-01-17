#include "tensor.h"

#include <fcntl.h>
#include <iostream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "table.h"
#include "types.h"
#include "stats.h"
#include "console.h"

template <typename... Indices>
size_t Tensor::flatten_indices(Indices... indices) const requires (std::integral<Indices> && ...) {
  static_assert(sizeof...(indices) > 0, "At least one index must be provided.");
  if (sizeof...(indices) >= rank) {
    throw std::invalid_argument("Number of indices does not match tensor rank.");
  }

  std::array<size_t, sizeof...(indices)> indices_ = {static_cast<size_t>(indices)...};
  constexpr auto indices_rank = sizeof...(indices);

  size_t offset = 0;
  size_t stride = 1;

  for (int i = rank - 1; i >= 0; --i) {
    if (i < indices_rank) {
      if (indices_[i] >= static_cast<size_t>(shape[i])) {
        throw std::out_of_range(std::format("Index {} out of bounds({}).", indices_[i], shape[i]));
      }
      offset += indices_[i] * stride;
    }
    stride *= shape[i];
  }
  return offset;
}

template<typename ... Indices> requires (std::unsigned_integral<Indices> && ...)
const float & Tensor::operator[](Indices... indices) const {
  auto idx = flatten_indices(indices...);
  return type.get_float(data, idx);
}

template <typename... Indices> requires (std::unsigned_integral<Indices> && ...)
std::vector<float32_t> Tensor::get_row(Indices... indices) const {
  if (sizeof...(indices) != rank - 1) {
    throw std::invalid_argument("Number of indices does must match rank - 1!");
  }
  auto idx = flatten_indices(indices...);
  auto values = std::vector<float32_t>(shape[rank - 1]);
  for (size_t i = 0; i < shape[rank - 1]; ++i) {
    values[i] = type.get_float(data, idx + i);
  }
  return values;
}

size_t Tensor::calculate_size(const Type type, const std::vector<int> &shape) {
  size_t num_elements = 1;
  for (const int dim : shape) {
    if (dim == 0) break;
    num_elements *= dim;
  }
  return num_elements * (type.bit_size / 8);
}

Tensor Tensor::view(void* data, const size_t size, const Type type, const std::vector<int> &shape, const std::string& name) {
  if (const size_t expected_size = calculate_size(type, shape); size != expected_size) {
    throw std::invalid_argument("External data size does not match expected size for the given shape and type.");
  }
  return Tensor(name, type, shape, data, size, false);
}

Tensor Tensor::zeroes(const Type type, const std::vector<int> &shape, const std::string &name) {
  const size_t size = calculate_size(type, shape);
  void* ptr = std::aligned_alloc(16, size);
  if (!ptr) {
    throw std::bad_alloc();
  }

  return Tensor(name, type, shape, ptr, size, true);
}

Tensor Tensor::uniform(const Type type, const std::vector<int> &shape, const float min, const float max, const std::string& name) {
  auto t = Tensor::zeroes(type, shape, name);

  std::default_random_engine generator(42);
  std::uniform_real_distribution distribution(min, max);

  switch (type) {
    case Type::F32: {
      auto* ptr = static_cast<float*>(t.data);
      for (size_t i = 0; i < t.linear_length; ++i) {
        ptr[i] = distribution(generator);
      }
      break;
    }
    case Type::F16: {
      auto* ptr = static_cast<float16_t*>(t.data);
      for (size_t i = 0; i < t.linear_length; ++i) {
        ptr[i] = distribution(generator);
      }
      break;
    }
    case Type::F8_E4M3: {
      auto* ptr = static_cast<f8e4m3_t*>(t.data);
      for (size_t i = 0; i < t.linear_length; ++i) {
        ptr[i] = f8e4m3_t::from(distribution(generator));
      }
      break;
    }
    default: throw std::invalid_argument("Unknown type.");
  }

  return t;
}

Tensor::Tensor(): rank(0), type(Type::Unknown), shape(), size(0) {

}

Tensor::~Tensor() {
  if (mem_owned && data) {
    std::free(data);
  }
}

Tensor::Tensor(std::string name, const Type type, const std::vector<int> &shape, void* data, const size_t size, const bool mem_owned)
  : rank(2), name(std::move(name)), type(type), size(size), mem_owned(mem_owned) {
  if (shape.size() > 4) {
    throw std::invalid_argument("Shape cannot have more than 4 dimensions");
  }

  this->rank = shape.size();
  this->shape.resize(rank);

  if (const size_t expected_size = calculate_size(type, shape); size != expected_size) {
    throw std::invalid_argument("External data size does not match expected size for the given shape and type.");
  }

  size_t numel = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] < 0) {
      throw std::invalid_argument("Shape dimensions must be positive");
    }
    this->shape[i] = shape[i];
    if (shape[i] != 0) {
      numel *= shape[i];
    }
  }
  for (size_t i = shape.size(); i < 4; i++) {
    this->shape[i] = 1;
  }

  const size_t dsize = type.bit_size / 8;
  this->linear_length = numel;
  this->size = numel * dsize;
  this->data = data;
}

int Tensor::from_json(const std::string& name, const json& val, void* bytes_ptr, const size_t bytes_size) {
  this->name = name;

  const std::string dtype_str = val.value("dtype", "");
  if (dtype_str == "F32") {
    this->type = Type::F32;
  } else if (dtype_str == "F16") {
    this->type = Type::F16;
  } else if (dtype_str == "BF16") {
    this->type = Type::BF16;
  } else if (dtype_str == "U8") {
    this->type = Type::U8;
  } else if (dtype_str == "F8_E4M3") {
    this->type = Type::F8_E4M3;
  } else if (dtype_str == "F8_E5M2") {
    this->type = Type::F8_E5M2;
  } else {;
    throw std::invalid_argument(std::format("bad dtype {}", dtype_str));
  }

  const size_t dsize = this->type.bit_size / 8;

  linear_length = 1;
  rank = val.at("shape").size();
  if (rank > 4) {
    throw std::invalid_argument("shape exceeds 4 dimensions");
  }

  shape.resize(rank);

  for (size_t i = 0; i < rank && i < 4; i++) {
    if (val.at("shape")[i].get<int>() != val.at("shape")[i]) {
      std::print("bad shape");
      throw std::bad_alloc();
    }
    shape[i] = val.at("shape")[i].get<int>();
    linear_length *= shape[i];
  }
  if (val.at("data_offsets").size() != 2) {
    std::print("{}", "bad offsets");
    throw std::bad_alloc();
  }
  const auto offset_start = static_cast<size_t>(val.at("data_offsets")[0]);
  const auto offset_end = static_cast<size_t>(val.at("data_offsets")[1]);
  if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
    throw std::invalid_argument(std::format("offset out of range"));
  }
  this->data = static_cast<char*>(bytes_ptr) + offset_start;

  #if defined(__AVX__) || defined(__AVX2__)
  constexpr int need_align = 32;
  #elif defined(__ARM_NEON)
  constexpr int need_align = std::alignment_of_v<float32x4_t>;
  #else
  constexpr int need_align = 1;
  #endif

  // std::printf("[%s] type=%s", name.c_str(), dtype_str.c_str());
  this->size = offset_end - offset_start;

  auto get_alignment = [](void* ptr) -> int {
    if (reinterpret_cast<uintptr_t>(ptr) % 128 == 0) return 128;
    if (reinterpret_cast<uintptr_t>(ptr) % 64 == 0) return 64;
    if (reinterpret_cast<uintptr_t>(ptr) % 32 == 0) return 32;
    if (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) return 16;
    if (reinterpret_cast<uintptr_t>(ptr) % 8 == 0) return 8;
    if (reinterpret_cast<uintptr_t>(ptr) % 4 == 0) return 4;
    if (reinterpret_cast<uintptr_t>(ptr) % 2 == 0) return 2;
    return 1;
  };

  if (false && get_alignment(this->data) < need_align) {
    std::print("{}: realign {} to {}...\n", this->name, get_alignment(this->data), need_align);

    void* aligned_data = std::aligned_alloc(need_align, (size / 16 + 1) * 16);
    if (!aligned_data) {
      throw std::invalid_argument(std::format("{}: failed to allocate buffer of size {}:{}!", this->name, size, need_align));
    }
    std::memcpy(aligned_data, this->data, this->size);
    this->data = aligned_data;
  }

  // validate the shape matches the size
  if (linear_length * dsize != this->size) {
    throw std::invalid_argument(std::format("bad size! {}: {} * {} = {}, expected {}", dtype_str, linear_length, dsize, linear_length * dsize, this->size));
  }

  // std::printf(" loaded.\n");

  return 0;
}

[[nodiscard]] std::string YALMData::format() const {
  auto tbl = table::make(
      column<int>{"#", -1, alignment::left, "{}", false},
      column<std::string>{"name", -1, alignment::left, "{}", true},
      column<std::string>{"type", -1, alignment::center, "{}", true},
      column<std::array<int, 2>>{"shape", -1, alignment::right, "{h}", true},
      column<size_t>{"size", -1, alignment::right, "{h}"}
  );

  int row_number = 0;
  for (const auto& [key, tensor]: tensors) {
    tbl.add(row_number++, key, std::string(tensor.type.name()), tensor.shape, tensor.size);;
  }

  return tbl.format(filename);
}

int YALMData::from_file(const std::string& filename) {
  this->filename = filename;
  std::cout << "loading data from file: " << filename << std::endl;
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    throw std::invalid_argument(std::format("failed to open file {}", filename));
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return -1;
  }
  
  size = st.st_size;
  data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    close(fd);
    return -1;
  }

#ifdef __linux__
  // increases readahead buffer size, resulting in faster cold loads
  posix_fadvise(fd, 0, size, POSIX_MADV_WILLNEED); // POSIX_FADV_SEQUENTIAL);
#elif defined(__APPLE__)
  madvise(data, size, MADV_WILLNEED); // | MADV_SEQUENTIAL MADV_WILLNEED);
#endif

  close(fd); // fd can be closed after mmap returns without invalidating the mapping

  // Parse the metadata JSON and the tensors
  if (size < sizeof(uint64_t)) {
    munmap(data, size);
    throw std::invalid_argument(std::format("bad size: {}", size));
  }

  const uint64_t json_size = *static_cast<uint64_t*>(data);
  if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
    munmap(data, size);
    throw std::invalid_argument(std::format("bad size: {}", size));
  }

  void* bytes_ptr = static_cast<char*>(data) + sizeof(uint64_t) + json_size;
  const size_t bytes_size = size - sizeof(uint64_t) - json_size;

  const std::string json_string(static_cast<char*>(data) + sizeof(uint64_t), json_size);
  const json header = json::parse(json_string);

  // std::print("{}", json_string);
  // std::flush(std::cout);

  for (auto& [key, val] : header.items()) {
    if (key == "__metadata__") {
      metadata = val;
    } else {
      Tensor& tensor = tensors[key];
      // printf("tensor: %s\n", key.c_str());
      // printf("tensor: %s\n",val.dump().c_str());

      if (tensor.from_json(key, val, bytes_ptr, bytes_size) != 0) {
        munmap(data, size);
        return -1;
      }
    }
  }

  return 0;
}

std::string Tensor::format(const size_t show_rows, const size_t show_columns) const {
  if (!data) {
    return "Error: Tensor data is null.";
  }

  if (rank == 0) {
    throw std::invalid_argument("rank 0!");
  }

  std::string shape_str = "[";
  for (size_t i = 0; i < rank; ++i) {
    shape_str += std::to_string(shape[i]);
    if (i + 1 < rank) {
      shape_str += ", ";
    }
  }
  shape_str += "]";

  auto tbl = table::make(
    column<size_t>{"row", -1, alignment::left, "{}", true},
    column<std::array<float32_t, 10>>{"col", -1, alignment::right, "{:.3f}", true},
    column<float32_t>{"sum", -1, alignment::right, "{:.3f}", false},
    //column<float32_t>{"mean", -1, alignment::right, "{:.3f}", false},
    column<float32_t>{"min", -1, alignment::right, "{:.3f}", false},
    column<float32_t>{"max", -1, alignment::right, "{:.3f}", true},
    column<std::string>{"histogram", 12, alignment::left, "{}", false},
    column<float32_t>{"offset", -1, alignment::right, "{:.4f}", false},
    column<float32_t>{"scale", -1, alignment::right, "{:.2f}", false}
    );

  [[maybe_unused]] const size_t num_columns = rank == 1 ? shape[0] : shape[1];
  [[maybe_unused]] const size_t num_rows = rank == 1 ? 1 : shape[0];

  for (size_t row = 0; row < std::min(num_rows, show_rows); ++row) {
    auto row_data = get_row(row);;

    stats::histogram_t histogram = stats::histogram(row_data,10);
    row_data.resize(10);

    tbl.add(row, row_data, histogram.sum, histogram.min, histogram.max, histogram.format(), histogram.calculate_offset(), histogram.calculate_scale());
  }

  return tbl.format(std::format("{} {}: {}", name,shape_str, type.name()));
}

[[nodiscard]] Tensor Tensor::operator*(const float32_t factor) const{
  auto result = Tensor::zeroes(type, shape);
  for (size_t i = 0; i < linear_length; ++i) {
    const float32_t v = type.get_float(data, i);
    result.type.set_float(result.data, i, v * factor);
  }

  return result;
}

[[nodiscard]] Tensor Tensor::operator-(const Tensor& other) const {
auto result = Tensor::zeroes(type, shape);
  for (size_t i = 0; i < linear_length; ++i) {
    const float32_t a = type.get_float(data, i);
    const float32_t b = other.type.get_float(other.data, i);
    result.type.set_float(result.data, i, a - b);
  }

  return result;
}

Tensor Tensor::convert_to(Type target_type) const {
  assert(data != nullptr && "Tensor data cannot be null");

  if (target_type != Type::F32 && target_type != Type::F16 && target_type != Type::F8_E2M5 && target_type != Type::F8_E3M4 && target_type != Type::F8_E4M3 && target_type != Type::F8_E5M2) {
    std::cerr << "convert_dtype only supports F32, F16, F8 conversions." << std::endl;
    exit(0);
  }

  if (type == target_type) {
    std::cerr << "Tensor is already of target dtype." << std::endl;
    exit(0);
  }

  std::vector<uint8_t> new_data(linear_length * target_type.bit_size / 8);

  if (type == Type::F16 && target_type == Type::F8_E4M3) {
    float16_t* src = static_cast<float16_t*>(data);
    f8e4m3_t* dst = reinterpret_cast<f8e4m3_t*>(new_data.data());
    for (size_t i = 0; i < this->linear_length; ++i) {
      dst[i] = f8e4m3_t::from(src[i]);
    }
  } else if (type == Type::BF16 && target_type == Type::F16) {
    uint16_t* src = static_cast<uint16_t*>(data);
    float16_t* dst = reinterpret_cast<float16_t*>(new_data.data());
    for (size_t i = 0; i < this->linear_length; ++i) {
      dst[i] = bf16_to_f32(src[i]);
    }
  } else if (type == Type::F16 && target_type == Type::F8_E5M2) {
    float16_t* src = static_cast<float16_t*>(data);
    f8e5m2_t* dst = reinterpret_cast<f8e5m2_t*>(new_data.data());
    for (size_t i = 0; i < this->linear_length; ++i) {
      dst[i] = f8e5m2_t::from(src[i]);
    }
  } else if (type == Type::F16 && target_type == Type::F8_E3M4) {
    float16_t* src = static_cast<float16_t*>(data);
    f8e3m4_t* dst = reinterpret_cast<f8e3m4_t*>(new_data.data());
    for (size_t i = 0; i < this->linear_length; ++i) {
      dst[i] = f8e3m4_t::from(src[i]);
    }
  } else if (type == Type::F16 && target_type == Type::F8_E2M5) {
    float16_t* src = static_cast<float16_t*>(data);
    f8e2m5_t* dst = reinterpret_cast<f8e2m5_t*>(new_data.data());
    for (size_t i = 0; i < this->linear_length; ++i) {
      dst[i] = f8e2m5_t::from(src[i]);
    }
  } else {
    std::cerr << "Unsupported dtype conversion." << std::endl;
    exit(0);
  }

  Tensor converted(this->name, target_type, this->shape, new uint8_t[new_data.size()], new_data.size(), true);
  std::memcpy(converted.data, new_data.data(), new_data.size());

  return converted;
}
