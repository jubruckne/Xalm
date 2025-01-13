#include "tensor.h"

#include <fcntl.h>
#include <iostream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include "table.h"

#include "types.h"

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

  size_t numel = 1;
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
    numel *= shape[i];
  }
  if (val.at("data_offsets").size() != 2) {
    std::printf("bad offsets");
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
  constexpr int need_align = 16;
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

  if (get_alignment(this->data) < need_align) {
    std::print("{}: realign {} to {}...\n", this->name, get_alignment(this->data), need_align);

    void* aligned_data = std::aligned_alloc(need_align, (size / 16 + 1) * 16);
    if (!aligned_data) {
      throw std::invalid_argument(std::format("{}: failed to allocate buffer of size {}:{}!", this->name, size, need_align));
    }
    std::memcpy(aligned_data, this->data, this->size);
    this->data = aligned_data;
  }

  // validate the shape matches the size
  if (numel * dsize != this->size) {
    throw std::invalid_argument(std::format("bad size! {}: {} * {} = {}, expected {}", dtype_str, numel, dsize, numel * dsize, this->size));
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

std::string Tensor::format(size_t show_rows, size_t show_columns, int groups_row, int groups_col) const {
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

 /* auto tbl = table::make(
    column<int>{"#", -1, alignment::left, "{}", false},
    column<std::string>{"name", -1, alignment::left, "{}", true},
    column<std::string>{"type", -1, alignment::center, "{}", true},
    column<std::array<int, 2>>{"shape", -1, alignment::right, "{h}", true},
    column<size_t>{"size", -1, alignment::right, "{h}"}
  );*/

  auto extra_columns = {"min", "max", "mean", "std", "unique"};

  int column_width = 9;
  std::string output = std::format("{}{}: {}\n", name, shape_str, type.name());

  size_t num_columns = rank == 1 ? shape[0] : shape[1];
  show_columns = std::min(show_columns, num_columns);

  std::string sep = std::string(column_width * (show_columns + extra_columns.size()) + 6 + (show_columns == 0 ? 0 : 3), '-') + "\n";

  output += sep;
  output += std::format("{:<{}}", "row", 6);

  struct column_t {
    std::string name;
    size_t i_begin = 0;
    size_t i_end = 0;
    bool show = true;
  };

  std::vector<column_t> col_def;

  for (size_t col = 0; col < num_columns; ++col) {
    column_t column;
    if (groups_col == 1) {
      column.i_begin = col;
      column.i_end = col;
      column.name = std::format("{:>{}}", std::format("col {}", col), column_width);
      column.show = col < show_columns;
    } else {
      column.i_begin = col * groups_col;
      column.i_end = col * (groups_col + 1) - 1;
      column.name = std::format("{:>{}}", std::format("col {}-{}", column.i_begin, column.i_end), column_width);
      column.show = col < show_columns;

      if (column.show && extra_columns.size() > 0) {
        // add stats columns
        for (std::string c: extra_columns) {
          col_def.push_back(column_t{c, 0, 0, true});
          output += std::format("{:>{}}", c, column_width);
        }
      }
    }
    column.show = col < show_columns;

    if (column.show) {
      output += std::format("{:>{}}", std::format("col {}", col), column_width);
    } else if (col == show_columns) {
      output += std::format("{:>{}}", "...", 3);
    }

    col_def.push_back(column);
  }

  for (auto c: extra_columns) {
    output += std::format("{:>{}}", c, column_width);
  }

  output += "\n" + sep;

  size_t num_rows = rank == 1 ? 1 : shape[0];
  show_rows = std::min(show_rows, num_rows);

  struct stats_t {
    float sum = 0;
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    float sum_abs = 0;
    std::unordered_set<float> unique_values;
  };

  stats_t all_stats{0,std::numeric_limits<float>::max(),std::numeric_limits<float>::min(), 0};

  for (size_t row = 0; row < show_rows; ++row) {
    output += std::format("{:<{}}", row, 6);

    for (size_t i = 0; i < num_columns; i += groups_col) {
      stats_t col_stats{0,std::numeric_limits<float>::max(),std::numeric_limits<float>::min(), 0};

      for (int gc = 0; gc < groups_col; ++gc) {
        auto val = type.get_float(data, num_columns * row + i);

        for (auto st: {&all_stats, &col_stats}) {
          st->sum += val;
          st->sum_abs += std::abs(val);
          st->min = std::min(st->min, val);
          st->max = std::max(st->max, val);
          st->unique_values.insert(val);
        }

        if (i < show_columns) {
          output += std::format("{:>{}.4f}", val*128.0f, column_width);
        }
      }
    }

    if (show_columns > 0) {
      output += std::format("{:>{}}", " ", 3);
    }

    output += "\n";
  }

  output += sep;
  { // final header
    output += std::format("{:<{}}", num_rows, 6);
    if (show_columns > 0) {
      output += std::format("{:>{}}", " ", 3);
    }
    for (std::string c: extra_columns) {
      if (c == "mean") {
        output += std::format("{:>{}.4f}", all_stats.sum / static_cast<float>(num_columns), column_width);
      } else if (c == "std") {
        output += std::format("{:>{}.4f}", static_cast<float>(666), column_width);
      } else if (c == "min") {
        output += std::format("{:>{}.4f}", all_stats.min, column_width);
      } else if (c == "max") {
        output += std::format("{:>{}.4f}", all_stats.max, column_width);
      } else if (c == "sum") {
        output += std::format("{:>{}.4f}", all_stats.sum, column_width);
      } else if (c == "sum_abs") {
        output += std::format("{:>{}.4f}", all_stats.sum_abs, column_width);
      } else if (c == "unique") {
        output += std::format("{:>{}}", all_stats.unique_values.size(), column_width);
      } else {
        throw std::invalid_argument(std::format("unknown column {}", c));
      }
    }

    output += "\n\n";
  }

  return output;
}

/*
Tensor Tensor::convert_to(DType target_dtype) const {
  assert(data != nullptr && "Tensor data cannot be null");

  // Check if target dtype is supported
  if (target_dtype != DType::F32 && target_dtype != DType::F16 && target_dtype != DType::F8) {
    std::cerr << "convert_dtype only supports F32, F16, F8 conversions." << std::endl;
    exit(0);
  }

  // If already the desired type, no conversion is needed
  if (dtype == target_dtype) {
    std::cerr << "Tensor is already of target dtype." << std::endl;
    exit(0);
  }

  size_t num_elements = size / dtype_size(dtype);
  std::vector<uint8_t> new_data(num_elements * dtype_size(target_dtype));

  if (dtype == DType::F32 && target_dtype == DType::F16) {
    // Convert from float (F32) to half-precision float (F16)
    float* src = static_cast<float*>(data);
    f16_t* dst = reinterpret_cast<f16_t*>(new_data.data());
    for (size_t i = 0; i < num_elements; ++i) {
      dst[i] = f16_t::from_float(src[i]);
    }
  } else if (dtype == DType::F16 && target_dtype == DType::F32) {
    // Convert from half-precision float (F16) to float (F32)
    f16_t* src = static_cast<f16_t*>(data);
    float* dst = reinterpret_cast<float*>(new_data.data());
    for (size_t i = 0; i < num_elements; ++i) {
      dst[i] = f16_t::to_float(src[i]);
    }
  } else if (dtype == DType::F16 && target_dtype == DType::F8) {
    // Convert from half-precision float (F16) to quarter float (f8)
    f16_t* src = static_cast<f16_t*>(data);
    f8_t* dst = reinterpret_cast<f8_t*>(new_data.data());
    for (size_t i = 0; i < num_elements; ++i) {
      dst[i] = f8_t::from(src[i]);
    }
  } else {
    std::cerr << "Unsupported dtype conversion." << std::endl;
    exit(0);
  }

  Tensor converted(this->name, target_dtype, this->shape, new uint8_t[new_data.size()], new_data.size());

  std::memcpy(data, new_data.data(), new_data.size());

  return converted;
}
*/