#include "tensor.h"

#include <fcntl.h>
#include <iostream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "types.h"

size_t Tensor::calculate_size(Type type, const std::array<int, 4>& shape) {
  size_t num_elements = 1;
  for (const int dim : shape) {
    if (dim == 0) break;
    num_elements *= dim;
  }
  return num_elements * type.bit_size() / 8;
}

Tensor Tensor::view(void* data, size_t size, Type type, const std::array<int, 4>& shape, const std::string& name) {
  if (const size_t expected_size = calculate_size(type, shape); size != expected_size) {
    throw std::invalid_argument("External data size does not match expected size for the given shape and type.");
  }
  return Tensor(name, type, shape, data, size, false);
}

Tensor Tensor::zeroes(Type type, const std::array<int, 4> &shape, const std::string &name) {
  const size_t size = calculate_size(type, shape);
  void* ptr = std::aligned_alloc(16, size);
  if (!ptr) {
    throw std::bad_alloc();
  }

  return Tensor(name, type, shape, ptr, size, true);
}

Tensor Tensor::uniform(Type type, const std::array<int, 4>& shape, const float min, const float max, const std::string& name) {
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
      auto* ptr = static_cast<f16_t*>(t.data);
      for (size_t i = 0; i < t.linear_length; ++i) {
        ptr[i] = f16_t::from_float(distribution(generator));
      }
      break;
    }
    case Type::F8: {
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

Tensor::Tensor(): type(Type::Unknown), size(0) {
  printf("Tensor::Tensor()\n");
  fflush(stdout);
  for (size_t i = 0; i < 4; i++) {
    shape[i] = 0;
  }
}

Tensor::~Tensor() {
  printf("Tensor::~Tensor()\n");
  fflush(stdout);

  if (mem_owned && data) {
    std::free(data);
  }
}

Tensor::Tensor(std::string name, const Type type, const std::array<int, 4>& shape, void* data, const size_t size, const bool mem_owned)
    : name(std::move(name)), type(type), size(size), mem_owned(mem_owned) {
  if (shape.size() > 4) {
    throw std::invalid_argument("Shape cannot have more than 4 dimensions");
  }

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

  const size_t dsize = type.bit_size() / 8;
  this->linear_length = numel;
  this->size = numel * dsize;
  this->data = data;
}

int Tensor::from_json(const std::string& name, const json& val, void* bytes_ptr, size_t bytes_size) {
  this->name = name;

  const std::string dtype_str = val.value("dtype", "");
  if (dtype_str == "F32") {
    this->type = Type::F32;
  } else if (dtype_str == "F16") {
    this->type = Type::F16;
  } else if (dtype_str == "U8") {
    this->type = Type::U8;
  } else if (dtype_str == "F8_E4M3") {
    this->type = Type::F8;
  } else {
    printf("bad dtype %s", dtype_str.c_str());
    return -1;
  }
  const size_t dsize = this->type.bit_size() / 8;

  size_t numel = 1;
  if (val.at("shape").size() > 4) {
    std::cerr << "shape exceeds 4 dimensions" << std::endl;
  }
  for (size_t i = 0; i < val.at("shape").size() && i < 4; i++) {
    if (val.at("shape")[i].get<int>() != val.at("shape")[i]) {
      std::cerr << "bad shape" << std::endl;
      return -1;
    }
    shape[i] = val.at("shape")[i].get<int>();
    numel *= shape[i];
  }
  if (val.at("data_offsets").size() != 2) {
    return -1;
  }
  const auto offset_start = static_cast<size_t>(val.at("data_offsets")[0]);
  const auto offset_end = static_cast<size_t>(val.at("data_offsets")[1]);
  if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
    std::cerr << "bad offsets" << std::endl;
    return -1;
  }
  this->data = (char*)bytes_ptr + offset_start;
  this->size = offset_end - offset_start;
  // validate the shape matches the size
  if (numel * dsize != this->size) {
    printf("bad size: (%zu * %zu = %zu), expected %zu", numel, dsize, numel * dsize, this->size);
    return -1;
  }
  return 0;
}

int YALMData::from_file(const std::string& filename) {
  std::cout << "loading data from file: " << filename << std::endl;
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    return -1;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    return -1;
  }
  
  size = st.st_size;
  data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    close(fd);
    return -1;
  }

#ifdef __linux__
  // increases readahead buffer size, resulting in faster cold loads
  posix_fadvise(fd, 0, size, POSIX_FADV_SEQUENTIAL);
#elif defined(__APPLE__)
    madvise(data, size, MADV_RANDOM | MADV_WILLNEED);
#endif

  close(fd); // fd can be closed after mmap returns without invalidating the mapping

  // Parse the metadata JSON and the tensors
  if (size < sizeof(uint64_t)) {
    munmap(data, size);
    return -1;
  }

  uint64_t json_size = *(uint64_t*)data;
  if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
    munmap(data, size);
    return -1;
  }

  char* json_ptr = (char*)data + sizeof(uint64_t);
  void* bytes_ptr = (char*)data + sizeof(uint64_t) + json_size;
  size_t bytes_size = size - sizeof(uint64_t) - json_size;

  json_ptr[json_size - 1] = 0; // null-terminate the JSON string
  json header = json::parse(json_ptr);

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