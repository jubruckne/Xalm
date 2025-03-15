#include "tensor.h"

#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "console.h"
#include "stats.h"
#include "table.h"
#include "types.h"

template<typename... Indices>
size_t Tensor::flatten_indices(Indices... indices) const
	requires(std::integral<Indices> && ...)
{
	static_assert(sizeof...(indices) > 0, "At least one index must be provided.");
	if (sizeof...(indices) > rank) {
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

template<typename... Indices>
	requires(std::unsigned_integral<Indices> && ...)
float Tensor::operator[](Indices... indices) const {
	auto idx = flatten_indices(indices...);
	return type.get_float(buffer, idx);
}

template<typename... Indices>
	requires(std::unsigned_integral<Indices> && ...)
std::vector<float32_t> Tensor::get_row(Indices... indices) const {
	if (sizeof...(indices) != rank - 1) {
		throw std::invalid_argument("Number of indices does must match rank - 1!");
	}
	auto idx = flatten_indices(indices...);
	auto values = std::vector<float32_t>(shape[rank - 1]);
	for (size_t i = 0; i < shape[rank - 1]; ++i) {
		values[i] = type.get_float(buffer, idx + i);
	}
	return values;
}

size_t Tensor::calculate_size(const Type type, const std::vector<int> &shape) {
	size_t num_elements = 1;
	for (const int dim: shape) {
		if (dim == 0)
			break;
		num_elements *= dim;
	}
	return num_elements * (type.bit_size / 8);
}

Tensor Tensor::view(buffer_t& data, const size_t size, const Type type, const std::vector<int> &shape,
					const std::string &name) {
	if (const size_t expected_size = calculate_size(type, shape); size != expected_size) {
		throw std::invalid_argument("External data size does not match expected size for the given shape and type.");
	}
	return {name, type, shape, data, size};
}

Tensor Tensor::zeroes(const Type type, const std::vector<int> &shape, const std::string &name) {
	const size_t size = calculate_size(type, shape);
	buffer_t buffer(size);

	return {name, type, shape, buffer, size};
}

Tensor Tensor::uniform(const Type type, const std::vector<int> &shape, const float min, const float max,
					   const std::string &name) {
	auto t = Tensor::zeroes(type, shape, name);

	std::default_random_engine generator(42);
	std::uniform_real_distribution distribution(min, max);

	switch (type) {
		case Type::F32: {
			auto data = t.buffer.span<float>();
			for (size_t i = 0; i < t.linear_length; ++i) {
				data[i] = distribution(generator);
			}
			break;
		}
		case Type::F16: {
			auto data = t.buffer.span<float16_t>();
			for (size_t i = 0; i < t.linear_length; ++i) {
				data[i] = distribution(generator);
			}
			break;
		}
		case Type::F8_E4M3: {
			auto data = t.buffer.span<f8e4m3_t>();
			for (size_t i = 0; i < t.linear_length; ++i) {
				data[i] = f8e4m3_t::from(distribution(generator));
			}
			break;
		}
		default:
			throw std::invalid_argument("Unknown type.");
	}

	return t;
}

//Tensor::Tensor() : rank(0), type(Type::Unknown), shape(), size(0) {}

Tensor::Tensor(std::string name, const Type type, const std::vector<int> &shape, buffer_t& data, const size_t size)
	:rank(2), name(std::move(name)), type(type), size(size), buffer(std::move(data)) {
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
}

std::string Tensor::format(const size_t show_rows, const size_t show_columns) const {
	if (!buffer.empty()) {
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

	auto tbl = table::make(column<size_t>{"row", -1, alignment::left, "{}", true},
						   column<std::array<float32_t, 10>>{"col", -1, alignment::right, "{:.3f}", true},
						   column<float32_t>{"sum", -1, alignment::right, "{:.3f}", false},
						   // column<float32_t>{"mean", -1, alignment::right, "{:.3f}", false},
						   column<float32_t>{"min", -1, alignment::right, "{:.3f}", false},
						   column<float32_t>{"max", -1, alignment::right, "{:.3f}", true},
						   column<std::string>{"histogram", 12, alignment::left, "{}", false},
						   column<float32_t>{"offset", -1, alignment::right, "{:.4f}", false},
						   column<float32_t>{"scale", -1, alignment::right, "{:.2f}", false});

	[[maybe_unused]] const size_t num_columns = rank == 1 ? shape[0] : shape[1];
	[[maybe_unused]] const size_t num_rows = rank == 1 ? 1 : shape[0];

	for (size_t row = 0; row < std::min(num_rows, show_rows); ++row) {
		auto row_data = get_row(row);
		;

		stats::histogram_t histogram = stats::histogram(row_data, 10);
		row_data.resize(10);

		tbl.add(row, row_data, histogram.sum, histogram.min, histogram.max, histogram.format(),
				histogram.calculate_offset(), histogram.calculate_scale());
	}

	return tbl.format(std::format("{} {}: {}", name, shape_str, type.name()));
}

[[nodiscard]] Tensor Tensor::operator*(const float32_t factor) const {
	auto result = Tensor::zeroes(type, shape);
	for (size_t i = 0; i < linear_length; ++i) {
		const float32_t v = type.get_float(buffer, i);
		result.type.set_float(result.buffer, i, v * factor);
	}

	return result;
}

[[nodiscard]] Tensor Tensor::operator-(const Tensor &other) const {
	auto result = Tensor::zeroes(type, shape);
	for (size_t i = 0; i < linear_length; ++i) {
		const float32_t a = type.get_float(buffer, i);
		const float32_t b = other.type.get_float(other.buffer, i);
		result.type.set_float(result.buffer, i, a - b);
	}

	return result;
}

Tensor Tensor::convert_to(const Type target_type) const {
	assert(!buffer.empty() && "Tensor data cannot be null");

	if (type == target_type) {
		std::cerr << "Tensor is already of target dtype." << std::endl;
		exit(0);
	}

	auto converted = Tensor::zeroes(target_type, this->shape, this->name);

	if (type == Type::F16 && target_type == Type::F8_E4M3) {
		const auto src = buffer.span<float16_t>();
		const auto dst = converted.buffer.span<f8e4m3_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			dst[i] = f8e4m3_t::from(src[i]);
		}
	} else if (type == Type::BF16 && target_type == Type::F16) {
		const auto src = buffer.span<uint16_t>();
		const auto dst = converted.buffer.span<float16_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			dst[i] = bf16_to_f32(src[i]);
		}
	} else if (type == Type::F16 && target_type == Type::F8_E5M2) {
		const auto src = buffer.span<float16_t>();
		const auto dst = converted.buffer.span<f8e5m2_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			dst[i] = f8e5m2_t::from(src[i]);
		}
	} else if (type == Type::F16 && target_type == Type::F8_E3M4) {
		const auto src = buffer.span<float16_t>();
		const auto dst = converted.buffer.span<f8e3m4_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			dst[i] = f8e3m4_t::from(src[i]);
		}
	} else if (type == Type::BF16 && target_type == Type::F8_E3M4) {
		const auto src = buffer.span<bfloat16_t>();
		const auto dst = converted.buffer.span<f8e3m4_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			dst[i] = f8e3m4_t::from(static_cast<float32_t>(src[i]));
		}
	} else if (type == Type::BF16 && target_type == Type::F8_E4M3) {
		const auto src = buffer.span<bfloat16_t>();
		const auto dst = converted.buffer.span<f8e4m3_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			dst[i] = f8e4m3_t::from(static_cast<float32_t>(src[i]));
		}
	} else if (type == Type::BF16 && target_type == Type::Q8) {
		const auto src = buffer.span<bfloat16_t>();
		const auto dst = converted.buffer.span<int8_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			Type::Q8.set_float(dst.data(), i, static_cast<float>(src[i]));
		}
	} else if (type == Type::F16 && target_type == Type::F8_E2M5) {
		const auto src = buffer.span<float16_t>();
		const auto dst = converted.buffer.span<f8e2m5_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			dst[i] = f8e2m5_t::from(src[i]);
		}
	} else if (type == Type::F8_E4M3 && target_type == Type::Q8) {
		const auto src = buffer.span<f8e4m3_t>();
		const auto dst = converted.buffer.span<int8_t>();
		for (size_t i = 0; i < this->linear_length; ++i) {
			Type::Q8.set_float(dst.data(), i, f8e4m3_t::to_float(src[i]));
		}
	} else {
		console::print("Unsupported type conversion {} -> {}!\n", type.name(), target_type.name());
		exit(0);
	}

	return converted;
}

void Tensor::save_to_csv(const std::string &filename) const {
	if (rank != 2) {
		throw std::invalid_argument("Tensor must be 2D to save as CSV.");
	}

	std::ofstream file(filename);
	if (!file.is_open()) {
		throw std::ios_base::failure("Failed to open file: " + filename);
	}

	const size_t rows = shape[0];
	const size_t cols = shape[1];

	for (size_t j = 0; j < cols; ++j) {
		if (j < cols - 1) {
			file << j << ",";
		}
	}
	file << "\n";

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			const auto v = (*this)[i, j];
			file << v;
			if (j < cols - 1) {
				file << ",";
			}
		}
		file << "\n";
	}

	file.close();
}
