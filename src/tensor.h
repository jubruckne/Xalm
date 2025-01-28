#pragma once

#include "json.hpp"

#include <string>
#include "types.h"
#include "buffer.h"

using json = nlohmann::json;

class Tensor {
public:
	using buffer_t = buffer<std::byte, sizeof(uint32x4_t)>;

	Tensor() = delete;
	Tensor(const Tensor& other) = delete;
	Tensor &operator=(const Tensor& other) = delete;

	Tensor(Tensor&& other) noexcept = default;
	Tensor &operator=(Tensor&& other) noexcept = default;

	uint8_t rank;
	std::string name;
	Type type = Type::Unknown;
	std::vector<int> shape = {};
	size_t size = 0;
	size_t linear_length = 0;

	template<class... Indices>
	size_t flatten_indices(Indices... indices) const
		requires(std::integral<Indices> && ...);

	template<typename... Indices>
		requires(std::unsigned_integral<Indices> && ...)
	float operator[](Indices... indices) const;

	template<typename... Indices>
		requires(std::unsigned_integral<Indices> && ...)
	std::vector<float32_t> get_row(Indices... indices) const;

	[[nodiscard]] Tensor operator*(float32_t factor) const;
	[[nodiscard]] Tensor operator-(const Tensor& other) const;

	static Tensor view(buffer_t& buffer, size_t size, Type type, const std::vector<int> &shape, const std::string &name = "");
	static Tensor zeroes(Type type, const std::vector<int> &shape, const std::string &name = "");
	static Tensor uniform(Type type, const std::vector<int> &shape, float min, float max, const std::string &name = "");

	[[nodiscard]] std::string format(size_t show_rows = 8, size_t show_columns = 8) const;

	[[nodiscard]] Tensor convert_to(Type target_type) const;

	void save_to_csv(const std::string &filename) const;

	[[nodiscard]] const buffer_t* get_buffer() const {
		return &buffer;
	}

private:
	buffer_t buffer;
	Tensor(std::string name, Type type, const std::vector<int> &shape, buffer_t& data, size_t size);
	static size_t calculate_size(Type type, const std::vector<int> &shape);
};
