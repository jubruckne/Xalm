#pragma once

#include "json.hpp"

#include <string>
#include "buffer.h"
#include "types.h"

#include <tuple>
#include <utility>

using json = nlohmann::json;

using type_code_t = int;

template <typename T>
concept ShapeLike = requires(T t) {
	{ t.size() } -> std::convertible_to<size_t>; // Must have a size() method
	requires std::is_same_v<std::remove_cvref_t<decltype(*t.begin())>, size_t>; // Elements must be size_t
};

template<typename T>
concept DataType = requires(std::byte* ptr, float32_t value) {
	{ T::name } -> std::convertible_to<std::string_view>;
	{ T::type_code } -> std::convertible_to<type_code_t>;
	{ T::bit_size } -> std::convertible_to<int>;
	{ T::load(ptr, &value) } -> std::same_as<void>;
	{ T::store(ptr, value) } -> std::same_as<void>;
};

namespace data_type {
	struct f32_t {
		constexpr static std::string_view name = "f32";
		constexpr static type_code_t type_code = 1;
		constexpr static int bit_size = sizeof(float32_t);

		static void load(std::byte* data, float32_t& value) {}
		static void store(std::byte* data, float32_t value) {}
	};

	constexpr static f32_t F32{};

	struct f16_t {
		constexpr static std::string_view name = "f16";
		constexpr static type_code_t type_code = 2;
		constexpr static int bit_size = sizeof(float16_t);

		static void load(std::byte* data, float32_t& value) {}
		static void store(std::byte* data, float32_t value) {}
	};

	constexpr static f16_t F16{};

	struct qi4_t {
		constexpr static std::string_view name = "qi4";
		constexpr static type_code_t type_code = 3;
		constexpr static int bit_size = 4;

		static void load(std::byte* data, float32_t& value) {}
		static void store(std::byte* data, float32_t value) {}
	};

	constexpr static qi4_t QI4{};

	struct type {
		const std::string_view name;
		const type_code_t type_code;
		explicit type(DataType auto t) noexcept : name(decltype(t)::name), type_code(decltype(t)::type_code) {}
	};
}

template<typename T>
concept TensorExpression = requires(T expr) {
	{ expr.rank } -> std::convertible_to<uint8_t>;
	{ expr.name } -> std::convertible_to<std::string>;
	{ expr.type } -> std::convertible_to<type_code_t>;
	{ expr.shape } -> std::convertible_to<std::span<size_t>>;
};

class tensor_base {
public:
	virtual ~tensor_base() = default;
	[[nodiscard]] virtual std::string_view name() const = 0;
	[[nodiscard]] virtual uint8_t rank() const = 0;
	[[nodiscard]] virtual std::span<size_t const> shape() const = 0;
	[[nodiscard]] virtual type_code_t type() const = 0;

	template<typename... Indices> requires(std::unsigned_integral<Indices> && ...)
	float operator[](Indices... indices) const {
		return get_float(indices...);
	}
};

template<DataType T, uint8_t Rank>
class typed_tensor final: public tensor_base {
public:
	using type_t = T;
	using buffer_t = buffer<std::byte, sizeof(uint32x4_t)>;

	static constexpr int type_code = T::type_code;

	using accessor_t = typename T::accessor;
private:
	std::string _name;
	buffer_t _buffer;
	size_t _linear_length;

	std::array<size_t, Rank> _shape;
public:
	typed_tensor() = delete;

	typed_tensor(const std::array<size_t, Rank>& shape, const buffer_t& buffer, const std::string& name) noexcept:
	_name(name), _buffer(buffer), _linear_length(0), _shape(shape) {
		size_t numel = 1;
		for (size_t i = 0; i < shape.size(); i++) {
			if (shape[i] != 0) {
				numel *= shape[i];
			}
		}

		const size_t dsize = type_t::bit_size / 8;
		_linear_length = numel;

		assert(numel * dsize == _buffer.bytes_size());
	}

	typed_tensor(const std::array<size_t, Rank>& shape, const std::string& name) noexcept:
		typed_tensor(shape, buffer_t{calculate_size(shape)}, name) {}

private:
	static constexpr size_t calculate_size(const std::array<size_t, Rank>& shape) {
		size_t num_elements = 1;
		for (const int dim: shape) {
			if (dim == 0)
				throw std::invalid_argument("Shape dimensions must be > 0!");
			num_elements *= dim;
		}
		return num_elements * (type_t::bit_size / 8);
	}

	template<typename... Indices>
	size_t flatten_indices(Indices... indices) const requires(std::integral<Indices> && ...){
		static_assert(sizeof...(indices) > 0, "At least one index must be provided.");
		if (sizeof...(indices) > Rank) {
			throw std::invalid_argument("Number of indices does not match tensor rank.");
		}

		std::array<size_t, sizeof...(indices)> indices_ = {static_cast<size_t>(indices)...};
		constexpr auto indices_rank = sizeof...(indices);

		size_t offset = 0;
		size_t stride = 1;

		for (int i = Rank - 1; i >= 0; --i) {
			if (i < indices_rank) {
				if (indices_[i] >= static_cast<size_t>(_shape[i])) {
					throw std::out_of_range(std::format("Index {} out of bounds({}).", indices_[i], _shape[i]));
				}
				offset += indices_[i] * stride;
			}
			stride *= _shape[i];
		}
		return offset;
	}

	template<typename... Indices> requires(std::unsigned_integral<Indices> && ...)
	float operator[](Indices... indices) const {
		auto idx = flatten_indices(indices...);
		return typename T::accessor::template load<float32_t>(_buffer, idx);
	}

public:
	[[nodiscard]] uint8_t rank() const override {
		return Rank;
	}

	[[nodiscard]] std::span<size_t const> shape() const override {
		return _shape;
	}

	[[nodiscard]] type_code_t type() const override {
		return type_code;
	}

	[[nodiscard]] std::string_view name() const override {
		return _name;
	}
};

class tensor final: public tensor_base {
	std::shared_ptr<tensor_base> _tensor;
public:
	tensor() = delete;

	tensor(const tensor&) = default;
	tensor(tensor&&) noexcept = default;
	tensor& operator=(const tensor&) = default;
	tensor& operator=(tensor&&) noexcept = default;

	explicit tensor(std::shared_ptr<tensor_base> tensor): _tensor(std::move(tensor)) {}

	template<DataType T, size_t Rank>
	static tensor create(const std::array<size_t, Rank>& shape, const std::string& name = "") {
		return{std::make_shared<typed_tensor<T, Rank>>(shape, name)};
	}

	template<size_t Rank>
	static tensor create(DataType auto type, const std::array<size_t, Rank>& shape, const std::string& name = "") {
		return tensor(std::make_shared<typed_tensor<decltype(type), Rank>>(shape, name));
	}

	static tensor create(DataType auto type, const std::vector<size_t>& shape, const std::string& name = "") {
		switch (shape.size()) {
			case 0:
				return tensor{std::make_shared<typed_tensor<decltype(type), 0>>(std::array<size_t, 0>{}, name)};
			case 1:
				return tensor(std::make_shared<typed_tensor<decltype(type), 1>>(std::array{shape[0]}, name));
			case 2:
				return tensor(std::make_shared<typed_tensor<decltype(type), 2>>(std::array{shape[0], shape[1]}, name));
			default: throw std::invalid_argument("Shape must be a scalar or vector.");
		}
	}

	[[nodiscard]] uint8_t rank() const override  {
		return _tensor->rank();
	}

	[[nodiscard]] std::span<size_t const> shape() const override {
		return _tensor->shape();
	}

	[[nodiscard]] type_code_t type() const override {
		return _tensor->type();
	}

	[[nodiscard]] std::string_view name() const override {
		return _tensor->name();
	}
};

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

	[[nodiscard]] const buffer_t* get_buffer() const noexcept {
		return &buffer;
	}

	template<typename T>
	[[nodiscard]] const T* get_data() const noexcept {
		return buffer.get<T>();
	}


private:
	buffer_t buffer;
	Tensor(std::string name, Type type, const std::vector<int> &shape, buffer_t& data, size_t size);
	static size_t calculate_size(Type type, const std::vector<int> &shape);
};
