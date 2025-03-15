#pragma once
#include <cassert>
#include <cstdint>
#include "console.h"
#include "types2.h"

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h> // Intel/AVX2
using float16_t = __fp16;
using float32_t = float;
using bfloat16_t = uint16_t;
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h> // ARM NEON
#endif

namespace DType {
	struct type {
		enum class type_code {
			int8,
			uint8,
			int16,
			int32,
			float16,
			float32,
			bfloat16,
			iq4
		};

		constexpr explicit type(const type_code code) : code(code) {}

		constexpr operator type_code() const { return code; }
		constexpr type_code val() const { return code; }

		type_code code;
	};

	inline constexpr type int8{type::type_code::int8};
	inline constexpr type uint8{type::type_code::uint8};
	inline constexpr type int16{type::type_code::int16};
	inline constexpr type int32{type::type_code::int32};
	inline constexpr type float16{type::type_code::float16};
	inline constexpr type float32{type::type_code::float32};
	inline constexpr type bfloat16{type::type_code::bfloat16};
	inline constexpr type iq4{type::type_code::iq4};
	type promote_types(const type& t1, const type& t2);

	template<typename T>
	concept ResultType = std::is_same_v<T, float32_t> || std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

	template<type T, ResultType R, size_t N=1>
	struct accessor;

	template<type TYPE>
	struct traits;

	template <>
	struct traits<float32> {
		using native_type = float32_t;
		static constexpr size_t bit_size = 32;
		static constexpr std::string_view name = "f32";
		static constexpr type type = float32;
		static constexpr type::type_code type_code = type.code;
		template<ResultType T, size_t N=1>
		using accessor = accessor<float32, T, N>;
	};

	template <>
	struct traits<float16> {
		using native_type = float16_t;
		static constexpr size_t bit_size = 16;
		static constexpr std::string_view name = "f16";
		static constexpr type type = float16;
		static constexpr type::type_code type_code = type.code;
		template<ResultType T, size_t N=1>
		using accessor = accessor<float16, T, N>;
	};

	template <>
	struct traits<bfloat16> {
		static constexpr size_t bit_size = 16;
		static constexpr std::string_view name = "bf16";
		static constexpr type type = bfloat16;
		static constexpr type::type_code type_code = type.code;
		using native_type = bfloat16_t;
		template<ResultType T, size_t N=1>
		using accessor = accessor<bfloat16, T, N>;
	};

	template <>
	struct traits<iq4> {
		static constexpr size_t bit_size = 4;
		static constexpr std::string_view name = "iq4";
		static constexpr type type = iq4;
		static constexpr type::type_code type_code = type.code;
		using native_type = void;
		template<ResultType T, size_t N=1>
		using accessor = accessor<iq4, T, N>;
	};

	template <ResultType R>
	struct accessor<float32, R, 1> {
		using native_type = float32_t;
		using result_type = float32_t;

		static result_type load(const native_type* data) {
			return *reinterpret_cast<R*>(data);
		}
		static void store(native_type* data, const result_type value) {
			*reinterpret_cast<R*>(data) = value;
		}
	};

	template <>
	struct accessor<float32, float32_t, 4> {
		using native_type = float32_t;
		using result_type = float32x4_t;

		static result_type load(const native_type* data) {
			return vld1q_f32(data);
		}
		static void store(native_type* data, const result_type value) {
			vst1q_f32(data, value);
		}
	};

	template <>
	struct accessor<float32, float16_t, 8> {
		using native_type = float32_t;
		using result_type = float16x8_t;

		static result_type load(const native_type* data) {
			const auto [val] = vld1q_f32_x2(data);
			return vcombine_f16(
				vcvt_f16_f32(val[0]),
				vcvt_f16_f32(val[1])
			);
		}
		
		static void store(native_type* data, const result_type value) {
			const float32x4x2_t hl = {
				vcvt_f32_f16(vget_low_f16(value)),
				vcvt_f32_f16(vget_high_f16(value))
			};

			vst1q_f32_x2(data, hl);
		}
	};


	template <>
	struct accessor<float32, float32_t, 16> {
		using native_type = float32_t;
		using result_type = float32x4x4_t;

		static result_type load(const native_type* data) {
			return vld1q_f32_x4(data);
		}
		static void store(native_type* data, const result_type value) {
			vst1q_f32_x4(data, value);
		}
	};

	template <>
	struct accessor<float16, float16_t, 8> {
		using native_type = float16_t;
		using result_type = float16x8_t;

		static result_type load(const native_type* data) {
			return vld1q_f16(data);
		}
		static void store(native_type* data, const result_type value) {
			vst1q_f16(data, value);
		}
	};

	template <>
	struct accessor<float16, float16_t, 16> {
		using native_type = float16_t;
		using result_type = float16x8x2_t;

		static result_type load(const native_type* data) {
			return vld1q_f16_x2(data);
		}
		static void store(native_type* data, const result_type value) {
			vst1q_f16_x2(data, value);
		}
	};

	template <>
	struct accessor<float16, float16_t, 32> {
		using native_type = float16_t;
		using result_type = float16x8x4_t;

		static result_type load(const native_type* data) {
			return vld1q_f16_x4(data);
		}
		static void store(native_type* data, const result_type value) {
			vst1q_f16_x4(data, value);
		}
	};

	template <>
	struct accessor<iq4, float16_t, 32> {
		using native_type = uint8_t;
		using result_type = float16x8x4_t;

		static result_type load(const native_type* data) {
			// Load 16 bytes (each byte contains two 4-bit values)
			const uint8x16_t q = vld1q_u8(data);

			// Extract lower 4 bits: AND with 0x0F
			const uint8x16_t lower = vandq_u8(q, vdupq_n_u8(0x0F));

			// Extract upper 4 bits: Shift right by 4
			const uint8x16_t upper = vshrq_n_u8(q, 4);

			// Convert to 16-bit
			const uint16x8_t lower_u16 = vmovl_u8(vget_low_u8(lower));
			const uint16x8_t upper_u16 = vmovl_u8(vget_low_u8(upper));
			const uint16x8_t lower_u16_h = vmovl_u8(vget_high_u8(lower));
			const uint16x8_t upper_u16_h = vmovl_u8(vget_high_u8(upper));

			// Convert to float16x8_t
			float16x8_t lower_f16 = vcvtq_f16_u16(lower_u16);
			float16x8_t upper_f16 = vcvtq_f16_u16(upper_u16);
			float16x8_t lower_f16_h = vcvtq_f16_u16(lower_u16_h);
			float16x8_t upper_f16_h = vcvtq_f16_u16(upper_u16_h);

			// Store in float16x8x4_t
			return {lower_f16, upper_f16, lower_f16_h, upper_f16_h};
		}
		static void store(native_type* data, const result_type value) {
		}
	};



	/*void hdhsfgdsh() {
		constexpr auto load = traits<float32>::accessor<float32_t, 4>::load;
		auto x = load(nullptr);

	}*/

}



template<int8_t N>
consteval float EXP2() {
	if constexpr (N == 0)
		return 1;
	if constexpr (N > 0)
		return EXP2<N - 1>() * 2;
	if constexpr (N < 0)
		return EXP2<N + 1>() / 2;
}

template<int8_t N>
consteval int EXP_I2()
	requires(N >= 0)
{
	if constexpr (N == 0)
		return 1;
	if constexpr (N > 0)
		return EXP_I2<N - 1>() * 2;
}

template<uint8_t E, uint8_t M>
	requires(M > 0 && E + M == 7)
struct f8_t {
private:
	uint8_t bits = 0;

	explicit f8_t(const uint8_t bits) noexcept : bits(bits) {}

	static constexpr int E_BIAS = EXP2<E - 1>() - 1;
	static constexpr float E_BIAS_MINUS_127 = EXP2<E_BIAS - 127>();
	static constexpr float E_127_MINUS_BIAS = EXP2<127 - E_BIAS>();
	static constexpr float max = (2 - EXP2<-M + 1>()) * EXP2<EXP_I2<E - 1>()>();
	static constexpr float min = EXP2<-M>() * EXP2<2 - EXP_I2<E - 1>()>();

public:
	static f8_t from(const float value) noexcept {
		union {
			float f;
			uint32_t bits;
		} in = {value};
		uint8_t bits = (in.bits >> 24) & 0x80;
		in.bits &= 0x7fffffff;
		if (in.f >= max) {
			bits |= 0x7E;
		} else if (in.f < min) {
		} else {
			in.f *= E_BIAS_MINUS_127;
			in.bits += 1 << (22 - M);
			bits |= (in.bits >> (23 - M)) & 0x7F;
		}

		return f8_t(bits);
	}

	static float to_float(const f8_t value) noexcept {
		union {
			float f;
			uint32_t bits;
		} out = {0};
		out.bits = value.bits & 0x80;
		out.bits <<= 24;
		uint32_t _bits = value.bits & 0x7F;
		_bits <<= (23 - M);
		out.bits |= _bits;
		out.f *= E_127_MINUS_BIAS;
		return out.f;
	}
};

using f8e2m5_t = f8_t<2, 5>;
using f8e3m4_t = f8_t<3, 4>;
using f8e4m3_t = f8_t<4, 3>;
using f8e5m2_t = f8_t<5, 2>;

static float32_t bf16_to_f32(const uint16_t h) {
	const uint32_t i = static_cast<uint32_t>(h) << 16;
	return *reinterpret_cast<const float32_t*>(&i);
}

static uint16_t f32_to_bf16(const float32_t s) {
	const uint32_t i = *reinterpret_cast<const uint32_t*>(&s);

	if ((i & 0x7fffffff) > 0x7f800000) {
		return (i >> 16) | 64;
	}

	return (i + (0x7fff + ((i >> 16) & 1))) >> 16;
}

struct Type {
	static const Type Unknown;
	static const Type F32;
	static const Type F16;
	static const Type BF16;
	static const Type F8_E2M5;
	static const Type F8_E3M4;
	static const Type F8_E4M3;
	static const Type F8_E5M2;
	static const Type U8;
	static const Type Q8;

	int id;
	uint8_t bit_size;

	constexpr Type(const int v, const size_t bit_size) noexcept : id(v), bit_size(bit_size) {}

	~Type() noexcept = default;

	constexpr operator int() const noexcept { return id; } // NOLINT(*-explicit-constructor, *-const-return-type)

	[[nodiscard]] constexpr std::string_view name() const {
		if (*this == Type::F32)
			return "F32";
		if (*this == Type::F16)
			return "F16";
		if (*this == Type::BF16)
			return "BF16";
		if (*this == Type::F8_E2M5)
			return "F8_E2M5";
		if (*this == Type::F8_E3M4)
			return "F8_E3M4";
		if (*this == Type::F8_E4M3)
			return "F8_E4M3";
		if (*this == Type::F8_E5M2)
			return "F8_E5M2";
		if (*this == Type::U8)
			return "U8";
		if (*this == Type::Q8)
			return "Q8";
		return "UNKNOWN";
	}

	[[nodiscard]] constexpr size_t byte_offset(const std::size_t offset) const {
		if (*this == Type::F32)
			return offset * sizeof(float32_t);
		if (*this == Type::F16)
			return offset * sizeof(float16_t);
		if (*this == Type::BF16)
			return offset * sizeof(bfloat16_t);
		if (*this == Type::F8_E2M5)
			return offset * sizeof(uint8_t);
		if (*this == Type::F8_E3M4)
			return offset * sizeof(uint8_t);
		if (*this == Type::F8_E4M3)
			return offset * sizeof(uint8_t);
		if (*this == Type::F8_E5M2)
			return offset * sizeof(uint8_t);
		if (*this == Type::U8)
			return offset * sizeof(uint8_t);
		if (*this == Type::Q8)
			return offset * sizeof(int8_t);
		return 0;
	}

	[[nodiscard]] constexpr const void* data_ptr(const void* data, const std::size_t offset) const {
		return static_cast<const uint8_t*>(data) + byte_offset(offset);
	}

	[[nodiscard]] float32_t get_float(const void* data, const std::size_t offset) const {
		auto d = data_ptr(data, offset);

		if (id == Type::F32.id)
			return *static_cast<const float32_t*>(d);
		if (id == Type::F16.id)
			return *static_cast<const float16_t*>(d);
		if (id == Type::BF16.id)
			return bf16_to_f32(*static_cast<const uint16_t*>(d));
		if (id == Type::F8_E2M5.id)
			return f8e2m5_t::to_float(*static_cast<const f8e2m5_t*>(d));
		if (id == Type::F8_E3M4.id)
			return f8e3m4_t::to_float(*static_cast<const f8e3m4_t*>(d));
		if (id == Type::F8_E4M3.id)
			return f8e4m3_t::to_float(*static_cast<const f8e4m3_t*>(d));
		if (id == Type::F8_E5M2.id)
			return f8e5m2_t::to_float(*static_cast<const f8e5m2_t*>(d));
		if (id == Type::Q8)
			return (1.f / 100.f) * static_cast<float32_t>(*static_cast<const int8_t*>(d));

		return 666.66f;
	}

	void set_float(void* data, const std::size_t offset, float32_t value) const {
		if (id == Type::F32.id) {
			static_cast<float32_t*>(data)[offset] = value;
			return;
		}
		if (id == Type::F16.id) {
			static_cast<float16_t*>(data)[offset] = static_cast<float16_t>(value);
			return;
		}
		if (id == Type::BF16.id) {
			static_cast<uint16_t*>(data)[offset] = f32_to_bf16(value);
			return;
		}
		if (id == Type::F8_E2M5.id) {
			static_cast<f8e2m5_t*>(data)[offset] = f8e2m5_t::from(value);
			return;
		}
		if (id == Type::F8_E3M4.id) {
			static_cast<f8e3m4_t*>(data)[offset] = f8e3m4_t::from(value);
			return;
		}
		if (id == Type::F8_E4M3.id) {
			static_cast<f8e4m3_t*>(data)[offset] = f8e4m3_t::from(value);
			return;
		}
		if (id == Type::F8_E5M2.id) {
			static_cast<f8e5m2_t*>(data)[offset] = f8e5m2_t::from(value);
			return;
		}
		if (id == Type::Q8.id) {
			static_cast<int8_t*>(data)[offset] = static_cast<int8_t>(std::ranges::clamp(
					std::round(value * 100.f), static_cast<float>(std::numeric_limits<int8_t>::lowest()),
					static_cast<float>(std::numeric_limits<int8_t>::max())));
			return;
		}

		throw std::invalid_argument("Invalid type");
	}

	[[nodiscard]] static Type parse(const std::string_view str) {
		auto to_upper = [](const std::string_view s) -> std::string {
			std::string result(s);
			std::ranges::transform(result, result.begin(), [](const unsigned char c) { return std::toupper(c); });
			return result;
		};

		const auto type_str = to_upper(str);

		if (type_str == "F32")
			return Type::F32;
		if (type_str == "F16")
			return Type::F16;
		if (type_str == "BF16")
			return Type::BF16;
		if (type_str == "F8_E2M5")
			return Type::F8_E2M5;
		if (type_str == "F8_E3M4")
			return Type::F8_E3M4;
		if (type_str == "F8_E4M3")
			return Type::F8_E4M3;
		if (type_str == "F8_E5M2")
			return Type::F8_E5M2;
		if (type_str == "U8")
			return Type::U8;
		if (type_str == "Q8")
			return Type::Q8;

		console::print("\nERROR: invalid type: {}\n", type_str);
		assert(false && "invalid type");
		return Type::Unknown;
	}

	constexpr bool operator==(const Type& other) const { return id == other.id; }
	constexpr bool operator!=(const Type& other) const { return id != other.id; }
};

constexpr Type Type::Unknown{0, 0};
constexpr Type Type::F32{1, sizeof(uint32_t) * 8};
constexpr Type Type::F16{2, sizeof(uint16_t) * 8};
constexpr Type Type::BF16{3, sizeof(uint16_t) * 8};
constexpr Type Type::F8_E2M5{4, sizeof(uint8_t) * 8};
constexpr Type Type::F8_E3M4{5, sizeof(uint8_t) * 8};
constexpr Type Type::F8_E4M3{6, sizeof(uint8_t) * 8};
constexpr Type Type::F8_E5M2{7, sizeof(uint8_t) * 8};
constexpr Type Type::U8{8, sizeof(uint8_t) * 8};
constexpr Type Type::Q8{9, sizeof(int8_t) * 8};

// A helper template to select a type based on an index
template<size_t Index, typename... Types>
struct conditional_select;

template<size_t Index, typename First, typename... Rest>
struct conditional_select<Index, First, Rest...> {
	using type = typename conditional_select<Index - 1, Rest...>::type;
};

template<typename First, typename... Rest>
struct conditional_select<0, First, Rest...> {
	using type = First;
};

template<size_t Index, typename... Types>
using conditional_select_t = typename conditional_select<Index, Types...>::type;


template<typename ScalarType, int Size>
struct Bundle {
	using scalar_t = ScalarType;
	static constexpr size_t size = Size;
};

template<>
struct Bundle<float16_t, 1> {
	using bundle_t = float16_t;
};

template<>
struct Bundle<float16_t, 4> {
	using bundle_t = float16x4_t;
};

template<>
struct Bundle<float16_t, 8> {
	using bundle_t = float16x8_t;
};

template<>
struct Bundle<float32_t, 1> {
	using bundle_t = float32_t;
};

template<>
struct Bundle<float32_t, 4> {
	using bundle_t = float32x4_t;
};

[[maybe_unused]] static int8x16_t vld1q(const int8_t* src) { return vld1q_s8(src); }
[[maybe_unused]] static uint8x16_t vld1q(const uint8_t* src) { return vld1q_u8(src); }
[[maybe_unused]] static int16x8_t vld1q(const int16_t* src) { return vld1q_s16(src); }
[[maybe_unused]] static uint16x8_t vld1q(const uint16_t* src) { return vld1q_u16(src); }
[[maybe_unused]] static int32x4_t vld1q(const int32_t* src) { return vld1q_s32(src); }
[[maybe_unused]] static uint32x4_t vld1q(const uint32_t* src) { return vld1q_u32(src); }
[[maybe_unused]] static int64x2_t vld1q(const int64_t* src) { return vld1q_s64(src); }
[[maybe_unused]] static uint64x2_t vld1q(const uint64_t* src) { return vld1q_u64(src); }
[[maybe_unused]] static float16x8_t vld1q(const float16_t* src) { return vld1q_f16(src); }
[[maybe_unused]] static float32x4_t vld1q(const float32_t* src) { return vld1q_f32(src); }
[[maybe_unused]] static float64x2_t vld1q(const float64_t* src) { return vld1q_f64(src); }

template<typename ScalarType, int Size>
static typename Bundle<ScalarType, Size>::bundle_t load(const ScalarType* src) {
	return vld1q(&src[0]);
}
/*
[[maybe_unused]] inline void lllllal() {
	[[maybe_unused]] auto s = Bundle<float, 2>::size;

	[[maybe_unused]] Bundle<float16_t, 4>::bundle_t xy = {};

	const float* data;

	auto d = load<float16_t, 4>(data);
}*/


constexpr float constexpr_exp2(const int exponent) {
	float result = 1.0f;
	if (exponent > 0) {
		for (int i = 0; i < exponent; ++i) {
			result *= 2.0f;
		}
	} else if (exponent < 0) {
		for (int i = 0; i > exponent; --i) {
			result /= 2.0f;
		}
	}
	return result;
}

constexpr float constexpr_log2(const float x) {
	if (x <= 0.0f)
		return std::numeric_limits<float>::quiet_NaN(); // log2(0) or log2(negative) is undefined
	if (x == 1.0f)
		return 0.0f; // log2(1) = 0

	// Use a simple iterative approximation (Newton-Raphson method)
	float y = 0.0f;
	float z = x;
	while (z >= 2.0f) {
		z /= 2.0f;
		y += 1.0f;
	}
	while (z < 1.0f) {
		z *= 2.0f;
		y -= 1.0f;
	}

	// Refine the result using Newton-Raphson iterations
	for (int i = 0; i < 4; ++i) {
		float exp_y = constexpr_exp2(y);
		y -= (exp_y - z) / (exp_y * 0.69314718056f); // 0.69314718056f = ln(2)
	}

	return y;
}

// Standard mantissa mapping presets
template<uint8_t M>
consteval auto linear_mapping() {
	std::array<float, 1 << M> mapping{};
	for (uint8_t i = 0; i < (1 << M); ++i) {
		mapping[i] = 1.0f + static_cast<float>(i) * constexpr_exp2(-static_cast<float>(M));
	}
	return mapping;
}

template<uint8_t M>
consteval auto logarithmic_mapping(const float min = 1.0f, const float max = 10.0f) {
	std::array<float, 1 << M> mapping{};
	const float log_min = constexpr_log2(min);
	const float log_max = constexpr_log2(max);
	const float step = (log_max - log_min) / ((1 << M) - 1);

	for (uint8_t i = 0; i < (1 << M); ++i) {
		mapping[i] = constexpr_exp2(log_min + i * step);
	}
	return mapping;
}

template<uint8_t M>
consteval auto piecewise_linear_mapping(const std::array<float, 2>& ranges = {1.0f, 4.0f},
										const std::array<float, 2>& slopes = {1.0f, 0.5f}) {
	std::array<float, 1 << M> mapping{};
	const float breakpoint = ranges[0];
	const float total_range = (ranges[1] - ranges[0]) * slopes[0] + (mapping.size() - breakpoint) * slopes[1];
	const float norm = (mapping.size() - 1) / total_range;

	float acc = 0.0f;
	for (uint8_t i = 0; i < (1 << M); ++i) {
		if (i < breakpoint) {
			acc += slopes[0];
		} else {
			acc += slopes[1];
		}
		mapping[i] = acc * norm;
	}
	return mapping;
}

template<uint8_t E, uint8_t M, bool Denormals = true, bool NaNs = true, int Bias = (E ? (1 << (E - 1)) - 1 : 0),
		 std::array<float, 1 << M> MantissaMapping = linear_mapping<M>()>
	requires((E + M + 1 <= 16) && (E == 0 || M > 0))
struct custom_float {
private:
	uint16_t bits = 0;
	explicit constexpr custom_float(uint16_t bits) noexcept : bits(bits) {}

	static constexpr size_t TOTAL_BITS = E + M + 1;
	static constexpr int EXP_MAX = NaNs ? (1 << E) - 2 : (1 << E) - 1;
	static constexpr float MAX_NORMAL = MantissaMapping.back() * constexpr_exp2(EXP_MAX - Bias);
	static constexpr float MIN_NORMAL = MantissaMapping[0] * constexpr_exp2(1 - Bias);
	static constexpr float MIN_DENORMAL = Denormals ? constexpr_exp2(static_cast<float>(1 - Bias - M)) : 0.0f;

public:
	static constexpr custom_float from(float value) noexcept {
		if constexpr (NaNs) {
			if (std::isnan(value))
				return create_nan(std::signbit(value));
		}

		const bool sign = std::signbit(value);
		value = std::abs(value);

		if (value == 0.0f)
			return custom_float(sign << (E + M));
		if (value >= MAX_NORMAL)
			return create_inf(sign);

		if (value < MIN_NORMAL) {
			if constexpr (Denormals) {
				return create_denormal(sign, value);
			} else {
				return custom_float(sign << (E + M));
			}
		}

		return create_normal(sign, value);
	}

	static constexpr float to_float(custom_float val) noexcept {
		const bool sign = val.bits >> (E + M);
		const uint8_t exp = (val.bits >> M) & ((1 << E) - 1);
		const uint8_t mant = val.bits & ((1 << M) - 1);

		if constexpr (NaNs) {
			if (exp == (1 << E) - 1)
				return sign ? -std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::quiet_NaN();
		}

		if (exp == EXP_MAX + (NaNs ? 0 : 1)) {
			return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
		}

		if (exp == 0) {
			if constexpr (Denormals) {
				return sign ? -(mant * MIN_DENORMAL) : mant * MIN_DENORMAL;
			} else {
				return sign ? -0.0f : 0.0f;
			}
		}

		return sign ? -MantissaMapping[mant] * std::exp2(exp - Bias) : MantissaMapping[mant] * std::exp2(exp - Bias);
	}

	static constexpr bool is_nan(auto x) noexcept {
		auto bits = x.bits;
		if constexpr (NaNs) {
			const uint8_t exp = (bits >> M) & ((1 << E) - 1);
			const uint8_t mant = bits & ((1 << M) - 1);
			return (exp == ((1 << E) - 1)) && (mant != 0);
		}
		return false;
	}

	static constexpr bool is_inf(auto x) noexcept {
		auto bits = x.bits;
		const uint8_t exp = (bits >> M) & ((1 << E) - 1);
		const uint8_t mant = bits & ((1 << M) - 1);
		return (exp == ((1 << E) - 1)) && (mant == 0);
	}

	static constexpr bool is_denorm(auto x) noexcept {
		auto bits = x.bits;

		if constexpr (Denormals) {
			const uint8_t exp = (bits >> M) & ((1 << E) - 1);
			return (exp == 0) && (bits != 0);
		}
		return false;
	}

	static constexpr std::string describe() {
		std::string s;

		// Format description
		s += std::format("Custom Floating-Point Format: f{}e{}m{}\n", TOTAL_BITS, E, M);
		s += std::format("  Bias: {}\n", Bias);
		s += std::format("  Denormals: {}\n", Denormals);
		s += std::format("  NaNs: {}\n", NaNs);
		s += "\n";

		// Bit layout
		s += "Bit Layout:\n";
		s += std::format("  [S:1|E:{}|M:{}]\n", E, M);
		s += "  S = Sign bit (0=positive, 1=negative)\n";
		s += std::format("  E = Exponent bits (biased by {})\n", Bias);
		s += "  M = Mantissa bits\n";
		s += "\n";

		// Value ranges
		s += "Value Ranges:\n";
		s += std::format("  Minimum Normal: {:.4f}\n", MIN_NORMAL);
		s += std::format("  Maximum Normal: {:.4f}\n", MAX_NORMAL);
		if constexpr (Denormals) {
			s += std::format("  Minimum Denormal: {:.4f}\n", MIN_DENORMAL);
		}
		s += "\n";

		// Special values
		s += "Special Values:\n";
		s += std::format("  Zero: {}\n", to_float(custom_float(0)));
		s += std::format("  Negative Zero: {}\n", to_float(custom_float(1 << (E + M))));
		if constexpr (NaNs) {
			s += std::format("  NaN: {}\n", to_float(custom_float((1 << (E + M)) | ((1 << E) - 1) << M | 1)));
		}
		// s += std::format("  +Infinity: {}\n", to_float(custom_float((1 << E) - 1) << M));
		// s += std::format("  -Infinity: {}\n", to_float(custom_float((1 << (E + M)) | ((1 << E) - 1) << M)));
		s += "\n";

		auto fmt = [](auto i, auto val) {
			auto flags = val.is_denorm(val) ? " (d)" : "";
			return std::format("{:<22}", std::format("0x{:02x}: {:.4f}{}", i, to_float(val), flags));
		};

		// All encodings
		s += "All Encodings:\n";
		constexpr size_t total_encodings = 1 << TOTAL_BITS;
		for (size_t i = 0; i < total_encodings; i += 4) {
			// First column
			custom_float val1(i);
			s += fmt(i, val1);

			// Second column (if exists)
			if (i + 1 < total_encodings) {
				custom_float val2(i + 1);
				s += fmt(i + 1, val2);
			}

			// Third column (if exists)
			if (i + 2 < total_encodings) {
				custom_float val3(i + 2);
				s += fmt(i + 2, val3);
			}

			// Third column (if exists)
			if (i + 3 < total_encodings) {
				custom_float val4(i + 3);
				s += fmt(i + 3, val4);
			}

			s += "\n";
		}

		return s;
	}

private:
	static constexpr custom_float create_inf(bool sign) noexcept {
		return custom_float((sign << (E + M)) | ((EXP_MAX + 1) << M));
	}

	static constexpr custom_float create_nan(bool sign) noexcept {
		return custom_float((sign << (E + M)) | ((EXP_MAX + 1) << M) | 1);
	}

	static constexpr custom_float create_denormal(bool sign, float value) noexcept {
		const float scaled = value / MIN_DENORMAL;
		const uint8_t mant = static_cast<uint8_t>(std::min(scaled, (1 << M) - 1.0f));
		return custom_float((sign << (E + M)) | mant);
	}

	static constexpr custom_float create_normal(bool sign, float value) noexcept {
		int exp = std::ilogb(value) + Bias;
		exp = std::clamp(exp, 1, EXP_MAX);
		const float scaled = std::scalbn(value, -exp);
		uint8_t mant = find_closest_mantissa(scaled);
		return custom_float((sign << (E + M)) | (exp << M) | mant);
	}

	static constexpr uint8_t find_closest_mantissa(float value) noexcept {
		uint8_t best = 0;
		float best_diff = std::abs(value - MantissaMapping[0]);
		for (uint8_t i = 1; i < MantissaMapping.size(); ++i) {
			const float diff = std::abs(value - MantissaMapping[i]);
			if (diff < best_diff)
				best = i, best_diff = diff;
		}
		return best;
	}

	template<typename>
	static constexpr bool dependent_false = false;
};

using f6e2m3_t = custom_float<2, 3, true, false, 3>; // Linear mapping
