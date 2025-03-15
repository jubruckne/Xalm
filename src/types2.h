#pragma once
#include <arm_neon.h>
#include <array>
#include <cstdint>
#include <functional>

namespace detail {
	template<typename T>
	concept SimdType =
	    std::is_same_v<T, float> ||
	    std::is_same_v<T, float16_t> ||
	    std::is_same_v<T, bfloat16_t> ||
	    std::is_same_v<T, int8_t> ||
	    std::is_same_v<T, int16_t>;

    template<SimdType T, size_t Size>
    struct simd_register_info;

    template<>
    struct simd_register_info<float32_t, 4> {
        using scalar_type = float32_t;
        using reg_type = float32x4_t;
        static constexpr size_t regs = 1;
    	static constexpr size_t size = 4;
    	static constexpr std::string_view name = "float32x4_t";

    	[[nodiscard]] static reg_type dup(const scalar_type v) {
    		return vdupq_n_f32(v);
    	}

    	[[nodiscard]] static reg_type add(const reg_type lhs, const reg_type rhs) {
    		return vaddq_f32(lhs, rhs);
    	}

    	[[nodiscard]] static std::array<scalar_type, size> store(const reg_type r) {
    		std::array<scalar_type, size> result{};
    		vst1q_f32(result.data(), r);
    		return result;
    	}
    };

	template<>
	struct simd_register_info<float32_t, 8> {
		using scalar_type = float32_t;
		using reg_type = float32x4x2_t;
		static constexpr size_t regs = 2;
		static constexpr size_t size = 8;
		static constexpr std::string_view name = "float32x4x2_t";

		[[nodiscard]] static reg_type dup(const scalar_type v) {
			const float32x4_t r = vdupq_n_f32(v);
			return {r, r};
		}

		[[nodiscard]] static reg_type add(const reg_type lhs, const reg_type rhs) {
			return {
				vaddq_f32(lhs.val[0], rhs.val[0]),
				vaddq_f32(lhs.val[1], rhs.val[1])
			};
		}

		[[nodiscard]] static std::array<scalar_type, size> store(const reg_type r) {
			std::array<scalar_type, size> result{};
    		vst1q_f32_x2(result.data(), r);
			return result;
		}
	};

	template<>
	struct simd_register_info<float32_t, 16> {
		using scalar_type = float32_t;
		using reg_type = float32x4x4_t;
		static constexpr size_t regs = 4;
		static constexpr size_t size = 16;
		static constexpr std::string_view name = "float32x4x4_t";

		[[nodiscard]] static reg_type dup(const scalar_type v) {
			const float32x4_t r = vdupq_n_f32(v);
			return {r, r, r, r};
		}

		[[nodiscard]] static reg_type add(const reg_type lhs, const reg_type rhs) {
			return {
				vaddq_f32(lhs.val[0], rhs.val[0]),
				vaddq_f32(lhs.val[1], rhs.val[1]),
				vaddq_f32(lhs.val[2], rhs.val[2]),
				vaddq_f32(lhs.val[3], rhs.val[3])
			};
		}

		[[nodiscard]] static std::array<scalar_type, size> store(const reg_type r) {
			std::array<scalar_type, size> result{};
			vst1q_f32_x4(result.data(), r);
			return result;
		}
	};

	template<size_t Size>
    concept ValidBundleSize =
        Size == 16 || Size == 32 || Size == 64 || Size == 128 || Size == 256;

	template<SimdType T>
	constexpr size_t bundle_alignment = 16;

	template<SimdType T, size_t Size>
	constexpr bool verify_bundle_alignment() {
		using traits = simd_register_info<T, Size>;
		constexpr size_t required_alignment = bundle_alignment<T>;
		constexpr size_t natural_alignment = alignof(typename traits::reg_array_type);
		return natural_alignment >= required_alignment;
	}
} // namespace detail


// A simple consteval function to concatenate string literals.
template <std::size_t N, std::size_t... Ns>
consteval auto concat(const char (&first)[N], const char (&... rest)[Ns]) {
	// Each literal includes a null terminator; ignore those except for one final one.
	constexpr std::size_t totalSize = ( (N - 1) + ... + (Ns - 1) ) + 1;
	std::array<char, totalSize> result{}; // will hold all characters plus a final '\0'

	std::size_t pos = 0;
	// A lambda that copies one string (excluding its null terminator).
	auto copy = [&pos, &result](auto str, std::size_t size) {
		for (std::size_t i = 0; i < size - 1; ++i)
			result[pos++] = str[i];
	};

	// Copy the first string and then the rest.
	copy(first, N);
	((copy(rest, Ns)), ...);

	// Ensure the result is null-terminated.
	result[totalSize - 1] = '\0';
	return std::string(result);
}

template<detail::SimdType T, size_t Size>
class bundle {
	static constexpr size_t alignment = detail::bundle_alignment<T>;
public:
	static constexpr size_t size = Size;

	using value_type = T;
	using bundle_type = bundle;
	using simd_type = detail::simd_register_info<value_type, Size>;
	using reg_type = typename detail::simd_register_info<value_type, Size>::reg_type;

	reg_type data;

	bundle() noexcept : bundle(0) {}
	explicit bundle(const reg_type val) noexcept : data(val) {}
	explicit bundle(const value_type val) noexcept { data = simd_type::dup(val); }

	[[nodiscard]] bundle operator+(const bundle& other) const {
		return bundle{simd_type::add(data, other.data)};
	}

	[[nodiscard]] std::array<value_type, Size> to_array() const {
		return simd_type::store(data);
	}
};

template<detail::SimdType T, size_t Size>
struct std::formatter<bundle<T, Size>> {
	constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

	template <typename FormatContext>
	auto format(const bundle<T, Size>& k, FormatContext& ctx) const {
		return std::format_to(ctx.out(), "bundle<{}, {}> [{}]", typeid(T).name(), k.size, k.to_array());
	}
};

inline void test2223() {
	float32_t v1 = 1.f;
	const bundle<float32_t, 16> b1{v1};
	const bundle<float32_t, 16> b2{2.2f};

	auto ggg = b1 + b2;
	console::print("{}", ggg);
}
/*

    [[nodiscard]] bundle operator*(const bundle& other) const {
        bundle result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        result.data.val[2] = vmulq_f32(data.val[2], other.data.val[2]);
        result.data.val[3] = vmulq_f32(data.val[3], other.data.val[3]);
        return result;
    }

    [[nodiscard]] value_type dot(const bundle& other) const {
        // Multiply corresponding elements
        float32x4_t mul0 = vmulq_f32(data.val[0], other.data.val[0]);
        float32x4_t mul1 = vmulq_f32(data.val[1], other.data.val[1]);
        float32x4_t mul2 = vmulq_f32(data.val[2], other.data.val[2]);
        float32x4_t mul3 = vmulq_f32(data.val[3], other.data.val[3]);

        // Add pairs within each vector
        float32x4_t sum01 = vpaddq_f32(mul0, mul1);
        float32x4_t sum23 = vpaddq_f32(mul2, mul3);

        // Add the results together
        float32x4_t sum = vpaddq_f32(sum01, sum23);

        // Final horizontal add
        return vaddvq_f32(sum);
    }

    [[nodiscard]] bundle fma(const bundle& mul, const bundle& add) const {
		bundle result;
		result.data.val[0] = vfmaq_f32(add.data.val[0], data.val[0], mul.data.val[0]);
		result.data.val[1] = vfmaq_f32(add.data.val[1], data.val[1], mul.data.val[1]);
		result.data.val[2] = vfmaq_f32(add.data.val[2], data.val[2], mul.data.val[2]);
		result.data.val[3] = vfmaq_f32(add.data.val[3], data.val[3], mul.data.val[3]);
		return result;
	}

    static bundle load(const value_type* ptr) {
        bundle result;
        result.data = vld1q_f32_x4(ptr);
        return result;
    }

    static void store(value_type* ptr, const bundle& b) {
        vst1q_f32_x4(ptr, b.data);
    }

    [[nodiscard]] std::array<value_type, SIZE> to_array() const {
        std::array<value_type, SIZE> result{};
        vst1q_f32_x4(result.data(), data);
        return result;
    }

    [[nodiscard]] bundle sqrt() const {
        bundle result;
        result.data.val[0] = vsqrtq_f32(data.val[0]);
        result.data.val[1] = vsqrtq_f32(data.val[1]);
        result.data.val[2] = vsqrtq_f32(data.val[2]);
        result.data.val[3] = vsqrtq_f32(data.val[3]);
        return result;
    }

    [[nodiscard]] bundle abs() const {
        bundle result;
        result.data.val[0] = vabsq_f32(data.val[0]);
        result.data.val[1] = vabsq_f32(data.val[1]);
        result.data.val[2] = vabsq_f32(data.val[2]);
        result.data.val[3] = vabsq_f32(data.val[3]);
        return result;
    }

	[[nodiscard]] static bundle from_float16(const bundle<float16_t>& other) noexcept;
};

template<>
class bundle<float16_t> {
	static constexpr size_t SIZE = 16;
	static constexpr size_t ALIGNMENT = 16;

public:
	using value_type = float16_t;
	using bundle_typer = bundle;
	using vector_type = float16x8x2_t;

	vector_type data{};

	bundle() noexcept : bundle(0) {}

	explicit bundle(value_type val) noexcept  {
		float16x8_t v = vdupq_n_f16(val);
		data.val[0] = v;
		data.val[1] = v;
	}

	explicit bundle(const vector_type val) noexcept : data(val) {}

	[[nodiscard]] bundle operator+(const bundle& other) const {
		bundle result;
		result.data.val[0] = vaddq_f16(data.val[0], other.data.val[0]);
		result.data.val[1] = vaddq_f16(data.val[1], other.data.val[1]);
		return result;
	}

	[[nodiscard]] bundle operator*(const bundle& other) const {
		bundle result;
		result.data.val[0] = vmulq_f16(data.val[0], other.data.val[0]);
		result.data.val[1] = vmulq_f16(data.val[1], other.data.val[1]);
		return result;
	}

	[[nodiscard]] value_type dot(const bundle& other) const {
		// Multiply corresponding elements
		float16x8_t mul0 = vmulq_f16(data.val[0], other.data.val[0]);
		float16x8_t mul1 = vmulq_f16(data.val[1], other.data.val[1]);

		// Convert to float32 for more accurate reduction
		const float32x4_t sum0_low = vcvt_f32_f16(vget_low_f16(mul0));
		const float32x4_t sum0_high = vcvt_f32_f16(vget_high_f16(mul0));
		const float32x4_t sum1_low = vcvt_f32_f16(vget_low_f16(mul1));
		const float32x4_t sum1_high = vcvt_f32_f16(vget_high_f16(mul1));

		// Add all parts
		const float32x4_t sum = vaddq_f32(vaddq_f32(sum0_low, sum0_high), vaddq_f32(sum1_low, sum1_high));

		// Convert back to float16
		float16x4_t result = vcvt_f16_f32(sum);
		return vget_lane_f16(result, 0);
	}

	[[nodiscard]] bundle fma(const bundle& mul, const bundle& add) const {
		bundle result;
		result.data.val[0] = vfmaq_f16(add.data.val[0], data.val[0], mul.data.val[0]);
		result.data.val[1] = vfmaq_f16(add.data.val[1], data.val[1], mul.data.val[1]);
		return result;
	}

	[[nodiscard]] static bundle load(const value_type* ptr) {
		bundle result;
		result.data = vld1q_f16_x2(ptr);
		return result;
	}

	static void store(value_type* ptr, const bundle& b) { vst1q_f16_x2(ptr, b.data); }

	[[nodiscard]] static bundle from_float32(const bundle<float>& other) noexcept;
};

template<>
class bundle<int8_t> {
    static constexpr size_t SIZE = 16;
    static constexpr size_t ALIGNMENT = 16;

	using value_type = int8_t;
	using bundle_typer = bundle;
	using vector_type = int8x16_t;

	vector_type data{};

public:
    bundle() noexcept : bundle(0) {}

	explicit bundle(const value_type val) noexcept : data(vdupq_n_s8(val)) {}
	explicit bundle(const vector_type val) noexcept : data(val) {}

    [[nodiscard]] bundle operator+(const bundle& other) const {
        return bundle{vaddq_s8(data, other.data)};
    }

    [[nodiscard]] bundle operator-(const bundle& other) const {
        return bundle(vsubq_s8(data, other.data));
    }

    [[nodiscard]] bundle operator*(const bundle& other) const {
        // NEON does not provide a direct multiplication for int8x16_t,
        // so we use vmull_s8 and then narrow back.
		const vector_type low = vmull_s8(vget_low_s8(data), vget_low_s8(other.data));
		const vector_type high = vmull_s8(vget_high_s8(data), vget_high_s8(other.data));
        return bundle(vcombine_s8(vmovn_s16(low), vmovn_s16(high)));
    }

	[[nodiscard]] int32_t dot(const bundle& other) const {
#ifdef __ARM_FEATURE_DOTPROD
    	int32x4_t acc = vdupq_n_s32(0);  // Initialize accumulator
    	acc = vdotq_s32(acc, data, other.data);  // Compute dot product
    	return vaddvq_s32(acc);  // Reduce to scalar
#else
    	const vector_type low = vmull_s8(vget_low_s8(data), vget_low_s8(other.data));
		const vector_type high = vmull_s8(vget_high_s8(data), vget_high_s8(other.data));
		const vector_type sum = vaddq_s16(low, high);
		const int32x4_t sum32 = vpaddlq_s16(sum);  // Pairwise add to int32x4_t
        return vaddvq_s32(sum32);  // Sum all elements
#endif
    }

    [[nodiscard]] bundle min(const bundle& other) const {
        return bundle(vminq_s8(data, other.data));
    }

    [[nodiscard]] bundle max(const bundle& other) const {
        return bundle(vmaxq_s8(data, other.data));
    }

    [[nodiscard]] static bundle load(const value_type* ptr) {
        return bundle(vld1q_s8(ptr));
    }

    static void store(value_type* ptr, const bundle& b) {
        vst1q_s8(ptr, b.data);
    }

    [[nodiscard]] std::array<value_type, SIZE> to_array() const {
        std::array<value_type, SIZE> result{};
        vst1q_s8(result.data(), data);
        return result;
    }

    [[nodiscard]] bundle abs() const {
        return bundle(vabsq_s8(data));
    }
};

template<>
class bundle<uint8_t> {
    static constexpr size_t SIZE = 16;
    static constexpr size_t ALIGNMENT = 16;

	using value_type = uint8_t;
	using bundle_typer = bundle;
	using vector_type = uint8x16_t;

	vector_type data{};

public:
    bundle() noexcept : bundle(0) {}

	explicit bundle(const value_type val) noexcept : data(vdupq_n_u8(val)) {}
	explicit bundle(const vector_type val) noexcept : data(val) {}

    [[nodiscard]] bundle operator+(const bundle& other) const {
        return bundle(vaddq_u8(data, other.data));
    }

    [[nodiscard]] bundle operator-(const bundle& other) const {
        return bundle(vsubq_u8(data, other.data));
    }

    [[nodiscard]] bundle operator*(const bundle& other) const {
        // NEON does not provide direct multiplication for uint8x16_t,
        // so we use vmull_u8 and then narrow back.
		const uint16x8_t low = vmull_u8(vget_low_u8(data), vget_low_u8(other.data));
		const uint16x8_t high = vmull_u8(vget_high_u8(data), vget_high_u8(other.data));

        return bundle(vcombine_u8(vmovn_u16(low), vmovn_u16(high)));
    }

	[[nodiscard]] uint32_t dot(const bundle& other) const {
#ifdef __ARM_FEATURE_DOTPROD
    	uint32x4_t acc = vdupq_n_u32(0);  // Initialize accumulator
    	acc = vdotq_u32(acc, data, other.data);  // Compute dot product
    	return vaddvq_u32(acc);  // Reduce to scalar
#else
		const uint16x8_t low = vmull_u8(vget_low_u8(data), vget_low_u8(other.data));
		const uint16x8_t high = vmull_u8(vget_high_u8(data), vget_high_u8(other.data));

		const uint16x8_t sum = vaddq_u16(low, high);
		const uint32x4_t sum32 = vpaddlq_u16(sum);  // Pairwise add to uint32x4_t
        return vaddvq_u32(sum32);  // Sum all elements
#endif
    }

    [[nodiscard]] bundle min(const bundle& other) const {
        return bundle(vminq_u8(data, other.data));
    }

    [[nodiscard]] bundle max(const bundle& other) const {
        return bundle(vmaxq_u8(data, other.data));
    }

    [[nodiscard]] static bundle load(const uint8_t* ptr) {
        return bundle(vld1q_u8(ptr));
    }

    static void store(uint8_t* ptr, const bundle& b) {
        vst1q_u8(ptr, b.data);
    }

    [[nodiscard]] std::array<value_type, SIZE> to_array() const {
        std::array<value_type, SIZE> result{};
        vst1q_u8(result.data(), data);
        return result;
    }
};

inline bundle<float32_t> bundle<float32_t>::from_float16(const bundle<float16_t>& other) noexcept {
	bundle result;
	// Convert each half of each float16x8_t to float32x4_t
	result.data.val[0] = vcvt_f32_f16(vget_low_f16(other.data.val[0]));
	result.data.val[1] = vcvt_f32_f16(vget_high_f16(other.data.val[0]));
	result.data.val[2] = vcvt_f32_f16(vget_low_f16(other.data.val[1]));
	result.data.val[3] = vcvt_f32_f16(vget_high_f16(other.data.val[1]));
	return result;
}

inline bundle<float16_t> bundle<float16_t>::from_float32(const bundle<float32_t>& other) noexcept {
    bundle result;
    // Convert pairs of float32x4_t to float16x8_t
    result.data.val[0] = vcombine_f16(
        vcvt_f16_f32(other.data.val[0]),
        vcvt_f16_f32(other.data.val[1])
    );
    result.data.val[1] = vcombine_f16(
        vcvt_f16_f32(other.data.val[2]),
        vcvt_f16_f32(other.data.val[3])
    );
    return result;
}
*/

struct type {
	const int id;
	// constexpr type(const int v) noexcept : id(v) {}

	template<int type_id>
	static consteval type make() {
		return type{type_id};
	}

	constexpr bool operator==(const type& other) const noexcept { return id == other.id; }
	constexpr bool operator!=(const type& other) const noexcept { return id != other.id; }
	constexpr operator int() const noexcept { return id; }

	/*	[[nodiscard]] static constexpr type parse(std::string_view str) {
			for (const auto& t : TypeRegistry) {
				if (t.name == str) return t;
			}
			throw std::invalid_argument("Invalid Type name");
		}*/
};

constexpr type F32 = type::make<1>();
constexpr type F16 = type::make<2>();

template <type T>
struct traits;

template<> struct traits<F32> {
	using storage_type = float32_t;
	using value_type = float32_t;
	static constexpr uint8_t bit_size = sizeof(value_type);
	static constexpr std::string_view name = "f32";

	[[nodiscard]] static float32_t get_float(const void* data, const std::size_t offset) {
		return static_cast<const value_type*>(data)[offset];
	}

	static void set_float(void* data, const std::size_t offset, const float32_t value) {
		static_cast<value_type*>(data)[offset] = value;
	}
};

template<> struct traits<F16> {
	using storage_type = float16_t;
	using value_type = float16_t;
	static constexpr uint8_t bit_size = sizeof(value_type);
	static constexpr std::string_view name = "f16";

	[[nodiscard]] static float32_t get_float(const void* data, const std::size_t offset) {
		return static_cast<const value_type*>(data)[offset];
	}

	static void set_float(void* data, const std::size_t offset, const float32_t value) {
		static_cast<value_type*>(data)[offset] = static_cast<value_type>(value);
	}
};

template <>
struct std::hash<type> {
	size_t operator()(const type& t) const noexcept {
		return std::hash<int>()(t.id);
	}
};