#pragma once
#include <arm_neon.h>
#include <array>
#include <concepts>
#include <cstdint>
#include <functional>

template<typename T>
concept BundleType = std::same_as<T, float32_t> || std::same_as<T, float16_t> || std::same_as<T, int8_t> || std::same_as<T, uint8_t>;

template<BundleType T>
class bundle;

template<>
class bundle<float32_t> {
    static constexpr size_t SIZE = 16;
    static constexpr size_t ALIGNMENT = 16;

public:
    using value_type = float32_t;
    using bundle_type = bundle;
	using vector_type = float32x4x4_t;

	vector_type data{};

    bundle() noexcept : bundle(0) {}

	explicit bundle(const vector_type val) noexcept : data(val) {}

    explicit bundle(const value_type val) noexcept {
		float32x4_t v = vdupq_n_f32(val);
		data.val[0] = v;
		data.val[1] = v;
		data.val[2] = v;
		data.val[3] = v;
	}

	[[nodiscard]] bundle operator+(const bundle& other) const {
        bundle result;
        result.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        result.data.val[2] = vaddq_f32(data.val[2], other.data.val[2]);
        result.data.val[3] = vaddq_f32(data.val[3], other.data.val[3]);
        return result;
    }

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


namespace types2 {
	enum class Type {
		unknown = 0,
		f32 = 1,
		f16 = 2,
		bf16 = 3,
		f8_e2m5 = 4,
		f8_e3m4 = 5,
		f8_e4m3 = 6,
		f8_e5m2 = 7,
		u8 = 8
	};

	template <Type TYPE>
	struct TypeDescriptor;

	template <>
	struct TypeDescriptor<Type::f32> {
		using scalar_type = float32_t;
		using native_type = float32_t;
		using bundle_type = bundle<native_type>;

		constexpr static auto type = Type::f32;
		constexpr static std::string_view name = "f32";
		constexpr static size_t bit_size = sizeof(float32_t) * 8;
		constexpr static bool is_floating_point = true;
		constexpr static bool is_native_type = true;

		static constexpr auto load_bundle = [](const bundle_type::value_type* ptr) {
			return bundle_type::load(ptr);
		};

		[[nodiscard]] static scalar_type load_scalar(const void* ptr) noexcept {
			return *static_cast<const scalar_type*>(ptr);
		}

		static void store_scalar(void* ptr, const scalar_type value) noexcept {
			*static_cast<scalar_type*>(ptr) = value;
		}
	};

	void test() {
		using type = TypeDescriptor<Type::f32>;
		const auto b = type::load_bundle(nullptr);
		console::print(type::name);
	}

}