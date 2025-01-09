#pragma once
#include <cstdint>
#include <cassert>

#if defined(__AVX2__) && defined(__F16C__)
  #include <immintrin.h> // Intel/AVX2
#elif defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>  // ARM NEON
#endif


template<int8_t N> consteval float EXP2() noexcept {
  if constexpr (N==0) return 1;
  if constexpr (N>0) return EXP2<N-1>()*2;
  if constexpr (N<0) return EXP2<N+1>()/2;
}

template<int8_t N> consteval int EXP_I2() noexcept requires (N >= 0) {
  if constexpr (N==0) return 1;
  if constexpr (N>0) return EXP_I2<N-1>()*2;
}

template<uint8_t E, uint8_t M> requires (M > 0 && E+M == 7)
struct f8_t {
private:
  uint8_t bits = 0;

  explicit f8_t(const uint8_t bits) noexcept: bits(bits) {}

  static constexpr int E_BIAS = EXP2<E-1>()-1;
  static constexpr float E_BIAS_MINUS_127 = EXP2<E_BIAS-127>();
  static constexpr float E_127_MINUS_BIAS = EXP2<127-E_BIAS>();
  static constexpr float max = (2-EXP2<-M+1>())*EXP2<EXP_I2<E-1>()>();
  static constexpr float min = EXP2<-M>()*EXP2<2-EXP_I2<E-1>()>();
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
      in.bits += 1 << (22-M);
      bits |= (in.bits >> (23-M)) & 0x7F;
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
    _bits <<= (23-M);
    out.bits |= _bits;
    out.f *= E_127_MINUS_BIAS;
    return out.f;
  }
};

using f8e4m3_t = f8_t<4, 3>;
using f8e5m2_t = f8_t<5, 2>;

struct qi4_t {
  static constexpr int block_length = 32;

  struct block {
    float16_t scale;
    uint8_t v[block_length / 2];
  };

  static constexpr int block_size = sizeof(block);


  static constexpr int8_t lut[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

  static constexpr float16_t lut2[16] = {
    -1.000000, -0.818898, -0.653543, -0.511811,
    -0.385827, -0.275591, -0.173228, -0.078740,
    0.007874, 0.102362, 0.196850, 0.309213,
    0.437323, 0.583307,  0.770787, 1.000000
  };

  static void store_f32(uint8_t* data, const size_t index, const float value) {
    // Quantization logic: map the float value to a 4-bit range (0 to 15)
    auto q = static_cast<uint8_t>(std::round(value * 15.0f));
    q = std::min<uint8_t>(15, q);

    const size_t byte_index = index / 2;
    const size_t nibble_index = index % 2;

    if (nibble_index == 0) {
      data[byte_index] = (data[byte_index] & 0x0F) | (q << 4);
    } else {
      data[byte_index] = (data[byte_index] & 0xF0) | (q & 0x0F);
    }
  }

  static float load_f32(const uint8_t* data, const size_t index) {
    const size_t byte_index = index / 2;
    const size_t nibble_index = index % 2;

    uint8_t quantized;
    if (nibble_index == 0) {
      quantized = (data[byte_index] >> 4) & 0x0F;
    } else {
      quantized = data[byte_index] & 0x0F;
    }

    return static_cast<float32_t>(quantized) / 15.0f;
  }
};

/*
namespace Types {
  template <typename T>
  concept Type = requires(T a, typename T::native_type* data, float& f32, float32x4_t& f32x4, float& f16, float16x4_t& f16x4) {
    typename T::type;
    typename T::native_type;
    { T::name } -> std::convertible_to<std::string_view>;
    { T::bit_size } -> std::convertible_to<size_t>;
    { T::load_f16_x4(data) } -> std::convertible_to<float16x4_t>;
    { T::store_f16_x4(data, f16x4) };
    { T::load_f32_x4(data) } -> std::convertible_to<float32x4_t>;
    { T::store_f32_x4(data, f32x4) };
  };

  struct f32_t {
    using type = f32_t;
    using native_type = float32_t;
    static constexpr size_t bit_size = sizeof(native_type) * 8;
    static constexpr std::string_view name = "F32";

    static float32x4_t load_f32_x4(const native_type* data) {
      return vld1q_f32(data);
    }

    static void store_f32_x4(native_type* data, const float32x4_t value) {
      vst1q_f32(data, value);
    }

    static float16x4_t load_f16_x4(const native_type* data) {
      return vcvt_f16_f32(vld1q_f32(data));
    }

    static void store_f16_x4(native_type* data, const float16x4_t value) {
      vst1q_f32(data, vcvt_f32_f16(value));
    }
  };

  struct f16_t {
    using type = f32_t;
    using native_type = float16_t;
    static constexpr size_t bit_size = sizeof(native_type) * 8;
    static constexpr std::string_view name = "F16";

    static float32x4_t load_f32_x4(const native_type* data) {
      auto f16_values = vld1_f16(data);
      return vcvt_f32_f16(f16_values);
    }

    static void store_f32_x4(native_type* data, const float32x4_t values) {
      vst1_f16(data, vcvt_f16_f32(values));
    }

    static float16x4_t load_f16_x4(void* data) {
      return vcvt_f16_f32(vld1q_f32(static_cast<float32_t*>(data)));
    }

    static void store_f16_x4(void* data, const float16x4_t value) {
      vst1q_f32(static_cast<float*>(data), vcvt_f32_f16(value));
    }
  };
};

namespace Type {
  using F32 = Types::f32_t;
  static_assert(Types::Type<F32>, "f32_t must implement the Type concept!");

  using F16 = Types::f16_t;
  static_assert(Types::Type<F16>, "f16_t must implement the Type concept!");
}

*/

struct Type {
  static const Type Unknown;
  static const Type F32;
  static const Type F16;
  static const Type F8;
  static const Type F8_E5M2;
  static const Type U8;
  static const Type QI4;

  int id;
  uint8_t bit_size;

  constexpr Type(const int v, const size_t bit_size) noexcept : id(v), bit_size(bit_size) {  }

  ~Type() noexcept = default;

  constexpr operator int() const noexcept { return id; }

  [[nodiscard]] constexpr std::string_view name() const {
    if (*this == Type::F32) return "F32";
    if (*this == Type::F16) return "F16";
    if (*this == Type::F8) return "F8";
    if (*this == Type::F8_E5M2) return "F8_E5M2";
    if (*this == Type::U8) return "U8";
    if (*this == Type::QI4) return "QF4";
    return "UNKNOWN";
  }

  [[nodiscard]] constexpr size_t byte_offset(const std::size_t offset) const {
    if (*this == Type::F32) return offset * sizeof(float32_t);
    if (*this == Type::F16) return offset * sizeof(float16_t);
    if (*this == Type::F8) return offset * sizeof(uint8_t);
    if (*this == Type::F8_E5M2) return offset * sizeof(uint8_t);
    if (*this == Type::U8) return offset * sizeof(uint8_t);
    if (*this == Type::QI4) return offset * sizeof(uint8_t) / 2;
    return 0;
  };


  [[nodiscard]] constexpr const void* data_ptr(const void* data, const std::size_t offset) const {
    return static_cast<const uint8_t*>(data) + byte_offset(offset);
  };

  float32_t get_float(const void* data, const std::size_t offset) const {
    auto d = data_ptr(data, offset);

    if (id == Type::F32.id) return *static_cast<const float32_t*>(d);
    if (id == Type::F16.id) return *static_cast<const float16_t*>(d);
    if (id == Type::F8.id) return f8e4m3_t::to_float(*static_cast<const f8e4m3_t*>(d));
    if (id == Type::F8_E5M2.id) return f8e5m2_t::to_float(*static_cast<const f8e5m2_t*>(d));

    return 666.66f;
  }

  [[nodiscard]] static Type parse(const std::string_view str) {
    auto to_upper = [](const std::string_view s) -> std::string {
      std::string result(s);
      std::ranges::transform(result, result.begin(),
                             [](const unsigned char c) { return std::toupper(c); });
      return result;
    };

    const auto type_str = to_upper(str);

    if (type_str == "F32") return Type::F32;
    if (type_str == "F16") return Type::F16;
    if (type_str == "F8") return Type::F8;
    if (type_str == "F8.E4M3") return Type::F8;
    if (type_str == "F8.E5M2") return Type::F8_E5M2;
    if (type_str == "U8") return Type::U8;
    if (type_str == "QF4") return Type::QI4;
    return Type::Unknown;
  }

  constexpr bool operator==(const Type& other) const {
    return id == other.id;
  }

  constexpr bool operator!=(const Type& other) const {
    return id != other.id;
  }
};

constexpr Type Type::Unknown{0, 0};
constexpr Type Type::F32{1, sizeof(uint32_t) * 8};
constexpr Type Type::F16{2, sizeof(uint16_t) * 8};
constexpr Type Type::F8{3, sizeof(uint8_t) * 8};
constexpr Type Type::F8_E5M2{4, sizeof(uint8_t) * 8};
constexpr Type Type::U8{5, sizeof(uint8_t) * 8};
constexpr Type Type::QI4{6, sizeof(uint8_t) * 4};

struct qi8_t {
  static constexpr int block_length = 8;
  static constexpr int block_size = sizeof(uint32_t) * 2;

  static void store_f32(uint8_t* data, const size_t index, const float32_t value) {
    const auto d = reinterpret_cast<int8_t*>(data);
    d[index] = static_cast<int8_t>(std::round(std::clamp(value * 8.0f, -128.0f, 127.0f)));
  }

  static float32_t load_f32(const uint8_t* data, const size_t index) {
    const auto q = reinterpret_cast<const int8_t*>(data)[index];
    return static_cast<float32_t>(q) / 8.0f;
  }
};


