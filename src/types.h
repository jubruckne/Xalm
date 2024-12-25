#pragma once
#include <cstdint>
#include <cmath>
#include <cassert>

#if defined(__AVX2__) && defined(__F16C__)
  #include <immintrin.h> // Intel/AVX2
#elif defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>  // ARM NEON
#endif

/*
enum struct DType {
  F32,
  F16,
  F8,
  U8,
};*/


struct Type {
  static const Type Unknown;
  static const Type F32;
  static const Type F16;
  static const Type F8;
  static const Type U8;

  constexpr Type(const int v, const size_t size) : _value(v), _size(size) {}
  ~Type() = default;

  constexpr operator int() const { return _value; }

  [[nodiscard]] size_t bit_size() const {
    return _size;
  }

  [[nodiscard]] constexpr std::string_view to_string() const {
    if (*this == Type::F32) return "F32";
    if (*this == Type::F16) return "F16";
    if (*this == Type::F8) return "F8";
    if (*this == Type::U8) return "U8";
    return "UNKNOWN";
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
    if (type_str == "U8") return Type::U8;
    return Type::Unknown;
  }

  constexpr bool operator==(const Type& other) const {
    return _value == other._value && _size == other._size;
  }

  constexpr bool operator!=(const Type& other) const {
    return !(*this == other);
  }

private:
  int _value;
  size_t _size;
};

constexpr Type Type::Unknown{0, 0};
constexpr Type Type::F32{1, sizeof(uint32_t) * 8};
constexpr Type Type::F16{2, sizeof(uint16_t) * 8};
constexpr Type Type::F8{3, sizeof(uint8_t) * 8};
constexpr Type Type::U8{4, sizeof(uint8_t) * 8};

/*
inline std::string_view dtype_to_string(DType dtype) {
  switch (dtype) {
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::F8: return "F8";
    case DType::U8: return "U8";
  }
  return "UNKNOWN";
}

inline size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::F8: return 1;
    case DType::U8: return 1;
  }
  return 0;
}*/

struct f16_t {
  uint16_t data;

  // Default constructor
  f16_t() : data(0) {}

  // Constructor from float
  explicit f16_t(float value) {
    *this = from_float(value);
  }

  // Convert from float to half-precision
  static f16_t from_float(float value) {
    f16_t result;

#if defined(__AVX2__) && defined(__F16C__)
    result.data = _cvtss_sh(value, 0); // AVX2 F16C path
#elif defined(__ARM_NEON) || defined(__aarch64__)
    __fp16 h = static_cast<__fp16>(value); // ARM path
    result.data = *reinterpret_cast<uint16_t*>(&h);
#else
    result.data = float_to_half_fallback(value);
#endif

    return result;
  }

  // Convert to float from half-precision
  static float to_float(f16_t value) {
#if defined(__AVX2__) && defined(__F16C__)
    return _cvtsh_ss(data); // AVX2 F16C path
#elif defined(__ARM_NEON) || defined(__aarch64__)
    __fp16 h = *reinterpret_cast<const __fp16*>(&value); // ARM path
    return static_cast<float>(h);
#else
    return half_to_float_fallback(data);
#endif
  }

private:
  static uint16_t float_to_half_fallback(float value) {
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);
    uint16_t sign = (bits >> 16) & 0x8000;
    uint16_t exponent = ((bits >> 23) & 0xFF) - 112;
    uint16_t mantissa = (bits >> 13) & 0x3FF;

    if (exponent <= 0) {
      return sign;        // Underflow
    }

    if (exponent >= 0x1F) {
      return sign | 0x7C00;   // Overflow
    }

    return sign | (exponent << 10) | mantissa;
  }

  static float half_to_float_fallback(uint16_t half) {
    uint32_t sign = (half & 0x8000) << 16;
    uint32_t exponent = ((half >> 10) & 0x1F) + 112;
    uint32_t mantissa = (half & 0x3FF) << 13;

    uint32_t bits = sign | (exponent << 23) | mantissa;
    return *reinterpret_cast<float*>(&bits);
  }

  bool operator==(const f16_t& other) const {
    return data == other.data;
  }

  bool operator!=(const f16_t& other) const {
    return !(*this == other);
  }
};

template<int8_t N> consteval float EXP2() {
  if constexpr (N==0) return 1;
  if constexpr (N>0) return EXP2<N-1>()*2;
  if constexpr (N<0) return EXP2<N+1>()/2;
}

template<int8_t N> consteval int EXP_I2() requires (N >= 0) {
  if constexpr (N==0) return 1;
  if constexpr (N>0) return EXP_I2<N-1>()*2;
}

template<uint8_t E, uint8_t M> requires (M > 0 && E+M == 7)
struct f8_t {
private:
  uint8_t bits = 0;

  explicit f8_t(const uint8_t bits) : bits(bits) {}

  static constexpr int E_BIAS = EXP2<E-1>()-1;
  static constexpr float E_BIAS_MINUS_127 = EXP2<E_BIAS-127>();
  static constexpr float E_127_MINUS_BIAS = EXP2<127-E_BIAS>();
  static constexpr float max = (2-EXP2<-M+1>())*EXP2<EXP_I2<E-1>()>();
  static constexpr float min = EXP2<-M>()*EXP2<2-EXP_I2<E-1>()>();
public:
  static f8_t from(const float value) {
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

  static float to_float(const f8_t value) {
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