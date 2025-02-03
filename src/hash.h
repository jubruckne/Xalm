#pragma once

#define XXH_INLINE_ALL
#define XXH_STATIC_LINKING_ONLY
#define XXH_NO_STREAM
#define XXH_NO_EXTERNC_GUARD
#include "../3rdparty/xxhash.h"

typedef uint64_t hash64_t;

struct xxhash3 {
	xxhash() = delete;

	template <typename T> requires std::is_scalar_v<T>
	static hash64_t hash(std::span<T> data) {
		return XXH3_64bits(data.data(), data.size());
	}

	static hash64_t hash(const std::string& s) {
		return XXH3_64bits(s.data(), s.size());
	}

	static hash64_t hash(const std::string_view s) {
		return XXH3_64bits(s.data(), s.size());
	}

	template <typename T> requires std::is_scalar_v<T>
	static hash64_t hash(const T& value) {
		return XXH3_64bits(&value, sizeof(T));
	}
};

template <>
struct std::formatter<hash64_t> {
	template <typename FormatContext>
	auto format(const hash64_t& k, FormatContext& ctx) {
		std::ostringstream oss;
		oss << std::hex << std::uppercase << std::setfill('0')
		    << std::setw(4) << ((k >> 48) & 0xFFFF) << '-'
		    << std::setw(4) << ((k >> 32) & 0xFFFF) << '-'
		    << std::setw(4) << ((k >> 16) & 0xFFFF) << '-'
		    << std::setw(4) << (k & 0xFFFF);
		return std::format_to(ctx.out(), "{}", oss.str());
	}
};
