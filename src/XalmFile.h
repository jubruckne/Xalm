#pragma once
/*
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <cstring>
#include "hash.h"

enum class kv_type : uint8_t {
    HEADER_TYPE = 0x00,
    STRING  = 0x01,
    INT32   = 0x02,
    INT64   = 0x03,
    UINT8   = 0x04,
    UINT32  = 0x05,
    UINT64  = 0x06,
    FLOAT32 = 0x07,
    FLOAT64  = 0x08,
    ALIGNED_BLOB  = 0x09
};

template <kv_type V> struct value_traits;
template<> struct value_traits<kv_type::STRING>  { using t = std::string;  };
template<> struct value_traits<kv_type::UINT8>   { using t = uint8_t;      };
template<> struct value_traits<kv_type::INT32>   { using t = int32_t;      };
template<> struct value_traits<kv_type::UINT32>  { using t = uint32_t;     };
template<> struct value_traits<kv_type::INT64>   { using t = int64_t;      };
template<> struct value_traits<kv_type::UINT64>  { using t = uint64_t;     };
template<> struct value_traits<kv_type::FLOAT32> { using t = float32_t;    };
template<> struct value_traits<kv_type::FLOAT64> { using t = float64_t;    };

#pragma pack(push, 1)
struct block_header {
	uint64_t key_hash;
	kv_type value_type;
	uint64_t value_count;
	uint64_t value_size;
};
#pragma pack(pop)

struct key_info {
	key_info() = delete;

    std::string name;
    hash64_t hash;
    kv_type type;
    uint64_t count;
private:
	uint64_t offset;
public:
    uint64_t size;
private:
	friend class tensor_file;
	std::shared_ptr<std::vector<uint8_t>> data{};
	key_info(const uint64_t hash, const kv_type type, const uint64_t count, const uint64_t offset, const uint64_t size):
		name(std::to_string(hash)), hash(hash), type(type), count(count), offset(offset), size(size) {}
};

class tensor_file {
private:
    static constexpr size_t ALIGNMENT = 32;
    static constexpr std::string_view DIRECTORY_KEY = "directory";
    static constexpr std::string_view HEADER_KEY = "header";
    static constexpr size_t LAZY_LOAD_THRESHOLD = 1024;

	std::unordered_map<uint64_t, key_info> key_list;

	static size_t align_size(const size_t size) {
        return ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }

public:
	const std::string_view file_name;
	std::ifstream file;

	explicit tensor_file(const std::string& filename) noexcept : file_name(filename) {
		file = std::ifstream{file_name, std::ios::binary};
	}

    void load() {
		if (!file) {
			throw std::runtime_error(std::format("Error opening file: {}", file_name));
		}

		block_header header{};
		file.read(reinterpret_cast<char*>(&header), sizeof(header));
		if (header.value_type != kv_type::HEADER_TYPE) {
			throw std::runtime_error("Invalid file format: missing Header block");
		}

		const uint64_t num_entries = header.value_count;
		for (uint64_t i = 0; i < num_entries; ++i) {
			block_header block{};
			file.read(reinterpret_cast<char*>(&block), sizeof(block));

			if (block.value_type == kv_type::ALIGNED_BLOB) {
				// seek to next alignment boundary
				const auto pos = file.tellg();
				const auto align_needed = align_size(pos) - pos;;
				file.seekg(align_needed, std::ios::beg);
			}

			auto key = key_info{block.key_hash, block.value_type, block.value_count, file.tellg(), block.value_size};
			if (block.value_type != kv_type::ALIGNED_BLOB) {
				key.data->assign(block.value_size, 0);
				file.read(reinterpret_cast<char*>(key.data->data()), block.value_size);
			}
			key_list.emplace(block.key_hash, std::move(key));
		}

		auto directory = get_string_array(DIRECTORY_KEY);

		for (const auto& name : directory) {
			const uint64_t hash = xxhash::hash(name);
			if (key_list.contains(hash)) {
				auto& key = key_list.at(name);
				key.name = name;
			}
		}
		file.close();
	}

    std::vector<key_info> keys() const {
		std::vector<key_info> result;
		result.reserve(key_list.size());
		for (const auto& info: key_list | std::views::values) {
			result.push_back(info);
		}
		return result;
	}

	template<kv_type V>
	typename value_traits<V>::t get(const std::string& key) {
		const auto hash = xxhash::hash(key);

		if (!key_list.contains(hash)) {
			throw std::runtime_error("Key not found: " + key);
		}

		const auto& info = key_list.at(hash);
		if (info.type == kv_type::ALIGNED_BLOB) {
			throw std::runtime_error("aligned blob must be read with load_data");
		}

		const auto raw_data = info.data.get();

		assert(info.count / sizeof(typename value_traits<V>::t) == info.size);

		typename value_traits<V>::t value;
		std::memcpy(&value, raw_data->data(), raw_data->size());
		return value;
	}

	std::vector<std::string> get_string_array(const std::string& key) const {
		std::vector<uint8_t> raw_data = get_tensor<uint8_t>(key);

		std::vector<std::string> result;
		size_t pos = 0;
		while (pos < raw_data.size()) {
			uint32_t length;
			std::memcpy(&length, raw_data.data() + pos, sizeof(length));
			pos += sizeof(length);
			std::string str(reinterpret_cast<char*>(raw_data.data() + pos), length - 1);
			pos += length;
			result.push_back(str);
		}

		return result;
	}
};

template <>
struct std::formatter<key_info> {
	template <typename FormatContext>
	auto format(const key_info& k, FormatContext& ctx) {
		return std::format_to(ctx.out(),
		    "{:<30} {:<21} {:<7} {:<8} {:>10} bytes",
		    k.name, k.hash, static_cast<int>(k.type), k.count, k.size);
	}
};

template <>
struct std::formatter<tensor_file> {
	template <typename FormatContext>
	auto format(const tensor_file& tf, FormatContext& ctx) {
		std::vector<key_info> key_info = tf.keys();

		std::string output = std::format(
		    "TensorFile \"{}\" contains {} blocks:\n"
		    "{:<30} {:<21} {:<7} {:<8} {:>10}\n"
		    "{}",
		    tf.file_name, key_info.size(),
		    "Key Name", "Hash", "Type", "Count", "Size",
		    std::string(80, '-'));

		for (const auto& k : key_info) {
			output += std::format("\n{}", k);
		}

		return std::format_to(ctx.out(), "{}", output);
	}
};

void test() {
	tensor_file f{""};
	auto keys = f.keys();

	auto n_dim = f.get<value_type::INT32>("n_dim");
}*/