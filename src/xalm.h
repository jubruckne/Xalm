#pragma once

#include <fstream>
#include <unistd.h>
#include "XalmFile.h"
#include "console.h"
#include "table.h"

#include <map>

struct Xalm {
	struct tensor_info {
		std::string name;
		Type type;
		std::vector<int> shape;
		std::string file_name;
		size_t offset;
		size_t size;
	};

	struct file_info {
		std::string file_name;
		json metadata;
		std::map<std::string, tensor_info> tensors;

		void load_tensor_data(const tensor_info& tensor_info, const std::span<std::byte> buffer) const {
			if (buffer.size() != tensor_info.size) {
				throw std::runtime_error("YAML buffer size mismatch");
			}

			stream.seekg(static_cast<std::streamoff>(tensor_info.offset), std::ios::beg);
			stream.read(reinterpret_cast<char*>(&buffer.front()), static_cast<std::streamoff>(tensor_info.size));
		}

		void load_tensor_data(const std::string& name, const std::span<std::byte> buffer) const {
			load_tensor_data(tensors.at(name), buffer);
		}

		Tensor load_tensor(const std::string& name) const {
			return load_tensor(tensors.at(name));
		}

		Tensor load_tensor(const tensor_info& tensor_info) const {
			auto t = Tensor::zeroes(tensor_info.type, tensor_info.shape, tensor_info.name);
			load_tensor_data(tensor_info, t.get_buffer()->span<std::byte>());
			return t;
		}

		[[nodiscard]] std::string format() const {
			auto tbl = table::make(column<int>{"#", -1, alignment::left, "{}", false},
								   column<std::string>{"name", -1, alignment::left, "{}", true},
								   column<std::string>{"type", -1, alignment::center, "{}", true},
								   column<std::array<int, 2>>{"shape", -1, alignment::right, "{h}", true},
								   column<size_t>{"size", -1, alignment::right, "{h}"});

			int row_number = 0;
			for (const auto& [key, tensor]: tensors) {
				tbl.add(row_number++, key, std::string(tensor.type.name()), tensor.shape, tensor.size);
			}

			return tbl.format(file_name);
		}

		~file_info() = default;
	private:
		mutable std::ifstream stream;

		friend Xalm;

	    file_info(std::string file_name, json metadata, std::map<std::string, tensor_info> tensors, std::ifstream stream):
			file_name(std::move(file_name)),
			metadata(std::move(metadata)),
			tensors(std::move(tensors)),
			stream(std::move(stream)) {}
	};

	static std::string expand_tilde(const std::string& path) {
		if (path.empty() || path[0] != '~') {
			return path;
		}

		const char* home = std::getenv("HOME");
		if (!home) {
			throw std::runtime_error("HOME environment variable not set");
		}

		return std::format("{}{}\n", home, path.substr(1));
	}

	[[nodiscard]] static file_info load(const std::string& file_name) {
		console::print("loading model {}\n", file_name);

		std::map<std::string, tensor_info> tensors;
		std::string json_string;
		json metadata;
		std::ifstream stream(expand_tilde(file_name),std::ios::binary);
		auto file_size = static_cast<std::streamoff>(std::filesystem::file_size(expand_tilde(file_name)));

		std::streamoff json_size = 0;

		stream.read(reinterpret_cast<std::istream::char_type *>(&json_size), sizeof(uint64_t));
		if (json_size == 0 || json_size > file_size - sizeof(uint64_t)) {
			throw std::invalid_argument(std::format("bad json size: {} for file size: {}", json_size, file_size));
		}

		json_size -= sizeof(uint64_t);

		std::vector<char> buf(json_size + 1, 0);
		stream.read(buf.data(), static_cast<uint32_t>(json_size));

		std::streamoff data_offset = static_cast<std::streamoff>(sizeof(uint64_t)) + json_size;
		std::streamoff data_end = data_offset + file_size;

		json_string = std::string(buf.data());

		const json header = json::parse(json_string);

		if (header.contains("xalm")) {
			auto ver = header.at("xalm").value("version", 0);

			if (ver != 1) {
				throw std::invalid_argument(std::format("xalm version mismatch: {}", ver));
			}
		} else {
			throw std::invalid_argument("invalid file format!");
		}

		// std::print("{}\n", json_string);
		// std::flush(std::cout);

		for (auto &[key, val]: header.items()) {
			//std::print("header {}\n", key);
			//std::flush(std::cout);
			if (key == "xalm") {
				continue;
			}

			auto model_arch = key;
			// printf("model arch: %s\n", model_arch.c_str());

			if (model_arch == "LlamaForCausalLM" || model_arch == "MistralForCausalLM") {
				metadata = val.at("config");
				for (auto &[key, val]: val.at("tensors").items()) {
					auto name = key;
					auto type_str = val.value("type", "<missing>");
					auto type = Type::parse(type_str);

					auto rank = val.at("shape").size();
					if (rank > 4) {
						throw std::invalid_argument("shape exceeds 4 dimensions");
					}

					auto shape = std::vector<int>(rank);

					for (size_t i = 0; i < rank && i < 4; i++) {
						if (val.at("shape")[i].get<int>() != val.at("shape")[i]) {
							std::print("bad shape");
							throw std::bad_alloc();
						}
						shape[i] = val.at("shape")[i].get<int>();
					}

					const auto offset = val.value("offset", -1ll);
					if (offset < 0) {
						throw std::invalid_argument("bad offset");
					}
					const auto size = val.value("size", -1ll);
					if (size < 0) {
						throw std::invalid_argument("bad size");
					}
					auto offset_end = offset + size;

					if (offset_end <= offset || offset_end > data_end) {
						throw std::invalid_argument(std::format("offset out of range"));
					}

					//printf("tensor: %s, offset: %llu, size: %llu\n", key.c_str(), offset + data_offset, size);

					tensors.emplace(
						name,
						tensor_info{name, type,shape,expand_tilde(file_name),
							static_cast<size_t>(offset + data_offset),
							static_cast<size_t>(size)}
					);
				}
			} else {
				console::error("unsupported model architecture: {}", model_arch);
			}
		}

		return file_info{expand_tilde(file_name), metadata, tensors, std::move(stream)};
	}
};

/*

inline void YALM::mmap_file(const std::string &filename) {
	this->filename = filename;
	std::cout << "loading data from file: " << filename << std::endl;

	std::string json_string;

	int fd = open(filename.c_str(), O_RDONLY);
	if (fd == -1) {
		throw std::invalid_argument(std::format("failed to open file {}", filename));
	}

	struct stat st{};
	if (fstat(fd, &st) != 0) {
		close(fd);
		throw std::invalid_argument(std::format("failed to stat file {}", filename));
	}

	size = st.st_size;
	data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (data == MAP_FAILED) {
		close(fd);
		throw std::invalid_argument(std::format("failed to mmap file {}", filename));
	}

#ifdef __linux__
	// increases readahead buffer size, resulting in faster cold loads
	posix_fadvise(fd, 0, size, POSIX_MADV_WILLNEED); // POSIX_FADV_SEQUENTIAL);
#elif defined(__APPLE__)
	madvise(data, size, MADV_WILLNEED); // | MADV_SEQUENTIAL MADV_WILLNEED);
#endif

	close(fd); // fd can be closed after mmap returns without invalidating the mapping

	// Parse the metadata JSON and the tensors
	if (size < sizeof(uint64_t)) {
		munmap(data, size);
		throw std::invalid_argument(std::format("bad size: {}", size));
	}


	const uint64_t json_size = *static_cast<uint64_t *>(data);
	if (json_size == 0 || json_size > size - sizeof(uint64_t)) {
		munmap(data, size);
		throw std::invalid_argument(std::format("bad size: {}", size));
	}

	std::byte* bytes_ptr = static_cast<std::byte*>(data) + sizeof(uint64_t) + json_size;
	const size_t bytes_size = size - sizeof(uint64_t) - json_size;

	json_string = std::string(static_cast<char *>(data) + sizeof(uint64_t), json_size);

	const json header = json::parse(json_string);

	// std::print("{}", json_string);
	// std::flush(std::cout);

	for (auto &[key, val]: header.items()) {
		if (key == "__metadata__") {
			metadata = val;
		} else {
			auto name = key;
			auto type_str = val.value("dtype", "");
			auto type = Type::parse(type_str);

			auto rank = val.at("shape").size();
			if (rank > 4) {
				throw std::invalid_argument("shape exceeds 4 dimensions");
			}

			auto shape = std::vector<int>(rank);

			for (size_t i = 0; i < rank && i < 4; i++) {
				if (val.at("shape")[i].get<int>() != val.at("shape")[i]) {
					std::print("bad shape");
					throw std::bad_alloc();
				}
				shape[i] = val.at("shape")[i].get<int>();
			}

			const auto offset_start = static_cast<size_t>(val.at("data_offsets")[0]);
			const auto offset_end = static_cast<size_t>(val.at("data_offsets")[1]);
			if (offset_start < 0 || offset_end <= offset_start || offset_end > bytes_size) {
				throw std::invalid_argument(std::format("offset out of range"));
			}

			auto data = static_cast<std::byte *>(bytes_ptr + offset_start);
			tensors.emplace(key, Tensor::view(data, bytes_size, type, shape, name));
			// printf("tensor: %s\n", key.c_str());
			// printf("tensor: %s\n",val.dump().c_str());
		}
	}
}
*/