#pragma once

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <span>

template<typename T>
concept aligned_buffer_type = std::is_standard_layout_v<T> &&	   // Standard layout for safe reinterpretation
							  std::is_trivially_copyable_v<T> &&   // Trivially copyable for bitwise operations
							  std::is_trivially_destructible_v<T>; // Trivially destructible for simple cleanup

template<typename T, uint32_t Alignment>
concept valid_aligned = Alignment > 0 &&						   // Positive alignment
						std::has_single_bit(Alignment) &&		   // Power-of-two alignment
						Alignment >= alignof(T);                   // Meets type's alignment requirements

template<aligned_buffer_type T = std::byte, int Alignment = alignof(T)>
	requires valid_aligned<T, Alignment>
class buffer {
	T* _data = nullptr;
	size_t _size = 0;

	constexpr static size_t aligned_allocation_size(const size_t count) noexcept {
		const size_t byte_size = count * sizeof(T);
		return (byte_size + Alignment - 1) & ~static_cast<size_t>(Alignment - 1);
	}

public:
	explicit buffer(const size_t size) : _size(size) {
		if (size > 0) {
			const size_t alloc_size = aligned_allocation_size(size);
			void* raw = std::aligned_alloc(Alignment, alloc_size);
			if (!raw)
				throw std::bad_alloc();
			_data = static_cast<T*>(raw);
		}
	}

	~buffer() { std::free(_data); }

	template<aligned_buffer_type V = T>
	[[nodiscard]] std::span<V> span(const size_t offset, const size_t count) const  {
		if constexpr(std::is_same_v<T, V>) {
			return std::span<V>{_data + offset, count};
		} else {
			V* vdata = reinterpret_cast<V*>(_data + offset);
			return std::span<V>{vdata + offset, count * sizeof(T) / sizeof(V)};
		}
	}

	template<aligned_buffer_type V = T>
	[[nodiscard]] std::span<V> span() const {
		return span<V>(0, _size);
	}

	template<aligned_buffer_type V = T>
	[[nodiscard]] std::span<V> span(const size_t offset) const {
		return span<V>(offset, _size - offset);
	}

	// Move semantics
	buffer(buffer&& other) noexcept : _data(other._data), _size(other._size) {
		other._data = nullptr;
		other._size = 0;
	}

	buffer& operator=(buffer&& other) noexcept {
		if (this != &other) {
			std::free(_data);
			_data = other._data;
			_size = other._size;
			other._data = nullptr;
			other._size = 0;
		}
		return *this;
	}

	template<aligned_buffer_type V = T>
	[[nodiscard]] auto get() const {
		return reinterpret_cast<V*>(_data);
	}

	// Implicit conversions
	[[nodiscard]] operator T*() noexcept { return _data; }
	[[nodiscard]] operator const T*() const noexcept { return _data; }

	// Accessors
	[[nodiscard]] size_t bytes_size() const noexcept { return _size * sizeof(T); }
	[[nodiscard]] size_t size() const noexcept { return _size; }
	[[nodiscard]] static constexpr size_t alignment() noexcept { return Alignment; }
	[[nodiscard]] bool empty() const noexcept { return _size == 0; }

	// Disallow copying
	buffer(const buffer&) = delete;
	buffer& operator=(const buffer&) = delete;
};
