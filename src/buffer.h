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

template <typename T, std::size_t alignment> class AlignedAllocator {
  public:
    using value_type = T;

	AlignedAllocator() noexcept = default;

    template <typename U> explicit AlignedAllocator(const AlignedAllocator<U, alignment>& other) noexcept {}

    template <typename U>
    bool operator==(const AlignedAllocator<U, alignment>& other) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const AlignedAllocator<U, alignment>& other) const noexcept {
        return false;
    }

    template <typename U> struct rebind {
        using other = AlignedAllocator<U, alignment>;
    };

	// ReSharper disable once CppMemberFunctionMayBeStatic
    [[nodiscard]] value_type* allocate(const std::size_t n) const {
		const auto ptr = std::aligned_alloc(alignment, sizeof(T) * n);
        if (ptr == nullptr)
            throw std::bad_alloc();
        return static_cast<value_type*>(ptr);
    };

	// ReSharper disable once CppMemberFunctionMayBeStatic
	void deallocate(value_type* const ptr, std::size_t n) const noexcept { std::free(ptr); }
};

template<aligned_buffer_type T = std::byte, int Alignment = alignof(T)>
	requires valid_aligned<T, Alignment>
class buffer {
	std::shared_ptr<std::vector<T, AlignedAllocator<T, Alignment>>> _data;
	size_t _offset;
	size_t _size;

	constexpr static size_t aligned_allocation_size(const size_t count) noexcept {
		const size_t byte_size = count * sizeof(T);
		return (byte_size + Alignment - 1) & ~static_cast<size_t>(Alignment - 1);
	}

public:
	buffer(buffer&& other) noexcept = default;
	buffer& operator=(buffer&& other) noexcept = default;
	buffer(const buffer&) = default;
	buffer& operator=(const buffer&) = default;

	explicit buffer(const size_t size) noexcept : _offset(0), _size(size) {
		if (size > 0) {
			_data = std::make_shared<std::vector<T, AlignedAllocator<T, Alignment>>>(size);
		} else {
			_data = nullptr;
		}
	}

	template<aligned_buffer_type V = T>
	[[nodiscard]] std::span<V> span(const size_t offset, const size_t count) const  {
		if constexpr(std::is_same_v<T, V>) {
			return std::span<V>{_data->data() + offset, count};
		} else {
			V* vdata = reinterpret_cast<V*>(_data->data() + offset);
			return std::span<V>{vdata + offset, count * sizeof(T) / sizeof(V)};
		}
	}

	template<aligned_buffer_type V = T>
	[[nodiscard]] std::span<V> span() const {
		return span<V>(_offset, _size);
	}

	template<aligned_buffer_type V = T>
	[[nodiscard]] std::span<V> span(const size_t offset) const {
		return span<V>(_offset + offset, _size - offset);
	}

	template<aligned_buffer_type V = T>
	[[nodiscard]] V* get() const {
		auto xxxx = _data->data();
		return reinterpret_cast<V*>(xxxx);
	}

	// Implicit conversions
	[[nodiscard]] operator T*() noexcept { return _data->data(); }
	[[nodiscard]] operator const T*() const noexcept { return _data->data(); }

	// Accessors
	[[nodiscard]] size_t bytes_size() const noexcept { return _size * sizeof(T); }
	[[nodiscard]] size_t size() const noexcept { return _size; }
	[[nodiscard]] static constexpr size_t alignment() noexcept { return Alignment; }
	[[nodiscard]] bool empty() const noexcept { return _size == 0; }
};
