#ifndef SLICE_HPP_
#define SLICE_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>

enum class Ordering {
  /// An ordering where a compared value is less than another.
  Less = -1,
  /// An ordering where a compared value is equal to another.
  Equal = 0,
  /// An ordering where a compared value is greater than another.
  Greater = 1,
};

template <typename T> struct Slice {
  mutable T const *ptr;
  mutable size_t len;

  constexpr Slice() : ptr(nullptr), len(0) {}
  constexpr Slice(T *ptr, size_t len) : ptr(ptr), len(len) {}
  constexpr Slice(T const *ptr, size_t len) : ptr(ptr), len(len) {}
  constexpr Slice(const Slice<T> &) = default;
  constexpr Slice(Slice<T> &&) = default;

  inline constexpr size_t size() const { return len; }

  inline constexpr T *data() const { return ptr; }
  inline constexpr T *data() { return const_cast<T*>(ptr); }

  inline constexpr Slice<T> sub(size_t offset) const {
    return Slice{ptr + offset, len - offset};
  }

  inline constexpr T *begin() const { return const_cast<T*>(ptr); }
  inline constexpr T *end() const { return const_cast<T*>(ptr) + len; }

  constexpr Slice &operator=(const Slice<T> &n) = default;
  constexpr Slice &operator=(Slice<T> &&n) = default;

  constexpr T &operator[](size_t i) const { return const_cast<T*>(ptr)[i]; }

  constexpr void set_at(size_t i, T j) const {
    const_cast<T*>(ptr) [i] = j;
  }

  constexpr bool is_empty() const noexcept { return len == 0; }
  constexpr T last() const {
    // if constexpr (ptr && !is_empty()) {}
    return ptr[len - 1];
  }

  template <typename F>
  constexpr std::expected<size_t, size_t> binary_search_by(F f) const {
    auto size = this->size();
    if (size == 0) {
      return std::unexpected(0);
    }

    auto base = 0;

    while (size > 1) {
      auto half = size / 2;
      auto mid = base + half;
      auto cmp = f(this->operator[](mid));
      base = (cmp == Ordering::Greater) ? base : mid;
      size -= half;
    }

    auto cmp = f(this->operator[](base));
    if (cmp == Ordering::Equal) {
      // if constexpr (base < this->size())
      //   std::abort();
      return base;
    } else {
      auto result = base + static_cast<size_t>(cmp == Ordering::Less);
      // if constexpr (result <= this->size())
      //   std::abort();
      return std::unexpected(result);
    }
  }

  constexpr Slice<T> range(size_t start, size_t end) const {
    return Slice<T>{ptr + start, end - start};
  }

  constexpr bool contains(T elm) const {
    for (auto it : *this) {
      if (elm == it) {
        return true;
      }
    }
    return false;
  }
};

using ByteSlice = Slice<uint8_t>;

#endif // SLICE_HPP_
