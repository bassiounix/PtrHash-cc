#ifndef SLICE_HPP_
#define SLICE_HPP_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <cassert>

enum class Ordering {
  /// An ordering where a compared value is less than another.
  Less = -1,
  /// An ordering where a compared value is equal to another.
  Equal = 0,
  /// An ordering where a compared value is greater than another.
  Greater = 1,
};

template <typename T> struct Slice {
  T *ptr;
  size_t len;

  Slice() : ptr(nullptr), len(0) {}
  Slice(T *ptr, size_t len) : ptr(ptr), len(len) {}
  Slice(const Slice<T> &other) = default;
  Slice(Slice<T> &) = default;
  Slice(Slice<T> &&) = default;

  inline size_t size() const { return len; }

  inline const T *data() const { return ptr; }

  inline Slice<T> sub(size_t offset) const {
    return Slice{ptr + offset, len - offset};
  }

  inline const T *begin() const { return ptr; }
  inline const T *end() const { return ptr + len; }

  inline T *begin() { return ptr; }
  inline T *end() { return ptr + len; }

  Slice &operator=(Slice<T> &n) = default;
  Slice &operator=(const Slice<T> &n) = default;
  Slice &operator=(Slice<T> &&n) = default;

  T &operator[](size_t i) { return ptr[i]; }
  const T &operator[](size_t i) const { return ptr[i]; }

  const bool is_empty() const noexcept { return len == 0; }
  std::optional<T> last() const {
    if (ptr && !is_empty()) {
      return ptr[len - 1];
    }
    return std::nullopt;
  }

  template <typename F> std::expected<size_t, size_t> binary_search_by(F f) {
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
      assert(base < this->size());
      return base;
    } else {
      auto result = base + static_cast<size_t>(cmp == Ordering::Less);
      assert(result <= this->size());
      return std::unexpected(result);
    }
  }

  Slice<T> range(size_t start, size_t end) const {
    return Slice<T>{ptr + start, end - start};
  }

  bool contains(T elm) {
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
