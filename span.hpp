#ifndef SPAN_HPP_
#define SPAN_HPP_

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

#define LIBC_INLINE inline
#define LIBC_INLINE_VAR inline

namespace cpp {
template <typename T> class span {
  template <typename U>
  LIBC_INLINE_VAR static constexpr bool is_const_view_v =
      !std::is_const_v<U> && std::is_const_v<T> &&
      std::is_same_v<U, std::remove_cv_t<T>>;

  template <typename U>
  LIBC_INLINE_VAR static constexpr bool is_compatible_v =
      std::is_same_v<U, T> || is_const_view_v<U>;

public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;
  using iterator = T *;

  LIBC_INLINE_VAR static constexpr size_type dynamic_extent =
      std::numeric_limits<size_type>::max();

  LIBC_INLINE constexpr span() : span_data(nullptr), span_size(0) {}

  LIBC_INLINE constexpr span(const span &) = default;

  LIBC_INLINE constexpr span(pointer first, size_type count)
      : span_data(first), span_size(count) {}

  LIBC_INLINE constexpr span(pointer first, pointer end)
      : span_data(first), span_size(static_cast<size_t>(end - first)) {}

  template <typename U, size_t N,
            std::enable_if_t<is_compatible_v<U>, bool> = true>
  LIBC_INLINE constexpr span(U (&arr)[N]) : span_data(arr), span_size(N) {}

  template <typename U, size_t N,
            std::enable_if_t<is_compatible_v<U>, bool> = true>
  LIBC_INLINE constexpr span(std::array<U, N> &arr)
      : span_data(arr.data()), span_size(arr.size()) {}

  template <typename U, std::enable_if_t<is_compatible_v<U>, bool> = true>
  LIBC_INLINE constexpr span(span<U> &s)
      : span_data(s.data()), span_size(s.size()) {}

  template <typename U, std::enable_if_t<is_compatible_v<U>, bool> = true>
  LIBC_INLINE constexpr span &operator=(span<U> &s) {
    span_data = s.data();
    span_size = s.size();
    return *this;
  }

  LIBC_INLINE ~span() = default;

  LIBC_INLINE constexpr reference operator[](size_type index) const {
    return data()[index];
  }

  LIBC_INLINE constexpr iterator begin() const { return data(); }
  LIBC_INLINE constexpr iterator end() const { return data() + size(); }
  LIBC_INLINE constexpr reference front() const { return (*this)[0]; }
  LIBC_INLINE constexpr reference back() const { return (*this)[size() - 1]; }
  LIBC_INLINE constexpr pointer data() const { return span_data; }
  LIBC_INLINE constexpr size_type size() const { return span_size; }
  LIBC_INLINE constexpr size_type size_bytes() const {
    return sizeof(T) * size();
  }
  LIBC_INLINE constexpr bool empty() const { return size() == 0; }

  LIBC_INLINE constexpr span<element_type>
  subspan(size_type offset, size_type count = dynamic_extent) const {
    return span<element_type>(data() + offset, count_to_size(offset, count));
  }

  LIBC_INLINE constexpr span<element_type> first(size_type count) const {
    return subspan(0, count);
  }

  LIBC_INLINE constexpr span<element_type> last(size_type count) const {
    return span<element_type>(data() + (size() - count), count);
  }

private:
  LIBC_INLINE constexpr size_type count_to_size(size_type offset,
                                                size_type count) const {
    if (count == dynamic_extent) {
      return size() - offset;
    }
    return count;
  }

  T *span_data;
  size_t span_size;
};
} // namespace cpp

#endif // SPAN_HPP_
