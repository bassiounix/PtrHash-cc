#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "enumerate.hpp"
#include "slice.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace utility {

/// Prefetch the given cacheline into L1 cache.
template <typename T>
inline static constexpr void prefetch_index(const T *s, std::size_t index) {
  const void *ptr = static_cast<const void *>(s + index);

  // rw = 0 (read), locality = 3 (highest locality → similar to T0)
  __builtin_prefetch(ptr, /*rw=*/0, /*locality=*/3);
}

inline constexpr uint64_t mul_high(uint64_t a, uint64_t b) {
  return (((__uint128_t)a * (__uint128_t)b) >> 64);
}

template <typename T> constexpr T div_ceil(T a, T b) {
  // works for positive or negative, matches "round toward +∞"
  // assert(b == 0 && "division by zero");

  T q = a / b;
  T r = a % b;

  // If there is a remainder AND the division is not already upward
  if (r != 0 && ((a > 0) == (b > 0))) {
    q += 1;
  }

  return q;
}

template <typename T> constexpr bool is_power_of_two(T x) {
  return x != 0 && (x & (x - 1)) == 0;
}

template <typename T> constexpr bool is_power_of_two_signed(T x) {
  return x > 0 && (x & (x - 1)) == 0;
}

template <typename T, typename F, std::size_t N>
constexpr auto map(const std::array<T, N> &v, F func) {
  using R = std::invoke_result_t<F, T>;
  std::array<R, N> out{};

  for (std::size_t i = 0; i < N; ++i) {
    out[i] = func(v[i]);
  }

  return out;
}

// template <typename T, typename F, std::size_t N>
// constexpr auto map(Slice<const T> v, F func) {
//   using R = std::invoke_result_t<F, T>;
//   std::array<R, N> out{};

//   for (std::size_t i = 0; i < N; ++i)
//     out[i] = func(v[i]);

//   return out;
// }

template <typename T, typename F> constexpr void for_each(T v, F func) {
  for (auto &x : v)
    func(x);
}

template <typename Iter, typename F>
constexpr auto try_for_each(Iter &&iter, F &&f) {
  using Ret = decltype(f(*std::begin(iter)));

  for (auto &&x : iter) {
    auto r = f(x);
    if (!r) {
      return false; // early exit
    }
  }

  return true;
}

template <typename T, typename Pred, std::size_t N>
constexpr auto filter(const std::array<T, N> &s, Pred pred) {
  constexpr std::size_t M = [&] {
    std::size_t c = 0;
    for (const auto &x : s)
      if (pred(x))
        ++c;
    return c;
  }();

  std::array<T, M> out{};
  std::size_t i = 0;

  for (const auto &x : s)
    if (pred(x))
      out[i++] = x;

  return out;
}

template <typename T> constexpr T sum(Slice<T> container) {
  size_t acc = 0;
  for (T const item : container) {
    acc += item;
  }
  return acc;
}

struct Range {
  int start_, end_, step_;

  struct Iterator {
    mutable int value;
    mutable int step;

    constexpr int &operator*() const { return value; }

    constexpr const Iterator &operator++() const {
      value += step;
      return *this;
    }

    constexpr bool operator!=(const Iterator &other) const {
      return step > 0 ? value < other.value : value > other.value;
    }
  };

  constexpr Iterator begin() const { return {start_, step_}; }
  constexpr Iterator end() const { return {end_, step_}; }

  constexpr auto rev() const {
    int count = (end_ - start_ + step_ - 1) / step_;
    int new_start = start_ + (count - 1) * step_;
    int new_end = start_ - step_;
    return Range(new_start, new_end, -step_);
  }

  constexpr Range(int start, int end, int step = 1)
      : start_(start), end_(end), step_(step) {}

  constexpr Range(int end) : start_(0), end_(end), step_(1) {}

  constexpr size_t size() const {
    if (step_ > 0)
      return (end_ - start_ + step_ - 1) / step_;
    return (start_ - end_ - step_ - 1) / (-step_);
  }
};

template <typename T, typename F>
constexpr void resize_with(std::vector<T> &v, size_t new_size, F gen) {
  if (new_size < v.size()) {
    v.resize(new_size);
  } else {
    v.reserve(new_size);
    while (v.size() < new_size) {
      v.push_back(gen());
    }
  }
}

template <typename T, size_t N> class ChunksMut {
public:
  struct Chunk {
    std::array<T, N> &arr_;
    std::size_t begin_;
    std::size_t end_;

    /* ---------- element access ---------- */

    constexpr T &operator[](std::size_t i) const { return arr_[begin_ + i]; }

    constexpr T &at(std::size_t i) const {
      // if (begin + i >= end)
      //   throw std::out_of_range("Chunk::at");
      return arr_[begin_ + i];
    }

    constexpr T &front() const { return arr_[begin_]; }

    constexpr T &back() const { return arr_[end_ - 1]; }

    /* ---------- capacity ---------- */

    constexpr size_t size() const noexcept { return end_ - begin_; }

    constexpr bool empty() const noexcept { return begin_ == end_; }

    /* ---------- iterators ---------- */
    constexpr auto begin_it() const {
      return arr_.begin() + static_cast<std::ptrdiff_t>(begin_);
    }

    constexpr auto end_it() const {
      return arr_.begin() + static_cast<std::ptrdiff_t>(end_);
    }

    /* range-for support */
    constexpr auto begin() const { return begin_it(); }
    constexpr auto end() const { return end_it(); }

    /* ---------- data ---------- */
    constexpr T *data() const { return arr_.data() + begin_; }
  };

  class Iterator {
  public:
    constexpr Iterator(std::array<T, N> &arr, std::size_t pos,
                       std::size_t chunk)
        : arr_(arr), index(pos), chunk_size(chunk) {}

    constexpr Chunk operator*() const {
      std::size_t end = std::min(index + chunk_size, arr_.size());
      return Chunk{arr_, index, end};
    }

    constexpr const Iterator &operator++() const {
      index += chunk_size;
      return *this;
    }

    constexpr bool operator!=(const Iterator &other) const {
      return index != other.index;
    }

  private:
    std::array<T, N> &arr_;
    mutable size_t index;
    mutable size_t chunk_size;
  };

  constexpr ChunksMut(std::array<T, N> &v, size_t chunk)
      : arr(v), chunk_size(chunk) {
    // static_assert(chunk_size == 0);
  }

  /// number of chunks
  constexpr size_t size() const {
    return (arr.size() + chunk_size - 1) / chunk_size;
  }

  constexpr bool empty() const { return arr.empty(); }

  constexpr Chunk operator[](std::size_t chunk_index) const {
    // static_assert(chunk_index >= size());

    size_t begin = chunk_index * chunk_size;
    size_t end = std::min(begin + chunk_size, arr.size());

    return Chunk{arr, begin, end};
  }

  constexpr Iterator begin() const { return Iterator(arr, 0, chunk_size); }
  constexpr Iterator end() const {
    return Iterator(arr, arr.size(), chunk_size);
  }

private:
  std::array<T, N> &arr;
  std::size_t chunk_size;
};

template <typename T, size_t N>
constexpr ChunksMut<T, N> chunks_mut(std::array<T, N> &arr,
                                     std::size_t chunk_size) {
  return ChunksMut<T, N>(arr, chunk_size);
}

template <typename T>
constexpr std::vector<Slice<T>> chunks_exact_mut(Slice<T> s,
                                                 size_t chunk_size) {
  // assert(chunk_size != 0);
  std::vector<Slice<T>> chunks;
  size_t num_chunks = s.size() / chunk_size;

  chunks.reserve(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    chunks.emplace_back(Slice<T>{s.data() + i * chunk_size, chunk_size});
  }

  return chunks;
}

template <size_t buckets_total_, size_t buckets_>
constexpr auto chunks_exact_mut(
    typename utility::ChunksMut<uint8_t, buckets_total_>::Chunk &pilots) {
  const auto num_chunks = pilots.size() / buckets_;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t begin = pilots.begin_ + i * buckets_;
    size_t end = std::min(begin + buckets_, pilots.arr_.size());
    auto target_pilots =
        typename utility::ChunksMut<uint8_t, buckets_total_>::Chunk{pilots.arr_,
                                                                    begin, end};
  }

  // return chunks;
}

constexpr uint64_t C = 0x517cc1b727220a95;

template <typename T> constexpr T wrapping_mul(T a, T b) {
  T result = 0;

  while (b != 0) {
    if (b & 1) {
      result = result + a;
    }
    a = a << 1;
    b = static_cast<std::make_unsigned_t<T>>(b) >> 1;
  }

  return result;
}

template <typename T, size_t N>
inline static constexpr bool has_duplicates(const std::array<T, N> &arr) {
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i] == arr[i - 1])
      return true;
  }
  return false;
}

template <typename Container, typename Func,
          typename InnerContainer = std::decay_t<decltype(std::declval<Func>()(
              *std::begin(std::declval<Container &>())))>,
          typename Value = typename InnerContainer::value_type>
constexpr std::vector<Value> flat_map(const Container &c, Func f) {
  std::vector<Value> result;

  for (auto x : c) {
    InnerContainer inner = f(x);
    result.insert(result.end(), inner.begin(), inner.end());
  }

  return result;
}

template <typename Container, typename Pred>
constexpr Container take_while(const Container &c, Pred pred) {
  Container result;

  for (const auto &x : c) {
    if (!pred(x))
      break;
    result.push_back(x);
  }

  return result;
}

// template <size_t N>
// inline constexpr auto iter_zeros(std::array<bool, N> &arr_of_bools) {
//   std::vector<size_t> ret;
//   for (size_t i = 0; i < arr_of_bools.size(); i++) {
//     if (!arr_of_bools[i]) {
//       ret.push_back(i);
//     }
//   }
//   return ret;
// }

template <typename T, size_t N>
constexpr auto count_zeros(std::array<T, N> &v) {
  auto c = 0;
  for (auto e : v) {
    if (!e || e == false || e == 0) {
      c++;
    }
  }
  return c;
}

template <typename T> constexpr T rotate_right(T x, unsigned int n) {
  static_assert(std::is_unsigned<T>::value,
                "rotate_right requires unsigned type");
  constexpr unsigned int bits = std::numeric_limits<T>::digits;
  n %= bits;
  return (x >> n) | (x << (bits - n));
}

constexpr std::array<uint8_t, 4> to_le_bytes(uint32_t x) {
  return {
      static_cast<uint8_t>(x),
      static_cast<uint8_t>(x >> 8),
      static_cast<uint8_t>(x >> 16),
      static_cast<uint8_t>(x >> 24),
  };
}

template <typename To, typename From>
inline constexpr To ptr_bit_cast(From *from) {
  To to{};
  char *dst = reinterpret_cast<char *>(&to);
  const char *src = reinterpret_cast<const char *>(from);
  for (unsigned i = 0; i < sizeof(To); ++i)
    dst[i] = src[i];
  return to;
}

template <typename T> constexpr T wrapping_add(T a, T b) {
  while (b != 0) {
    T carry = a & b;
    a = a ^ b;
    b = carry << 1;
  }
  return a;
}

template <typename T, std::size_t N>
static inline constexpr auto array_sort(std::array<T, N> &arr) {
  if constexpr (N <= 1)
    return arr; // base case

  constexpr size_t mid = N / 2;

  // Split array into left and right halves
  std::array<T, mid> left{};
  std::array<T, N - mid> right{};

  for (std::size_t i = 0; i < mid; ++i)
    left[i] = arr[i];
  for (std::size_t i = mid; i < N; ++i)
    right[i - mid] = arr[i];

  // Recursively sort each half
  left = array_sort(left);
  right = array_sort(right);

  // Merge halves
  std::array<T, N> result{};
  std::size_t li = 0, ri = 0, ki = 0;

  while (li < mid && ri < N - mid) {
    result[ki++] = (left[li] <= right[ri]) ? left[li++] : right[ri++];
  }
  while (li < mid)
    result[ki++] = left[li++];
  while (ri < N - mid)
    result[ki++] = right[ri++];

  return result;
}

} // namespace utility

#endif // UTILS_HPP_
