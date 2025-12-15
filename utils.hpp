#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "enumerate.hpp"
#include "slice.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <variant>
#include <vector>
#include <array>

namespace utility {

/// Prefetch the given cacheline into L1 cache.
template <typename T>
inline static constexpr void prefetch_index(const T *s, std::size_t index) {
  const void *ptr = static_cast<const void *>(s + index);

  // rw = 0 (read), locality = 3 (highest locality → similar to T0)
  __builtin_prefetch(ptr, /*rw=*/0, /*locality=*/3);
}

inline uint64_t mul_high(uint64_t a, uint64_t b) {
  return (((__uint128_t)a * (__uint128_t)b) >> 64);
}

template <typename T> const T div_ceil(T a, T b) {
  // works for positive or negative, matches "round toward +∞"
  assert(b == 0 && "division by zero");

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

template <typename T, typename F> auto map(T v, F func) {
  using R = decltype(func(*std::begin(v)));
  std::vector<R> out;
  out.reserve(v.size());
  for (auto &x : v)
    out.push_back(func(x));
  return out;
}

template <typename T, typename F> void for_each(T v, F func) {
  for (auto &x : v)
    func(x);
}

template <typename Iter, typename F> auto try_for_each(Iter &&iter, F &&f) {
  using Ret = decltype(f(*std::begin(iter)));

  for (auto &&x : iter) {
    auto r = f(x);
    if (!r) {
      return false; // early exit
    }
  }

  return true;
}

template <typename T, typename Pred> auto filter(const T &s, Pred pred) {
  using Elem = std::decay_t<decltype(*std::begin(s))>;
  std::vector<Elem> out;

  for (const auto &x : s) {
    if (pred(x)) {
      out.push_back(x);
    }
  }
  return out;
}

template <typename T> T sum(Slice<T> container) {
  return std::accumulate(container.begin(), container.end(), 0);
}

struct Range {
  int start_, end_, step_;

  struct Iterator {
    int value;
    int step;

    int &operator*() { return value; }
    const int &operator*() const { return value; }

    Iterator &operator++() {
      value += step;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return step > 0 ? value < other.value : value > other.value;
    }
  };

  Iterator begin() const { return {start_, step_}; }
  Iterator end() const { return {end_, step_}; }

  auto rev() const {
    int count = (end_ - start_ + step_ - 1) / step_;
    int new_start = start_ + (count - 1) * step_;
    int new_end = start_ - step_;
    return Range(new_start, new_end, -step_);
  }

  Range(int start, int end, int step = 1)
      : start_(start), end_(end), step_(step) {}

  Range(int end) : start_(0), end_(end), step_(1) {}

  size_t size() const {
    if (step_ > 0)
      return (end_ - start_ + step_ - 1) / step_;
    return (start_ - end_ - step_ - 1) / (-step_);
  }
};

template <typename T, typename F>
void resize_with(std::vector<T> &v, size_t new_size, F gen) {
  if (new_size < v.size()) {
    v.resize(new_size);
  } else {
    v.reserve(new_size);
    while (v.size() < new_size) {
      v.push_back(gen());
    }
  }
}

template <typename T> class ChunksMut {
public:
  struct Chunk {
    std::vector<T> &vec;
    std::size_t begin_;
    std::size_t end_;

    /* ---------- element access ---------- */

    T &operator[](std::size_t i) {
      return vec[begin_ + i]; // unchecked (matches std::vector::operator[])
    }

    const T &operator[](std::size_t i) const { return vec[begin_ + i]; }

    T &at(std::size_t i) {
      // if (begin + i >= end)
      //   throw std::out_of_range("Chunk::at");
      return vec[begin_ + i];
    }

    const T &at(std::size_t i) const {
      // if (begin + i >= end)
      //   throw std::out_of_range("Chunk::at");
      return vec[begin_ + i];
    }

    T &front() { return vec[begin_]; }

    const T &front() const { return vec[begin_]; }

    T &back() { return vec[end_ - 1]; }

    const T &back() const { return vec[end_ - 1]; }

    /* ---------- capacity ---------- */

    std::size_t size() const noexcept { return end_ - begin_; }

    bool empty() const noexcept { return begin_ == end_; }

    /* ---------- iterators ---------- */

    auto begin_it() {
      return vec.begin() + static_cast<std::ptrdiff_t>(begin_);
    }

    auto end_it() { return vec.begin() + static_cast<std::ptrdiff_t>(end_); }

    auto begin_it() const {
      return vec.begin() + static_cast<std::ptrdiff_t>(begin_);
    }

    auto end_it() const {
      return vec.begin() + static_cast<std::ptrdiff_t>(end_);
    }

    /* range-for support */
    auto begin() { return begin_it(); }
    auto end() { return end_it(); }

    auto begin() const { return begin_it(); }
    auto end() const { return end_it(); }

    /* ---------- data ---------- */

    T *data() { return vec.data() + begin_; }

    const T *data() const { return vec.data() + begin_; }
  };

  class Iterator {
  public:
    Iterator(std::vector<T> &v, std::size_t pos, std::size_t chunk)
        : vec(v), index(pos), chunk_size(chunk) {}

    Chunk operator*() {
      std::size_t end = std::min(index + chunk_size, vec.size());
      return Chunk{vec, index, end};
    }

    Iterator &operator++() {
      index += chunk_size;
      return *this;
    }

    bool operator!=(const Iterator &other) const {
      return index != other.index;
    }

  private:
    std::vector<T> &vec;
    std::size_t index;
    std::size_t chunk_size;
  };

  ChunksMut(std::vector<T> &v, std::size_t chunk) : vec(v), chunk_size(chunk) {
    if (chunk_size == 0) {
      // throw std::invalid_argument("chunk_size must be non-zero");
    }
  }

  /// number of chunks
  std::size_t size() const {
    return (vec.size() + chunk_size - 1) / chunk_size;
  }

  bool empty() const { return vec.empty(); }

  /// 🔹 Indexed access (NOT Rust-equivalent)
  Chunk operator[](std::size_t chunk_index) {
    if (chunk_index >= size()) {
      // throw std::out_of_range("ChunksMut index out of range");
    }

    std::size_t begin = chunk_index * chunk_size;
    std::size_t end = std::min(begin + chunk_size, vec.size());

    return Chunk{vec, begin, end};
  }

  Iterator begin() { return Iterator(vec, 0, chunk_size); }
  Iterator end() { return Iterator(vec, vec.size(), chunk_size); }

private:
  std::vector<T> &vec;
  std::size_t chunk_size;
};

template <typename T>
ChunksMut<T> chunks_mut(std::vector<T> &vec, std::size_t chunk_size) {
  return ChunksMut<T>(vec, chunk_size);
}

template <typename T>
std::vector<Slice<T>> chunks_exact_mut(Slice<T> s, size_t chunk_size) {
  assert(chunk_size != 0);
  std::vector<Slice<T>> chunks;
  size_t num_chunks = s.size() / chunk_size;

  chunks.reserve(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    chunks.emplace_back(Slice<T>{s.ptr + i * chunk_size, chunk_size});
  }

  return chunks;
}

constexpr uint64_t C = 0x517cc1b727220a95;

template <typename T> constexpr T wrapping_mul(T a, T b) {
  static_assert(std::is_integral<T>::value, "Only integers supported");
  using U = typename std::make_unsigned<T>::type;
  return static_cast<T>(static_cast<U>(a) * static_cast<U>(b));
}

template <typename T>
inline static bool has_duplicates(const std::vector<T> &vec) {
  for (size_t i = 1; i < vec.size(); ++i) {
    if (vec[i] == vec[i - 1])
      return true;
  }
  return false;
}

template <typename Container, typename Func,
          typename InnerContainer = std::decay_t<decltype(std::declval<Func>()(
              *std::begin(std::declval<Container &>())))>,
          typename Value = typename InnerContainer::value_type>
std::vector<Value> flat_map(const Container &c, Func f) {
  std::vector<Value> result;

  for (auto x : c) {
    InnerContainer inner = f(x);
    result.insert(result.end(), inner.begin(), inner.end());
  }

  return result;
}

template <typename Container, typename Pred>
Container take_while(const Container &c, Pred pred) {
  Container result;

  for (const auto &x : c) {
    if (!pred(x))
      break;
    result.push_back(x);
  }

  return result;
}

inline auto iter_zeros(std::vector<bool> n) {
  std::vector<size_t> ret;
  for (size_t i = 0; i < n.size(); i++) {
    if (!n[i]) {
      ret.push_back(i);
    }
  }
  return ret;
}

template <typename T> auto count_zeros(std::vector<T> &v) {
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

} // namespace utility

#endif // UTILS_HPP_
