#ifndef PORT_COMMON_HPP_
#define PORT_COMMON_HPP_

#include "enumerate.hpp"
#include "fxhash.hpp"
#include "slice.hpp"
#include "utils.hpp"
// #include <cassert>
#include <cstdio>
#include <limits>

// inline static constexpr uint64_t mul_high(uint64_t a, uint64_t b) {
//   uint64_t hi = 0;
//   __builtin_mul_overflow(a, b, &hi);
//   return hi;
// }

class Packed {
  virtual constexpr uint64_t index(size_t index) const = 0;
  virtual constexpr void prefetch(size_t _index) const {}
  virtual constexpr size_t size_in_bytes() const = 0;
  static constexpr std::optional<Packed> try_new(Slice<uint64_t> vals);
};

template <typename T, size_t N> class DynamicContainer : public Packed {
private:
  std::array<T, N> i_;

public:
  // DynamicContainer(Slice<T> i) : i_(i) {}
  constexpr DynamicContainer(Slice<T> &i) {
    for (auto [i, e] : enumerate(i)) {
      i_[i] = e;
    }
  }
  constexpr DynamicContainer(Slice<T> &&i) : DynamicContainer(i) {}
  constexpr DynamicContainer(std::array<T, N> &v) : i_(v) {}
  constexpr DynamicContainer() = default;
  constexpr DynamicContainer(const DynamicContainer<T, N> &) = default;
  constexpr DynamicContainer(DynamicContainer<T, N> &&) = default;

  constexpr DynamicContainer<T, N> &
  operator=(const DynamicContainer<T, N> &) = default;
  // DynamicContainer<T>& operator=(DynamicContainer<T>&&) = default;

  constexpr uint64_t index(size_t index) const override {
    // assert(index < this->i_.size() && "Index out of bounds accessing Slice");
    return this->i_[index];
  }

  constexpr void prefetch(size_t _index) const override {
    utility::prefetch_index(this->i_.data(), _index);
  }

  constexpr size_t size_in_bytes() const override { return this->i_.size(); }

  static constexpr DynamicContainer
  try_new(Slice<uint64_t> const vals) {
    // for (auto i : vals) {
    //   if (i > std::numeric_limits<T>::max()) {
    //     fprintf(stderr, "values are larger than backing type can hold\n");
    //     std::abort();
    //   }
    // }
    std::array<T, N> n;

    for (size_t i = 0; i < N; i++) {
      n[i] = vals[i];
    }

    return DynamicContainer(n);
  }
};

inline constexpr uint64_t low(uint64_t x) { return x; }
inline constexpr uint64_t high(uint64_t x) { return x; }

inline constexpr uint64_t low(__uint128_t x) { return (uint64_t)x; }
inline constexpr uint64_t high(__uint128_t x) { return (uint64_t)(x >> 64); }

template <typename Key, typename Ret> class KeyHasher {
public:
  using H = Ret;
  static constexpr H hash(Key x, uint64_t seed);
};

template <typename Key>
class KeyHasherDefaultImpl : public KeyHasher<Key, uint64_t> {
public:
  static constexpr KeyHasher<Key, uint64_t>::H hash(Key &x, uint64_t seed) {
    return FxHasherDecl::hash64(x) ^ seed;
  }
};

class Reduce {
public:
  virtual constexpr size_t reduce(uint64_t h) const = 0;
  virtual constexpr std::pair<size_t, uint64_t>
  reduce_with_remainder(uint64_t _h) const = 0;
};

struct FastReduce : public Reduce {
  uint64_t d;

  constexpr FastReduce(uint64_t d) : d(d) {}
  constexpr FastReduce() = default;

  constexpr operator uint64_t() const { return d; }

  constexpr size_t reduce(uint64_t h) const override {
    return utility::mul_high(this->d, h);
  }

  constexpr std::pair<size_t, uint64_t>
  reduce_with_remainder(uint64_t h) const override {
    auto r = (__uint128_t)this->d * (__uint128_t)h;
    return {r >> 64, r};
  }
};

struct FM32 : public Reduce {
  uint64_t d;
  uint64_t m;

  constexpr FM32(size_t d)
      : d(d), m(std::numeric_limits<uint64_t>::max() / d + 1) {
    // assert(d <= std::numeric_limits<uint32_t>::max());
  }
  constexpr FM32() = default;

  constexpr size_t reduce(uint64_t h) const override {
    auto lowbits = m * h;
    return (static_cast<__uint128_t>(lowbits) * static_cast<__uint128_t>(d)) >>
           64;
  }

  constexpr std::pair<size_t, uint64_t>
  reduce_with_remainder(uint64_t h) const override {
    return {};
  }
};

using Rp = FastReduce;
using Rb = FastReduce;
using RemSlots = FM32;
using Pilot = uint64_t;
using PilotHash = uint64_t;

#endif // PORT_COMMON_HPP_
