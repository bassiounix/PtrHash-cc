#ifndef PTR_HASH_HPP_
#define PTR_HASH_HPP_

#include "expected.hpp"
#include "span.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <variant>

#define LIBC_ASSERT(cond)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

enum class Ordering {
  /// An ordering where a compared value is less than another.
  Less = -1,
  /// An ordering where a compared value is equal to another.
  Equal = 0,
  /// An ordering where a compared value is greater than another.
  Greater = 1,
};

template <typename T> struct Slice : public cpp::span<T> {
  LIBC_INLINE constexpr Slice() : cpp::span<T>() {}

  LIBC_INLINE constexpr Slice(T *ptr, size_t len) : cpp::span<T>(ptr, len) {}
  LIBC_INLINE constexpr Slice(T const *ptr, size_t len)
      : cpp::span<T>(ptr, len) {}

  template <typename U, size_t N>
  LIBC_INLINE constexpr Slice(std::array<U, N> &arr) : cpp::span<T>(arr) {}

  LIBC_INLINE constexpr Slice(const Slice<T> &) = default;
  LIBC_INLINE constexpr Slice(Slice<T> &&) = default;

  LIBC_INLINE constexpr Slice(const cpp::span<T> &s) : cpp::span<T>(s) {}
  LIBC_INLINE constexpr Slice(cpp::span<T> &&s) : cpp::span<T>(s) {}

  LIBC_INLINE constexpr Slice &operator=(const Slice<T> &n) = default;
  LIBC_INLINE constexpr Slice &operator=(Slice<T> &&n) = default;

  LIBC_INLINE constexpr Slice &operator=(const cpp::span<T> &n) {
    cpp::span<T>::operator=(n);
    return *this;
  }
  LIBC_INLINE constexpr Slice &operator=(cpp::span<T> &&n) {
    cpp::span<T>::operator=(n);
    return *this;
  }

  // Binary searches this slice with a comparator function.
  //
  // The comparator function should return an order code that indicates whether
  // its argument is `Less`, `Equal` or `Greater` the desired target.
  // If the slice is not sorted or if the comparator function does not
  // implement an order consistent with the sort order of the underlying
  // slice, the returned result is unspecified and meaningless.
  //
  // If the value is found then `cpp::expected<size_t>` is returned,
  // containing the index of the matching element. If there are multiple
  // matches, then any one of the matches could be returned. The index is chosen
  // deterministically.
  // If the value is not found then `cpp::unexpected<size_t>` is returned,
  // containing the index where a matching element could be inserted while
  // maintaining sorted order.
  template <typename Fn>
  LIBC_INLINE constexpr cpp::expected<size_t, size_t>
  binary_search_by(Fn func) const {
    auto size = this->size();
    if (size == 0) {
      return cpp::unexpected<size_t>(0);
    }

    size_t base = 0;

    while (size > 1) {
      auto half = size / 2;
      auto mid = base + half;
      auto cmp = func(this->operator[](mid));
      base = (cmp == Ordering::Greater) ? base : mid;
      size -= half;
    }

    auto cmp = func(this->operator[](base));
    if (cmp == Ordering::Equal) {
      LIBC_ASSERT(base < this->size());
      return base;
    } else {
      auto result = base + static_cast<size_t>(cmp == Ordering::Less);
      LIBC_ASSERT(result <= this->size());
      return cpp::unexpected(result);
    }
  }

  LIBC_INLINE constexpr Slice<T> slice_form_range(size_t start,
                                                  size_t end) const {
    LIBC_ASSERT(start <= end && end <= this->size());
    return Slice<T>(this->data() + start, end - start);
  }

  LIBC_INLINE constexpr bool contains(T elm) const {
    for (auto it : *this) {
      if (elm == it) {
        return true;
      }
    }
    return false;
  }

  LIBC_INLINE constexpr void copy_from_slice(Slice<T> other) const {
    for (size_t i = 0; i < std::min(this->size(), other.size()); i++) {
      this->data()[i] = other.data()[i];
    }
  }
};

namespace utility {

inline constexpr uint64_t mul_high(uint64_t a, uint64_t b) {
  return (((__uint128_t)a * (__uint128_t)b) >> 64);
}

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

} // namespace utility

namespace rngs {

static inline constexpr size_t BLOCK = 16;
static inline constexpr uint64_t BLOCK64 = BLOCK;
static inline constexpr uint64_t LOG2_BUFBLOCKS = 2;
static inline constexpr uint64_t BUFBLOCKS = 1 << LOG2_BUFBLOCKS;
static inline constexpr uint64_t BUFSZ64 = BLOCK64 * BUFBLOCKS;
static inline constexpr size_t BUFSZ = BUFSZ64;

union vec128_storage {
  mutable std::array<uint32_t, 4> u32x4;

  constexpr vec128_storage(std::array<uint32_t, 4> &&x) : u32x4(x) {}
  constexpr vec128_storage(std::array<uint32_t, 4> &x) : u32x4(x) {}
  constexpr vec128_storage() : u32x4() {}

  constexpr operator std::array<uint32_t, 4>() const { return this->u32x4; }

  inline constexpr std::array<uint32_t, 4> to_lanes() const {
    return this->u32x4;
  }

  inline static constexpr vec128_storage
  from_lanes(std::array<uint32_t, 4> &&xs) {
    // std::array<uint64_t, 2> x = {
    //     static_cast<uint64_t>(xs[0]) | (static_cast<uint64_t>(xs[1]) << 32),
    //     static_cast<uint64_t>(xs[2]) |
    //         static_cast<uint64_t>(static_cast<uint64_t>(xs[3]) << 32)};

    return vec128_storage(xs);
  }

  inline static constexpr auto from_lanes(std::array<uint64_t, 2> &&xs) {
    return vec128_storage(std::array{
        static_cast<uint32_t>(xs[0]), static_cast<uint32_t>(xs[0] >> 32),
        static_cast<uint32_t>(xs[1]), static_cast<uint32_t>(xs[1] >> 32)});
  }

  inline static constexpr auto read_le(Slice<uint8_t> x) {
    // static_assert(x.size() == 16);
    vec128_storage v = std::array<uint32_t, 4>{0};
    uint32_t *dst = v.u32x4.data();
    uint8_t *src = x.data();
    for (uint8_t i = 0; i < 4; ++i)
      dst[i] = src[i * 4] | (src[i * 4 + 1] << 8) | (src[i * 4 + 2] << 16) |
               (src[i * 4 + 3] << 24);
    return v;
  }
};

union vec256_storage {
  mutable std::array<uint32_t, 8> u32x8;
  // mutable std::array<vec128_storage, 2> sse2;

  constexpr operator std::array<uint32_t, 8>() const { return this->u32x8; }
  // constexpr operator std::array<vec128_storage, 2>() const {
  //   return this->sse2;
  // }

  constexpr vec256_storage() : u32x8() {}
  static inline constexpr vec256_storage
  construct_from_vec128(vec128_storage &&lo, vec128_storage &&hi) {
    vec256_storage r{{}};
    for (size_t i = 0; i < 4; i++) {
      r.u32x8[i] = lo.u32x4[i];
    }
    for (size_t i = 0; i < 4; i++) {
      r.u32x8[i + 4] = hi.u32x4[i];
    }
    return r;
  }

  constexpr vec256_storage(std::array<uint32_t, 8> &&x) : u32x8(x) {}

  inline static constexpr vec256_storage mm256_shuffle_epi32(vec256_storage a,
                                                             int imm) {
    vec256_storage r{{}};

    // lower half (elements 0..3)
    for (int i = 0; i < 4; ++i) {
      int src = (imm >> (2 * i)) & 0x3;
      r.u32x8[i] = a.u32x8[src];
    }

    // upper half (elements 4..7)
    for (int i = 0; i < 4; ++i) {
      int src = (imm >> (2 * i)) & 0x3;
      r.u32x8[4 + i] = a.u32x8[4 + src];
    }

    return r;
  }

  constexpr vec256_storage shuffle_lane_words3012() const {
    return mm256_shuffle_epi32(*this, 0b0011'1001);
  }

  constexpr vec256_storage shuffle_lane_words2301() const {
    return mm256_shuffle_epi32(*this, 0b0100'1110);
  }

  constexpr vec256_storage shuffle_lane_words1230() const {
    return mm256_shuffle_epi32(*this, 0b1001'0011);
  }

  inline static constexpr vec128_storage
  mm256_extracti128_si256(const vec256_storage &V, int M) {
    const int base = (M & 1) * 4;
    return {{V.u32x8[base + 0], V.u32x8[base + 1], V.u32x8[base + 2],
             V.u32x8[base + 3]}};
  }

  constexpr vec256_storage to_lanes() const {
    auto lo = mm256_extracti128_si256(*this, 0);
    auto hi = mm256_extracti128_si256(*this, 1);
    return {std::array{
        lo.u32x4[0],
        lo.u32x4[1],
        lo.u32x4[2],
        lo.u32x4[3],
        hi.u32x4[0],
        hi.u32x4[1],
        hi.u32x4[2],
        hi.u32x4[3],
    }};
  }
};

union vec512_storage {
  mutable std::array<uint32_t, 16> u32x16;
  // mutable std::array<vec128_storage, 4> sse2;
  // mutable std::array<vec256_storage, 2> avx;

  // constexpr vec512_storage() : u32x16() {}
  // constexpr vec512_storage(std::array<vec128_storage, 4> &sse2) : sse2(sse2)
  // {} constexpr vec512_storage(std::array<vec128_storage, 4> &&sse2) :
  // sse2(sse2) {} constexpr vec512_storage(std::array<vec256_storage, 2> &avx)
  // : avx(avx) {} constexpr vec512_storage(std::array<vec256_storage, 2> &&avx)
  // : avx(avx) {}

  inline static constexpr vec512_storage
  construct_from_vec256(const vec256_storage &lo, const vec256_storage &hi) {
    vec512_storage r{{}};
    for (size_t i = 0; i < 8; i++) {
      r.u32x16[i] = lo.u32x8[i];
    }
    for (size_t i = 0; i < 8; i++) {
      r.u32x16[8 + i] = hi.u32x8[i];
    }
    return r;
  }

  constexpr operator std::array<uint32_t, 16>() const { return this->u32x16; }
  // constexpr operator std::array<vec128_storage, 4>() const {
  //   return this->sse2;
  // }
  // constexpr operator std::array<vec256_storage, 2>() const { return
  // this->avx; }

  inline static constexpr vec512_storage new128(std::array<uint32_t, 16> &&xs) {
    return vec512_storage{xs};
  }
  inline static constexpr vec512_storage new128(vec128_storage i,
                                                vec128_storage j,
                                                vec128_storage k,
                                                vec128_storage l) {
    vec512_storage r{{}};
    for (size_t a = 0; a < 4; a++) {
      r.u32x16[a] = i.u32x4[a];
      r.u32x16[a + 4] = j.u32x4[a];
      r.u32x16[a + 8] = k.u32x4[a];
      r.u32x16[a + 12] = l.u32x4[a];
    }
    return r;
  }

  inline constexpr vec512_storage unpack() const { return *this; }

  static inline constexpr vec256_storage
  mm256_add_epi32(const vec256_storage &a, const vec256_storage &b) {
    vec256_storage r{{}};
    for (int i = 0; i < 8; ++i) {
      r.u32x8[i] = a.u32x8[i] + b.u32x8[i]; // modulo 2^32
    }
    return r;
  }

  static inline constexpr vec512_storage
  mm256_add_epi32(const vec512_storage &a, const vec512_storage &b) {
    vec512_storage r{{}};
    for (int i = 0; i < 16; ++i) {
      r.u32x16[i] = a.u32x16[i] + b.u32x16[i]; // modulo 2^32
    }
    return r;
  }

  constexpr auto &operator+=(vec512_storage &rhs) const {
    this->u32x16 = mm256_add_epi32(*this, rhs).u32x16;
    return *this;
  }

  constexpr vec512_storage operator+(vec512_storage &rhs) const {
    return mm256_add_epi32(*this, rhs);
  }

  constexpr vec512_storage operator+(const vec512_storage &rhs) const {
    return mm256_add_epi32(*this, rhs);
  }

  static inline constexpr vec256_storage
  mm256_xor_si256(const std::array<uint32_t, 8> &a,
                  const std::array<uint32_t, 8> &b) {
    std::array<uint32_t, 8> r{};
    for (int i = 0; i < 8; ++i) {
      r[i] = a[i] ^ b[i];
    }
    return r;
  }

  static inline constexpr vec512_storage
  mm256_xor_si256(const std::array<uint32_t, 16> &a,
                  const std::array<uint32_t, 16> &b) {
    vec512_storage r{.u32x16 = {}};
    for (int i = 0; i < 16; ++i) {
      r.u32x16[i] = a[i] ^ b[i];
    }
    return r;
  }

  constexpr auto operator^(vec512_storage &rhs) const {
    return mm256_xor_si256(*this, rhs);
  }

  static inline constexpr vec256_storage
  mm256_shuffle_epi8(const vec256_storage &a, const vec256_storage &b) {
    vec256_storage r{{}};
    for (size_t k = 0; k < 8; k++) {
      r.u32x8[k] = 0;
    }

    // Helper for 128-bit lane (16 bytes)
    auto shuffle_128 = [](const uint32_t *src, const uint32_t *ctrl,
                          uint32_t *dst) {
      // dst must be zero-initialized by caller
      for (int i = 0; i < 16; ++i) {
        uint8_t c = (ctrl[i / 4] >> ((i % 4) * 8)) & 0xFF;

        if (c & 0x80) {
          // zero byte → already zero
          continue;
        }

        int k = c & 0x0F;
        uint8_t byte = (src[k / 4] >> ((k % 4) * 8)) & 0xFF;

        dst[i / 4] |= static_cast<uint32_t>(byte) << ((i % 4) * 8);
      }
    };

    // Shuffle lower 128-bit lane
    shuffle_128(&a.u32x8[0], &b.u32x8[0], &r.u32x8[0]);
    // Shuffle upper 128-bit lane
    shuffle_128(&a.u32x8[4], &b.u32x8[4], &r.u32x8[4]);

    return r;
  }

  inline static constexpr vec256_storage
  mm256_set_epi64x(long long a, long long b, long long c, long long d) {
    vec256_storage v{{}};

    // Lower 128-bit lane (d, c)
    v.u32x8[0] = static_cast<uint32_t>(d);       // d[31:0]
    v.u32x8[1] = static_cast<uint32_t>(d >> 32); // d[63:32]
    v.u32x8[2] = static_cast<uint32_t>(c);       // c[31:0]
    v.u32x8[3] = static_cast<uint32_t>(c >> 32); // c[63:32]

    // Upper 128-bit lane (b, a)
    v.u32x8[4] = static_cast<uint32_t>(b);       // b[31:0]
    v.u32x8[5] = static_cast<uint32_t>(b >> 32); // b[63:32]
    v.u32x8[6] = static_cast<uint32_t>(a);       // a[31:0]
    v.u32x8[7] = static_cast<uint32_t>(a >> 32); // a[63:32]

    return v;
  }

  constexpr vec512_storage rotate_each_word_right16() const {
    auto constexpr k0 = 0x0d0c'0f0e'0908'0b0a;
    auto constexpr k1 = 0x0504'0706'0100'0302;

    vec256_storage lo{{}};
    vec256_storage hi{{}};

    for (size_t i = 0; i < 8; i++) {
      lo.u32x8[i] = this->u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      hi.u32x8[i] = this->u32x16[8 + i];
    }
    lo = mm256_shuffle_epi8(lo, mm256_set_epi64x(k0, k1, k0, k1));
    hi = mm256_shuffle_epi8(hi, mm256_set_epi64x(k0, k1, k0, k1));

    vec512_storage ret{{}};
    for (size_t i = 0; i < 8; i++) {
      ret.u32x16[i] = lo.u32x8[i];
    }

    for (size_t i = 0; i < 8; i++) {
      ret.u32x16[8 + i] = hi.u32x8[i];
    }

    return ret;
  }

  inline static constexpr vec256_storage
  mm256_or_si256(const vec256_storage &a, const vec256_storage &b) {
    vec256_storage r{{}};
    for (int i = 0; i < 8; ++i) {
      r.u32x8[i] = a.u32x8[i] | b.u32x8[i];
    }
    return r;
  }

  static inline constexpr vec256_storage
  mm256_srli_epi32(const vec256_storage &a, int count) {
    vec256_storage r{{}};

    // Cap the shift count at 31, as larger shifts produce zero
    const int c = count & 0x1F;

    for (int i = 0; i < 8; ++i) {
      r.u32x8[i] = a.u32x8[i] >> c;
    }

    return r;
  }

  static inline constexpr vec256_storage
  mm256_slli_epi32(const vec256_storage &a, int count) {
    vec256_storage r{{}};

    // Cap the shift count at 31, as larger shifts produce zero
    const int c = count & 0x1F;

    for (int i = 0; i < 8; ++i) {
      r.u32x8[i] = a.u32x8[i] << c;
    }

    return r;
  }

  constexpr vec512_storage rotate_each_word_right20() const {
    constexpr int32_t i = 20;

    vec256_storage lo{{}};
    vec256_storage hi{{}};

    for (size_t i = 0; i < 8; i++) {
      lo.u32x8[i] = this->u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      hi.u32x8[i] = this->u32x16[8 + i];
    }

    lo = mm256_or_si256(mm256_srli_epi32(lo, i), mm256_slli_epi32(lo, 32 - i));
    hi = mm256_or_si256(mm256_srli_epi32(hi, i), mm256_slli_epi32(hi, 32 - i));

    vec512_storage r{{}};

    for (size_t i = 0; i < 8; i++) {
      r.u32x16[i] = lo.u32x8[i];
    }
    for (size_t i = 0; i < 8; i++) {
      r.u32x16[8 + i] = hi.u32x8[i];
    }

    return r;
  }

  constexpr vec512_storage rotate_each_word_right24() const {
    auto constexpr k0 = 0x0e0d'0c0f'0a09'080b;
    auto constexpr k1 = 0x0605'0407'0201'0003;

    vec256_storage lo{{}};
    vec256_storage hi{{}};

    for (size_t i = 0; i < 8; i++) {
      lo.u32x8[i] = this->u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      hi.u32x8[i] = this->u32x16[8 + i];
    }

    lo = mm256_shuffle_epi8(lo, mm256_set_epi64x(k0, k1, k0, k1));
    hi = mm256_shuffle_epi8(hi, mm256_set_epi64x(k0, k1, k0, k1));

    vec512_storage r{{}};

    for (size_t i = 0; i < 8; i++) {
      r.u32x16[i] = lo.u32x8[i];
    }
    for (size_t i = 0; i < 8; i++) {
      r.u32x16[8 + i] = hi.u32x8[i];
    }

    return r;
  }

  constexpr vec512_storage rotate_each_word_right25() const {
    constexpr int32_t i = 25;
    vec256_storage lo{{}};
    vec256_storage hi{{}};

    for (size_t i = 0; i < 8; i++) {
      lo.u32x8[i] = this->u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      hi.u32x8[i] = this->u32x16[8 + i];
    }

    lo = mm256_or_si256(mm256_srli_epi32(lo, i), mm256_slli_epi32(lo, 32 - i));
    hi = mm256_or_si256(mm256_srli_epi32(hi, i), mm256_slli_epi32(hi, 32 - i));

    vec512_storage r{{}};

    for (size_t i = 0; i < 8; i++) {
      r.u32x16[i] = lo.u32x8[i];
    }
    for (size_t i = 0; i < 8; i++) {
      r.u32x16[8 + i] = hi.u32x8[i];
    }

    return r;
  }

  constexpr vec512_storage shuffle_lane_words3012() const {
    vec256_storage lo{{}};
    vec256_storage hi{{}};

    for (size_t i = 0; i < 8; i++) {
      lo.u32x8[i] = this->u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      hi.u32x8[i] = this->u32x16[8 + i];
    }
    lo = lo.shuffle_lane_words3012();
    hi = hi.shuffle_lane_words3012();

    vec512_storage r{{}};

    for (size_t i = 0; i < 8; i++) {
      r.u32x16[i] = lo.u32x8[i];
    }
    for (size_t i = 0; i < 8; i++) {
      r.u32x16[8 + i] = hi.u32x8[i];
    }

    return r;
  }

  constexpr vec512_storage shuffle_lane_words2301() const {
    vec256_storage lo{{}};
    vec256_storage hi{{}};

    for (size_t i = 0; i < 8; i++) {
      lo.u32x8[i] = this->u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      hi.u32x8[i] = this->u32x16[8 + i];
    }
    lo = lo.shuffle_lane_words2301();
    hi = hi.shuffle_lane_words2301();

    vec512_storage r{{}};

    for (size_t i = 0; i < 8; i++) {
      r.u32x16[i] = lo.u32x8[i];
    }
    for (size_t i = 0; i < 8; i++) {
      r.u32x16[8 + i] = hi.u32x8[i];
    }

    return r;
  }

  constexpr vec512_storage shuffle_lane_words1230() const {
    vec256_storage lo{{}};
    vec256_storage hi{{}};

    for (size_t i = 0; i < 8; i++) {
      lo.u32x8[i] = this->u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      hi.u32x8[i] = this->u32x16[8 + i];
    }

    lo = lo.shuffle_lane_words1230();
    hi = hi.shuffle_lane_words1230();

    vec512_storage r{{}};

    for (size_t i = 0; i < 8; i++) {
      r.u32x16[i] = lo.u32x8[i];
    }
    for (size_t i = 0; i < 8; i++) {
      r.u32x16[8 + i] = hi.u32x8[i];
    }

    return r;
  }

  static inline constexpr vec256_storage
  mm256_permute2x128_si256(const vec256_storage &V1, const vec256_storage &V2,
                           int M) {
    vec256_storage r{{}};

    // For each 128-bit destination half
    for (int half = 0; half < 2; ++half) {
      int control = (M >> (half * 4)) & 0xF;
      int dst_base = half * 4;

      if (control & 0x8) {
        // bit 3 set → zero this 128-bit half
        for (int i = 0; i < 4; ++i) {
          r.u32x8[dst_base + i] = 0;
        }
      } else {
        // bits [1:0] select source half
        const vec256_storage *src{};
        int src_base{};

        switch (control & 0x3) {
        case 0: // V1 lower
          src = &V1;
          src_base = 0;
          break;
        case 1: // V1 upper
          src = &V1;
          src_base = 4;
          break;
        case 2: // V2 lower
          src = &V2;
          src_base = 0;
          break;
        case 3: // V2 upper
          src = &V2;
          src_base = 4;
          break;
        }

        for (int i = 0; i < 4; ++i) {
          r.u32x8[dst_base + i] = src->u32x8[src_base + i];
        }
      }
    }

    return r;
  }

  static constexpr std::array<vec512_storage, 4>
  transpose4(const vec512_storage &a, const vec512_storage &b,
             const vec512_storage &c, const vec512_storage &d) {
    /*
     * a00:a01 a10:a11
     * b00:b01 b10:b11
     * c00:c01 c10:c11
     * d00:d01 d10:d11
     *       =>
     * a00:b00 c00:d00
     * a01:b01 c01:d01
     * a10:b10 c10:d10
     * a11:b11 c11:d11
     */
    vec256_storage a_lo{{}};
    vec256_storage b_lo{{}};

    for (size_t i = 0; i < 8; i++) {
      a_lo.u32x8[i] = a.u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      b_lo.u32x8[i] = b.u32x16[i];
    }
    auto const ab00 = mm256_permute2x128_si256(a_lo, b_lo, 0x20);
    auto const ab01 = mm256_permute2x128_si256(a_lo, b_lo, 0x31);

    vec256_storage a_hi{{}};
    vec256_storage b_hi{{}};

    for (size_t i = 0; i < 8; i++) {
      a_hi.u32x8[i] = a.u32x16[8 + i];
    }
    for (size_t i = 0; i < 8; i++) {
      b_hi.u32x8[i] = b.u32x16[8 + i];
    }
    auto const ab10 = mm256_permute2x128_si256(a_hi, b_hi, 0x20);
    auto const ab11 = mm256_permute2x128_si256(a_hi, b_hi, 0x31);

    vec256_storage c_lo{{}};
    vec256_storage d_lo{{}};

    for (size_t i = 0; i < 8; i++) {
      c_lo.u32x8[i] = c.u32x16[i];
    }
    for (size_t i = 0; i < 8; i++) {
      d_lo.u32x8[i] = d.u32x16[i];
    }
    auto const cd00 = mm256_permute2x128_si256(c_lo, d_lo, 0x20);
    auto const cd01 = mm256_permute2x128_si256(c_lo, d_lo, 0x31);

    vec256_storage c_hi{{}};
    vec256_storage d_hi{{}};

    for (size_t i = 0; i < 8; i++) {
      c_hi.u32x8[i] = c.u32x16[8 + i];
    }
    for (size_t i = 0; i < 8; i++) {
      d_hi.u32x8[i] = d.u32x16[8 + i];
    }
    auto const cd10 = mm256_permute2x128_si256(c_hi, d_hi, 0x20);
    auto const cd11 = mm256_permute2x128_si256(c_hi, d_hi, 0x31);

    auto r1 = vec512_storage::construct_from_vec256(ab00, cd00);
    auto r2 = vec512_storage::construct_from_vec256(ab01, cd01);
    auto r3 = vec512_storage::construct_from_vec256(ab10, cd10);
    auto r4 = vec512_storage::construct_from_vec256(ab11, cd11);

    return {r1, r2, r3, r4};
  }

  constexpr Slice<uint32_t> to_scalars() const {
    return Slice<uint32_t>(this->u32x16.data(), this->u32x16.size());
  }

  constexpr vec512_storage to_lanes() const { return *this; }
};

struct ChaCha {
  vec128_storage b;
  vec128_storage c;
  vec128_storage d;
};

template <class V> struct State {
  V a, b, c, d;
};

template <typename V = vec512_storage>
static constexpr inline State<V> round(State<V> x) {
  x.a += x.b;
  x.d = (x.d ^ x.a).rotate_each_word_right16();
  x.c += x.d;
  x.b = (x.b ^ x.c).rotate_each_word_right20();
  x.a += x.b;
  x.d = (x.d ^ x.a).rotate_each_word_right24();
  x.c += x.d;
  x.b = (x.b ^ x.c).rotate_each_word_right25();
  return x;
}

template <typename V = vec512_storage>
inline constexpr State<V> diagonalize(State<V> x) {
  x.b = x.b.shuffle_lane_words3012();
  x.c = x.c.shuffle_lane_words2301();
  x.d = x.d.shuffle_lane_words1230();
  return x;
}

template <typename V = vec512_storage>
inline constexpr State<V> undiagonalize(State<V> x) {
  x.b = x.b.shuffle_lane_words1230();
  x.c = x.c.shuffle_lane_words2301();
  x.d = x.d.shuffle_lane_words3012();
  return x;
}

inline constexpr std::array<uint32_t, 4>
add_epi64(const std::array<uint32_t, 4> &a, const std::array<uint32_t, 4> &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

inline constexpr vec128_storage add_pos(vec128_storage &d, uint64_t i) {
  auto const d0 = d.u32x4;
  auto const incr = vec128_storage::from_lanes(std::array<uint64_t, 2>{i, 0});
  return add_epi64(d0, incr.u32x4);
}

// a_lo and a_hi are each 128-bit vectors represented as 4 x 32-bit integers
inline constexpr vec256_storage mm256_setr_m128i(const vec128_storage &lo,
                                                 const vec128_storage &hi) {
  return {std::array{
      lo.u32x4[0],
      lo.u32x4[1],
      lo.u32x4[2],
      lo.u32x4[3],
      hi.u32x4[0],
      hi.u32x4[1],
      hi.u32x4[2],
      hi.u32x4[3],
  }};
}

inline constexpr vec128_storage mm_add_epi64(const std::array<uint32_t, 4> &a,
                                             const std::array<uint32_t, 4> &b) {
  return {std::array<uint32_t, 4>{
      a[0] + b[0],
      a[1] + b[1],
      a[2] + b[2],
      a[3] + b[3],
  }};
}

// Rotates bits to the right for unsigned integral types
template <typename T>
LIBC_INLINE static constexpr T rotate_right(T number, size_t rotation) {
  static_assert(std::is_unsigned_v<T>, "rotate_right requires unsigned type");

  constexpr size_t BITS = std::numeric_limits<T>::digits;
  rotation %= BITS;
  return static_cast<T>((number >> rotation) | (number << (BITS - rotation)));
}

// Converts a 32-bit unsigned integer to an array of 4 little-endian bytes
LIBC_INLINE static constexpr std::array<uint8_t, 4>
to_le_bytes(uint32_t number) {
  return {
      static_cast<uint8_t>(number),
      static_cast<uint8_t>(number >> 8),
      static_cast<uint8_t>(number >> 16),
      static_cast<uint8_t>(number >> 24),
  };
}

struct StdRng {
  mutable ChaCha core_state;
  mutable std::array<uint, 64> results;
  mutable uint8_t index;

  static constexpr std::array<uint8_t, 4> pcg32(uint64_t &state) {
    constexpr uint64_t MUL = 6364136223846793005ull;
    constexpr uint64_t INC = 11634580027462260723ull;

    state = utility::wrapping_mul(state, MUL) + INC;
    // Use PCG output function with to_le to generate x:
    uint32_t xorshifted = (((state >> 18) ^ state) >> 27);
    uint32_t rot = state >> 59;
    uint32_t x = rotate_right(xorshifted, rot);
    return to_le_bytes(x);
  }

  static constexpr StdRng from_seed(uint64_t state = 31415) {
    std::array<uint8_t, 32> key = {0};
    auto s = Slice(key.data(), 32);
    size_t chunk_size = 4;
    size_t num_chunks = s.size() / chunk_size;

    for (size_t i = 0; i < num_chunks; ++i) {
      auto chunk = Slice{s.data() + i * chunk_size, chunk_size};
      std::array<uint8_t, 4> x = pcg32(state);
      uint8_t *dst = chunk.data();
      uint8_t *src = x.data();

      for (unsigned j = 0; j < chunk.size(); ++j)
        dst[j] = src[j];
    }
    auto key0 = vec128_storage::read_le(
        Slice(key.data(), key.size()).slice_form_range(0, 16));
    auto key1 =
        vec128_storage::read_le(Slice(key.data(), key.size()).subspan(16));

    auto core_state = ChaCha{
        .b = key0,
        .c = key1,
        .d = vec128_storage(std::array<uint32_t, 4>{0, 0, 0, 0}),
    };
    auto results = std::array<uint, 64>{0};
    uint8_t index = results.size();
    return StdRng{core_state, results, index};
  }

  inline constexpr void generate_and_set(size_t index) const {
    // assert(index < this->results.size());
    this->generate(this->results);
    this->index = index;
  }

  inline static constexpr vec512_storage d0123(vec128_storage &d) {
    auto x1 = vec256_storage::construct_from_vec128(
        vec128_storage::from_lanes(std::array<uint64_t, 2>{0, 0}),
        vec128_storage::from_lanes(std::array<uint64_t, 2>{1, 0}));
    auto x2 = vec256_storage::construct_from_vec128(
        vec128_storage::from_lanes(std::array<uint64_t, 2>{2, 0}),
        vec128_storage::from_lanes(std::array<uint64_t, 2>{3, 0}));

    vec512_storage incr = vec512_storage::construct_from_vec256(x1, x2);

    vec128_storage p1, p2, p3, p4;
    for (size_t x = 0; x < 4; x++) {
      p1.u32x4[x] = incr.u32x16[x];
      p2.u32x4[x] = incr.u32x16[x + 4];
      p3.u32x4[x] = incr.u32x16[x + 8];
      p4.u32x4[x] = incr.u32x16[x + 12];
    }
    vec128_storage i = mm_add_epi64(d, p1);
    vec128_storage j = mm_add_epi64(d, p2);
    vec128_storage k = mm_add_epi64(d, p3);
    vec128_storage l = mm_add_epi64(d, p4);

    auto v = vec512_storage::new128(i, j, k, l);

    return v;
  }

  inline constexpr void
  refill_wide_impl(uint32_t drounds, std::array<uint32_t, BUFSZ> &out) const {
    auto k = vec128_storage::from_lanes(
        {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574});
    vec128_storage b = core_state.b;
    vec128_storage c = core_state.c;
    auto x = State<vec512_storage>{
        .a = vec512_storage::construct_from_vec256(mm256_setr_m128i(k, k),
                                                   mm256_setr_m128i(k, k)),
        .b = vec512_storage::construct_from_vec256(mm256_setr_m128i(b, b),
                                                   mm256_setr_m128i(b, b)),
        .c = vec512_storage::construct_from_vec256(mm256_setr_m128i(c, c),
                                                   mm256_setr_m128i(c, c)),
        .d = d0123(core_state.d),
    };

    for (size_t i = 0; i < drounds; i++) {
      x = round(x);
      x = undiagonalize(round(diagonalize(x)));
    }

    auto const kk = vec512_storage::construct_from_vec256(
        mm256_setr_m128i(k, k), mm256_setr_m128i(k, k));
    auto const sb1 = core_state.b;
    auto sb = vec512_storage::construct_from_vec256(mm256_setr_m128i(sb1, sb1),
                                                    mm256_setr_m128i(sb1, sb1));
    auto const sc1 = core_state.c;
    auto sc = vec512_storage::construct_from_vec256(mm256_setr_m128i(sc1, sc1),
                                                    mm256_setr_m128i(sc1, sc1));
    auto const sd = d0123(core_state.d);
    auto const &[ra, rb, rc, rd] =
        vec512_storage::transpose4(kk + x.a, x.b + sb, x.c + sc, x.d + sd);

    Slice<uint32_t> sout(out.data(), out.size());
    sout.slice_form_range(0, 16).copy_from_slice(ra.to_scalars());
    sout.slice_form_range(16, 32).copy_from_slice(rb.to_scalars());
    sout.slice_form_range(32, 48).copy_from_slice(rc.to_scalars());
    sout.slice_form_range(48, 64).copy_from_slice(rd.to_scalars());
    vec128_storage rx{{}};
    auto tc = sd.to_lanes().u32x16;
    for (size_t z = 0; z < 4; z++) {
      rx.u32x4[z] = tc[z];
    }
    core_state.d = add_pos(rx, 4);
  }

  inline constexpr void generate(std::array<uint32_t, BUFSZ> &out) const {
    refill_wide_impl(6, out);
  }

  inline constexpr uint64_t next_u64() const {
    constexpr auto read_u64 = [](Slice<uint32_t> results, size_t index) {
      auto data = results.slice_form_range(index, index + 2);
      return (static_cast<uint64_t>(data[1]) << 32) |
             static_cast<uint64_t>(data[0]);
    };
    const auto len = this->results.size();
    const auto index = this->index;

    if (index < len - 1) {
      this->index += 2;
      // Read an u64 from the current index
      return read_u64(Slice(this->results.data(), this->results.size()), index);
    } else if (index >= len) {
      this->generate_and_set(2);
      return read_u64(Slice(this->results.data(), this->results.size()), 0);
    } else {
      const uint64_t x = this->results[len - 1];
      this->generate_and_set(1);
      const uint64_t y = this->results[0];
      return (y << 32) | x;
    }
  }

  constexpr uint64_t random() const { return next_u64(); }
};

} // namespace rngs

template <typename BucketFnImpl> class BucketFn {
public:
  constexpr static bool LINEAR = false;
  constexpr static bool B_OUTPUT = false;
  constexpr void set_buckets_per_part(uint64_t) const {}
  constexpr uint64_t call(uint64_t x) const {
    return static_cast<const BucketFnImpl *>(this)->call_impl(x);
  }
};

class Linear : public BucketFn<Linear> {
public:
  constexpr static bool LINEAR = true;
  constexpr uint64_t call_impl(uint64_t x) const { return x; }
};

/// A 2-piece-wise linear function; as used in FCH and PTHash.
///
/// |              .
/// |             .
/// |         ....---< gamma
/// |    .....   |
/// |....        |
/// +------------^--
///              beta
///
/// line1: y = x * (gamma / beta)
///                ~~~ slope1 ~~~
/// line2: y = x * ((1 - gamma) / (1 - beta)) + (gamma - beta) / (1 - beta)
///                ~~~~~~~~~ slope2 ~~~~~~~~~   ~~~~~~~~~~ offset ~~~~~~~~~
class Skewed : public BucketFn<Skewed> {
public:
  mutable double beta_f;
  mutable double gamma_f;
  /// buckets per part
  mutable uint64_t b;
  mutable uint64_t beta;
  mutable uint64_t slope1;
  mutable uint64_t slope2;
  mutable uint64_t neg_offset;

  constexpr Skewed(double beta = 0.6, double gamma = 0.3)
      : beta_f(beta), gamma_f(gamma), b(0), beta(0), slope1(0), slope2(0),
        neg_offset(0) {
    // assert(beta > gamma && "Beta={beta} must be larger than gamma={gamma}");
  }

  static constexpr bool B_OUTPUT = true;

  constexpr void set_buckets_per_part(uint64_t b) const {
    auto beta = this->beta_f;
    auto gamma = this->gamma_f;
    this->b = b;
    constexpr auto as_u64 = [](double x) -> uint64_t {
      return x * static_cast<double>(~static_cast<uint64_t>(0));
    };
    this->slope1 = utility::mul_high(as_u64(gamma / beta), this->b);
    this->slope2 = utility::mul_high(as_u64((1. - gamma) / (1. - beta) / 8.),
                                     this->b << 3);
    this->neg_offset = utility::mul_high(
        as_u64((beta - gamma) / (1. - beta) / 8.), this->b << 3);
    this->beta = as_u64(beta);
  }

  constexpr uint64_t call_impl(uint64_t x) const {
    // NOTE: There is a lot of MOV/CMOV going on here.
    auto is_large = x >= this->beta;
    auto slope = is_large ? this->slope2 : this->slope1;
    return utility::mul_high(x, slope) - is_large * this->neg_offset;
    // assert(!is_large || this->p2 <= b, "p2 {} <= b {}", this->p2, b);
    // assert(!is_large || b < this->b, "b {} < p2 {}", b, this->b);
    // assert(is_large || b < this->p2, "b {} < p2 {}", b, this->p2);
  }
};

class Optimal : public BucketFn<Optimal> {
public:
  double eps;

  constexpr uint64_t call_impl(uint64_t _x) const {
    double constexpr p32 = (1ULL << 32);
    constexpr auto p64 = p32 * p32;
    constexpr auto p64inv = 1. / p64;
    auto x = ((double)_x) * p64inv;
    auto y = x + (1. - this->eps) * (1. - x) * std::log(1. - x);

    return y * p64;
  }
};

class Square : public BucketFn<Square> {
public:
  constexpr uint64_t call_impl(uint64_t x) const {
    return utility::mul_high(x, x);
  }
};

class SquareEps : public BucketFn<SquareEps> {
public:
  constexpr uint64_t call_impl(uint64_t x) const {
    return utility::mul_high(x, x) / 256 * 255 + x / 256;
  }
};

class Cubic : public BucketFn<Cubic> {
public:
  constexpr uint64_t call_impl(uint64_t x) const {
    // x * x * (1 + x) / 2
    return utility::mul_high(utility::mul_high(x, x), (x >> 1) | (1ULL << 63));
  }
};

class CubicEps : public BucketFn<CubicEps> {
public:
  constexpr uint64_t call_impl(uint64_t x) const {
    // x * x * (1 + x) / 2
    return utility::mul_high(utility::mul_high(x, x), (x >> 1) | (1ULL << 63)) /
               256 * 255 +
           x / 256;
  }
};

// static constexpr void test_skewed() {
//   constexpr Skewed skewed(0.6, 0.3);
//   skewed.set_buckets_per_part(1000000000);
//   auto last_y = 0;
//   auto n = 100;
//   for (size_t i = 0; i < 100; i++) {
//     auto x = ~(uint64_t)0 / n * i;
//     auto y = skewed.call(x);
//     assert(y >= last_y);
//     last_y = y;
//   }
// }

enum class ShardingType { None, Memory, Disk, Hybrid };

struct Sharding {
  ShardingType type = ShardingType::None;
  size_t mem = 0;
};

template <typename BF> struct PtrHashParams {
  bool remap : 1;
  double alpha;
  double lambda;
  BF bucket_fn;
  size_t keys_per_shard;
  Sharding sharding;
  bool single_part : 1;

  constexpr PtrHashParams<Linear>()
      : PtrHashParams(PtrHashParams::default_fast()) {}

  constexpr PtrHashParams(bool remap, double alpha, double lambda, BF bucket_fn,
                          size_t keys_per_shard, Sharding sharding,
                          bool single_part)
      : remap(remap), alpha(alpha), lambda(lambda), bucket_fn(bucket_fn),
        keys_per_shard(keys_per_shard), sharding(sharding),
        single_part(single_part) {}

  static constexpr PtrHashParams default_fast() {
    return PtrHashParams(true, 0.99, 3.0, Linear(), (size_t)1 << 31,
                         Sharding{ShardingType::None}, false);
  };

  static constexpr PtrHashParams default_square() {
    return PtrHashParams(true, 0.99, 3.5, SquareEps(), (size_t)1 << 31,
                         Sharding{ShardingType::None}, false);
  }

  static constexpr PtrHashParams default_balanced() {
    return PtrHashParams(true, 0.99, 3.5, CubicEps(), (size_t)1 << 31,
                         Sharding{ShardingType::None}, false);
  }

  static constexpr PtrHashParams default_compact() {
    return PtrHashParams(true, 0.99, 3.9, CubicEps(), (size_t)1 << 31,
                         Sharding{ShardingType::None}, false);
  }
};

template <typename HasherImpl> class Hasher {
public:
  constexpr uint64_t finish() const {
    return static_cast<const HasherImpl *>(this)->finish_impl();
  }
  constexpr void write(Slice<uint8_t> bytes) const {
    static_cast<const HasherImpl *>(this)->write_imp(bytes);
  }
};

namespace FxHasherDecl {

constexpr uint32_t ROTATE = 5;
constexpr uint64_t SEED64 = 0x517cc1b727220a95;
constexpr uint32_t SEED32 = (uint32_t)(SEED64 & 0xFFFF'FFFF);
constexpr auto SEED = (sizeof(size_t) == 4) ? static_cast<size_t>(SEED32)
                                            : static_cast<size_t>(SEED64);

template <typename T, T SEED_VALUE> class HashWord {
public:
  static_assert(std::is_unsigned_v<T>,
                "Hash word only accepts unsigned numbers");
  mutable T value;
  inline constexpr void hash_word(T word) const {
    this->value = rotate_left(this->value, ROTATE) ^ word;
    this->value *= SEED_VALUE; // wrapping multiplication
  }

  // Portable rotate-left
  inline static constexpr T rotate_left(T x, unsigned int n) {
    constexpr unsigned int bits = sizeof(T) * 8;
    return (x << n) | (x >> (bits - n));
  }

  constexpr operator T() const { return this->value; }

  constexpr HashWord(T initial) : value(initial) {}
  constexpr HashWord &operator=(T new_value) {
    this->value = new_value;
    return *this;
  }
};

// Type aliases for convenience
using HashWordU64 = HashWord<uint64_t, SEED64>;
using HashWordU32 = HashWord<uint32_t, SEED32>;
using HashWordUSize = HashWord<size_t, SEED>;

// Bit-casts a pointer of one type to another type. This is different from the
// cpp::bit_cast in that it copies the bytes manually to local variable.
template <typename To, typename From>
LIBC_INLINE static constexpr To ptr_bit_cast(From *from) {
  To to{};
  char *dst = reinterpret_cast<char *>(&to);
  const char *src = reinterpret_cast<const char *>(from);
  for (unsigned i = 0; i < sizeof(To); ++i)
    dst[i] = src[i];
  return to;
}

inline constexpr uint32_t write32(uint32_t _hash, Slice<uint8_t> bytes) {
  auto hash = HashWordU32{_hash};
  while (bytes.size() >= 4) {
    auto n = ptr_bit_cast<uint32_t>(bytes.data());
    hash.hash_word(n);
    bytes = bytes.subspan(4);
  }

  for (size_t i = 0; i < bytes.size(); i++) {
    hash.hash_word(bytes.data()[i]);
  }

  return hash;
}

inline constexpr uint64_t write64(uint64_t _hash, Slice<uint8_t> bytes) {
  auto hash = HashWordU64{_hash};
  while (bytes.size() >= 8) {
    auto n = ptr_bit_cast<uint64_t>(bytes.data());
    hash.hash_word(n);
    bytes = bytes.subspan(8);
  }

  if (bytes.size() >= 4) {
    auto n = ptr_bit_cast<uint32_t>(bytes.data());
    hash.hash_word((uint64_t)n);
    bytes = bytes.subspan(4);
  }

  for (size_t i = 0; i < bytes.size(); i++) {
    hash.hash_word(static_cast<uint64_t>(bytes.data()[i]));
  }
  return hash;
}

inline constexpr size_t write(size_t hash, Slice<uint8_t> bytes) {
  if constexpr (sizeof(size_t) == 4) {
    return static_cast<size_t>(write32(static_cast<uint32_t>(hash), bytes));
  } else {
    return static_cast<size_t>(write64(static_cast<uint64_t>(hash), bytes));
  }
}

class FxHasher : public Hasher<FxHasher> {
public:
  mutable HashWordUSize hash;

  constexpr FxHasher() : hash(0) {}

  constexpr void write_impl(Slice<uint8_t> bytes) const {
    this->hash = FxHasherDecl::write(this->hash, bytes);
  }

  inline constexpr void write(uint8_t i) const { this->hash.hash_word(i); }
  inline constexpr void write(uint16_t i) const { this->hash.hash_word(i); }
  inline constexpr void write(uint32_t i) const { this->hash.hash_word(i); }
  inline constexpr void write(uint64_t i) const {
    if constexpr (sizeof(size_t) == 4) {
      this->hash.hash_word((size_t)i);
      this->hash.hash_word((size_t)(i >> 32));
    } else {
      this->hash.hash_word((size_t)i);
    }
  }

  inline constexpr uint64_t finish_impl() const { return this->hash; }
};

class FxHasher64 : public Hasher<FxHasher64> {
public:
  mutable HashWordU64 hash;

  constexpr FxHasher64() : hash(0) {}

  inline constexpr void write_impl(Slice<uint8_t> bytes) const {
    this->hash = write64(this->hash, bytes);
  }

  inline constexpr void write(uint8_t i) const {
    this->hash.hash_word((uint64_t)i);
  }
  inline constexpr void write(uint16_t i) const {
    this->hash.hash_word((uint64_t)i);
  }
  inline constexpr void write(uint32_t i) const {
    this->hash.hash_word((uint64_t)i);
  }
  constexpr void write(uint64_t i) const { this->hash.hash_word(i); }
  inline constexpr uint64_t finish_impl() const { return this->hash; }
};

class FxHasher32 : public Hasher<FxHasher32> {
public:
  mutable HashWordU32 hash;

  constexpr FxHasher32() : hash(0) {}

  inline constexpr void write_impl(Slice<uint8_t> bytes) const {
    this->hash = write32(this->hash, bytes);
  }

  inline constexpr void write(uint8_t i) const {
    this->hash.hash_word((uint32_t)i);
  }
  inline constexpr void write(uint16_t i) const {
    this->hash.hash_word((uint32_t)i);
  }
  inline constexpr void write(uint32_t i) const { this->hash.hash_word(i); }
  inline constexpr void write(uint64_t i) const {
    this->hash.hash_word((uint32_t)i);
    this->hash.hash_word((uint32_t)(i >> 32));
  }

  inline constexpr uint64_t finish_impl() const { return this->hash; }
};

template <typename T> inline constexpr uint64_t hash64(T v) {
  constexpr auto state = FxHasher64();
  state.write(v);
  return state.finish();
}

template <typename T> inline constexpr uint32_t hash32(T v) {
  constexpr auto state = FxHasher32();
  state.write(v);
  return state.finish();
}

template <typename T> inline constexpr size_t hash(T v) {
  constexpr auto state = FxHasher();
  state.write(v);
  return state.finish();
}

} // namespace FxHasherDecl

namespace fastrand {

// This seed value is very important for different inputs. Bad values are known
// to cause compilation errors and/or incorrect computations in some cases.
// Defaulted to 0xEF6F79ED30BA75A in the original implementation, but this is
// not sufficient. 0x64a727ea04c46a32 is another viable seed.
constexpr uint64_t DEFAULT_RNG_SEED = 0xeec13c9f1362aa74;

template <typename T> constexpr T wrapping_add(T a, T b) {
  while (b != 0) {
    T carry = a & b;
    a = a ^ b;
    b = carry << 1;
  }
  return a;
}

class Rng {
  mutable uint64_t seed_;

public:
  constexpr Rng() : Rng(DEFAULT_RNG_SEED) {}

  constexpr Rng(uint64_t seed) : seed_(seed) {}
  constexpr Rng(const Rng &) = default;
  constexpr Rng(Rng &&) = default;

  constexpr Rng &operator=(const Rng &) = default;
  constexpr Rng &operator=(Rng &&) = default;

  inline constexpr uint64_t gen() const {
    constexpr uint64_t WY_CONST_0 = 0x2d35'8dcc'aa6c'78a5;
    constexpr uint64_t WY_CONST_1 = 0x8bb8'4b93'962e'acc9;

    auto s = wrapping_add(seed_, WY_CONST_0);
    seed_ = s;
    auto const t =
        static_cast<__uint128_t>(s) * static_cast<__uint128_t>(s ^ WY_CONST_1);
    return static_cast<uint64_t>(t) ^ static_cast<uint64_t>(t >> 64);
  }

  constexpr uint8_t gen_byte() const {
    return static_cast<uint8_t>(this->gen());
  }

  constexpr Rng fork() const { return Rng(this->gen()); }
  constexpr void set(uint64_t i) const { this->seed_ = i; }

  constexpr Rng replace(Rng &&n) const {
    auto ret = seed_;
    seed_ = n.seed_;
    return ret;
  }
};

} // namespace fastrand

template <typename Iterable> struct Enumerate {
  Iterable iterable;

  LIBC_INLINE constexpr Enumerate(Iterable &&iter)
      : iterable(std::forward<Iterable>(iter)) {}

  struct Iterator {
    size_t index;
    decltype(iterable.begin()) it;

    LIBC_INLINE constexpr auto operator*() const {
      return std::tuple<size_t, decltype(*it)>(index, *it);
    }

    LIBC_INLINE constexpr Iterator &operator++() {
      ++index;
      ++it;
      return *this;
    }

    LIBC_INLINE constexpr bool operator!=(const Iterator &other) const {
      return it != other.it;
    }
  };

  LIBC_INLINE constexpr Iterator begin() const { return {0, iterable.begin()}; }

  LIBC_INLINE constexpr Iterator end() const { return {0, iterable.end()}; }
};

template <typename Iterable>
LIBC_INLINE static constexpr Enumerate<Iterable>
enumerate(Iterable &&iterable) {
  return Enumerate<Iterable>(std::forward<Iterable>(iterable));
}

class BucketIdx {
  mutable uint32_t i_;

public:
  constexpr BucketIdx() : i_(0) {}

  constexpr operator uint32_t() const { return i_; }

  constexpr BucketIdx(uint32_t i) : i_(i) {}

  constexpr bool operator==(const BucketIdx other) const {
    return this->i_ == other.i_;
  }

  constexpr BucketIdx operator+(size_t rhs) const {
    return BucketIdx(this->i_ + static_cast<uint32_t>(rhs));
  }

  constexpr BucketIdx operator-(size_t rhs) const {
    return BucketIdx(this->i_ - static_cast<uint32_t>(rhs));
  }

  constexpr bool operator<(const BucketIdx &other) const {
    return this->i_ < other.i_;
  }
  constexpr bool operator>(const BucketIdx &other) const {
    return this->i_ > other.i_;
  }

  static constexpr auto NONE = ~(uint32_t)0;

  constexpr bool is_some() const { return this->i_ != ~(uint32_t)0; }
  constexpr bool is_none() const { return this->i_ == ~(uint32_t)0; }
};

template <typename T = std::pair<size_t, BucketIdx>, std::size_t MaxSize = 5>
class BinaryHeap {
private:
  mutable std::array<T, MaxSize> data{};
  mutable size_t current_size{};

  constexpr void heapify_up(std::size_t index) const {
    while (index > 0) {
      std::size_t parent = (index - 1) / 2;
      if (data[index] <= data[parent])
        break;
      std::swap(data[index], data[parent]);
      index = parent;
    }
  }

  constexpr void heapify_down(std::size_t index) const {
    while (true) {
      std::size_t left = 2 * index + 1;
      std::size_t right = 2 * index + 2;
      std::size_t largest = index;

      if (left < current_size && data[left] > data[largest])
        largest = left;
      if (right < current_size && data[right] > data[largest])
        largest = right;

      if (largest == index)
        break;

      std::swap(data[index], data[largest]);
      index = largest;
    }
  }

public:
  constexpr BinaryHeap() = default;

  constexpr void push(const T &value) const {
    if (current_size >= MaxSize)
      return; // Optional: handle overflow
    data[current_size] = value;
    heapify_up(current_size);
    ++current_size;
  }

  constexpr void push(T &&value) const {
    if (current_size >= MaxSize)
      return;
    data[current_size].first = std::move(value.first);
    data[current_size].second = std::move(value.second);
    heapify_up(current_size);
    ++current_size;
  }

  // constexpr void push(std::pair<size_t, BucketIdx> &&value) const {
  //   if (current_size >= MaxSize)
  //     return;
  //   data[current_size] = {value.first, value.second};
  //   heapify_up(current_size);
  //   ++current_size;
  // }

  constexpr T pop() const {
    if (current_size == 0)
      return T{}; // Optional: handle underflow
    T top = data[0];
    data[0].first = data[current_size - 1].first;
    data[0].second = data[current_size - 1].second;
    --current_size;
    if (current_size > 0)
      heapify_down(0);
    return top;
  }

  constexpr const T &peek() const { return data[0]; }

  constexpr size_t size() const { return current_size; }

  constexpr bool empty() const { return current_size == 0; }

  constexpr bool full() const { return current_size == MaxSize; }
};

template <typename PackedImpl> class Packed {
public:
  constexpr uint64_t index(size_t index) const {
    return static_cast<const PackedImpl *>(this)->index_impl(index);
  }
  constexpr size_t size_in_bytes() const {
    return static_cast<const PackedImpl *>(this)->size_in_bytes_impl();
  }
  static constexpr std::optional<Packed> try_new(Slice<uint64_t> vals);
};

template <typename T, size_t N>
class StaticContainer : public Packed<StaticContainer<T, N>> {
private:
  std::array<T, N> i_;

public:
  // DynamicContainer(Slice<T> i) : i_(i) {}
  constexpr StaticContainer(Slice<T> &i) {
    for (auto [i, e] : enumerate(i)) {
      i_[i] = e;
    }
  }
  constexpr StaticContainer(Slice<T> &&i) : StaticContainer(i) {}
  constexpr StaticContainer(std::array<T, N> &v) : i_(v) {}
  constexpr StaticContainer() = default;
  constexpr StaticContainer(const StaticContainer<T, N> &) = default;
  constexpr StaticContainer(StaticContainer<T, N> &&) = default;

  constexpr StaticContainer<T, N> &
  operator=(const StaticContainer<T, N> &) = default;
  // DynamicContainer<T>& operator=(DynamicContainer<T>&&) = default;

  constexpr uint64_t index_impl(size_t index) const {
    // assert(index < this->i_.size() && "Index out of bounds accessing Slice");
    return this->i_[index];
  }

  constexpr size_t size_in_bytes_impl() const { return this->i_.size(); }

  static constexpr StaticContainer try_new(Slice<uint64_t> const vals) {
    std::array<T, N> n{};

    for (size_t i = 0; i < N; i++) {
      n[i] = vals[i];
    }

    return StaticContainer(n);
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
  static constexpr typename KeyHasher<Key, uint64_t>::H hash(Key &x,
                                                             uint64_t seed) {
    return FxHasherDecl::hash64(x) ^ seed;
  }
};

template <typename ReduceImpl> class Reduce {
public:
  constexpr size_t reduce(uint64_t h) const {
    return static_cast<const ReduceImpl *>(this)->reduce_impl(h);
  }
  constexpr std::pair<size_t, uint64_t>
  reduce_with_remainder(uint64_t _h) const {
    return static_cast<const ReduceImpl *>(this)->reduce_with_remainder_impl(
        _h);
  }
};

struct FastReduce : public Reduce<FastReduce> {
  uint64_t d;

  constexpr FastReduce(uint64_t d) : d(d) {}
  constexpr FastReduce() : d(0) {}

  constexpr operator uint64_t() const { return d; }

  constexpr size_t reduce_impl(uint64_t h) const {
    return utility::mul_high(this->d, h);
  }

  constexpr std::pair<size_t, uint64_t>
  reduce_with_remainder_impl(uint64_t h) const {
    auto r = (__uint128_t)this->d * (__uint128_t)h;
    return {r >> 64, r};
  }
};

struct FM32 : public Reduce<FM32> {
  uint64_t d;
  uint64_t m;

  constexpr FM32(size_t d)
      : d(d), m(std::numeric_limits<uint64_t>::max() / d + 1) {
    // assert(d <= std::numeric_limits<uint32_t>::max());
  }
  constexpr FM32() : d(0), m(0) {}

  constexpr size_t reduce_impl(uint64_t h) const {
    auto lowbits = m * h;
    return (static_cast<__uint128_t>(lowbits) * static_cast<__uint128_t>(d)) >>
           64;
  }

  constexpr std::pair<size_t, uint64_t>
  reduce_with_remainder_impl(uint64_t) const {
    return {};
  }
};

using Rp = FastReduce;
using Rb = FastReduce;
using RemSlots = FM32;
using Pilot = uint64_t;
using PilotHash = uint64_t;

namespace ptrhash {

template <typename T> constexpr bool is_power_of_two(T x) {
  static_assert(std::is_unsigned_v<T>,
                "is_power_of_two requires unsigned type");
  return x != 0 && (x & (x - 1)) == 0;
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

template <typename Iter, typename F>
constexpr auto try_for_each(Iter &&iter, F &&f) {
  for (auto &&x : iter) {
    auto r = f(x);
    if (!r) {
      return false; // early exit
    }
  }

  return true;
}

template <typename T> constexpr T sum(Slice<T> container) {
  size_t acc = 0;
  for (T const item : container) {
    acc += item;
  }
  return acc;
}

// Counts the number of zero elements in an array
template <typename T, size_t N>
LIBC_INLINE static constexpr auto count_zeros(std::array<T, N> &container) {
  size_t counter = 0;

  for (auto element : container) {
    if (!element) {
      counter++;
    }
  }

  return counter;
}

template <typename T, size_t N>
LIBC_INLINE static constexpr auto array_sort(std::array<T, N> &arr) {
  if constexpr (N <= 1) {
    return arr; // base case
  } else {
    constexpr size_t MID = N / 2;

    std::array<T, MID> left{};
    std::array<T, N - MID> right{};

    for (size_t i = 0; i < MID; ++i)
      left[i] = arr[i];
    for (size_t i = MID; i < N; ++i)
      right[i - MID] = arr[i];

    left = array_sort(left);
    right = array_sort(right);

    std::array<T, N> result{};
    size_t li = 0, ri = 0, ki = 0;

    while (li < MID && ri < N - MID)
      result[ki++] = (left[li] <= right[ri]) ? left[li++] : right[ri++];
    while (li < MID)
      result[ki++] = left[li++];
    while (ri < N - MID)
      result[ki++] = right[ri++];

    return result;
  }
}

struct Range {
  int start_range, end_range, step_range;

  struct Iterator {
    mutable int value;
    mutable int step;

    LIBC_INLINE constexpr int &operator*() const { return value; }

    LIBC_INLINE constexpr const Iterator &operator++() const {
      value += step;
      return *this;
    }

    LIBC_INLINE constexpr bool operator!=(const Iterator &other) const {
      return step > 0 ? value < other.value : value > other.value;
    }
  };

  LIBC_INLINE constexpr Iterator begin() const {
    return {start_range, step_range};
  }
  LIBC_INLINE constexpr Iterator end() const { return {end_range, step_range}; }

  LIBC_INLINE constexpr auto rev() const {
    int count = (end_range - start_range + step_range - 1) / step_range;
    int new_start = start_range + (count - 1) * step_range;
    int new_end = start_range - step_range;
    return Range(new_start, new_end, -step_range);
  }

  LIBC_INLINE constexpr Range(int start, int end, int step = 1)
      : start_range(start), end_range(end), step_range(step) {}

  LIBC_INLINE constexpr Range(int end)
      : start_range(0), end_range(end), step_range(1) {}

  LIBC_INLINE constexpr size_t size() const {
    if (step_range > 0)
      return (end_range - start_range + step_range - 1) / step_range;
    return (start_range - end_range - step_range - 1) / (-step_range);
  }
};

template <typename T, size_t N> class ChunksMut {
public:
  struct Chunk {
    std::array<T, N> &arr;
    size_t chunk_begin;
    size_t chunk_end;

    LIBC_INLINE constexpr T &operator[](size_t i) const {
      return arr[chunk_begin + i];
    }

    LIBC_INLINE constexpr T &at(size_t i) const {
      // if (begin + i >= end)
      //   throw std::out_of_range("Chunk::at");
      return arr[chunk_begin + i];
    }

    LIBC_INLINE constexpr T &front() const { return arr[chunk_begin]; }

    LIBC_INLINE constexpr T &back() const { return arr[chunk_end - 1]; }

    LIBC_INLINE constexpr size_t size() const noexcept {
      return chunk_end - chunk_begin;
    }

    LIBC_INLINE constexpr bool empty() const noexcept {
      return chunk_begin == chunk_end;
    }

    LIBC_INLINE constexpr auto begin_it() const {
      return arr.begin() + static_cast<ptrdiff_t>(chunk_begin);
    }
    LIBC_INLINE constexpr auto end_it() const {
      return arr.begin() + static_cast<ptrdiff_t>(chunk_end);
    }

    LIBC_INLINE constexpr auto begin() const { return begin_it(); }

    LIBC_INLINE constexpr auto end() const { return end_it(); }

    LIBC_INLINE constexpr T *data() const { return arr.data() + chunk_begin; }
  };

  class Iterator {
  public:
    LIBC_INLINE constexpr Iterator(std::array<T, N> &arr, size_t pos,
                                   size_t chunk)
        : arr(arr), index(pos), chunk_size(chunk) {}

    LIBC_INLINE constexpr Chunk operator*() const {
      size_t end = std::min(index + chunk_size, arr.size());
      return Chunk{arr, index, end};
    }

    LIBC_INLINE constexpr const Iterator &operator++() const {
      index += chunk_size;
      return *this;
    }

    LIBC_INLINE constexpr bool operator!=(const Iterator &other) const {
      return index != other.index;
    }

  private:
    std::array<T, N> &arr;
    mutable size_t index;
    mutable size_t chunk_size;
  };

  LIBC_INLINE constexpr ChunksMut(std::array<T, N> &v, size_t chunk)
      : arr(v), chunk_size(chunk) {
    // static_assert(chunk_size == 0);
  }

  /// number of chunks
  LIBC_INLINE constexpr size_t size() const {
    return (arr.size() + chunk_size - 1) / chunk_size;
  }

  LIBC_INLINE constexpr bool empty() const { return arr.empty(); }

  LIBC_INLINE constexpr Chunk operator[](size_t chunk_index) const {
    // static_assert(chunk_index >= size());

    size_t begin = chunk_index * chunk_size;
    size_t end = std::min(begin + chunk_size, arr.size());

    return Chunk{arr, begin, end};
  }

  LIBC_INLINE constexpr Iterator begin() const {
    return Iterator(arr, 0, chunk_size);
  }
  LIBC_INLINE constexpr Iterator end() const {
    return Iterator(arr, arr.size(), chunk_size);
  }

private:
  std::array<T, N> &arr;
  size_t chunk_size;
};

template <typename T, size_t N>
LIBC_INLINE static constexpr ChunksMut<T, N> chunks_mut(std::array<T, N> &arr,
                                                        size_t chunk_size) {
  return ChunksMut<T, N>(arr, chunk_size);
}

template <size_t buckets_total_, size_t buckets_>
LIBC_INLINE static constexpr auto
chunks_exact_mut(typename ChunksMut<uint8_t, buckets_total_>::Chunk &pilots) {
  const auto num_chunks = pilots.size() / buckets_;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t begin = pilots.begin_ + i * buckets_;
    size_t end = std::min(begin + buckets_, pilots.arr_.size());
    auto target_pilots = typename ChunksMut<uint8_t, buckets_total_>::Chunk{
        pilots.arr_, begin, end};
  }
}

template <
    typename BF, const PtrHashParams<BF> &params_, size_t n_, size_t parts_,
    size_t shards_, size_t parts_per_shard_, size_t slots_total_,
    size_t buckets_total_, size_t slots_, size_t buckets_,
    const Rp &rem_shards_, const Rp &rem_parts_, const Rb &rem_buckets_,
    const Rb &rem_buckets_total_, const RemSlots &rem_slots_,
    typename Key = uint64_t,
    typename F = StaticContainer<uint32_t, slots_total_ - n_>, // or Vec<u32>
    typename Hx = KeyHasherDefaultImpl<Key>, // -> FxHasherDecl::FxHasher64,
    typename PilotsTypeV = std::array<uint8_t, buckets_total_>>
class PtrHash {
public:
  // static_assert(std::is_base_of_v<KeyT, Key>, "Key must implement KeyT");
  static_assert(std::is_base_of_v<BucketFn<BF>, BF>,
                "BF must implement BucketFn");
  static_assert(std::is_base_of_v<Packed<F>, F>, "F must implement Packed");
  static_assert(std::is_base_of_v<KeyHasher<Key, uint64_t>, Hx>,
                "Hx must implement KeyHasher<Key>");
  static_assert(
      std::is_same_v<PilotsTypeV, Slice<uint8_t>> ||
          std::is_same_v<PilotsTypeV, std::array<uint8_t, buckets_total_>>,
      "V must be a byte slice or byte vector");

  uint64_t seed_;
  PilotsTypeV pilots_;
  F remap_;

  constexpr PtrHash(uint64_t seed_, PilotsTypeV pilots_, F remap_)
      : seed_(seed_), pilots_(pilots_), remap_(remap_) {}

  constexpr PtrHash(const PtrHash &) = default;
  constexpr PtrHash(PtrHash &&) = default;

  constexpr PtrHash &operator=(const PtrHash &) = default;
  constexpr PtrHash &operator=(PtrHash &&) = default;

  inline constexpr size_t index(Key key) const {
    auto slot = this->index_no_remap(key);

    if (slot < n_) {
      return slot;
    }

    return this->remap_.index(slot - n_);
  }

  inline constexpr size_t index_no_remap(Key key) const {
    auto hx = this->hash_key(key);
    auto b = this->bucket(hx);
    auto pilot = this->pilots_[b];
    return this->slot(hx, pilot);
  }

  constexpr size_t slot(typename Hx::H hx, uint64_t pilot) const {
    return (this->part(hx) * slots_) + this->slot_in_part(hx, pilot);
  }

  constexpr size_t slot_in_part(typename Hx::H hx, Pilot pilot) const {
    return this->slot_in_part_hp(hx, this->hash_pilot(pilot));
  }

  constexpr std::optional<std::tuple<uint64_t, PilotsTypeV, F>>
  compute_pilots(const std::array<Key, n_> &keys) {
    std::array<std::array<bool, slots_>, parts_> taken{};
    for (std::array<bool, slots_> &t : taken) {
      for (size_t i = 0; i < slots_; i++) {
        t[i] = 0;
      }
    }
    PilotsTypeV pilots{};

    size_t tries = 0;
    constexpr size_t max_tries = 10;

    constexpr auto rng = rngs::StdRng::from_seed(31415);
    while (true) {
      bool contd = false;
      tries += 1;
      // std::println("Try num {}", tries);
      if (tries > max_tries) {
        return {};
      }

      this->seed_ = rng.random();

      pilots = PilotsTypeV{0};

      for (auto &t : taken) {
        for (size_t e = 0; e < pilots.size(); e++) {
          t[e] = false;
        }
      }

      auto shard_hashes = this->shards(keys);

      ChunksMut<uint8_t, buckets_total_> shard_pilots =
          chunks_mut(pilots, std::max(buckets_ * parts_per_shard_,
                                      static_cast<size_t>(1)));
      ChunksMut<std::array<bool, slots_>, parts_> shard_taken =
          chunks_mut(taken, parts_per_shard_);

      for (size_t shard = 0;
           shard < std::min({shard_hashes.size(), shard_pilots.size(),
                             shard_taken.size()});
           shard++) {
        std::array<typename Hx::H, n_> hashes = shard_hashes[shard];
        typename ChunksMut<uint8_t, buckets_total_>::Chunk pilots =
            shard_pilots[shard];
        typename ChunksMut<std::array<bool, slots_>, parts_>::Chunk taken =
            shard_taken[shard];
        std::optional<std::pair<std::array<typename Hx::H, n_>,
                                std::array<uint32_t, parts_per_shard_ + 1>>>
            sorted_parts = this->sort_parts(shard, hashes);
        if (!sorted_parts) {
          contd = true;
          break;
        }

        auto &[new_hashes, part_starts] = sorted_parts.value();

        if (!this->build_shard(shard, new_hashes, part_starts, pilots, taken)) {
          contd = true;
          break;
        }
      }
      if (contd) {
        continue;
      }

      auto const remap = this->remap_free_slots(taken);

      if (!remap) {
        continue;
      }
      break;
    }
    this->pilots_ = pilots;

    return {{this->seed_, this->pilots_, this->remap_}};
  }

  constexpr cpp::expected<std::monostate, std::nullopt_t>
  remap_free_slots(std::array<std::array<bool, slots_>, parts_> &taken) {
    auto val = map(taken, [&](auto t) { return count_zeros(t); });
    if (sum(Slice(val.data(), val.size())) != slots_total_ - n_) {

      fprintf(stderr,
              "Not the right number of free slots left!\n total slots %zu - n "
              "%zu\n",
              slots_total_, n_);
      return cpp::unexpected(std::nullopt);
    }

    if (!params_.remap || slots_total_ == n_) {
      return std::monostate{};
    }

    std::array<uint64_t, slots_total_ - n_> v{};
    size_t v_idx = 0;

    auto const get = [&](std::array<std::array<bool, slots_>, parts_> &t,
                         size_t idx) { return t[idx / slots_][idx % slots_]; };

    for (const auto &[p, t] : enumerate(taken)) {
      auto const offset = p * slots_;
      for (size_t idx = 0; idx < t.size(); idx++) {
        if (!t[idx]) {
          auto result = offset + idx;
          if (result < n_) {
            while (!get(taken, n_ + v_idx)) {
              v[v_idx++] = result;
            }
            v[v_idx++] = result;
          }
        }
      }
    }
    this->remap_ = F::try_new(Slice(v.data(), v.size()));
    return std::monostate{};
  }

  constexpr auto shards(const std::array<Key, n_> &keys) const {
    switch (params_.sharding.type) {
    case ShardingType::None:
      return this->no_sharding(keys);
      // case ShardingType::Memory:
      //   return this->shard_keys_in_memory(keys);
      // We don't need this for compile time data
      // case ShardingType::Disk:
      //   return this->shard_keys_hybrid(std::numeric_limits<size_t>::max(),
      //   keys);
      // case ShardingType::Hybrid:
      //   return this->shard_keys_hybrid(params_.sharding.mem, keys);
    }
  }

  constexpr std::array<std::array<typename Hx::H, n_>, 1>
  no_sharding(const std::array<Key, n_> &keys) const {
    return {map(keys, [&](Key key) { return this->hash_key(key); })};
  }

  // constexpr auto shard_keys_in_memory(std::array<Key, n_> &keys) const {
  //   constexpr utility::Range r(shards_);
  //   return map(r, [&](size_t shard) {
  //     std::array<typename Hx::H, n_> res = utility::map<std::array<Key, n_>>(
  //         keys, [&](typename Hx::H key) -> typename Hx::H {
  //           return this->hash_key(key);
  //         });
  //     std::vector<typename Hx::H> hashes = utility::filter(
  //         res, [&](typename Hx::H h) { return this->shard(h) == shard; });
  //     return hashes;
  //   });
  // }

  constexpr typename Hx::H hash_key(Key x) const {
    return Hx::hash(x, this->seed_);
  }

  constexpr size_t shard(typename Hx::H hx) const {
    return rem_shards_.reduce(high(hx));
  }

  constexpr std::optional<std::pair<std::array<typename Hx::H, n_>,
                                    std::array<uint32_t, parts_per_shard_ + 1>>>
  sort_parts(size_t shard, std::array<typename Hx::H, n_> hashes) const {
    hashes = array_sort(hashes);

    bool distinct = true;
    for (size_t i = 1; i < hashes.size(); ++i) {
      if (hashes[i] == hashes[i - 1]) {
        distinct = false;
        break;
      }
    }

    if (!distinct) {
      fprintf(stderr, "Hashes are not distinct\n");
      return std::nullopt;
    }

    if (!hashes.empty()) {
      // assert(shard * parts_per_shard_ <= this->part(hashes[0]));
      // assert(this->part(hashes.last()) < (shard + 1) *
      // parts_per_shard_);
    }

    std::array<uint32_t, parts_per_shard_ + 1> part_starts{};

    for (auto part_in_shard : Range(1, parts_per_shard_ + 1)) {
      part_starts[part_in_shard] =
          Slice(hashes.data(), hashes.size())
              .binary_search_by([&](typename Hx::H h) {
                if (this->part(h) < shard * parts_per_shard_ + part_in_shard) {
                  return Ordering::Less;
                } else {
                  return Ordering::Greater;
                }
              })
              .error();
    }
    size_t max_part_len = 0;
    for (size_t i = 0; i + 1 < part_starts.size(); ++i) {
      auto start = part_starts[i];
      auto end = part_starts[i + 1];

      size_t len = end - start;
      max_part_len = std::max<size_t>(max_part_len, len);
    }

    if (max_part_len > slots_) {
      return std::nullopt;
    }

    return std::pair(hashes, part_starts);
  }

  // TODO: get back here for pilots
  constexpr bool build_shard(
      size_t shard, std::array<typename Hx::H, n_> &hashes,
      std::array<uint32_t, parts_per_shard_ + 1> &part_starts,
      typename ChunksMut<uint8_t, buckets_total_>::Chunk pilots,
      typename ChunksMut<std::array<bool, slots_>, parts_>::Chunk taken) const {
    // auto pilots_per_part =
    //     chunks_exact_mut<buckets_total_, buckets_>(pilots);

    // auto parts_done = shard * parts_per_shard_;

    auto ok = try_for_each(enumerate(taken), [&](auto e) constexpr -> bool {
      const auto num_chunks = pilots.size() / buckets_;
      for (size_t i = 0; i < num_chunks; ++i) {
        size_t begin = pilots.chunk_begin + i * buckets_;
        size_t end = std::min(begin + buckets_, pilots.arr.size());
        auto target_pilots = typename ChunksMut<uint8_t, buckets_total_>::Chunk{
            pilots.arr, begin, end};
        auto &[part_in_shard, taken] = e;
        auto part = shard * parts_per_shard_ + part_in_shard;

        auto _cnt = this->build_part(
            part,
            Slice(hashes.data(), hashes.size())
                .slice_form_range(part_starts[part_in_shard],
                                  part_starts[part_in_shard + 1]),
            Slice(target_pilots.data(), target_pilots.size()), taken);
        if (!_cnt) {
          return false;
        }
      }
      // parts_done++;
      return true;
    });

    if (!ok) {
      return false;
    }

    // assert(parts_done == (shard + 1) * parts_per_shard_);
    return true;
  }

  constexpr std::optional<size_t>
  build_part(size_t part, Slice<typename Hx::H> hashes, Slice<uint8_t> pilots,
             std::array<bool, slots_> &taken) const {
    std::pair<std::array<uint32_t, buckets_ + 1>,
              std::array<BucketIdx, buckets_>>
        sorted_buckets = this->sort_buckets(part, hashes);
    std::array<uint32_t, buckets_ + 1> starts = sorted_buckets.first;
    std::array<BucketIdx, buckets_> bucket_order = sorted_buckets.second;

    auto kmax = 256;

    std::array<BucketIdx, slots_> slots{};
    for (size_t i = 0; i < slots_; i++) {
      slots[i] = BucketIdx::NONE;
    }

    constexpr size_t one = 1;
    auto bucket_len = [&](BucketIdx b) constexpr -> size_t {
      return starts[b + one] - starts[b];
    };

    constexpr BinaryHeap<std::pair<size_t, BucketIdx>> stack{};

    auto duplicate_slots = [&](BucketIdx b, Pilot p) constexpr {
      auto hp = this->hash_pilot(p);
      auto hashes_range = hashes.slice_form_range(starts[b], starts[b + one]);

      for (auto const &[i, e1] : enumerate(hashes_range)) {
        auto hx = this->slot_in_part_hp(e1, hp);
        for (auto e2 : hashes_range.subspan(i + 1)) {
          auto hy = this->slot_in_part_hp(e2, hp);
          if (hx == hy) {
            return true;
          }
        }
      }
      return false;
    };

    std::array<BucketIdx, 16> recent{
        {BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE,
         BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE,
         BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE,
         BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE}};
    size_t total_evictions = 0;

    constexpr auto rng = fastrand::Rng();

    for (auto const &[iter_num, new_b] : enumerate(bucket_order)) {
      auto const new_bucket =
          hashes.slice_form_range(starts[new_b], starts[new_b + one]);
      if (new_bucket.empty()) {
        pilots[new_b] = 0;
        continue;
      }
      auto const new_b_len = new_bucket.size();
      size_t evictions = 0;

      stack.push({new_b_len, new_b});
      // std::print("stack(push): ");
      // for (auto x : stack) {
      //   std::print("({} {}) ", std::get<0>(x), std::get<1>(x).i_);
      // }
      // std::println();
      for (auto const &[i, _] : enumerate(recent)) {
        recent[i] = BucketIdx::NONE;
      }
      auto recent_idx = 0;
      recent[0] = new_b;

      while (!stack.empty()) {
        auto const &[b_len, b] = stack.peek();
        stack.pop();
        // std::print("stack(pop): ");
        // for (auto x : stack.data) {
        //   std::print("({} {}) ", std::get<0>(x), std::get<1>(x).i_);
        // }
        // std::println();
        if (evictions > slots_ && is_power_of_two((evictions))) {
          if (evictions >= 10 * slots_) {
            std::cout << "iter num " << iter_num << std::endl;
            return std::nullopt;
          }
        }
        auto const bucket = hashes.slice_form_range(starts[b], starts[b + one]);
        if (auto fpilot = this->find_pilot(kmax, bucket, taken)) {
          auto &[p, hp] = fpilot.value();
          pilots[b] = static_cast<uint8_t>(p);
          for (auto &item :
               hashes.slice_form_range(starts[b], starts[b + one])) {
            auto p = this->slot_in_part_hp(item, hp);
            slots[p] = b;
          }
          continue;
        }
        uint64_t p0 = rng.gen_byte();
        std::pair best = {std::numeric_limits<size_t>::max(),
                          std::numeric_limits<uint64_t>::max()};
        // std::println("rng.u8 {}", p0);
        for (auto delat : Range(kmax)) {
          bool build_part_loop_continue_inner_continue = false;
          auto const p = (p0 + delat) % kmax;
          auto const hp = this->hash_pilot(p);
          size_t collision_score = 0;
          for (auto &item :
               hashes.slice_form_range(starts[b], starts[b + one])) {
            auto p = this->slot_in_part_hp(item, hp);
            auto const s = slots[p];
            size_t new_score = 0;
            if (s.is_none()) {
              continue;
            } else if (Slice(recent.data(), recent.size()).contains(s)) {
              build_part_loop_continue_inner_continue = true;
              break;
            } else {
              auto const len = bucket_len(s);
              new_score = len * len;
            }
            collision_score += new_score;
            if (collision_score >= std::get<0>(best)) {
              build_part_loop_continue_inner_continue = true;
              break;
            }
          }
          if (build_part_loop_continue_inner_continue) {
            continue;
          }
          if (!duplicate_slots(b, p)) {
            best.first = collision_score;
            best.second = p;
            if (collision_score == new_b_len * new_b_len) {
              break;
            }
          }
        }
        if (best == std::pair{std::numeric_limits<size_t>::max(),
                              std::numeric_limits<uint64_t>::max()}) {
          return std::nullopt;
        }

        auto const &[_collision_score, p] = best;
        pilots[b] = static_cast<uint8_t>(p);
        auto const hp = this->hash_pilot(p);
        for (auto &item : hashes.slice_form_range(starts[b], starts[b + one])) {
          auto slot = this->slot_in_part_hp(item, hp);
          auto const b2 = slots[slot];
          if (b2.is_some()) {
            // assert(b2 != b);
            stack.push({bucket_len(b2), b2});
            // std::print("stack(push): ");
            // for (auto x : stack.data) {
            //   std::print("({} {}) ", std::get<0>(x), std::get<1>(x).i_);
            // }
            // std::println();
            evictions++;

            auto hp = this->hash_pilot(static_cast<Pilot>(pilots[b2]));
            for (auto &item :
                 hashes.slice_form_range(starts[b2], starts[b2 + one])) {
              auto p2 = this->slot_in_part_hp(item, hp);
              slots[p2] = BucketIdx::NONE;
              taken[p2] = false;
            }
          }
          slots[slot] = b;
          taken[slot] = true;
        }

        recent_idx++;
        recent_idx %= recent.size();
        recent[recent_idx] = b;
      }

      total_evictions += evictions;
    }
    return total_evictions;
  }

  constexpr std::optional<std::pair<Pilot, PilotHash>>
  find_pilot(uint64_t kmax, Slice<typename Hx::H> bucket,
             std::array<bool, slots_> &taken) const {
    switch (bucket.size()) {
    case 1:
      return this->find_pilot_array<1>(kmax, bucket, taken);
    case 2:
      return this->find_pilot_array<2>(kmax, bucket, taken);
    case 3:
      return this->find_pilot_array<3>(kmax, bucket, taken);
    case 4:
      return this->find_pilot_array<4>(kmax, bucket, taken);
    case 5:
      return this->find_pilot_array<5>(kmax, bucket, taken);
    case 6:
      return this->find_pilot_array<6>(kmax, bucket, taken);
    case 7:
      return this->find_pilot_array<7>(kmax, bucket, taken);
    case 8:
      return this->find_pilot_array<8>(kmax, bucket, taken);
    default:
      return this->find_pilot_slice(kmax, bucket, taken);
    };
  }

  template <const size_t L>
  constexpr std::optional<std::pair<Pilot, PilotHash>>
  find_pilot_array(uint64_t kmax, Slice<typename Hx::H> bucket,
                   std::array<bool, slots_> &taken) const {
    auto cpy = Slice(bucket.data(), L);
    return this->find_pilot_slice(kmax, cpy, taken);
  }

  inline constexpr std::optional<std::pair<Pilot, PilotHash>>
  find_pilot_slice(uint64_t kmax, Slice<typename Hx::H> bucket,
                   std::array<bool, slots_> &taken) const {
    auto const r = bucket.size() / 4 * 4;
    for (auto p : Range(kmax)) {
      bool find_pilot_continue = false;
      auto const hp = this->hash_pilot(p);
      auto const check = [&](typename Hx::H hx) {
        return taken[this->slot_in_part_hp(hx, hp)];
      };
      auto bad = false;
      for (auto i : Range(0, r, 4)) {
        std::array<bool, 4> checks{{
            check(bucket[i]),
            check(bucket[i + 1]),
            check(bucket[i + 2]),
            check(bucket[i + 3]),
        }};
        for (auto bad : checks) {
          if (bad) {
            find_pilot_continue = true;
            break;
          }
        }
        if (find_pilot_continue) {
          break;
          ;
        }
      }
      if (find_pilot_continue) {
        continue;
      }
      for (auto hx : bucket.subspan(r)) {
        bad |= check(hx);
      }
      if (bad) {
        continue;
      }

      if (this->try_take_pilot(bucket, hp, taken)) {
        return std::pair(p, hp);
      }
    }
    return std::nullopt;
  }

  constexpr bool try_take_pilot(Slice<typename Hx::H> bucket, PilotHash hp,
                                std::array<bool, slots_> &taken) const {
    for (auto [i, hx] : enumerate(bucket)) {
      auto const slot = this->slot_in_part_hp(hx, hp);
      if (taken[slot]) {
        for (auto hx : bucket.slice_form_range(0, i)) {
          taken[this->slot_in_part_hp(hx, hp)] = false;
        }
        return false;
      }
      taken[slot] = true;
    }
    return true;
  }

  constexpr PilotHash hash_pilot(Pilot p) const {
    constexpr uint64_t C = 0x517cc1b727220a95;
    return utility::wrapping_mul(C, p ^ this->seed_);
  }

  constexpr size_t slot_in_part_hp(typename Hx::H hx, PilotHash hp) const {
    return rem_slots_.reduce(low(hx) ^ hp);
  }

  constexpr std::pair<std::array<uint32_t, buckets_ + 1>,
                      std::array<BucketIdx, buckets_>>
  sort_buckets(size_t part, Slice<typename Hx::H> hashes) const {
    std::array<uint32_t, buckets_ + 1> bucket_starts{};
    size_t bucket_starts_idx = 0;
    std::array<BucketIdx, buckets_> order{};
    for (auto const &[i, _] : enumerate(order)) {
      order[i] = BucketIdx::NONE;
    }
    std::array<size_t, 32> bucket_len_cnt = {0};

    size_t end = 0;
    bucket_starts[bucket_starts_idx++] = end;

    for (auto b : Range(buckets_)) {
      auto start = end;
      while (end < hashes.size() &&
             this->bucket(hashes[end]) == part * buckets_ + b) {
        end++;
      }

      auto l = end - start;
      // 32 seems to be fine as a size for now, update bucket_len_cnt in the
      // future if necessary
      // if (l >= bucket_len_cnt.size()) {
      //   utility::resize_with(bucket_len_cnt, l + 1, []() { return 0; });
      // }
      bucket_len_cnt[l]++;
      bucket_starts[bucket_starts_idx++] = end;
    }

    // assert(end == hashes.size());

    auto max_bucket_size = bucket_len_cnt.size() - 1;
    // if (false) {
    //   auto const expected_bucket_size = slots_ / buckets_;
    //   static_assert(max_bucket_size <= (20. * expected_bucket_size),
    //                 "Part {part}: Bucket size {max_bucket_size} is too much "
    //                 "larger than the expected size of
    //                 {expected_bucket_size}.");
    // }
    auto acc = 0;
    for (auto i : Range(max_bucket_size + 1).rev()) {
      auto tmp = bucket_len_cnt[i];
      bucket_len_cnt[i] = acc;
      acc += tmp;
    }
    constexpr size_t one = 1;
    for (auto &b : Range(buckets_)) {
      size_t l = bucket_starts[b + one] - bucket_starts[b];
      order[bucket_len_cnt[l]] = b;
      bucket_len_cnt[l] += 1;
    }

    return {bucket_starts, order};
  }

  constexpr size_t part(typename Hx::H hx) const {
    return rem_parts_.reduce(high(hx));
  }

  constexpr size_t bucket_in_part(uint64_t x) const {
    if (BF::LINEAR) {
      return rem_buckets_.reduce(x);
    } else if (BF::B_OUTPUT) {
      return params_.bucket_fn.call(x);
    } else {
      return rem_buckets_.reduce(params_.bucket_fn.call(x));
    }
  }

  constexpr size_t bucket(typename Hx::H hx) const {
    if (BF::LINEAR) {
      return rem_buckets_total_.reduce(high(hx));
    }
    auto [part, nhx] = rem_parts_.reduce_with_remainder(high(hx));
    auto bucket = this->bucket_in_part(nhx);
    return part * buckets_ + bucket;
  }
};

template <typename BF = Linear>
constexpr PtrHashParams<BF> params = PtrHashParams<BF>();

template <typename T> static constexpr T div_ceil(T a, T b) {
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

constexpr double constexpr_ln(double x) {
  if (x <= 0.0)
    return 0.0; // or static_assert in C++23

  // Range reduction: bring x to [0.5, 2)
  int k = 0;
  while (x > 2.0) {
    x *= 0.5;
    ++k;
  }
  while (x < 0.5) {
    x *= 2.0;
    --k;
  }

  // ln(x) ≈ 2 * sum_{n=0}^∞ (1/(2n+1)) * ((x-1)/(x+1))^(2n+1)
  const double y = (x - 1.0) / (x + 1.0);
  const double y2 = y * y;

  double term = y;
  double sum = 0.0;

  // 20 iterations ≈ double precision accuracy
  for (int n = 1; n <= 39; n += 2) {
    sum += term / n;
    term *= y2;
  }

  // ln(x) = 2 * sum + k * ln(2)
  return 2.0 * sum + k * 0.693147180559945309417232121458176568;
}

template <typename T> constexpr T constexpr_ceil(T x) {
  static_assert(std::is_floating_point_v<T>);
  long long i = static_cast<long long>(x);
  return (static_cast<T>(i) == x)
             ? x
             : (x > T{0} ? static_cast<T>(i + 1) : static_cast<T>(i));
}

template <size_t n, typename BF>
constexpr size_t shards = (params<BF>.single_part == true) ? 1
                          : (params<BF>.sharding.type == ShardingType::None)
                              ? 1
                              : div_ceil(n, params<BF>.keys_per_shard);

template <size_t n, typename BF = Linear>
static inline constexpr size_t get_parts() {
  size_t parts = 0;
  if (params<BF>.single_part) {
    parts = 1;
  } else {
    auto eps = (1.0 - params<BF>.alpha) / 2.0;
    auto x = n * eps * eps / 2.0;
    size_t target_parts = x / constexpr_ln(x);
    auto parts_per_shard = target_parts / shards<n, BF>;
    parts = ((parts_per_shard > 1) ? parts_per_shard : 1) * shards<n, BF>;
  }
  return parts;
}

template <size_t keys_per_part, typename BF>
static constexpr auto get_slots_per_part() {
  size_t slots_per_part = keys_per_part / params<BF>.alpha;
  if (is_power_of_two(slots_per_part)) {
    slots_per_part += 1;
  }
  return slots_per_part;
}

namespace ptrhash_config {

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t parts = get_parts<n, BF>();

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t keys_per_part = n / parts<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t parts_per_shard = parts<n, Key, BF, Hx> / shards<n, BF>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t slots_per_part =
    get_slots_per_part<keys_per_part<n, Key, BF, Hx>, BF>();

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t slots_total =
    parts<n, Key, BF, Hx> * slots_per_part<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t buckets_per_part =
    constexpr_ceil(keys_per_part<n, Key, BF, Hx> / params<BF>.lambda) + 3;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t buckets_total =
    parts<n, Key, BF, Hx> * buckets_per_part<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr Rp rem_shards = shards<n, BF>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr Rp rem_parts = parts<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr Rb rem_buckets_per_part = buckets_per_part<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr Rb rem_buckets_total = buckets_total<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr RemSlots rem_slots_per_part =
    std::max(slots_per_part<n, Key, BF, Hx>, static_cast<size_t>(1));

} // namespace ptrhash_config

template <size_t n, typename Key = uint64_t, const std::array<Key, n> &keys,
          typename BF = Linear,
          typename Hx = KeyHasherDefaultImpl<Key>> // FxHasherDecl::FxHasher64,
static constexpr inline auto init_hasher() {
  using namespace ptrhash_config;

  params<BF>.bucket_fn.set_buckets_per_part(buckets_per_part<n, Key, BF, Hx>);

  using F = StaticContainer<uint32_t, slots_total<n, Key, BF, Hx> - n>;
  using PilotsTypeV = std::array<uint8_t, buckets_total<n, Key, BF, Hx>>;

  auto p =
      PtrHash<BF, params<BF>, n, parts<n, Key, BF, Hx>, shards<n, BF>,
              parts_per_shard<n, Key, BF, Hx>, slots_total<n, Key, BF, Hx>,
              buckets_total<n, Key, BF, Hx>, slots_per_part<n, Key, BF, Hx>,
              buckets_per_part<n, Key, BF, Hx>, rem_shards<n, Key, BF, Hx>,
              rem_parts<n, Key, BF, Hx>, rem_buckets_per_part<n, Key, BF, Hx>,
              rem_buckets_total<n, Key, BF, Hx>,
              rem_slots_per_part<n, Key, BF, Hx>, Key, F, Hx, PilotsTypeV>(
          0, PilotsTypeV(), F());
  auto result = p.compute_pilots(keys);

  if (!result) {
    fprintf(stderr, "Unable to construct PtrHash after 10 tries. Try using a "
                    "better hash or decreasing lambda.\n");
    std::abort();
  }

  auto &[seed, pilots, remap] = result.value();

  return PtrHash<BF, params<BF>, n, parts<n, Key, BF, Hx>, shards<n, BF>,
                 parts_per_shard<n, Key, BF, Hx>, slots_total<n, Key, BF, Hx>,
                 buckets_total<n, Key, BF, Hx>, slots_per_part<n, Key, BF, Hx>,
                 buckets_per_part<n, Key, BF, Hx>, rem_shards<n, Key, BF, Hx>,
                 rem_parts<n, Key, BF, Hx>,
                 rem_buckets_per_part<n, Key, BF, Hx>,
                 rem_buckets_total<n, Key, BF, Hx>,
                 rem_slots_per_part<n, Key, BF, Hx>, Key, F, Hx, PilotsTypeV>(
      seed, pilots, remap);
}

} // namespace ptrhash

#endif // PTR_HASH_HPP_
