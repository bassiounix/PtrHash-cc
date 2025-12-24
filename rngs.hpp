#ifndef RNGS_HPP_
#define RNGS_HPP_

#include "utils.hpp"
#include <array>
#include <cstdint>
#include <cstring>

namespace rngs {

static constexpr size_t BLOCK = 16;
static constexpr uint64_t BLOCK64 = BLOCK;
static constexpr uint64_t LOG2_BUFBLOCKS = 2;
static constexpr uint64_t BUFBLOCKS = 1 << LOG2_BUFBLOCKS;
static constexpr uint64_t BUFSZ64 = BLOCK64 * BUFBLOCKS;
static constexpr size_t BUFSZ = BUFSZ64;

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
    std::array<uint64_t, 2> x = {
        static_cast<uint64_t>(xs[0]) | (static_cast<uint64_t>(xs[1]) << 32),
        static_cast<uint64_t>(xs[2]) |
            static_cast<uint64_t>(static_cast<uint64_t>(xs[3]) << 32)};

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

class StdRng {
public:
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
    uint32_t x = utility::rotate_right(xorshifted, rot);
    return utility::to_le_bytes(x);
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
    auto key0 =
        vec128_storage::read_le(Slice(key.data(), key.size()).range(0, 16));
    auto key1 = vec128_storage::read_le(Slice(key.data(), key.size()).sub(16));

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
    sout.range(0, 16).copy_from_slice(ra.to_scalars());
    sout.range(16, 32).copy_from_slice(rb.to_scalars());
    sout.range(32, 48).copy_from_slice(rc.to_scalars());
    sout.range(48, 64).copy_from_slice(rd.to_scalars());
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
      auto data = results.range(index, index + 2);
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

#endif // RNGS_HPP_
