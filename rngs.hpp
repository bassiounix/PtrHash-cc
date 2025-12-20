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
  mutable std::array<uint8_t, 16> u8x16;
  mutable std::array<uint32_t, 4> u32x4;
  mutable std::array<uint64_t, 2> u64x2;
  mutable std::array<__uint128_t, 1> u128x1;

  constexpr vec128_storage(std::array<uint32_t, 4> &&x) : u32x4(x) {}
  constexpr vec128_storage(std::array<uint32_t, 4> &x) : u32x4(x) {}
  constexpr vec128_storage(std::array<uint64_t, 2> &x) : u64x2(x) {}
  constexpr vec128_storage(std::array<uint64_t, 2> &&x) : u64x2(x) {}
  constexpr vec128_storage() = default;

  inline constexpr std::array<uint32_t, 4> to_lanes() const {
    uint64_t x = this->u64x2[0];
    uint64_t y = this->u64x2[1];
    return {static_cast<uint32_t>(x), static_cast<uint32_t>(x >> 32),
            static_cast<uint32_t>(y), static_cast<uint32_t>(y >> 32)};
  }

  inline static constexpr vec128_storage
  from_lanes(std::array<uint32_t, 4> &&xs) {
    std::array<uint64_t, 2> x = {
        static_cast<uint64_t>(xs[0]) | (static_cast<uint64_t>(xs[1]) << 32),
        static_cast<uint64_t>(xs[2]) |
            static_cast<uint64_t>(static_cast<uint64_t>(xs[3]) << 32)};

    return vec128_storage(x);
  }

  inline static constexpr auto from_lanes(std::array<uint64_t, 2> &&xs) {
    return vec128_storage(xs);
  }

  inline static constexpr auto read_le(Slice<uint8_t> x) {
    // static_assert(x.size() == 16);
    vec128_storage v;
    uint8_t *dst = v.u8x16.data();
    uint8_t *src = x.data();
    for (uint8_t i = 0; i < 16; ++i)
      dst[i] = src[i];
    return v;
  }
};

union vec256_storage {
  mutable std::array<uint32_t, 8> u32x8;
  mutable std::array<uint64_t, 4> u64x4;
  mutable std::array<__uint128_t, 2> u128x2;
  mutable std::array<vec128_storage, 2> sse2;

  constexpr vec256_storage() = default;
  constexpr vec256_storage(std::array<vec128_storage, 2> &&x) : sse2(x) {}

  inline static constexpr vec256_storage mm256_shuffle_epi32(vec256_storage a,
                                                             int imm) {
    vec256_storage r;

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

  inline static vec128_storage mm256_extracti128_si256(const vec256_storage &V,
                                                       int M) {
    return V.sse2[M & 1];
  }

  constexpr vec256_storage to_lanes() const {
    return std::array{mm256_extracti128_si256(*this, 0),
                      mm256_extracti128_si256(*this, 1)};
  }
};

union vec512_storage {
  mutable std::array<uint32_t, 16> u32x16;
  mutable std::array<uint64_t, 8> u64x8;
  mutable std::array<__uint128_t, 4> u128x4;
  mutable std::array<vec128_storage, 4> sse2;
  mutable std::array<vec256_storage, 2> avx;

  constexpr vec512_storage() = default;
  constexpr vec512_storage(std::array<uint32_t, 16> &u32x16) : u32x16(u32x16) {}
  constexpr vec512_storage(std::array<uint32_t, 16> &&u32x16)
      : u32x16(u32x16) {}
  constexpr vec512_storage(std::array<uint64_t, 8> &u64x8) : u64x8(u64x8) {}
  constexpr vec512_storage(std::array<uint64_t, 8> &&u64x8) : u64x8(u64x8) {}
  constexpr vec512_storage(std::array<vec128_storage, 4> &sse2) : sse2(sse2) {}
  constexpr vec512_storage(std::array<vec128_storage, 4> &&sse2) : sse2(sse2) {}
  constexpr vec512_storage(std::array<vec256_storage, 2> &avx) : avx(avx) {}
  constexpr vec512_storage(std::array<vec256_storage, 2> &&avx) : avx(avx) {}

  inline static constexpr vec512_storage
  new128(std::array<vec128_storage, 4> &&xs) {
    return vec512_storage{xs};
  }

  inline constexpr vec512_storage unpack() const {
    return vec512_storage{
        std::array<vec256_storage, 2>{this->avx[0], this->avx[1]}};
  }

  static inline vec256_storage mm256_add_epi32(const vec256_storage &a,
                                               const vec256_storage &b) {
    vec256_storage r;
    for (int i = 0; i < 8; ++i) {
      r.u32x8[i] = a.u32x8[i] + b.u32x8[i]; // modulo 2^32
    }
    return r;
  }

  constexpr auto &operator+=(vec512_storage &rhs) const {
    this->avx[0] = mm256_add_epi32(this->avx[0], rhs.avx[0]);
    this->avx[1] = mm256_add_epi32(this->avx[1], rhs.avx[1]);
    return *this;
  }

  constexpr vec512_storage operator+(vec512_storage &rhs) const {
    return {{mm256_add_epi32(this->avx[0], rhs.avx[0]),
             mm256_add_epi32(this->avx[1], rhs.avx[1])}};
  }

  constexpr vec512_storage operator+(const vec512_storage &rhs) const {
    return {{mm256_add_epi32(this->avx[0], rhs.avx[0]),
             mm256_add_epi32(this->avx[1], rhs.avx[1])}};
  }
  // simd_xor
  constexpr vec512_storage operator+(vec512_storage &rhs) {
    return {{mm256_add_epi32(this->avx[0], rhs.avx[0]),
             mm256_add_epi32(this->avx[1], rhs.avx[1])}};
  }

  static inline vec256_storage mm256_xor_si256(const vec256_storage &a,
                                               const vec256_storage &b) {
    vec256_storage r;
    for (int i = 0; i < 8; ++i) {
      r.u32x8[i] = a.u32x8[i] ^ b.u32x8[i];
    }
    return r;
  }

  constexpr auto operator^(vec512_storage &rhs) const {
    return vec512_storage{{mm256_xor_si256(this->avx[0], rhs.avx[0]),
                           mm256_xor_si256(this->avx[1], rhs.avx[1])}};
  }

  static inline vec256_storage mm256_shuffle_epi8(const vec256_storage &a,
                                                  const vec256_storage &b) {
    vec256_storage r;

    // Helper for 128-bit lane (16 bytes)
    auto shuffle_128 = [](const vec128_storage &src,
                          const vec128_storage &ctrl) {
      vec128_storage res;
      for (int i = 0; i < 16; ++i) {
        uint8_t c = ctrl.u32x4[i / 4] >> ((i % 4) * 8) & 0xFF;
        if (c & 0x80) {
          // top bit set → zero
          res.u32x4[i / 4] &= ~(0xFFu << ((i % 4) * 8)); // clear byte
        } else {
          int k = (c & 0x0F);
          uint8_t byte = (src.u32x4[k / 4] >> ((k % 4) * 8)) & 0xFF;
          // place byte in result
          res.u32x4[i / 4] &= ~(0xFFu << ((i % 4) * 8)); // clear target byte
          res.u32x4[i / 4] |= static_cast<uint32_t>(byte) << ((i % 4) * 8);
        }
      }
      return res;
    };

    // Shuffle lower 128-bit lane
    r.sse2[0] = shuffle_128(a.sse2[0], b.sse2[0]);

    // Shuffle upper 128-bit lane
    r.sse2[1] = shuffle_128(a.sse2[1], b.sse2[1]);

    return r;
  }

  inline static vec256_storage mm256_set_epi64x(long long a, long long b,
                                                long long c, long long d) {
    vec256_storage v;

    // Lower 128-bit lane
    v.sse2[0].u64x2[0] = static_cast<uint64_t>(d); // bits [63:0]
    v.sse2[0].u64x2[1] = static_cast<uint64_t>(c); // bits [127:64]

    // Upper 128-bit lane
    v.sse2[1].u64x2[0] = static_cast<uint64_t>(b); // bits [191:128]
    v.sse2[1].u64x2[1] = static_cast<uint64_t>(a); // bits [255:192]

    return v;
  }

  constexpr vec512_storage rotate_each_word_right16() const {
    auto constexpr k0 = 0x0d0c'0f0e'0908'0b0a;
    auto constexpr k1 = 0x0504'0706'0100'0302;

    return {
        {mm256_shuffle_epi8(this->avx[0], mm256_set_epi64x(k0, k1, k0, k1)),
         mm256_shuffle_epi8(this->avx[1], mm256_set_epi64x(k0, k1, k0, k1))}};
  }

  inline static constexpr vec256_storage
  mm256_or_si256(const vec256_storage &a, const vec256_storage &b) {
    vec256_storage r;
    for (int i = 0; i < 4; ++i) {
      r.u64x4[i] = a.u64x4[i] | b.u64x4[i];
    }
    return r;
  }

  static inline vec256_storage mm256_srli_epi32(const vec256_storage &a,
                                                int count) {
    vec256_storage r;

    // Cap the shift count at 31, as larger shifts produce zero
    const int c = count & 0x1F;

    for (int i = 0; i < 8; ++i) {
      r.u32x8[i] = a.u32x8[i] >> c;
    }

    return r;
  }

  static inline vec256_storage mm256_slli_epi32(const vec256_storage &a,
                                                int count) {
    vec256_storage r;

    // Cap the shift count at 31, as larger shifts produce zero
    const int c = count & 0x1F;

    for (int i = 0; i < 8; ++i) {
      r.u32x8[i] = a.u32x8[i] << c;
    }

    return r;
  }

  constexpr vec512_storage rotate_each_word_right20() const {
    constexpr int32_t i = 20;

    return {{mm256_or_si256(mm256_srli_epi32(this->avx[0], i),
                            mm256_slli_epi32(this->avx[0], 32 - i)),
             mm256_or_si256(mm256_srli_epi32(this->avx[1], i),
                            mm256_slli_epi32(this->avx[1], 32 - i))}};
  }

  constexpr vec512_storage rotate_each_word_right24() const {
    auto constexpr k0 = 0x0e0d'0c0f'0a09'080b;
    auto constexpr k1 = 0x0605'0407'0201'0003;

    return {
        {mm256_shuffle_epi8(this->avx[0], mm256_set_epi64x(k0, k1, k0, k1)),
         mm256_shuffle_epi8(this->avx[1], mm256_set_epi64x(k0, k1, k0, k1))}};
  }

  constexpr vec512_storage rotate_each_word_right25() const {
    constexpr int32_t i = 25;

    return {{mm256_or_si256(mm256_srli_epi32(this->avx[0], i),
                            mm256_slli_epi32(this->avx[0], 32 - i)),
             mm256_or_si256(mm256_srli_epi32(this->avx[1], i),
                            mm256_slli_epi32(this->avx[1], 32 - i))}};
  }

  constexpr vec512_storage shuffle_lane_words3012() const {
    return {{this->avx[0].shuffle_lane_words3012(),
             this->avx[1].shuffle_lane_words3012()}};
  }

  constexpr vec512_storage shuffle_lane_words2301() const {
    return {{this->avx[0].shuffle_lane_words2301(),
             this->avx[1].shuffle_lane_words2301()}};
  }

  constexpr vec512_storage shuffle_lane_words1230() const {
    return {{this->avx[0].shuffle_lane_words1230(),
             this->avx[1].shuffle_lane_words1230()}};
  }

  static inline vec256_storage
  mm256_permute2x128_si256(const vec256_storage &V1, const vec256_storage &V2,
                           int M) {
    vec256_storage r;

    // Process each 128-bit half (0 = lower, 1 = upper)
    for (int i = 0; i < 2; ++i) {
      int control = (M >> (i * 4)) & 0xF;

      if (control & 0x8) {
        // bit 3 set → zero this 128-bit half
        r.sse2[i].u64x2[0] = 0;
        r.sse2[i].u64x2[1] = 0;
      } else {
        // select source 128-bit half based on bits [1:0]
        int src = control & 0x3;
        switch (src) {
        case 0:
          r.sse2[i] = V1.sse2[0];
          break; // lower half of V1
        case 1:
          r.sse2[i] = V1.sse2[1];
          break; // upper half of V1
        case 2:
          r.sse2[i] = V2.sse2[0];
          break; // lower half of V2
        case 3:
          r.sse2[i] = V2.sse2[1];
          break; // upper half of V2
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
    auto const ab00 = mm256_permute2x128_si256(a.avx[0], b.avx[0], 0x20);
    auto const ab01 = mm256_permute2x128_si256(a.avx[0], b.avx[0], 0x31);
    auto const ab10 = mm256_permute2x128_si256(a.avx[1], b.avx[1], 0x20);
    auto const ab11 = mm256_permute2x128_si256(a.avx[1], b.avx[1], 0x31);
    auto const cd00 = mm256_permute2x128_si256(c.avx[0], d.avx[0], 0x20);
    auto const cd01 = mm256_permute2x128_si256(c.avx[0], d.avx[0], 0x31);
    auto const cd10 = mm256_permute2x128_si256(c.avx[1], d.avx[1], 0x20);
    auto const cd11 = mm256_permute2x128_si256(c.avx[1], d.avx[1], 0x31);
    return {vec512_storage{{ab00, cd00}}, vec512_storage{{ab01, cd01}},
            vec512_storage{{ab10, cd10}}, vec512_storage{{ab11, cd11}}};
  }

  constexpr Slice<uint32_t> to_scalars() const {
    return Slice<uint32_t>(this->u32x16.data(), this->u32x16.size());
  }

  constexpr vec512_storage to_lanes() const {
    auto const [a, b] = this->avx[0].to_lanes().sse2;
    auto const [c, d] = this->avx[1].to_lanes().sse2;
    return {{a, b, c, d}};
  }
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
static inline State<V> round(State<V> x) {
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

inline std::array<uint64_t, 2> add_epi64(const std::array<uint64_t, 2> &a,
                                         const std::array<uint64_t, 2> &b) {
  return {a[0] + b[0], a[1] + b[1]};
}

inline constexpr vec128_storage add_pos(vec128_storage &d, uint64_t i) {
  auto const d0 = d.u64x2;
  auto const incr = vec128_storage::from_lanes(std::array<uint64_t, 2>{i, 0});
  return add_epi64(d0, incr.u64x2);
}

// a_lo and a_hi are each 128-bit vectors represented as 4 x 32-bit integers
inline vec256_storage mm256_setr_m128i(const vec128_storage &lo,
                                       const vec128_storage &hi) {
  vec256_storage r;

  // lower 128-bit lane
  r.sse2[0] = lo;

  // upper 128-bit lane
  r.sse2[1] = hi;

  return r;
}

inline vec128_storage mm_add_epi64(const vec128_storage &a,
                                   const vec128_storage &b) {
  vec128_storage r;
  // Element-wise addition of the two 64-bit integers
  r.u64x2[0] = a.u64x2[0] + b.u64x2[0];
  r.u64x2[1] = a.u64x2[1] + b.u64x2[1];
  return r;
}

class StdRng {
public:
  mutable ChaCha core_state;
  mutable std::array<uint, 64> results;
  mutable uint8_t index;

  constexpr StdRng(std::array<uint8_t, 32> &key) {
    auto key0 =
        vec128_storage::read_le(Slice(key.data(), key.size()).range(0, 16));
    auto key1 = vec128_storage::read_le(Slice(key.data(), key.size()).sub(16));

    this->core_state = ChaCha{
        .b = key0,
        .c = key1,
        .d = vec128_storage({0, 0, 0, 0}),
    };
    this->results = std::array<uint, 64>{0};
    this->index = this->results.size();
  }

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
    std::array<uint8_t, 32> seed{{0}};
    for (auto chunk : utility::chunks_exact_mut(Slice(seed.data(), 32), 4)) {
      std::array<uint8_t, 4> x = pcg32(state);
      uint8_t *dst = chunk.data();
      uint8_t *src = x.data();
      for (unsigned i = 0; i < chunk.size(); ++i)
        dst[i] = src[i];
    }
    return StdRng(seed);
  }

  inline constexpr void generate_and_set(size_t index) const {
    // assert(index < this->results.size());
    this->generate(this->results);
    this->index = index;
  }

  inline static constexpr vec512_storage d0123(vec128_storage &d) {
    vec512_storage incr = vec512_storage{
        {vec128_storage::from_lanes(std::array<uint64_t, 2>{0, 0}),
         vec128_storage::from_lanes(std::array<uint64_t, 2>{1, 0}),
         vec128_storage::from_lanes(std::array<uint64_t, 2>{2, 0}),
         vec128_storage::from_lanes(std::array<uint64_t, 2>{3, 0})}};

    auto v = vec512_storage::new128(
        {mm_add_epi64(d, incr.sse2[0]), mm_add_epi64(d, incr.sse2[1]),
         mm_add_epi64(d, incr.sse2[2]), mm_add_epi64(d, incr.sse2[3])});

    return v.unpack();
  }

  inline constexpr void
  refill_wide_impl(uint32_t drounds, std::array<uint32_t, BUFSZ> &out) const {
    auto k = vec128_storage::from_lanes(
        {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574});
    vec128_storage b = core_state.b;
    vec128_storage c = core_state.c;
    auto x = State<vec512_storage>{
        .a = {{mm256_setr_m128i(k.u32x4, k.u32x4), mm256_setr_m128i(k, k)}},
        .b = {{mm256_setr_m128i(b.u32x4, b.u32x4), mm256_setr_m128i(b, b)}},
        .c = {{mm256_setr_m128i(c.u32x4, c.u32x4), mm256_setr_m128i(c, c)}},
        .d = d0123(core_state.d),
    };

    for (size_t i = 0; i < drounds; i++) {
      x = round(x);
      x = undiagonalize(round(diagonalize(x)));
    }

    auto const kk =
        vec512_storage{{mm256_setr_m128i(k, k), mm256_setr_m128i(k, k)}};
    auto const sb1 = core_state.b;
    auto sb = vec512_storage{
        {mm256_setr_m128i(sb1, sb1), mm256_setr_m128i(sb1, sb1)}};
    auto const sc1 = core_state.c;
    auto sc = vec512_storage{
        {mm256_setr_m128i(sc1, sc1), mm256_setr_m128i(sc1, sc1)}};
    auto const sd = d0123(core_state.d);
    auto const &[ra, rb, rc, rd] =
        vec512_storage::transpose4(kk + x.a, x.b + sb, x.c + sc, x.d + sd);

    Slice<uint32_t> sout(out.data(), out.size());
    sout.range(0, 16).copy_from_slice(ra.to_scalars());
    sout.range(16, 32).copy_from_slice(rb.to_scalars());
    sout.range(32, 48).copy_from_slice(rc.to_scalars());
    sout.range(48, 64).copy_from_slice(rd.to_scalars());
    core_state.d = add_pos(sd.to_lanes().sse2[0], 4);
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
