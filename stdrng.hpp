// this file is experimentation with advanced random algorithms
// (incomplete & not used)

#ifndef STDRNG_HPP_
#define STDRNG_HPP_

#include "utils.hpp"
#include <array>
#include <cstdint>
#include <cstring>

// ChaCha12 RNG (like Rust StdRng's algorithm)
class ChaCha12Rng {
public:
  // 512-bit state (16 x 32-bit words)
  std::array<uint32_t, 16> state{};

private:
  // Buffer for output (16 words = 64 bytes)
  std::array<uint32_t, 16> buffer{};
  size_t index = 16; // Next index in buffer (16 triggers refill)

  static constexpr int ROUNDS = 12; // 12 rounds for ChaCha12

  // Rotate left
  static inline uint32_t rotl(uint32_t value, int bits) {
    return (value << bits) | (value >> (32 - bits));
  }

  // ChaCha quarter round
  static inline void quarter_round(uint32_t &a, uint32_t &b, uint32_t &c,
                                   uint32_t &d) {
    a += b;
    d ^= a;
    d = rotl(d, 16);
    c += d;
    b ^= c;
    b = rotl(b, 12);
    a += b;
    d ^= a;
    d = rotl(d, 8);
    c += d;
    b ^= c;
    b = rotl(b, 7);
  }

  // Generate next 16-word block into buffer
  void refill() {
    // Copy state to working block
    std::array<uint32_t, 16> working = state;

    // Apply 12 rounds (6 double rounds)
    for (int i = 0; i < ROUNDS; i += 2) {
      // Column rounds
      quarter_round(working[0], working[4], working[8], working[12]);
      quarter_round(working[1], working[5], working[9], working[13]);
      quarter_round(working[2], working[6], working[10], working[14]);
      quarter_round(working[3], working[7], working[11], working[15]);
      // Diagonal rounds
      quarter_round(working[0], working[5], working[10], working[15]);
      quarter_round(working[1], working[6], working[11], working[12]);
      quarter_round(working[2], working[7], working[8], working[13]);
      quarter_round(working[3], working[4], working[9], working[14]);
    }

    // Add working block back into state to form output
    for (int i = 0; i < 16; i++) {
      buffer[i] = working[i] + state[i];
    }

    // Increment 64-bit counter (words 12 and 13)
    state[12] += 1;
    if (state[12] == 0) {
      state[13] += 1;
    }

    index = 0;
  }

public:
  // Seed using a 256-bit seed array (like Rust SeedableRng)
  // Rust's StdRng seed is 256 bits (32 bytes)
  explicit ChaCha12Rng(const std::array<char, 32> &seed) {
    // Constants "expand 32‑byte k"
    static constexpr uint32_t constants[4] = {0x61707865, 0x3320646e,
                                              0x79622d32, 0x6b206574};

    // Initialize state
    state[0] = constants[0];
    state[1] = constants[1];
    state[2] = constants[2];
    state[3] = constants[3];

    // Load 256‑bit seed into state[4..11]
    for (int i = 0; i < 8; ++i) {
      state[4 + i] =
          uint32_t(seed[i * 4 + 0]) | (uint32_t(seed[i * 4 + 1]) << 8) |
          (uint32_t(seed[i * 4 + 2]) << 16) | (uint32_t(seed[i * 4 + 3]) << 24);
    }

    // Initialize counter/nonce
    state[12] = 0;
    state[13] = 0;
    state[14] = 0;
    state[15] = 0;
  }

  // Produce next random 32‑bit word
  uint32_t next_u32() {
    if (index >= buffer.size()) {
      refill();
    }
    return buffer[index++];
  }

  // Produce next random 64‑bit value
  uint64_t next_u64() {
    uint64_t lo = next_u32();
    uint64_t hi = next_u32();
    return (hi << 32) | lo;
  }

  // Generate a random number in [0, max)
  uint32_t gen_range(uint32_t max) {
    // Rejection could be used to reduce bias, but simple modulus:
    return next_u32() % max;
  }
};

std::array<char, 32> seed = {
    '\x93', '\xf9', '\U0000001a', 'J',    '\xd8', '\xd6', 'b',    ';',
    '\xed', '\'',   '\xdb',       '\xf1', '\xdb', 'f',    'c',    '\xfd',
    '\x95', '\x9a', '\xe0',       'y',    'Z',    '\xa5', 'I',    '#',
    'H',    '\xf7', '\0',         'y',    '\x8c', '\xb1', '\xde', '\xf3'};

ChaCha12Rng rng(seed);

uint64_t static fastseed = 0x4d595df4d0f33173;
inline uint64_t gen_u64() {
  const uint64_t WY_CONST_0 = 0x2d35'8dcc'aa6c'78a5;
  const uint64_t WY_CONST_1 = 0x8bb8'4b93'962e'acc9;

  auto const s = fastseed + (WY_CONST_0);
  fastseed = s;
  auto const t =
      static_cast<__uint128_t>(s) * static_cast<__uint128_t>(s ^ WY_CONST_1);
  return static_cast<uint64_t>(t) ^ static_cast<uint64_t>(t >> 64);
}

static constexpr size_t BLOCK = 16;
static constexpr uint64_t BLOCK64 = BLOCK;
static constexpr uint64_t LOG2_BUFBLOCKS = 2;
static constexpr uint64_t BUFBLOCKS = 1 << LOG2_BUFBLOCKS;
static constexpr uint64_t BUFSZ64 = BLOCK64 * BUFBLOCKS;
static constexpr size_t BUFSZ = BUFSZ64;

class RRNG126 {
public:
  std::array<uint32_t, BUFSZ> results;
  struct ChaCha {
    std::array<uint64_t, 2> b = {4279218820399954323, 18258550409131665389u};
    std::array<uint64_t, 2> c = {2542745272637758101, 17572678012929898312u};
    std::array<uint64_t, 2> d = {0, 0};
  } state;
  std::array<uint8_t, 32> seed;
  size_t index = results.size();

  RRNG126(std::array<uint8_t, 32> seed) : seed(seed) {}

  auto random() { return next_u64(); }

  inline uint64_t next_u64() {
    auto read_u64 = [](Slice<uint32_t> results, size_t index) -> uint64_t {
      auto data = results.range(index, index + 1);
      return (static_cast<uint64_t>(data[1]) << 32) |
             static_cast<uint64_t>(data[0]);
    };

    auto len = this->results.size();

    auto index = this->index;
    if (index < len - 1) {
      this->index += 2;
      // Read an u64 from the current index
      return read_u64(
          Slice<uint32_t>(this->results.data(), this->results.size()), index);
    } else if (index >= len) {
      this->generate_and_set(2);
      return read_u64(
          Slice<uint32_t>(this->results.data(), this->results.size()), 0);
    } else {
      uint64_t x = this->results[len - 1];
      this->generate_and_set(1);
      uint64_t y = this->results[0];
      return (y << 32) | x;
    }
  }

  inline void generate_and_set(size_t index) {
    // assert(index < this->results.size());
    this->generate(this->results);
    this->index = index;
  }

  inline void generate(std::array<uint32_t, BUFSZ> &out) {
    refill_wide_impl(this->state, 6, out);
  }

  static inline uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
  }

  static inline void quarter_round(uint32_t &a, uint32_t &b, uint32_t &c,
                                   uint32_t &d) {
    a += b;
    d ^= a;
    d = rotl(d, 16);
    c += d;
    b ^= c;
    b = rotl(b, 12);
    a += b;
    d ^= a;
    d = rotl(d, 8);
    c += d;
    b ^= c;
    b = rotl(b, 7);
  }

  static inline void split_u64(uint64_t x, uint32_t &lo, uint32_t &hi) {
    lo = static_cast<uint32_t>(x);
    hi = static_cast<uint32_t>(x >> 32);
  }

  void refill_wide_impl(ChaCha &state, uint32_t drounds,
                        std::array<uint32_t, BUFSZ> &out) {
    constexpr uint32_t K[4] = {0x61707865u, 0x3320646eu, 0x79622d32u,
                               0x6b206574u};

    uint32_t key[4];
    uint32_t nonce[4];
    uint32_t counter[4];

    split_u64(state.b[0], key[0], key[1]);
    split_u64(state.b[1], key[2], key[3]);

    split_u64(state.c[0], nonce[0], nonce[1]);
    split_u64(state.c[1], nonce[2], nonce[3]);

    split_u64(state.d[0], counter[0], counter[1]);
    split_u64(state.d[1], counter[2], counter[3]);

    uint32_t x[4][16];
    uint32_t w[4][16];

    for (int lane = 0; lane < 4; ++lane) {
      x[lane][0] = K[0];
      x[lane][1] = K[1];
      x[lane][2] = K[2];
      x[lane][3] = K[3];

      x[lane][4] = key[0];
      x[lane][5] = key[1];
      x[lane][6] = key[2];
      x[lane][7] = key[3];

      x[lane][8] = nonce[0];
      x[lane][9] = nonce[1];
      x[lane][10] = nonce[2];
      x[lane][11] = nonce[3];

      x[lane][12] = counter[0] + lane;
      x[lane][13] = counter[1];
      x[lane][14] = counter[2];
      x[lane][15] = counter[3];

      for (int i = 0; i < 16; ++i)
        w[lane][i] = x[lane][i];
    }

    for (uint32_t r = 0; r < drounds; ++r) {
      for (int lane = 0; lane < 4; ++lane) {
        auto &v = w[lane];

        quarter_round(v[0], v[4], v[8], v[12]);
        quarter_round(v[1], v[5], v[9], v[13]);
        quarter_round(v[2], v[6], v[10], v[14]);
        quarter_round(v[3], v[7], v[11], v[15]);

        quarter_round(v[0], v[5], v[10], v[15]);
        quarter_round(v[1], v[6], v[11], v[12]);
        quarter_round(v[2], v[7], v[8], v[13]);
        quarter_round(v[3], v[4], v[9], v[14]);
      }
    }

    for (int lane = 0; lane < 4; ++lane) {
      for (int i = 0; i < 16; ++i) {
        out[lane * 16 + i] = w[lane][i] + x[lane][i];
      }
    }

    uint64_t ctr = state.d[0];
    ctr += 4;
    state.d[0] = ctr;
    if (ctr < 4) {
      state.d[1] += 1;
    }
  }

  static constexpr std::array<uint8_t, 4> pcg32(uint64_t state) {
    constexpr uint64_t MUL = 6364136223846793005ull;
    constexpr uint64_t INC = 11634580027462260723ull;

    state = utility::wrapping_mul(state, MUL) + INC;
    // Use PCG output function with to_le to generate x:
    uint32_t xorshifted = (((state >> 18) ^ state) >> 27);
    uint32_t rot = state >> 59;
    uint32_t x = utility::rotate_right(xorshifted, rot);
    return utility::to_le_bytes(x);
  }

  static auto seed_from_u64(uint64_t state = 31415) {
    std::array<uint8_t, 32> seed;
    for (auto chunk : utility::chunks_exact_mut(Slice(seed.data(), 32), 4)) {
      std::memcpy((void *)chunk.data(), pcg32(state).data(), chunk.size());
    }
    return RRNG126(seed);
  }
};

#endif // STDRNG_HPP_
