#ifndef FXHASH_HPP_
#define FXHASH_HPP_

#include "hasher_interface.hpp"
#include "slice.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

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
  T value;
  inline void hash_word(T word) {
    this->value = rotate_left(this->value, ROTATE) ^ word;
    this->value *= SEED_VALUE; // wrapping multiplication
  }

  // Portable rotate-left
  inline T rotate_left(T x, unsigned int n) {
    constexpr unsigned int bits = sizeof(T) * 8;
    return (x << n) | (x >> (bits - n));
  }

  operator T() { return this->value; }
  operator T() const { return this->value; }

  HashWord(T initial) : value(initial) {}
  HashWord &operator=(T new_value) {
    this->value = new_value;
    return *this;
  }
};

// Type aliases for convenience
using HashWordU64 = HashWord<uint64_t, SEED64>;
using HashWordU32 = HashWord<uint32_t, SEED32>;
using HashWordUSize = HashWord<size_t, SEED>;

namespace NativeEndian {

inline uint32_t read_u32(const uint8_t *bytes) {
  uint32_t n;
  std::memcpy(&n, bytes, sizeof(n)); // copies exactly 4 bytes
  return n;                          // native-endian
}

inline uint64_t read_u64(const uint8_t *bytes) {
  uint64_t n;
  std::memcpy(&n, bytes, sizeof(n)); // copies 8 bytes safely
  return n;                          // native-endian
}

} // namespace NativeEndian

inline uint32_t write32(uint32_t _hash, Slice<uint8_t> bytes) {
  auto hash = HashWordU32{_hash};
  while (bytes.size() >= 4) {
    auto n = NativeEndian::read_u32(bytes.data());
    hash.hash_word(n);
    bytes = bytes.sub(4);
  }

  for (size_t i = 0; i < bytes.size(); i++) {
    hash.hash_word(bytes.data()[i]);
  }

  return hash;
}

inline uint64_t write64(uint64_t _hash, Slice<uint8_t> bytes) {
  auto hash = HashWordU64{_hash};
  while (bytes.size() >= 8) {
    auto n = NativeEndian::read_u64(bytes.data());
    hash.hash_word(n);
    bytes = bytes.sub(8);
  }

  if (bytes.size() >= 4) {
    auto n = NativeEndian::read_u32(bytes.data());
    hash.hash_word((uint64_t)n);
    bytes = bytes.sub(4);
  }

  for (size_t i = 0; i < bytes.size(); i++) {
    hash.hash_word(static_cast<uint64_t>(bytes.data()[i]));
  }
  return hash;
}

inline size_t write(size_t hash, Slice<uint8_t> bytes) {
  if constexpr (sizeof(size_t) == 4) {
    return static_cast<size_t>(write32(static_cast<uint32_t>(hash), bytes));
  } else {
    return static_cast<size_t>(write64(static_cast<uint64_t>(hash), bytes));
  }
}

class FxHasher : public Hasher {
public:
  HashWordUSize hash;

  FxHasher() : hash(0) {}

  void write(Slice<uint8_t> bytes) override {
    this->hash = FxHasherDecl::write(this->hash, bytes);
  }

  inline void write(uint8_t i) { this->hash.hash_word(i); }
  inline void write(uint16_t i) { this->hash.hash_word(i); }
  inline void write(uint32_t i) { this->hash.hash_word(i); }
  inline void write(uint64_t i) {
    if constexpr (sizeof(size_t) == 4) {
      this->hash.hash_word((size_t)i);
      this->hash.hash_word((size_t)(i >> 32));
    } else {
      this->hash.hash_word((size_t)i);
    }
  }

  inline uint64_t finish() override { return this->hash; }
};

class FxHasher64 : public Hasher {
public:
  HashWordU64 hash;

  FxHasher64() : hash(0) {}

  inline void write(Slice<uint8_t> bytes) override {
    this->hash = write64(this->hash, bytes);
  }

  inline void write(uint8_t i) { this->hash.hash_word((uint64_t)i); }
  inline void write(uint16_t i) { this->hash.hash_word((uint64_t)i); }
  inline void write(uint32_t i) { this->hash.hash_word((uint64_t)i); }
  void write(uint64_t i) { this->hash.hash_word(i); }
  inline uint64_t finish() override { return this->hash; }
};

class FxHasher32 : public Hasher {
public:
  HashWordU32 hash;

  FxHasher32() : hash(0) {}

  inline void write(Slice<uint8_t> bytes) override {
    this->hash = write32(this->hash, bytes);
  }

  inline void write(uint8_t i) { this->hash.hash_word((uint32_t)i); }
  inline void write(uint16_t i) { this->hash.hash_word((uint32_t)i); }
  inline void write(uint32_t i) { this->hash.hash_word(i); }
  inline void write(uint64_t i) {
    this->hash.hash_word((uint32_t)i);
    this->hash.hash_word((uint32_t)(i >> 32));
  }

  inline uint64_t finish() override { return this->hash; }
};

template<typename T>
inline uint64_t hash64(T v) {
  auto state = FxHasher64();
  state.write(v);
  return state.finish();
}

template<typename T>
inline uint32_t hash32(T v) {
  auto state = FxHasher32();
  state.write(v);
  return state.finish();
}
template<typename T>
inline size_t hash(T v) {
  auto state = FxHasher();
  state.write(v);
  return state.finish();
}

} // namespace FxHasherDecl

#endif // FXHASH_HPP_
