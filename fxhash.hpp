#ifndef FXHASH_HPP_
#define FXHASH_HPP_

#include "hasher_interface.hpp"
#include "slice.hpp"
#include "utils.hpp"
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

inline constexpr uint32_t write32(uint32_t _hash, Slice<uint8_t> bytes) {
  auto hash = HashWordU32{_hash};
  while (bytes.size() >= 4) {
    auto n = utility::ptr_bit_cast<uint32_t>(bytes.data());
    hash.hash_word(n);
    bytes = bytes.sub(4);
  }

  for (size_t i = 0; i < bytes.size(); i++) {
    hash.hash_word(bytes.data()[i]);
  }

  return hash;
}

inline constexpr uint64_t write64(uint64_t _hash, Slice<uint8_t> bytes) {
  auto hash = HashWordU64{_hash};
  while (bytes.size() >= 8) {
    auto n = utility::ptr_bit_cast<uint64_t>(bytes.data());
    hash.hash_word(n);
    bytes = bytes.sub(8);
  }

  if (bytes.size() >= 4) {
    auto n = utility::ptr_bit_cast<uint32_t>(bytes.data());
    hash.hash_word((uint64_t)n);
    bytes = bytes.sub(4);
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

#endif // FXHASH_HPP_
