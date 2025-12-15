#ifndef GXHASH_HASHER_INTERFACE_HPP_
#define GXHASH_HASHER_INTERFACE_HPP_

#include "slice.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

template <typename T> inline Slice<uint8_t> to_ne_bytes(T value) {
  Slice<uint8_t> bytes = Slice<uint8_t>();
  std::memcpy((void *)bytes.data(), &value, sizeof(T)); // native-endian
  return bytes;
}

class Hasher {
public:
  virtual uint64_t finish() = 0;

  virtual void write(Slice<uint8_t> bytes) = 0;

  /// Writes a single `u8` into this hasher.
  inline auto write(uint8_t i) {
    uint8_t bytes[] = {i};
    return this->write(Slice<uint8_t>(bytes, 1));
  }

  /// Writes a single `u16` into this hasher.
  inline auto write(uint16_t i) { return this->write(to_ne_bytes(i)); }
  /// Writes a single `u32` into this hasher.
  inline auto write(uint32_t i) { return this->write(to_ne_bytes(i)); }
  /// Writes a single `u64` into this hasher.
  inline auto write(uint64_t i) { return this->write(to_ne_bytes(i)); }
  /// Writes a single `u128` into this hasher.
  inline auto write(__uint128_t i) { return this->write(to_ne_bytes(i)); }
  /// Writes a single `usize` into this hasher.
  // inline auto write(std::size_t i) { return this->write(to_ne_bytes(i)); }

  /// Writes a single `i8` into this hasher.
  inline auto write(int8_t i) { return this->write((uint8_t)i); }
  /// Writes a single `i16` into this hasher.
  inline auto write(int16_t i) { return this->write((uint16_t)i); }
  /// Writes a single `i32` into this hasher.
  inline auto write(int32_t i) { return this->write((uint32_t)i); }
  /// Writes a single `i64` into this hasher.
  inline auto write(int64_t i) { return this->write((uint64_t)i); }
  /// Writes a single `i128` into this hasher.
  inline auto write(__int128_t i) {
    return this->write((__uint128_t)i);
  }
  /// Writes a single `isize` into this hasher.
  // inline auto write(intptr_t i) {
  //   return this->write((std::size_t)i);
  // }

  inline auto write_length_prefix(size_t len) { return this->write(len); }

  inline auto write(std::string_view s) {
    this->write(Slice<uint8_t>{(uint8_t *)s.data(), s.size()});
    this->write(0xff);
  }
};

class Hash {
public:
  virtual void hash(Hasher &state) = 0;

  static void hash_slice(Slice<Hash> data, Hasher &state) {
    for (auto &piece : data) {
      piece.hash(state);
    }
  }
};

/// Type of the hasher that will be created.
class BuildHasher {
public:
  virtual Hasher &build_hasher() = 0;

  uint64_t hash_one(Hash &x) {
    auto &hasher = this->build_hasher();
    x.hash(hasher);
    return hasher.finish();
  }
};

#endif // GXHASH_HASHER_INTERFACE_HPP_
