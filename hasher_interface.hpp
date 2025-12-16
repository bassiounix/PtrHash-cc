#ifndef GXHASH_HASHER_INTERFACE_HPP_
#define GXHASH_HASHER_INTERFACE_HPP_

#include "slice.hpp"
#include <cstdint>

class Hasher {
public:
  virtual constexpr uint64_t finish() const = 0;
  virtual constexpr void write(Slice<uint8_t> bytes) const = 0;
};

#endif // GXHASH_HASHER_INTERFACE_HPP_
