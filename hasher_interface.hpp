#ifndef GXHASH_HASHER_INTERFACE_HPP_
#define GXHASH_HASHER_INTERFACE_HPP_

#include "slice.hpp"
#include <cstdint>

template<typename HasherImpl>
class Hasher {
public:
  constexpr uint64_t finish() const {
    return static_cast<const HasherImpl*>(this)->finish_impl();
  }
  constexpr void write(Slice<uint8_t> bytes) const {
    static_cast<const HasherImpl*>(this)->write_imp(bytes);
  }
};

#endif // GXHASH_HASHER_INTERFACE_HPP_
