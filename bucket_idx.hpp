#ifndef BUCKET_IDX_HPP_
#define BUCKET_IDX_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

class BucketIdx {
public:
  mutable uint32_t i_;

  constexpr BucketIdx() = default;

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

#endif // BUCKET_IDX_HPP_
