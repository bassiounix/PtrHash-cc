#ifndef BUCKET_IDX_HPP_
#define BUCKET_IDX_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

class BucketIdx {
public:
  uint32_t i_;

  operator uint32_t() { return i_; }
  operator uint32_t() const { return i_; }

  BucketIdx(uint32_t i) : i_(i) {}

  bool operator==(const BucketIdx other) const { return this->i_ == other.i_; }

  BucketIdx operator+(size_t rhs) const {
    return BucketIdx(this->i_ + static_cast<uint32_t>(rhs));
  }

  BucketIdx operator-(size_t rhs) const {
    return BucketIdx(this->i_ - static_cast<uint32_t>(rhs));
  }

  bool operator<(const BucketIdx &other) const { return this->i_ < other.i_; }
  bool operator>(const BucketIdx &other) const { return this->i_ > other.i_; }

  static constexpr auto NONE = ~(uint32_t)0;

  static std::vector<BucketIdx> range(size_t num_buckets) {
    std::vector<BucketIdx> out;
    out.reserve(num_buckets);

    for (uint32_t i = 0; i < num_buckets; ++i) {
      out.emplace_back(i);
    }
    return out;
  }

  bool is_some() const { return this->i_ != ~(uint32_t)0; }
  bool is_none() const { return this->i_ == ~(uint32_t)0; }
};

#endif // BUCKET_IDX_HPP_
