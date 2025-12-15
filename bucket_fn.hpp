#ifndef BUCKET_FN_HPP_
#define BUCKET_FN_HPP_

#include "utils.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>

class BucketFn {
public:
  constexpr static bool LINEAR = false;
  constexpr static bool B_OUTPUT = false;
  void set_buckets_per_part(uint64_t _b) {}
  virtual uint64_t call(uint64_t x) = 0;
};

class Linear : public BucketFn {
public:
  constexpr static bool LINEAR = true;
  uint64_t call(uint64_t x) override { return x; }
};

/// A 2-piece-wise linear function; as used in FCH and PTHash.
///
/// |              .
/// |             .
/// |         ....---< gamma
/// |    .....   |
/// |....        |
/// +------------^--
///              beta
///
/// line1: y = x * (gamma / beta)
///                ~~~ slope1 ~~~
/// line2: y = x * ((1 - gamma) / (1 - beta)) + (gamma - beta) / (1 - beta)
///                ~~~~~~~~~ slope2 ~~~~~~~~~   ~~~~~~~~~~ offset ~~~~~~~~~
class Skewed : public BucketFn {
public:
  double beta_f;
  double gamma_f;
  /// buckets per part
  uint64_t b;
  uint64_t beta;
  uint64_t slope1;
  uint64_t slope2;
  uint64_t neg_offset;

  Skewed(double beta = 0.6, double gamma = 0.3)
      : beta_f(beta), gamma_f(gamma), b(0), beta(0), slope1(0), slope2(0),
        neg_offset(0) {
    assert(beta > gamma && "Beta={beta} must be larger than gamma={gamma}");
  }

  static constexpr bool B_OUTPUT = true;

  void set_buckets_per_part(uint64_t b) {
    auto beta = this->beta_f;
    auto gamma = this->gamma_f;
    this->b = b;
    auto as_u64 = [](double x) {
      return (uint64_t)(x * (double)(~(uint64_t)(0)));
    };
    this->slope1 = utility::mul_high(as_u64(gamma / beta), this->b);
    this->slope2 =
        utility::mul_high(as_u64((1. - gamma) / (1. - beta) / 8.), this->b << 3);
    this->neg_offset =
        utility::mul_high(as_u64((beta - gamma) / (1. - beta) / 8.), this->b << 3);
    this->beta = as_u64(beta);
  }

  uint64_t call(uint64_t x) override {
    // NOTE: There is a lot of MOV/CMOV going on here.
    auto is_large = x >= this->beta;
    auto slope = is_large ? this->slope2 : this->slope1;
    return utility::mul_high(x, slope) - (uint64_t)is_large * this->neg_offset;
    // assert(!is_large || this->p2 <= b, "p2 {} <= b {}", this->p2, b);
    // assert(!is_large || b < this->b, "b {} < p2 {}", b, this->b);
    // assert(is_large || b < this->p2, "b {} < p2 {}", b, this->p2);
  }
};

class Optimal : public BucketFn {
public:
  double eps;

  uint64_t call(uint64_t _x) override {
    double p32 = (1ULL << 32);
    auto p64 = p32 * p32;
    auto p64inv = 1. / p64;
    auto x = ((double)_x) * p64inv;
    auto y = x + (1. - this->eps) * (1. - x) * std::log(1. - x);

    return (uint64_t)(y * p64);
  }
};

class Square : public BucketFn {
public:
  uint64_t call(uint64_t x) override { return utility::mul_high(x, x); }
};

class SquareEps : public BucketFn {
public:
  uint64_t call(uint64_t x) override {
    return utility::mul_high(x, x) / 256 * 255 + x / 256;
  }
};

class Cubic : public BucketFn {
public:
  uint64_t call(uint64_t x) override {
    // x * x * (1 + x) / 2
    return utility::mul_high(utility::mul_high(x, x), (x >> 1) | (1ULL << 63));
  }
};

class CubicEps : public BucketFn {
public:
  uint64_t call(uint64_t x) override {
    // x * x * (1 + x) / 2
    return utility::mul_high(utility::mul_high(x, x), (x >> 1) | (1ULL << 63)) / 256 * 255 +
           x / 256;
  }
};

static void test_skewed() {
  Skewed skewed(0.6, 0.3);
  skewed.set_buckets_per_part(1000000000);

  auto last_y = 0;
  auto n = 100;
  for (size_t i = 0; i < 100; i++) {
    auto x = ~(uint64_t)0 / n * i;
    auto y = skewed.call(x);
    assert(y >= last_y);
    last_y = y;
  }
}

#endif // BUCKET_FN_HPP_
