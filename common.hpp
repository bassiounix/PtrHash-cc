#ifndef PORT_COMMON_HPP_
#define PORT_COMMON_HPP_

#include "fxhash.hpp"
#include "hasher_interface.hpp"
#include "slice.hpp"
#include "utils.hpp"
#include "zip.hpp"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <numeric>
#include <optional>
#include <variant>
#include <vector>

// inline static constexpr uint64_t mul_high(uint64_t a, uint64_t b) {
//   uint64_t hi = 0;
//   __builtin_mul_overflow(a, b, &hi);
//   return hi;
// }

template <class T> class Number {
public:
  static_assert(std::is_arithmetic_v<T>,
                "Number class only works with numbers");
  T number;

  operator T() { return this->number; }
  operator T() const { return this->number; }
};

// class KeyT : public Hash {};

class Packed {
  virtual uint64_t index(size_t index) = 0;
  virtual void prefetch(size_t _index) {}
  virtual size_t size_in_bytes() = 0;
  static std::optional<Packed> try_new(Slice<uint64_t> vals);
};

template <typename T> class DynamicContainer : public Packed {
private:
  std::vector<T> i_;

public:
  // DynamicContainer(Slice<T> i) : i_(i) {}
  DynamicContainer(Slice<T> &i) {
    for (auto e : i) {
      i_.push_back(e);
    }
  }
  DynamicContainer(Slice<T> &&i) {
    for (auto e : i) {
      i_.push_back(e);
    }
  }
  DynamicContainer(std::vector<T> &v) : i_(v) {}
  DynamicContainer() = default;
  DynamicContainer(const DynamicContainer<T> &) = default;
  DynamicContainer(DynamicContainer<T> &&) = default;

  DynamicContainer<T> &operator=(const DynamicContainer<T> &) = default;
  // DynamicContainer<T>& operator=(DynamicContainer<T>&&) = default;

  uint64_t index(size_t index) override {
    assert(index < this->i_.size() && "Index out of bounds accessing Slice");
    return this->i_[index];
  }

  void prefetch(size_t _index) override {
    utility::prefetch_index(this->i_.data(), _index);
  }

  size_t size_in_bytes() override { return this->i_.size(); }

  static constexpr std::optional<DynamicContainer<T>>
  try_new(Slice<uint64_t> const vals) {
    for (auto i : vals) {
      bool const cond = i <= std::numeric_limits<T>::max();
      if (!cond) {
        fprintf(stderr, "values are larger than backing type can hold");
        return std::nullopt;
      }
    }
    std::vector<T> n;
    n.reserve(vals.size());

    for (auto e : vals) {
      n.push_back(e);
    }

    return DynamicContainer<T>(n);
  }
};

inline uint64_t low(uint64_t x) { return x; }
inline uint64_t high(uint64_t x) { return x; }

inline uint64_t low(__uint128_t x) { return (uint64_t)x; }
inline uint64_t high(__uint128_t x) { return (uint64_t)(x >> 64); }

template <typename Key, typename Ret> class KeyHasher {
public:
  // static_assert(std::is_base_of_v<Hash, Ret>, "Return number must impl
  // Hash");
  using H = Ret;
  static H hash(Key x, uint64_t seed);
};

template <typename Key>
class KeyHasherDefaultImpl : public KeyHasher<Key, uint64_t> {
public:
  static KeyHasher<Key, uint64_t>::H hash(Key &x, uint64_t seed) {
    return FxHasherDecl::hash64(x) ^ seed;
  }
};

class Reduce {
public:
  virtual size_t reduce(uint64_t h) = 0;
  virtual std::pair<size_t, uint64_t> reduce_with_remainder(uint64_t _h) = 0;
};

struct FastReduce : public Reduce {
  uint64_t d;

  FastReduce(uint64_t d) : d(d) {}
  FastReduce() : d(0) {}

  operator uint64_t() { return d; }
  operator uint64_t() const { return d; }

  size_t reduce(uint64_t h) override { return utility::mul_high(this->d, h); }

  std::pair<size_t, uint64_t> reduce_with_remainder(uint64_t h) override {
    auto r = (__uint128_t)this->d * (__uint128_t)h;
    return {r >> 64, r};
  }
};

struct FM32 : public Reduce {
  uint64_t d;
  uint64_t m;

  FM32(size_t d) : d(d), m(std::numeric_limits<uint64_t>::max() / d + 1) {
        assert(d <= std::numeric_limits<uint32_t>::max());
  }
  FM32() : FM32(0) {}

  size_t reduce(uint64_t h) override {
    auto lowbits = m * h;
    return (static_cast<__uint128_t>(lowbits) * static_cast<__uint128_t>(d)) >>
           64;
  }

  std::pair<size_t, uint64_t> reduce_with_remainder(uint64_t h) override {
    return {};
  }
};

using Rp = FastReduce;
using Rb = FastReduce;
using RemSlots = FM32;
using Pilot = uint64_t;
using PilotHash = uint64_t;

struct Row {
  size_t buckets;
  size_t elements;
  size_t elements_max;
  Pilot pilot_sum;
  Pilot pilot_max;
  size_t evictions;
  size_t evictions_max;

  void add(size_t bucket_len, Pilot pilot, size_t evictions) {
    this->buckets += 1;
    this->elements += bucket_len;
    this->elements_max = std::max(this->elements_max, bucket_len);
    this->pilot_sum += pilot;
    this->pilot_max = std::max(this->pilot_max, pilot);
    this->evictions += evictions;
    this->evictions_max = std::max(this->evictions_max, evictions);
  }
};

struct BucketStats {
  std::vector<Row> by_pct;
  std::vector<Row> by_bucket_len;

  BucketStats() : by_pct(100, Row()), by_bucket_len(100, Row()) {}

  void merge(BucketStats &other) {
    this->by_pct.resize(100);
    this->by_bucket_len.resize(
        std::max(by_bucket_len.size(), other.by_bucket_len.size()));
    for (const auto &[a, b] : zip(this->by_pct, other.by_pct)) {
      a.buckets += b.buckets;
      a.elements += b.elements;
      a.elements_max = std::max(a.elements_max, b.elements_max);
      a.pilot_sum += b.pilot_sum;
      a.pilot_max = std::max(a.pilot_max, b.pilot_max);
      a.evictions += b.evictions;
      a.evictions_max = std::max(a.evictions_max, b.evictions_max);
    }

    for (const auto &[a, b] : zip(this->by_bucket_len, other.by_bucket_len)) {
      a.buckets += b.buckets;
      a.elements += b.elements;
      a.elements_max = std::max(a.elements_max, b.elements_max);
      a.pilot_sum += b.pilot_sum;
      a.pilot_max = std::max(a.pilot_max, b.pilot_max);
      a.evictions += b.evictions;
      a.evictions_max = std::max(a.evictions_max, b.evictions_max);
    }
  }

  void add(size_t bucket_id, size_t buckets_total, size_t bucket_len,
           Pilot pilot, size_t evictions) {
    auto pct = bucket_id * 100 / buckets_total;
    this->by_pct[pct].add(bucket_len, pilot, evictions);
    if (this->by_bucket_len.size() <= bucket_len) {
      this->by_bucket_len.resize(bucket_len + 1);
    }
    this->by_bucket_len[bucket_len].add(bucket_len, pilot, evictions);
  }

  void print() {
    printf("\n");
    BucketStats::print_rows(
        Slice<Row>{this->by_pct.data(), this->by_pct.size()}, false);
    printf("\n");
  }

  static void print_rows(Slice<Row> rows, bool reverse) {
    auto x = utility::map(rows, [](auto &r) { return r.buckets; });
    auto b_total = std::accumulate(x.begin(), x.end(), 0L,
                                   [](auto a, auto &r) { return a + r; });
    auto y = utility::map(rows, [](auto &r) { return r.elements; });
    auto n = std::accumulate(y.begin(), y.end(), 0L,
                             [](auto a, auto &r) { return a + r; });
    fprintf(stderr, "%4s  %11s %7s %6s %6s %6s %10s %10s %10s %10s\n", "sz",
            "cnt", "bucket%", "cuml%", "elem%", "cuml%", "avg p", "max p",
            "avg evict", "max evict");

    auto bucket_cuml = 0;
    auto elem_cuml = 0;
    auto process_row = [&](Row &row) {
      if (row.buckets == 0) {
        return;
      }
      bucket_cuml += row.buckets;
      elem_cuml += row.elements;
      fprintf(stderr,
              "%4zu: %11zu %7.2f %6.2f %6.2f %6.2f %10.1f %10lu %10.5f %10zu\n",
              row.elements_max, row.buckets,
              (float)row.buckets / (float)b_total * 100.0f,
              (float)bucket_cuml / (float)b_total * 100.0f,
              (float)row.elements / (float)n * 100.0f,
              (float)elem_cuml / (float)n * 100.0f,
              (float)row.pilot_sum / (float)row.buckets, row.pilot_max,
              (float)row.evictions / (float)row.buckets, row.evictions_max);
    };

    if (reverse) {
      auto rev = rows;
      std::reverse(rev.begin(), rev.end());
      utility::for_each(rev, [&](auto &r) { process_row(r); });
    } else {
      utility::for_each(rows, [&](auto &r) { process_row(r); });
    }

    Pilot sum_pilots =
        std::accumulate(rows.begin(), rows.end(), 0L,
                        [](auto a, const Row &r) { return a + r.pilot_sum; });
    auto max_pilot = std::max_element(rows.begin(), rows.end(),
                                      [](const Row &a, const Row &b) {
                                        return a.pilot_max < b.pilot_max;
                                      })
                         ->pilot_max;
    size_t sum_evictions =
        std::accumulate(rows.begin(), rows.end(), 0L,
                        [](auto a, const Row &r) { return a + r.evictions; });
    auto max_evictions =
        std::max_element(rows.begin(), rows.end(),
                         [](const Row &a, const Row &b) {
                           return a.evictions_max < b.evictions_max;
                         })
            ->evictions_max;

    fprintf(stderr,
            "%4s: %11ld %7.2f %6.2f %6.2f %6.2f %10.1f %10lu %10.5f %10zu\n",
            "", b_total, 100.0f, 100.0f, 100.0f, 100.0f,
            static_cast<float>(sum_pilots) / static_cast<float>(b_total),
            max_pilot,
            static_cast<float>(sum_evictions) / static_cast<float>(b_total),
            max_evictions);
  }
};

#endif // PORT_COMMON_HPP_
