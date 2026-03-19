#ifndef PTR_HASH_HPP_
#define PTR_HASH_HPP_

#include "expected.hpp"
#include "span.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <variant>

namespace ptrhash {

LIBC_INLINE_VAR constexpr size_t shards = 1;

class fastrand {
public:
  // This seed value is very important for different inputs. Bad values are
  // known to cause compilation errors and/or incorrect computations in some
  // cases. Defaulted to 0xEF6F79ED30BA75A in the original implementation, but
  // this is not sufficient. 0x64a727ea04c46a32 is another viable seed.
  constexpr fastrand() : seed_(0xeec13c9f1362aa74) {}

  constexpr uint8_t gen_byte() { return static_cast<uint8_t>(this->gen()); }

  LIBC_INLINE constexpr uint64_t gen() {
    constexpr uint64_t WY_CONST_0 = 0x2d35'8dcc'aa6c'78a5;
    constexpr uint64_t WY_CONST_1 = 0x8bb8'4b93'962e'acc9;

    auto s = wrapping_add(seed_, WY_CONST_0);
    seed_ = s;
    auto const t =
        static_cast<__uint128_t>(s) * static_cast<__uint128_t>(s ^ WY_CONST_1);
    return static_cast<uint64_t>(t) ^ static_cast<uint64_t>(t >> 64);
  }

private:
  template <typename T> LIBC_INLINE static constexpr T wrapping_add(T a, T b) {
    while (b != 0) {
      T carry = a & b;
      a = a ^ b;
      b = carry << 1;
    }
    return a;
  }

private:
  uint64_t seed_;
};

LIBC_INLINE_VAR constexpr auto BucketIdxNONE = ~static_cast<uint32_t>(0);

template <size_t MaxSize = 5> class BinaryHeap {
public:
  constexpr BinaryHeap() = default;

  constexpr void push(const std::pair<size_t, uint32_t> &value) {
    if (current_size >= MaxSize)
      return;
    data[current_size++] = value;
  }

  constexpr std::pair<size_t, uint32_t> pop() {
    if (current_size == 0)
      return {};
    size_t max_idx = 0;
    for (size_t i = 1; i < current_size; ++i) {
      if (data[i] > data[max_idx])
        max_idx = i;
    }
    auto top = data[max_idx];
    data[max_idx] = data[current_size - 1];
    --current_size;
    return top;
  }

  constexpr const std::pair<size_t, uint32_t> &peek() const {
    size_t max_idx = 0;
    for (size_t i = 1; i < current_size; ++i) {
      if (data[i] > data[max_idx])
        max_idx = i;
    }
    return data[max_idx];
  }

  constexpr bool empty() const { return current_size == 0; }

private:
  std::array<std::pair<size_t, uint32_t>, MaxSize> data{};
  size_t current_size{};
};

template <typename T> constexpr bool is_power_of_two(T x) {
  static_assert(std::is_unsigned_v<T>,
                "is_power_of_two requires unsigned type");
  return x != 0 && (x & (x - 1)) == 0;
}

constexpr double constexpr_ln(double x) {
  if (x <= 0.0)
    return 0.0;

  // Range reduction: bring x to [0.5, 2)
  int k = 0;
  while (x > 2.0) {
    x *= 0.5;
    ++k;
  }
  while (x < 0.5) {
    x *= 2.0;
    --k;
  }

  // ln(x) ≈ 2 * sum_{n=0}^∞ (1/(2n+1)) * ((x-1)/(x+1))^(2n+1)
  const double y = (x - 1.0) / (x + 1.0);
  const double y2 = y * y;

  double term = y;
  double sum = 0.0;

  // 20 iterations ≈ double precision accuracy
  for (int n = 1; n <= 39; n += 2) {
    sum += term / n;
    term *= y2;
  }

  // ln(x) = 2 * sum + k * ln(2)
  return 2.0 * sum + k * 0.693147180559945309417232121458176568;
}

template <typename T> constexpr T constexpr_ceil(T x) {
  static_assert(std::is_floating_point_v<T>);
  long long i = static_cast<long long>(x);
  return (static_cast<T>(i) == x)
             ? x
             : (x > T{0} ? static_cast<T>(i + 1) : static_cast<T>(i));
}

LIBC_INLINE constexpr size_t get_parts(size_t n) {
  size_t parts = 0;
  auto eps = 0.01 / 2.0;
  auto x = n * eps * eps / 2.0;
  size_t target_parts = x / constexpr_ln(x);
  auto parts_per_shard = target_parts / shards;
  parts = ((parts_per_shard > 1) ? parts_per_shard : 1) * shards;
  return parts;
}

LIBC_INLINE constexpr size_t get_slots_per_part(size_t keys_per_part) {
  size_t slots_per_part = keys_per_part / 0.99;
  if (is_power_of_two(slots_per_part)) {
    slots_per_part += 1;
  }
  return slots_per_part;
}

template <size_t n> class ptrhash_config {
public:
  LIBC_INLINE_VAR static constexpr size_t parts = get_parts(n);
  LIBC_INLINE_VAR static constexpr size_t keys_per_part = n / parts;
  LIBC_INLINE_VAR static constexpr size_t parts_per_shard = parts / shards;
  LIBC_INLINE_VAR static constexpr size_t slots_per_part =
      get_slots_per_part(keys_per_part);
  LIBC_INLINE_VAR static constexpr size_t slots_total = parts * slots_per_part;
  LIBC_INLINE_VAR static constexpr size_t buckets_per_part =
      constexpr_ceil(keys_per_part / 3.0) + 3;
  LIBC_INLINE_VAR static constexpr size_t buckets_total =
      parts * buckets_per_part;
};

template <size_t n_, size_t parts_, size_t parts_per_shard_,
          size_t slots_total_, size_t buckets_total_, size_t slots_,
          size_t buckets_, typename Key = uint64_t,
          typename F = std::array<uint32_t, slots_total_ - n_>,
          typename PilotsTypeV = std::array<uint8_t, buckets_total_>>
class PtrHash {
public:
  static_assert(
      std::is_same_v<PilotsTypeV, cpp::span<uint8_t>> ||
          std::is_same_v<PilotsTypeV, std::array<uint8_t, buckets_total_>>,
      "V must be a byte slice or byte vector");

  uint64_t seed_;
  PilotsTypeV pilots_;
  F remap_;

  constexpr PtrHash(uint64_t seed_, PilotsTypeV pilots_, F remap_)
      : seed_(seed_), pilots_(pilots_), remap_(remap_) {}

  constexpr PtrHash(const PtrHash &) = default;
  constexpr PtrHash(PtrHash &&) = default;

  constexpr PtrHash &operator=(const PtrHash &) = default;
  constexpr PtrHash &operator=(PtrHash &&) = default;

  LIBC_INLINE constexpr size_t index(Key key) const {
    auto slot = this->index_no_remap(key);

    if (slot < n_) {
      return slot;
    }

    return this->remap_[slot - n_];
  }

  LIBC_INLINE constexpr size_t index_no_remap(Key key) const {
    auto hx = this->hash_key(key);
    auto b = this->bucket(hx);
    auto pilot = this->pilots_[b];
    return this->slot(hx, pilot);
  }

  constexpr size_t slot(uint64_t hx, uint64_t pilot) const {
    return (this->part(hx) * slots_) + this->slot_in_part(hx, pilot);
  }

  constexpr size_t slot_in_part(uint64_t hx, uint64_t pilot) const {
    return this->slot_in_part_hp(hx, this->hash_pilot(pilot));
  }

  constexpr std::optional<std::tuple<uint64_t, PilotsTypeV, F>>
  compute_pilots(const std::array<Key, n_> &keys) {
    std::array<std::array<bool, slots_>, parts_> taken{};
    for (std::array<bool, slots_> &t : taken) {
      for (size_t i = 0; i < slots_; i++) {
        t[i] = 0;
      }
    }
    PilotsTypeV pilots{};

    size_t tries = 0;
    constexpr size_t max_tries = 10;

    // hard code random numbers for the generator to make it simpler
    constexpr uint64_t stdrng[max_tries] = {
        0x1a275d28e2768536, 0x72737b411117ac11, 0xeb08f8fcd423148f,
        0x1d6f85975d49be9e, 0xf03250d1c097577,  0xac6e884d8db1fa90,
        0x4415d98c0c03a79f, 0xa36bfbcfddf4d5e6, 0x154aef1f436d8e98,
        0xd21f78471475f18e};

    while (true) {
      bool contd = false;
      tries += 1;
      if (tries > max_tries) {
        return {};
      }

      this->seed_ = stdrng[tries - 1];
      pilots = PilotsTypeV{0};

      for (auto &t : taken)
        for (size_t e = 0; e < pilots.size(); e++)
          t[e] = false;

      auto shard_hashes = this->shards(keys);

      const size_t pilots_chunk_size =
          std::max(buckets_ * parts_per_shard_, static_cast<size_t>(1));
      const size_t taken_chunk_size = parts_per_shard_;
      const size_t num_pilots_chunks =
          (pilots.size() + pilots_chunk_size - 1) / pilots_chunk_size;
      const size_t num_taken_chunks =
          (taken.size() + taken_chunk_size - 1) / taken_chunk_size;

      for (size_t shard = 0;
           shard <
           std::min({shard_hashes.size(), num_pilots_chunks, num_taken_chunks});
           shard++) {
        std::array<uint64_t, n_> hashes = shard_hashes[shard];

        size_t pilots_begin = shard * pilots_chunk_size;
        size_t pilots_end =
            std::min(pilots_begin + pilots_chunk_size, pilots.size());

        size_t taken_begin = shard * taken_chunk_size;
        size_t taken_end =
            std::min(taken_begin + taken_chunk_size, taken.size());

        std::optional<std::pair<std::array<uint64_t, n_>,
                                std::array<uint32_t, parts_per_shard_ + 1>>>
            sorted_parts = this->sort_parts(shard, hashes);
        if (!sorted_parts) {
          contd = true;
          break;
        }

        auto &[new_hashes, part_starts] = sorted_parts.value();

        if (!this->build_shard(shard, new_hashes, part_starts, pilots,
                               pilots_begin, pilots_end, taken_begin, taken_end,
                               taken)) {
          contd = true;
          break;
        }
      }
      if (contd) {
        continue;
      }

      auto const remap = this->remap_free_slots(taken);

      if (!remap) {
        continue;
      }
      break;
    }
    this->pilots_ = pilots;

    return {{this->seed_, this->pilots_, this->remap_}};
  }

  constexpr cpp::expected<std::monostate, std::nullopt_t>
  remap_free_slots(std::array<std::array<bool, slots_>, parts_> &taken) {
    std::array<size_t, parts_> val{};
    for (size_t i = 0; i < taken.size(); ++i) {
      size_t counter = 0;
      for (auto element : taken[i]) {
        if (!element) {
          counter++;
        }
      }
      val[i] = counter;
    }

    size_t acc = 0;
    for (const auto &item : val) {
      acc += item;
    }

    if (acc != slots_total_ - n_) {
      fprintf(stderr,
              "Not the right number of free slots left!\n total slots %zu - n "
              "%zu\n",
              slots_total_, n_);
      return cpp::unexpected(std::nullopt);
    }

    if (slots_total_ == n_) {
      return std::monostate{};
    }

    std::array<uint64_t, slots_total_ - n_> v{};
    size_t v_idx = 0;

    auto const get = [&](std::array<std::array<bool, slots_>, parts_> &t,
                         size_t idx) { return t[idx / slots_][idx % slots_]; };

    size_t p = 0;
    for (const auto &t : taken) {
      auto const offset = p * slots_;
      for (size_t idx = 0; idx < t.size(); idx++) {
        if (!t[idx]) {
          auto result = offset + idx;
          if (result < n_) {
            while (!get(taken, n_ + v_idx)) {
              v[v_idx++] = result;
            }
            v[v_idx++] = result;
          }
        }
      }
      p++;
    }

    for (size_t i = 0; i < v.size(); i++) {
      this->remap_[i] = v[i];
    }

    return std::monostate{};
  }

  constexpr auto shards(const std::array<Key, n_> &keys) const {
    return this->no_sharding(keys);
  }

  constexpr std::array<std::array<uint64_t, n_>, 1>
  no_sharding(const std::array<Key, n_> &keys) const {
    std::array<uint64_t, n_> ret;
    for (size_t i = 0; i < keys.size(); i++) {
      ret[i] = this->hash_key(keys[i]);
    }
    return {ret};
  }

  constexpr uint64_t hash_key(Key x) const {
    uint64_t value = 0;
    constexpr uint64_t bits = sizeof(uint64_t) * 8;
    value = ((value << 5) | (value >> (bits - 5))) ^ x;
    value *= 0x517cc1b727220a95;
    return value ^ this->seed_;
  }

  constexpr std::optional<std::pair<std::array<uint64_t, n_>,
                                    std::array<uint32_t, parts_per_shard_ + 1>>>
  sort_parts(size_t shard, std::array<uint64_t, n_> hashes) const {
    for (size_t i = 0; i < hashes.size(); i++) {
      for (size_t j = i + 1; j < hashes.size(); j++) {
        if (hashes[i] > hashes[j]) {
          auto temp = hashes[i];
          hashes[i] = hashes[j];
          hashes[j] = temp;
        }
      }
    }

    bool distinct = true;
    for (size_t i = 1; i < hashes.size(); ++i) {
      if (hashes[i] == hashes[i - 1]) {
        distinct = false;
        break;
      }
    }

    if (!distinct) {
      fprintf(stderr, "Hashes are not distinct\n");
      return std::nullopt;
    }

    if (!hashes.empty()) {
      assert(shard * parts_per_shard_ <= this->part(hashes[0]));
      assert(this->part(hashes[hashes.size() - 1]) <
             (shard + 1) * parts_per_shard_);
    }

    std::array<uint32_t, parts_per_shard_ + 1> part_starts{};

    for (size_t part_in_shard = 1; part_in_shard < parts_per_shard_ + 1;
         ++part_in_shard) {
      auto it = std::lower_bound(
          hashes.begin(), hashes.end(),
          shard * parts_per_shard_ + part_in_shard,
          [&](uint64_t h, uint64_t k) { return this->part(h) < k; });

      part_starts[part_in_shard] = std::distance(hashes.begin(), it);
    }
    size_t max_part_len = 0;
    for (size_t i = 0; i + 1 < part_starts.size(); ++i) {
      auto start = part_starts[i];
      auto end = part_starts[i + 1];

      size_t len = end - start;
      max_part_len = std::max<size_t>(max_part_len, len);
    }

    if (max_part_len > slots_) {
      return std::nullopt;
    }

    return std::pair(hashes, part_starts);
  }

  constexpr bool
  build_shard(size_t shard, std::array<uint64_t, n_> &hashes,
              std::array<uint32_t, parts_per_shard_ + 1> &part_starts,
              PilotsTypeV &pilots, size_t pilots_begin, size_t pilots_end,
              size_t taken_begin, size_t taken_end,
              std::array<std::array<bool, slots_>, parts_> &taken) const {

    size_t pilots_chunk_size = pilots_end - pilots_begin;

    auto part_in_shard = 0;
    for (size_t taken_idx = taken_begin; taken_idx < taken_end; ++taken_idx) {
      const auto num_chunks = pilots_chunk_size / buckets_;
      for (size_t i = 0; i < num_chunks; ++i) {
        size_t target_pilots_begin = pilots_begin + i * buckets_;
        size_t target_pilots_end =
            std::min(target_pilots_begin + buckets_, pilots_end);
        auto part = shard * parts_per_shard_ + part_in_shard;

        auto _cnt = this->build_part(
            part,
            cpp::span<uint64_t>(hashes).subspan(part_starts[part_in_shard],
                                                part_starts[part_in_shard + 1] -
                                                    part_starts[part_in_shard]),
            cpp::span(
                const_cast<uint8_t *>(pilots.data() + target_pilots_begin),
                target_pilots_end - target_pilots_begin),
            taken[taken_idx]);
        if (!_cnt) {
          return false;
        }
      }
      part_in_shard++;
    }

    return true;
  }

  constexpr std::optional<size_t>
  build_part(size_t part, cpp::span<uint64_t> hashes, cpp::span<uint8_t> pilots,
             std::array<bool, slots_> &taken) const {
    std::pair<std::array<uint32_t, buckets_ + 1>,
              std::array<uint32_t, buckets_>>
        sorted_buckets = this->sort_buckets(part, hashes);
    std::array<uint32_t, buckets_ + 1> starts = sorted_buckets.first;
    std::array<uint32_t, buckets_> bucket_order = sorted_buckets.second;

    auto kmax = 256u;

    std::array<uint32_t, slots_> slots{};
    for (size_t i = 0; i < slots_; i++) {
      slots[i] = BucketIdxNONE;
    }

    auto bucket_len = [&](uint32_t b) constexpr -> size_t {
      return starts[b + 1] - starts[b];
    };

    auto heap = BinaryHeap();

    auto duplicate_slots = [&](uint32_t b, uint64_t p) constexpr {
      auto hp = this->hash_pilot(p);
      auto hashes_range = hashes.subspan(starts[b], starts[b + 1] - starts[b]);

      auto i = 0;
      for (auto const &e1 : hashes_range) {
        auto hx = this->slot_in_part_hp(e1, hp);
        for (auto e2 : hashes_range.subspan(i + 1)) {
          auto hy = this->slot_in_part_hp(e2, hp);
          if (hx == hy) {
            return true;
          }
        }
        i++;
      }
      return false;
    };

    std::array<uint32_t, 16> recent{
        {BucketIdxNONE, BucketIdxNONE, BucketIdxNONE, BucketIdxNONE,
         BucketIdxNONE, BucketIdxNONE, BucketIdxNONE, BucketIdxNONE,
         BucketIdxNONE, BucketIdxNONE, BucketIdxNONE, BucketIdxNONE,
         BucketIdxNONE, BucketIdxNONE, BucketIdxNONE, BucketIdxNONE}};
    size_t total_evictions = 0;

    auto rng = fastrand();

    for (size_t iter_num = 0; iter_num < bucket_order.size(); iter_num++) {
      auto const &new_b = bucket_order[iter_num];
      auto const new_bucket =
          hashes.subspan(starts[new_b], starts[new_b + 1] - starts[new_b]);
      if (new_bucket.empty()) {
        pilots[new_b] = 0;
        continue;
      }
      auto const new_b_len = new_bucket.size();
      size_t evictions = 0;

      heap.push({new_b_len, new_b});
      for (size_t i = 0; i < recent.size(); ++i) {
        recent[i] = BucketIdxNONE;
      }
      auto recent_idx = 0;
      recent[0] = new_b;

      while (!heap.empty()) {
        auto const &[b_len, b] = heap.peek();
        heap.pop();
        if (evictions > slots_ && is_power_of_two((evictions))) {
          if (evictions >= 10 * slots_) {
            std::cout << "iter num " << iter_num << std::endl;
            return std::nullopt;
          }
        }
        auto const bucket =
            hashes.subspan(starts[b], starts[b + 1] - starts[b]);
        if (auto fpilot = this->find_pilot(kmax, bucket, taken)) {
          auto &[p, hp] = fpilot.value();
          pilots[b] = static_cast<uint8_t>(p);
          for (auto &item :
               hashes.subspan(starts[b], starts[b + 1] - starts[b])) {
            auto p = this->slot_in_part_hp(item, hp);
            slots[p] = b;
          }
          continue;
        }
        uint64_t p0 = rng.gen_byte();
        std::pair best = {std::numeric_limits<size_t>::max(),
                          std::numeric_limits<uint64_t>::max()};
        for (size_t delat = 0; delat < kmax; ++delat) {
          bool build_part_loop_continue_inner_continue = false;
          auto const p = (p0 + delat) % kmax;
          auto const hp = this->hash_pilot(p);
          size_t collision_score = 0;
          for (auto &item :
               hashes.subspan(starts[b], starts[b + 1] - starts[b])) {
            auto p = this->slot_in_part_hp(item, hp);
            auto const s = slots[p];
            size_t new_score = 0;
            if (s == BucketIdxNONE) {
              continue;
            } else {
              bool found = false;
              for (auto it : recent) {
                found = found || it == s;
              }
              if (found) {
                build_part_loop_continue_inner_continue = true;
                break;
              } else {
                auto const len = bucket_len(s);
                new_score = len * len;
              }
            }
            collision_score += new_score;
            if (collision_score >= std::get<0>(best)) {
              build_part_loop_continue_inner_continue = true;
              break;
            }
          }
          if (build_part_loop_continue_inner_continue) {
            continue;
          }
          if (!duplicate_slots(b, p)) {
            best.first = collision_score;
            best.second = p;
            if (collision_score == new_b_len * new_b_len) {
              break;
            }
          }
        }
        if (best == std::pair{std::numeric_limits<size_t>::max(),
                              std::numeric_limits<uint64_t>::max()}) {
          return std::nullopt;
        }

        auto const &[_collision_score, p] = best;
        pilots[b] = static_cast<uint8_t>(p);
        auto const hp = this->hash_pilot(p);
        for (auto &item :
             hashes.subspan(starts[b], starts[b + 1] - starts[b])) {
          auto slot = this->slot_in_part_hp(item, hp);
          auto const b2 = slots[slot];
          if (b2 != BucketIdxNONE) {
            assert(b2 != b);
            heap.push({bucket_len(b2), b2});
            evictions++;

            auto hp = this->hash_pilot(static_cast<uint64_t>(pilots[b2]));
            for (auto &item :
                 hashes.subspan(starts[b2], starts[b2 + 1] - starts[b2])) {
              auto p2 = this->slot_in_part_hp(item, hp);
              slots[p2] = BucketIdxNONE;
              taken[p2] = false;
            }
          }
          slots[slot] = b;
          taken[slot] = true;
        }

        recent_idx++;
        recent_idx %= recent.size();
        recent[recent_idx] = b;
      }

      total_evictions += evictions;
    }
    return total_evictions;
  }

  LIBC_INLINE constexpr std::optional<std::pair<uint64_t, uint64_t>>
  find_pilot(uint64_t kmax, cpp::span<uint64_t> bucket,
             std::array<bool, slots_> &taken) const {
    auto const r = bucket.size() / 4 * 4;
    for (size_t p = 0; p < kmax; p++) {
      bool find_pilot_continue = false;
      auto const hp = this->hash_pilot(p);
      auto const check = [&](uint64_t hx) {
        return taken[this->slot_in_part_hp(hx, hp)];
      };
      auto bad = false;
      for (size_t i = 0; i < r; i++) {
        if (check(bucket[i])) {
          find_pilot_continue = true;
          break;
        }
      }
      if (find_pilot_continue) {
        continue;
      }
      for (auto hx : bucket.subspan(r)) {
        bad |= check(hx);
      }
      if (bad) {
        continue;
      }

      if (this->try_take_pilot(bucket, hp, taken)) {
        return std::pair(p, hp);
      }
    }
    return std::nullopt;
  }

  constexpr bool try_take_pilot(cpp::span<uint64_t> bucket, uint64_t hp,
                                std::array<bool, slots_> &taken) const {
    for (size_t i = 0; i < bucket.size(); i++) {
      size_t hx = bucket[i];
      auto const slot = this->slot_in_part_hp(hx, hp);
      if (taken[slot]) {
        for (auto hx : bucket.subspan(0, i)) {
          taken[this->slot_in_part_hp(hx, hp)] = false;
        }
        return false;
      }
      taken[slot] = true;
    }
    return true;
  }

  constexpr uint64_t hash_pilot(uint64_t p) const {
    uint64_t C = 0x517cc1b727220a95;
    uint64_t b = p ^ this->seed_;
    uint64_t result = 0;

    while (b != 0) {
      if (b & 1) {
        result = result + C;
      }
      C = C << 1;
      b >>= 1;
    }

    return result;
  }

  constexpr size_t slot_in_part_hp(uint64_t hx, uint64_t hp) const {
    uint64_t d =
        std::max(ptrhash_config<n_>::slots_per_part, static_cast<size_t>(1));
    uint64_t m = std::numeric_limits<uint64_t>::max() / d + 1;
    auto lowbits = m * (hx ^ hp);
    return (static_cast<__uint128_t>(lowbits) * static_cast<__uint128_t>(d)) >>
           64;
  }

  constexpr std::pair<std::array<uint32_t, buckets_ + 1>,
                      std::array<uint32_t, buckets_>>
  sort_buckets(size_t part, cpp::span<uint64_t> hashes) const {
    std::array<uint32_t, buckets_ + 1> bucket_starts{};
    size_t bucket_starts_idx = 0;
    std::array<uint32_t, buckets_> order{};
    for (size_t i = 0; i < order.size(); ++i) {
      order[i] = BucketIdxNONE;
    }
    std::array<size_t, 32> bucket_len_cnt = {0};

    size_t end = 0;
    bucket_starts[bucket_starts_idx++] = end;

    for (size_t b = 0; b < buckets_; b++) {
      auto start = end;
      while (end < hashes.size() &&
             this->bucket(hashes[end]) == part * buckets_ + b) {
        end++;
      }

      auto l = end - start;
      bucket_len_cnt[l]++;
      bucket_starts[bucket_starts_idx++] = end;
    }

    assert(end == hashes.size());

    auto max_bucket_size = bucket_len_cnt.size() - 1;
    auto const expected_bucket_size = slots_ / buckets_;
    //  "Part {part}: Bucket size {max_bucket_size} is too much "
    //  "larger than the expected size of {expected_bucket_size}."
    assert(max_bucket_size <= (20. * expected_bucket_size));
    auto acc = 0;
    for (int i = max_bucket_size; i > -1; i--) {
      auto tmp = bucket_len_cnt[i];
      bucket_len_cnt[i] = acc;
      acc += tmp;
    }
    for (size_t b = 0; b < buckets_; b++) {
      size_t l = bucket_starts[b + 1] - bucket_starts[b];
      order[bucket_len_cnt[l]] = b;
      bucket_len_cnt[l] += 1;
    }

    return {bucket_starts, order};
  }

  constexpr size_t part(uint64_t hx) const {

    return (static_cast<__uint128_t>(ptrhash_config<n_>::parts) *
            static_cast<__uint128_t>(hx)) >>
           64;
  }

  constexpr size_t bucket(uint64_t hx) const {
    return (static_cast<__uint128_t>(ptrhash_config<n_>::buckets_total) *
            static_cast<__uint128_t>(hx)) >>
           64;
  }
};

template <size_t n, typename Key = uint64_t>
LIBC_INLINE constexpr auto get_params(const std::array<Key, n> &keys) {
  using F = std::array<uint32_t, ptrhash_config<n>::slots_total - n>;
  using PilotsTypeV = std::array<uint8_t, ptrhash_config<n>::buckets_total>;

  auto p =
      PtrHash<n, ptrhash_config<n>::parts, ptrhash_config<n>::parts_per_shard,
              ptrhash_config<n>::slots_total, ptrhash_config<n>::buckets_total,
              ptrhash_config<n>::slots_per_part,
              ptrhash_config<n>::buckets_per_part, Key, F, PilotsTypeV>(
          0, PilotsTypeV(), F());
  auto result = p.compute_pilots(keys);

  assert(result && "Unable to construct PtrHash after 10 tries. Try using a "
                   "better hash or decreasing lambda.\n");

  auto &[seed, pilots, remap] = result.value();

  return std::tuple<uint64_t, PilotsTypeV, F>{seed, pilots, remap};
}

template <size_t n,
          typename PilotsTypeV =
              std::array<uint8_t, ptrhash_config<n>::buckets_total>,
          typename F = std::array<uint32_t, ptrhash_config<n>::slots_total - n>>
LIBC_INLINE constexpr auto init_hasher(size_t seed, PilotsTypeV pilots,
                                       F remap) {
  return PtrHash<
      n, ptrhash_config<n>::parts, ptrhash_config<n>::parts_per_shard,
      ptrhash_config<n>::slots_total, ptrhash_config<n>::buckets_total,
      ptrhash_config<n>::slots_per_part, ptrhash_config<n>::buckets_per_part>(
      seed, pilots, remap);
}

template <size_t Capacity, class Hasher> class PerfectHashMap {
public:
  struct Entry {
    wint_t key : 21;
    wint_t value : 21;
    constexpr Entry() = default;
    constexpr Entry(wint_t key, wint_t value) : key(key), value(value) {}
  };

  constexpr PerfectHashMap(
      const std::array<std::array<wint_t, 2>, Capacity> &pairs,
      const Hasher &hasher_)
      : hasher_(hasher_) {
    for (auto &[key, value] : pairs) {
      auto const idx = hasher_.index(key);
      assert(idx < Capacity && "Index out of bounds");
      this->entries_[idx] = Entry{key, value};
    }
  }

  LIBC_INLINE constexpr std::optional<wint_t> find(const wint_t key) const {
    size_t idx = hasher_.index(key);
    if (idx >= Capacity)
      return std::nullopt;

    const Entry &e = entries_[idx];
    if (e.key != key)
      return std::nullopt;

    return e.value;
  }

  LIBC_INLINE constexpr bool contains(const wint_t key) const {
    return this->find(key).has_value();
  }

  LIBC_INLINE constexpr size_t size() { return Capacity; }

private:
  Entry entries_[Capacity];
  const Hasher &hasher_;
};

} // namespace ptrhash

#endif // PTR_HASH_HPP_
