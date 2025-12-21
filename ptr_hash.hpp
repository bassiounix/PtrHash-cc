#ifndef PORT_PTR_HASH_HPP_
#define PORT_PTR_HASH_HPP_

#include "binary_heap.hpp"
#include "bucket_fn.hpp"
#include "bucket_idx.hpp"
#include "common.hpp"
#include "enumerate.hpp"
#include "fastrand.hpp"
#include "math.hpp"
#include "ptr_hash_params.hpp"
#include "rngs.hpp"
#include "slice.hpp"
#include "zip.hpp"
#include <array>
#include <cstdlib>
#include <print>
#include <variant>

template <typename BF, PtrHashParams<BF> params_, size_t n_, size_t parts_,
          size_t shards_, size_t parts_per_shard_, size_t slots_total_,
          size_t buckets_total_, size_t slots_, size_t buckets_, Rp rem_shards_,
          Rp rem_parts_, Rb rem_buckets_, Rb rem_buckets_total_,
          RemSlots rem_slots_, typename Key = uint64_t,
          typename F = DynamicContainer<uint32_t>, // or Vec<u32>
          typename Hx =
              KeyHasherDefaultImpl<Key>,    // -> FxHasherDecl::FxHasher64,
          typename V = std::vector<uint8_t> // or Vec<u8>,
          >
class PtrHash {
public:
  // static_assert(std::is_base_of_v<KeyT, Key>, "Key must implement KeyT");
  static_assert(std::is_base_of_v<BucketFn, BF>, "BF must implement BucketFn");
  static_assert(std::is_base_of_v<Packed, F>, "F must implement Packed");
  static_assert(std::is_base_of_v<KeyHasher<Key, uint64_t>, Hx>,
                "Hx must implement KeyHasher<Key>");
  static_assert(std::is_same_v<V, Slice<uint8_t>> ||
                    std::is_same_v<V, std::vector<uint8_t>>,
                "V must be a byte slice or byte vector");

  mutable uint64_t seed_;
  mutable V pilots_;
  mutable F remap_;

  constexpr PtrHash(uint64_t seed_, V pilots_, F remap_)
      : seed_(seed_), pilots_(pilots_), remap_(remap_) {}

  constexpr PtrHash(const PtrHash &) = default;
  constexpr PtrHash(PtrHash &&) = default;

  constexpr PtrHash &operator=(const PtrHash &) = default;
  constexpr PtrHash &operator=(PtrHash &&) = default;

  inline constexpr size_t index(Key key) const {
    auto slot = this->index_no_remap(key);

    if (slot < n_) {
      return slot;
    }

    return this->remap_.index(slot - n_);
  }

  inline constexpr size_t index_no_remap(Key key) const {
    auto hx = this->hash_key(key);
    auto b = this->bucket(hx);
    auto pilot = this->pilots_[b];
    return this->slot(hx, pilot);
  }

  constexpr size_t slot(typename Hx::H hx, uint64_t pilot) const {
    return (this->part(hx) * slots_) + this->slot_in_part(hx, pilot);
  }

  constexpr size_t slot_in_part(typename Hx::H hx, Pilot pilot) const {
    return this->slot_in_part_hp(hx, this->hash_pilot(pilot));
  }

  constexpr std::optional<std::monostate>
  compute_pilots(Slice<Key> keys) const {
    std::vector<std::vector<bool>> taken{};
    std::vector<uint8_t> pilots{};

    auto tries = 0;
    constexpr size_t max_tries = 10;

    constexpr auto rng = rngs::StdRng::from_seed(31415);
    while (true) {
      bool contd = false;
      tries += 1;
      // std::println("Try num {}", tries);
      if (tries > max_tries) {
        return std::nullopt;
      }

      this->seed_ = rng.random();
      // std::println("seed is {}", this->seed_);
      pilots.clear();
      utility::resize_with(pilots, buckets_total_, []() { return 0; });
      for (auto &t : taken) {
        t.clear();
        utility::resize_with(t, slots_, []() { return false; });
      }
      utility::resize_with(taken, parts_,
                           [&]() { return std::vector<bool>(slots_, 0); });
      auto shard_hashes = this->shards(keys);
      auto shard_pilots =
          utility::chunks_mut(pilots, std::max(buckets_ * parts_per_shard_,
                                               static_cast<size_t>(1)));
      auto shard_taken = utility::chunks_mut(taken, parts_per_shard_);

      for (size_t shard = 0;
           shard < std::min({shard_hashes.size(), shard_pilots.size(),
                             shard_taken.size()});
           shard++) {
        auto hashes = shard_hashes[shard];
        auto pilots = shard_pilots[shard];
        auto taken = shard_taken[shard];
        auto sorted_parts =
            this->sort_parts(shard, Slice(hashes.data(), hashes.size()));
        if (!sorted_parts) {
          contd = true;
          break;
        }

        auto &[new_hashes, part_starts] = sorted_parts.value();

        if (!this->build_shard(shard, new_hashes,
                               Slice(part_starts.data(), part_starts.size()),
                               Slice(pilots.data(), pilots.size()),
                               Slice(taken.data(), taken.size()))) {
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

    return std::monostate{};
  }

  constexpr std::expected<std::monostate, std::nullopt_t>
  remap_free_slots(std::vector<std::vector<bool>> &taken) const {
    auto val =
        utility::map(taken, [&](auto &t) { return utility::count_zeros(t); });
    if (utility::sum(Slice(val.data(), val.size())) != slots_total_ - n_) {

      fprintf(stderr,
              "Not the right number of free slots left!\n total slots %zu - n "
              "%zu\n",
              slots_total_, n_);
      assert(0);
    }

    if (!params_.remap || slots_total_ == n_) {
      return std::monostate{};
    }

    std::vector<uint64_t> v;
    v.reserve(slots_total_ - n_);

    auto const get = [&](std::vector<std::vector<bool>> &t, size_t idx) {
      return t[idx / slots_][idx % slots_];
    };

    for (auto i : utility::take_while(
             utility::flat_map(enumerate(taken),
                               [&](auto x) {
                                 auto [p, t] = x;
                                 auto const offset = p * slots_;
                                 return utility::map(
                                     utility::iter_zeros(t),
                                     [&](size_t i) { return offset + i; });
                               }),
             [&](auto &i) { return i < n_; })) {
      while (!get(taken, n_ + v.size())) {
        v.push_back(i);
      }
      v.push_back(i);
    }
    this->remap_ = F::try_new(Slice(v.data(), v.size()));
    return std::monostate{};
  }

  constexpr auto shards(Slice<Key> keys) const {
    switch (params_.sharding.type) {
    case ShardingType::None:
      return this->no_sharding(keys);
    case ShardingType::Memory:
      return this->shard_keys_in_memory(keys);
      // We don't need this for compile time data
      // case ShardingType::Disk:
      //   return this->shard_keys_hybrid(std::numeric_limits<size_t>::max(),
      //   keys);
      // case ShardingType::Hybrid:
      //   return this->shard_keys_hybrid(params_.sharding.mem, keys);
    }
  }

  constexpr std::vector<std::vector<typename Hx::H>>
  no_sharding(Slice<Key> keys) const {
    return {utility::map(keys, [&](Key key) { return this->hash_key(key); })};
  }

  constexpr std::vector<std::vector<typename Hx::H>>
  shard_keys_in_memory(Slice<Key> keys) const {
    return utility::map(utility::Range(shards_), [&](size_t shard) {
      std::vector<typename Hx::H> res =
          utility::map(keys, [&](auto key) -> typename Hx::H {
            return this->hash_key(key);
          });
      std::vector<typename Hx::H> hashes = utility::filter(
          res, [&](typename Hx::H h) { return this->shard(h) == shard; });
      return hashes;
    });
  }

  constexpr typename Hx::H hash_key(Key x) const {
    return Hx::hash(x, this->seed_);
  }

  constexpr size_t shard(typename Hx::H hx) const {
    return rem_shards_.reduce(high(hx));
  }

  constexpr std::optional<
      std::pair<Slice<typename Hx::H>, std::vector<uint32_t>>>
  sort_parts(size_t shard, Slice<typename Hx::H> hashes) const {
    std::sort(hashes.begin(), hashes.end());

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

    if (!hashes.is_empty()) {
      // assert(shard * parts_per_shard_ <= this->part(hashes[0]));
      // assert(this->part(hashes.last()) < (shard + 1) *
      // parts_per_shard_);
    }

    std::vector<uint32_t> part_starts(parts_per_shard_ + 1, 0);

    for (auto part_in_shard : utility::Range(1, parts_per_shard_ + 1)) {
      part_starts[part_in_shard] =
          hashes
              .binary_search_by([&](typename Hx::H h) {
                if (this->part(h) < shard * parts_per_shard_ + part_in_shard) {
                  return Ordering::Less;
                } else {
                  return Ordering::Greater;
                }
              })
              .error();
    }
    auto max_part_len = 0;
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

  constexpr std::optional<std::monostate>
  build_shard(size_t shard, Slice<typename Hx::H> hashes,
              Slice<uint32_t> part_starts, Slice<uint8_t> pilots,
              Slice<std::vector<bool>> taken) const {
    auto pilots_per_part = utility::chunks_exact_mut(pilots, buckets_);
    auto parts_done = shard * parts_per_shard_;
    auto ok = utility::try_for_each(
        enumerate(zip(pilots_per_part, taken)),
        [&](auto e) -> std::optional<std::monostate> {
          auto &[part_in_shard, y] = e;
          auto &[pilots, taken] = y;
          auto part = shard * parts_per_shard_ + part_in_shard;
          hashes = hashes.range(part_starts[part_in_shard],
                                part_starts[part_in_shard + 1]);
          auto _cnt = this->build_part(part, hashes, pilots, taken);
          if (!_cnt) {
            return std::nullopt;
          }
          parts_done++;
          return std::monostate();
        });

    if (!ok) {
      return std::nullopt;
    }

    // assert(parts_done == (shard + 1) * parts_per_shard_);
    return std::monostate{};
  }

  constexpr std::optional<size_t> build_part(size_t part,
                                             Slice<typename Hx::H> hashes,
                                             Slice<uint8_t> pilots,
                                             std::vector<bool> &taken) const {
    auto const &[starts, bucket_order] = this->sort_buckets(part, hashes);

    auto kmax = 256;

    std::vector<BucketIdx> slots(slots_, BucketIdx::NONE);
    constexpr size_t one = 1;
    auto bucket_len = [&](BucketIdx b) -> size_t {
      return starts[b + one] - starts[b];
    };

    auto max_bucket_len = bucket_len(bucket_order[0]);
    constexpr BinaryHeap<std::pair<size_t, BucketIdx>> stack;

    auto slots_for_bucket = [&](BucketIdx b, Pilot p) {
      auto hp = this->hash_pilot(p);
      return utility::map(
          hashes.range(starts[b], starts[b + one]),
          [&](typename Hx::H hx) { return this->slot_in_part_hp(hx, hp); });
    };
    std::vector<size_t> slots_tmp(max_bucket_len, 0);
    auto duplicate_slots = [&](BucketIdx b, Pilot p) {
      slots_tmp.clear();
      auto slots = slots_for_bucket(b, p);
      slots_tmp.insert(slots_tmp.end(), slots.begin(), slots.end());
      std::sort(slots_tmp.begin(), slots_tmp.end());
      return utility::has_duplicates(slots_tmp);
    };

    std::array<BucketIdx, 16> recent{
        {BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE,
         BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE,
         BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE,
         BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE, BucketIdx::NONE}};
    size_t total_evictions = 0;

    constexpr auto rng = fastrand::Rng();

    for (auto const &[iter_num, new_b] : enumerate(bucket_order)) {
      auto const new_bucket = hashes.range(starts[new_b], starts[new_b + one]);
      if (new_bucket.is_empty()) {
        pilots[new_b] = 0;
        continue;
      }
      auto const new_b_len = new_bucket.size();
      size_t evictions = 0;

      stack.push({new_b_len, new_b});
      // std::print("stack(push): ");
      // for (auto x : stack) {
      //   std::print("({} {}) ", std::get<0>(x), std::get<1>(x).i_);
      // }
      // std::println();
      recent.fill(BucketIdx::NONE);
      auto recent_idx = 0;
      recent[0] = new_b;

      while (!stack.empty()) {
        auto const &[b_len, b] = stack.peek();
        stack.pop();
        // std::print("stack(pop): ");
        // for (auto x : stack.data) {
        //   std::print("({} {}) ", std::get<0>(x), std::get<1>(x).i_);
        // }
        // std::println();
        bool continue_outer_loop = false;
        if (evictions > slots_ && utility::is_power_of_two((evictions))) {
          if (evictions >= 10 * slots_) {
            std::println("iter num {}", iter_num);
            return std::nullopt;
          }
        }
        auto const bucket = hashes.range(starts[b], starts[b + one]);
        auto const b_slots = [&](PilotHash hp) {
          return utility::map(bucket, [&](typename Hx::H hx) {
            return this->slot_in_part_hp(hx, hp);
          });
        };
        if (auto fpilot = this->find_pilot(kmax, bucket, taken)) {
          auto &[p, hp] = fpilot.value();
          pilots[b] = static_cast<uint8_t>(p);
          for (auto &p : b_slots(hp)) {
            slots[p] = b;
          }
          continue;
        }
        uint64_t p0 = rng.u8();
        std::pair best = {std::numeric_limits<size_t>::max(),
                          std::numeric_limits<uint64_t>::max()};
        // std::println("rng.u8 {}", p0);
        for (auto delat : utility::Range(kmax)) {
          bool build_part_loop_continue_inner_continue = false;
          auto const p = (p0 + delat) % kmax;
          auto const hp = this->hash_pilot(p);
          auto collision_score = 0;
          for (auto &p : b_slots(hp)) {
            auto const s = slots[p];
            size_t new_score = 0;
            if (s.is_none()) {
              continue;
            } else if (Slice(recent.data(), recent.size()).contains(s)) {
              build_part_loop_continue_inner_continue = true;
              break;
            } else {
              auto const len = bucket_len(s);
              new_score = len * len;
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
            best = {collision_score, p};
            if (collision_score == new_b_len * new_b_len) {
              break;
            }
          }
        }
        if (best == std::pair{std::numeric_limits<size_t>::max(),
                              std::numeric_limits<uint64_t>::max()}) {
          auto const slots = b_slots(0);
          auto const len = bucket.size();
          auto const num_slots = slots_;
          fprintf(stderr,
                  "part %zu: bucket of size %zu with %zu slots: "
                  "Indistinguishable hashes in bucket!",
                  part, len, num_slots);
          for (auto [hx, slot] : zip(bucket, slots)) {
            fprintf(stderr, "%zu -> slot %zu", hx, slot);
          }
          fprintf(stderr,
                  "part %zu: bucket of size %zu with %zu slots: "
                  "Indistinguishable hashes in bucket!",
                  part, len, num_slots);
          return std::nullopt;
        }

        auto const &[_collision_score, p] = best;
        pilots[b] = static_cast<uint8_t>(p);
        auto const hp = this->hash_pilot(p);
        for (auto &slot : b_slots(hp)) {
          auto const b2 = slots[slot];
          if (b2.is_some()) {
            assert(b2 != b);
            stack.push({bucket_len(b2), b2});
            // std::print("stack(push): ");
            // for (auto x : stack.data) {
            //   std::print("({} {}) ", std::get<0>(x), std::get<1>(x).i_);
            // }
            // std::println();
            evictions++;
            for (auto &p2 :
                 slots_for_bucket(b2, static_cast<Pilot>(pilots[b2]))) {
              slots[p2] = BucketIdx::NONE;
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

  constexpr std::optional<std::pair<Pilot, PilotHash>>
  find_pilot(uint64_t kmax, Slice<typename Hx::H> bucket,
             std::vector<bool> &taken) const {
    switch (bucket.size()) {
    case 1:
      return this->find_pilot_array<1>(kmax, bucket, taken);
    case 2:
      return this->find_pilot_array<2>(kmax, bucket, taken);
    case 3:
      return this->find_pilot_array<3>(kmax, bucket, taken);
    case 4:
      return this->find_pilot_array<4>(kmax, bucket, taken);
    case 5:
      return this->find_pilot_array<5>(kmax, bucket, taken);
    case 6:
      return this->find_pilot_array<6>(kmax, bucket, taken);
    case 7:
      return this->find_pilot_array<7>(kmax, bucket, taken);
    case 8:
      return this->find_pilot_array<8>(kmax, bucket, taken);
    default:
      return this->find_pilot_slice(kmax, bucket, taken);
    };
  }

  template <const size_t L>
  constexpr std::optional<std::pair<Pilot, PilotHash>>
  find_pilot_array(uint64_t kmax, Slice<typename Hx::H> bucket,
                   std::vector<bool> &taken) const {
    auto cpy = bucket;
    cpy.len = L;
    return this->find_pilot_slice(kmax, cpy, taken);
  }

  inline constexpr std::optional<std::pair<Pilot, PilotHash>>
  find_pilot_slice(uint64_t kmax, Slice<typename Hx::H> bucket,
                   std::vector<bool> &taken) const {
    auto const r = bucket.size() / 4 * 4;
    for (auto p : utility::Range(kmax)) {
      bool find_pilot_continue = false;
      auto const hp = this->hash_pilot(p);
      auto const check = [&](typename Hx::H hx) {
        return taken[this->slot_in_part_hp(hx, hp)];
      };
      auto bad = false;
      for (auto i : utility::Range(0, r, 4)) {
        std::array<bool, 4> checks{{
            check(bucket[i]),
            check(bucket[i + 1]),
            check(bucket[i + 2]),
            check(bucket[i + 3]),
        }};
        for (auto bad : checks) {
          if (bad) {
            find_pilot_continue = true;
            break;
          }
        }
        if (find_pilot_continue) {
          break;
          ;
        }
      }
      if (find_pilot_continue) {
        continue;
      }
      for (auto hx : bucket.sub(r)) {
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

  constexpr bool try_take_pilot(Slice<typename Hx::H> bucket, PilotHash hp,
                                std::vector<bool> &taken) const {
    for (auto [i, hx] : enumerate(bucket)) {
      auto const slot = this->slot_in_part_hp(hx, hp);
      if (taken[slot]) {
        for (auto hx : bucket.range(0, i)) {
          taken[this->slot_in_part_hp(hx, hp)] = false;
        }
        return false;
      }
      taken[slot] = true;
    }
    return true;
  }

  constexpr PilotHash hash_pilot(Pilot p) const {
    return utility::wrapping_mul(utility::C, p ^ this->seed_);
  }

  constexpr size_t slot_in_part_hp(typename Hx::H hx, PilotHash hp) const {
    return rem_slots_.reduce(low(hx) ^ hp);
  }

  constexpr std::pair<std::vector<uint32_t>, std::vector<BucketIdx>>
  sort_buckets(size_t part, Slice<typename Hx::H> hashes) const {
    std::vector<uint32_t> bucket_starts;
    bucket_starts.reserve(buckets_ + 1);
    std::vector<BucketIdx> order(buckets_, BucketIdx::NONE);
    std::vector<size_t> bucket_len_cnt(32, 0);

    auto end = 0;
    bucket_starts.push_back(end);

    for (auto b : utility::Range(buckets_)) {
      auto start = end;
      while (end < hashes.size() &&
             this->bucket(hashes[end]) == part * buckets_ + b) {
        end++;
      }

      auto l = end - start;
      if (l >= bucket_len_cnt.size()) {
        utility::resize_with(bucket_len_cnt, l + 1, []() { return 0; });
      }
      bucket_len_cnt[l]++;
      bucket_starts.push_back(end);
    }

    // assert(end == hashes.size());

    auto max_bucket_size = bucket_len_cnt.size() - 1;
    // if (false) {
    //   auto const expected_bucket_size = slots_ / buckets_;
    //   static_assert(max_bucket_size <= (20. * expected_bucket_size),
    //                 "Part {part}: Bucket size {max_bucket_size} is too much "
    //                 "larger than the expected size of
    //                 {expected_bucket_size}.");
    // }
    auto acc = 0;
    for (auto i : utility::Range(max_bucket_size + 1).rev()) {
      auto tmp = bucket_len_cnt[i];
      bucket_len_cnt[i] = acc;
      acc += tmp;
    }
    constexpr size_t one = 1;
    for (auto &b : utility::Range(buckets_)) {
      size_t l = bucket_starts[b + one] - bucket_starts[b];
      order[bucket_len_cnt[l]] = b;
      bucket_len_cnt[l] += 1;
    }

    return {bucket_starts, order};
  }

  constexpr size_t part(typename Hx::H hx) const {
    return rem_parts_.reduce(high(hx));
  }

  constexpr size_t bucket_in_part(uint64_t x) const {
    if (BF::LINEAR) {
      return rem_buckets_.reduce(x);
    } else if (BF::B_OUTPUT) {
      return params_.bucket_fn.call(x);
    } else {
      return rem_buckets_.reduce(params_.bucket_fn.call(x));
    }
  }

  const size_t bucket(typename Hx::H hx) const {
    if (BF::LINEAR) {
      return rem_buckets_total_.reduce(high(hx));
    }
    auto [part, nhx] = rem_parts_.reduce_with_remainder(high(hx));
    auto bucket = this->bucket_in_part(nhx);
    return part * buckets_ + bucket;
  }
};

template <typename BF = Linear>
constexpr PtrHashParams<BF> params = PtrHashParams<BF>();

template <size_t n, typename BF>
constexpr size_t shards = (params<BF>.single_part == true) ? 1
                          : (params<BF>.sharding.type == ShardingType::None)
                              ? 1
                              : utility::div_ceil(n, params<BF>.keys_per_shard);

template <size_t n, typename BF = Linear>
static inline constexpr size_t get_parts() {
  size_t parts = 0;
  if (params<BF>.single_part) {
    parts = 1;
  } else {
    auto eps = (1.0 - params<BF>.alpha) / 2.0;
    auto x = n * eps * eps / 2.0;
    size_t target_parts = x / constexpr_ln(x);
    auto parts_per_shard = target_parts / shards<n, BF>;
    parts = ((parts_per_shard > 1) ? parts_per_shard : 1) * shards<n, BF>;
  }
  return parts;
}

template <size_t keys_per_part, typename BF>
static constexpr auto get_slots_per_part() {
  size_t slots_per_part = keys_per_part / params<BF>.alpha;
  if (utility::is_power_of_two(slots_per_part)) {
    slots_per_part += 1;
  }
  return slots_per_part;
}

template <size_t n, typename Key = uint64_t, typename BF = Linear,
          typename F = DynamicContainer<uint32_t>, // or Vec<u32>
          typename Hx = KeyHasherDefaultImpl<Key>, // FxHasherDecl::FxHasher64,
          typename V = std::vector<uint8_t>>
static constexpr inline auto init_hasher() {
  size_t constexpr parts = get_parts<n, BF>();

  size_t constexpr keys_per_part = n / parts;
  size_t constexpr parts_per_shard = parts / shards<n, BF>;
  size_t constexpr slots_per_part = get_slots_per_part<keys_per_part, BF>();

  size_t constexpr slots_total = parts * slots_per_part;
  size_t constexpr buckets_per_part =
      constexpr_ceil(keys_per_part / params<BF>.lambda) + 3;
  size_t constexpr buckets_total = parts * buckets_per_part;

  params<BF>.bucket_fn.set_buckets_per_part(buckets_per_part);

  return PtrHash<BF, params<BF>, n, parts, shards<n, BF>, parts_per_shard,
                 slots_total, buckets_total, slots_per_part, buckets_per_part,
                 Rp(shards<n, BF>), Rp(parts), Rb(buckets_per_part),
                 Rb(buckets_total),
                 RemSlots(std::max(slots_per_part, static_cast<size_t>(1))),
                 Key, F, Hx, V>(0, V(), F());
}

#endif // PORT_PTR_HASH_HPP_
