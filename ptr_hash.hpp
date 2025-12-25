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
#include <array>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <variant>

namespace ptrhash {

template <
    typename BF, const PtrHashParams<BF> &params_, size_t n_, size_t parts_,
    size_t shards_, size_t parts_per_shard_, size_t slots_total_,
    size_t buckets_total_, size_t slots_, size_t buckets_,
    const Rp &rem_shards_, const Rp &rem_parts_, const Rb &rem_buckets_,
    const Rb &rem_buckets_total_, const RemSlots &rem_slots_,
    typename Key = uint64_t,
    typename F = StaticContainer<uint32_t, slots_total_ - n_>, // or Vec<u32>
    typename Hx = KeyHasherDefaultImpl<Key>, // -> FxHasherDecl::FxHasher64,
    typename PilotsTypeV = std::array<uint8_t, buckets_total_>>
class PtrHash {
public:
  // static_assert(std::is_base_of_v<KeyT, Key>, "Key must implement KeyT");
  static_assert(std::is_base_of_v<BucketFn<BF>, BF>,
                "BF must implement BucketFn");
  static_assert(std::is_base_of_v<Packed<F>, F>, "F must implement Packed");
  static_assert(std::is_base_of_v<KeyHasher<Key, uint64_t>, Hx>,
                "Hx must implement KeyHasher<Key>");
  static_assert(
      std::is_same_v<PilotsTypeV, Slice<uint8_t>> ||
          std::is_same_v<PilotsTypeV, std::array<uint8_t, buckets_total_>>,
      "V must be a byte slice or byte vector");

  uint64_t seed_;
  PilotsTypeV pilots_;
  F remap_;

  constexpr PtrHash(uint64_t seed_, PilotsTypeV pilots_, F remap_,
                    const std::array<Key, n_> &keys)
      : seed_(seed_), pilots_(pilots_), remap_(remap_) {
    this->compute_pilots(keys);
  }

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

  constexpr bool compute_pilots(const std::array<Key, n_> &keys) {
    std::array<std::array<bool, slots_>, parts_> taken{};
    for (std::array<bool, slots_> &t : taken) {
      for (size_t i = 0; i < slots_; i++) {
        t[i] = 0;
      }
    }
    PilotsTypeV pilots{};

    auto tries = 0;
    constexpr size_t max_tries = 10;

    constexpr auto rng = rngs::StdRng::from_seed(31415);
    while (true) {
      bool contd = false;
      tries += 1;
      // std::println("Try num {}", tries);
      if (tries > max_tries) {
        return false;
      }

      this->seed_ = rng.random();

      pilots = PilotsTypeV{0};

      for (auto &t : taken) {
        for (size_t e = 0; e < pilots.size(); e++) {
          t[e] = false;
        }
      }

      auto shard_hashes = this->shards(keys);

      utility::ChunksMut<uint8_t, buckets_total_> shard_pilots =
          utility::chunks_mut(pilots, std::max(buckets_ * parts_per_shard_,
                                               static_cast<size_t>(1)));
      utility::ChunksMut<std::array<bool, slots_>, parts_> shard_taken =
          utility::chunks_mut(taken, parts_per_shard_);

      for (size_t shard = 0;
           shard < std::min({shard_hashes.size(), shard_pilots.size(),
                             shard_taken.size()});
           shard++) {
        std::array<typename Hx::H, n_> hashes = shard_hashes[shard];
        typename utility::ChunksMut<uint8_t, buckets_total_>::Chunk pilots =
            shard_pilots[shard];
        typename utility::ChunksMut<std::array<bool, slots_>, parts_>::Chunk
            taken = shard_taken[shard];
        std::optional<std::pair<std::array<typename Hx::H, n_>,
                                std::array<uint32_t, parts_per_shard_ + 1>>>
            sorted_parts = this->sort_parts(shard, hashes);
        if (!sorted_parts) {
          contd = true;
          break;
        }

        auto &[new_hashes, part_starts] = sorted_parts.value();

        if (!this->build_shard(shard, new_hashes, part_starts, pilots, taken)) {
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

    return true;
  }

  constexpr cpp::expected<std::monostate, std::nullopt_t>
  remap_free_slots(std::array<std::array<bool, slots_>, parts_> &taken) {
    auto val =
        utility::map(taken, [&](auto t) { return utility::count_zeros(t); });
    if (utility::sum(Slice(val.data(), val.size())) != slots_total_ - n_) {

      fprintf(stderr,
              "Not the right number of free slots left!\n total slots %zu - n "
              "%zu\n",
              slots_total_, n_);
      return cpp::unexpected(std::nullopt);
    }

    if (!params_.remap || slots_total_ == n_) {
      return std::monostate{};
    }

    std::array<uint64_t, slots_total_ - n_> v{};
    size_t v_idx = 0;

    auto const get = [&](std::array<std::array<bool, slots_>, parts_> &t,
                         size_t idx) { return t[idx / slots_][idx % slots_]; };

    for (const auto &[p, t] : enumerate(taken)) {
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
    }
    this->remap_ = F::try_new(Slice(v.data(), v.size()));
    return std::monostate{};
  }

  constexpr auto shards(const std::array<Key, n_> &keys) const {
    switch (params_.sharding.type) {
    case ShardingType::None:
      return this->no_sharding(keys);
      // case ShardingType::Memory:
      //   return this->shard_keys_in_memory(keys);
      // We don't need this for compile time data
      // case ShardingType::Disk:
      //   return this->shard_keys_hybrid(std::numeric_limits<size_t>::max(),
      //   keys);
      // case ShardingType::Hybrid:
      //   return this->shard_keys_hybrid(params_.sharding.mem, keys);
    }
  }

  constexpr std::array<std::array<typename Hx::H, n_>, 1>
  no_sharding(const std::array<Key, n_> &keys) const {
    return {utility::map(keys, [&](Key key) { return this->hash_key(key); })};
  }

  // constexpr auto shard_keys_in_memory(std::array<Key, n_> &keys) const {
  //   constexpr utility::Range r(shards_);
  //   return utility::map(r, [&](size_t shard) {
  //     std::array<typename Hx::H, n_> res = utility::map<std::array<Key, n_>>(
  //         keys, [&](typename Hx::H key) -> typename Hx::H {
  //           return this->hash_key(key);
  //         });
  //     std::vector<typename Hx::H> hashes = utility::filter(
  //         res, [&](typename Hx::H h) { return this->shard(h) == shard; });
  //     return hashes;
  //   });
  // }

  constexpr typename Hx::H hash_key(Key x) const {
    return Hx::hash(x, this->seed_);
  }

  constexpr size_t shard(typename Hx::H hx) const {
    return rem_shards_.reduce(high(hx));
  }

  constexpr std::optional<std::pair<std::array<typename Hx::H, n_>,
                                    std::array<uint32_t, parts_per_shard_ + 1>>>
  sort_parts(size_t shard, std::array<typename Hx::H, n_> hashes) const {
    hashes = utility::array_sort(hashes);

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
      // assert(shard * parts_per_shard_ <= this->part(hashes[0]));
      // assert(this->part(hashes.last()) < (shard + 1) *
      // parts_per_shard_);
    }

    std::array<uint32_t, parts_per_shard_ + 1> part_starts{};

    for (auto part_in_shard : utility::Range(1, parts_per_shard_ + 1)) {
      part_starts[part_in_shard] =
          Slice(hashes.data(), hashes.size())
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

  // TODO: get back here for pilots
  constexpr bool build_shard(
      size_t shard, std::array<typename Hx::H, n_> &hashes,
      std::array<uint32_t, parts_per_shard_ + 1> &part_starts,
      typename utility::ChunksMut<uint8_t, buckets_total_>::Chunk pilots,
      typename utility::ChunksMut<std::array<bool, slots_>, parts_>::Chunk
          taken) const {
    // auto pilots_per_part =
    //     utility::chunks_exact_mut<buckets_total_, buckets_>(pilots);

    // auto parts_done = shard * parts_per_shard_;

    auto ok =
        utility::try_for_each(enumerate(taken), [&](auto e) constexpr -> bool {
          const auto num_chunks = pilots.size() / buckets_;
          for (size_t i = 0; i < num_chunks; ++i) {
            size_t begin = pilots.begin_ + i * buckets_;
            size_t end = std::min(begin + buckets_, pilots.arr_.size());
            auto target_pilots =
                typename utility::ChunksMut<uint8_t, buckets_total_>::Chunk{
                    pilots.arr_, begin, end};
            auto &[part_in_shard, taken] = e;
            auto part = shard * parts_per_shard_ + part_in_shard;

            auto _cnt = this->build_part(
                part,
                Slice(hashes.data(), hashes.size())
                    .range(part_starts[part_in_shard],
                           part_starts[part_in_shard + 1]),
                Slice(target_pilots.data(), target_pilots.size()), taken);
            if (!_cnt) {
              return false;
            }
          }
          // parts_done++;
          return true;
        });

    if (!ok) {
      return false;
    }

    // assert(parts_done == (shard + 1) * parts_per_shard_);
    return true;
  }

  constexpr std::optional<size_t>
  build_part(size_t part, Slice<typename Hx::H> hashes, Slice<uint8_t> pilots,
             std::array<bool, slots_> &taken) const {
    std::pair<std::array<uint32_t, buckets_ + 1>,
              std::array<BucketIdx, buckets_>>
        sorted_buckets = this->sort_buckets(part, hashes);
    std::array<uint32_t, buckets_ + 1> starts = sorted_buckets.first;
    std::array<BucketIdx, buckets_> bucket_order = sorted_buckets.second;

    auto kmax = 256;

    std::array<BucketIdx, slots_> slots{};
    for (size_t i = 0; i < slots_; i++) {
      slots[i] = BucketIdx::NONE;
    }

    constexpr size_t one = 1;
    auto bucket_len = [&](BucketIdx b) constexpr -> size_t {
      return starts[b + one] - starts[b];
    };

    auto max_bucket_len = bucket_len(bucket_order[0]);
    constexpr BinaryHeap<std::pair<size_t, BucketIdx>> stack{};

    auto duplicate_slots = [&](BucketIdx b, Pilot p) constexpr {
      auto hp = this->hash_pilot(p);
      auto hashes_range = hashes.range(starts[b], starts[b + one]);

      for (auto const &[i, e1] : enumerate(hashes_range)) {
        auto hx = this->slot_in_part_hp(e1, hp);
        for (auto e2 : hashes_range.sub(i+1)) {
          auto hy = this->slot_in_part_hp(e2, hp);
          if (hx == hy) {
            return true;
          }
        }
      }
      return false;
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
      for (auto const &[i, _] : enumerate(recent)) {
        recent[i] = BucketIdx::NONE;
      }
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
            std::cout << "iter num " << iter_num << std::endl;
            return std::nullopt;
          }
        }
        auto const bucket = hashes.range(starts[b], starts[b + one]);
        if (auto fpilot = this->find_pilot(kmax, bucket, taken)) {
          auto &[p, hp] = fpilot.value();
          pilots[b] = static_cast<uint8_t>(p);
          for (auto &item : hashes.range(starts[b], starts[b + one])) {
            auto p = this->slot_in_part_hp(item, hp);
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
          for (auto &item : hashes.range(starts[b], starts[b + one])) {
            auto p = this->slot_in_part_hp(item, hp);
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
        for (auto &item : hashes.range(starts[b], starts[b + one])) {
          auto slot = this->slot_in_part_hp(item, hp);
          auto const b2 = slots[slot];
          if (b2.is_some()) {
            // assert(b2 != b);
            stack.push({bucket_len(b2), b2});
            // std::print("stack(push): ");
            // for (auto x : stack.data) {
            //   std::print("({} {}) ", std::get<0>(x), std::get<1>(x).i_);
            // }
            // std::println();
            evictions++;

            auto hp = this->hash_pilot(static_cast<Pilot>(pilots[b2]));
            for (auto &item : hashes.range(starts[b2], starts[b2 + one])) {
              auto p2 = this->slot_in_part_hp(item, hp);
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
             std::array<bool, slots_> &taken) const {
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
                   std::array<bool, slots_> &taken) const {
    auto cpy = bucket;
    cpy.len = L;
    return this->find_pilot_slice(kmax, cpy, taken);
  }

  inline constexpr std::optional<std::pair<Pilot, PilotHash>>
  find_pilot_slice(uint64_t kmax, Slice<typename Hx::H> bucket,
                   std::array<bool, slots_> &taken) const {
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
                                std::array<bool, slots_> &taken) const {
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

  constexpr std::pair<std::array<uint32_t, buckets_ + 1>,
                      std::array<BucketIdx, buckets_>>
  sort_buckets(size_t part, Slice<typename Hx::H> hashes) const {
    std::array<uint32_t, buckets_ + 1> bucket_starts{};
    size_t bucket_starts_idx = 0;
    std::array<BucketIdx, buckets_> order{};
    for (auto const &[i, _] : enumerate(order)) {
      order[i] = BucketIdx::NONE;
    }
    std::array<size_t, 32> bucket_len_cnt = {0};

    auto end = 0;
    bucket_starts[bucket_starts_idx++] = end;

    for (auto b : utility::Range(buckets_)) {
      auto start = end;
      while (end < hashes.size() &&
             this->bucket(hashes[end]) == part * buckets_ + b) {
        end++;
      }

      auto l = end - start;
      // 32 seems to be fine as a size for now, update bucket_len_cnt in the
      // future if necessary
      // if (l >= bucket_len_cnt.size()) {
      //   utility::resize_with(bucket_len_cnt, l + 1, []() { return 0; });
      // }
      bucket_len_cnt[l]++;
      bucket_starts[bucket_starts_idx++] = end;
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

  constexpr size_t bucket(typename Hx::H hx) const {
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

namespace ptrhash_config {

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t parts = get_parts<n, BF>();

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t keys_per_part = n / parts<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t parts_per_shard = parts<n, Key, BF, Hx> / shards<n, BF>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t slots_per_part =
    get_slots_per_part<keys_per_part<n, Key, BF, Hx>, BF>();

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t slots_total =
    parts<n, Key, BF, Hx> * slots_per_part<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t buckets_per_part =
    constexpr_ceil(keys_per_part<n, Key, BF, Hx> / params<BF>.lambda) + 3;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr size_t buckets_total =
    parts<n, Key, BF, Hx> * buckets_per_part<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr Rp rem_shards = shards<n, BF>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr Rp rem_parts = parts<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr Rb rem_buckets_per_part = buckets_per_part<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr Rb rem_buckets_total = buckets_total<n, Key, BF, Hx>;

template <size_t n, typename Key, typename BF, typename Hx>
constexpr RemSlots rem_slots_per_part =
    std::max(slots_per_part<n, Key, BF, Hx>, static_cast<size_t>(1));

} // namespace ptrhash_config

template <size_t n, typename Key = uint64_t, const std::array<Key, n> &keys,
          typename BF = Linear,
          typename Hx = KeyHasherDefaultImpl<Key>> // FxHasherDecl::FxHasher64,
static constexpr inline auto init_hasher() {
  using namespace ptrhash_config;

  params<BF>.bucket_fn.set_buckets_per_part(buckets_per_part<n, Key, BF, Hx>);

  using F = StaticContainer<uint32_t, slots_total<n, Key, BF, Hx> - n>;
  using PilotsTypeV = std::array<uint8_t, buckets_total<n, Key, BF, Hx>>;

  auto p =
      PtrHash<BF, params<BF>, n, parts<n, Key, BF, Hx>, shards<n, BF>,
              parts_per_shard<n, Key, BF, Hx>, slots_total<n, Key, BF, Hx>,
              buckets_total<n, Key, BF, Hx>, slots_per_part<n, Key, BF, Hx>,
              buckets_per_part<n, Key, BF, Hx>, rem_shards<n, Key, BF, Hx>,
              rem_parts<n, Key, BF, Hx>, rem_buckets_per_part<n, Key, BF, Hx>,
              rem_buckets_total<n, Key, BF, Hx>,
              rem_slots_per_part<n, Key, BF, Hx>, Key, F, Hx, PilotsTypeV>(
          0, PilotsTypeV(), F(), keys);
  if (!p.compute_pilots(keys)) {
    fprintf(stderr, "Unable to construct PtrHash after 10 tries. Try "
                    "using a better hash or decreasing lambda.\n");
    std::abort();
  }
  return p;
}

} // namespace ptrhash

#endif // PORT_PTR_HASH_HPP_
