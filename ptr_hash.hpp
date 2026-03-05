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

template <typename Iterable> struct Enumerate {
  Iterable iterable;

  LIBC_INLINE constexpr Enumerate(Iterable &&iter)
      : iterable(std::forward<Iterable>(iter)) {}

  struct Iterator {
    size_t index;
    decltype(iterable.begin()) it;

    LIBC_INLINE constexpr auto operator*() const {
      return std::tuple<size_t, decltype(*it)>(index, *it);
    }

    LIBC_INLINE constexpr Iterator &operator++() {
      ++index;
      ++it;
      return *this;
    }

    LIBC_INLINE constexpr bool operator!=(const Iterator &other) const {
      return it != other.it;
    }
  };

  LIBC_INLINE constexpr Iterator begin() const { return {0, iterable.begin()}; }

  LIBC_INLINE constexpr Iterator end() const { return {0, iterable.end()}; }
};

template <typename Iterable>
LIBC_INLINE constexpr Enumerate<Iterable> enumerate(Iterable &&iterable) {
  return Enumerate<Iterable>(std::forward<Iterable>(iterable));
}

LIBC_INLINE_VAR constexpr auto BucketIdxNONE = ~static_cast<uint32_t>(0);

template <size_t MaxSize = 5> class BinaryHeap {
public:
  constexpr BinaryHeap() = default;

  constexpr void push(const std::pair<size_t, uint32_t> &value) {
    if (current_size >= MaxSize)
      return; // Optional: handle overflow
    data[current_size] = value;
    heapify_up(current_size);
    ++current_size;
  }

  constexpr std::pair<size_t, uint32_t> pop() {
    if (current_size == 0)
      return std::pair<size_t, uint32_t>{}; // Optional: handle underflow
    std::pair<size_t, uint32_t> top = data[0];
    data[0].first = data[current_size - 1].first;
    data[0].second = data[current_size - 1].second;
    --current_size;
    if (current_size > 0)
      heapify_down(0);
    return top;
  }

  constexpr const std::pair<size_t, uint32_t> &peek() const { return data[0]; }

  constexpr bool empty() const { return current_size == 0; }

private:
  std::array<std::pair<size_t, uint32_t>, MaxSize> data{};
  size_t current_size{};

  constexpr void heapify_up(size_t index) {
    while (index > 0) {
      size_t parent = (index - 1) / 2;
      if (data[index] <= data[parent])
        break;
      std::swap(data[index], data[parent]);
      index = parent;
    }
  }

  constexpr void heapify_down(size_t index) {
    while (true) {
      size_t left = 2 * index + 1;
      size_t right = 2 * index + 2;
      size_t largest = index;

      if (left < current_size && data[left] > data[largest])
        largest = left;
      if (right < current_size && data[right] > data[largest])
        largest = right;

      if (largest == index)
        break;

      std::swap(data[index], data[largest]);
      index = largest;
    }
  }
};

struct FastReduce {
  uint64_t d;

  constexpr FastReduce(uint64_t d) : d(d) {}
  constexpr FastReduce() : d(0) {}

  constexpr operator uint64_t() const { return d; }

  constexpr size_t reduce(uint64_t h) const {
    return (static_cast<__uint128_t>(this->d) * static_cast<__uint128_t>(h)) >>
           64;
  }

  constexpr std::pair<size_t, uint64_t>
  reduce_with_remainder(uint64_t h) const {
    __uint128_t r =
        static_cast<__uint128_t>(this->d) * static_cast<__uint128_t>(h);
    return {r >> 64, r};
  }
};

struct RemSlotsFM32 {
  uint64_t d;
  uint64_t m;

  constexpr RemSlotsFM32(size_t d)
      : d(d), m(std::numeric_limits<uint64_t>::max() / d + 1) {
    assert(d <= std::numeric_limits<uint32_t>::max());
  }
  constexpr RemSlotsFM32() : d(0), m(0) {}

  constexpr size_t reduce(uint64_t h) const {
    auto lowbits = m * h;
    return (static_cast<__uint128_t>(lowbits) * static_cast<__uint128_t>(d)) >>
           64;
  }
};

template <typename T> constexpr bool is_power_of_two(T x) {
  static_assert(std::is_unsigned_v<T>,
                "is_power_of_two requires unsigned type");
  return x != 0 && (x & (x - 1)) == 0;
}

template <typename T, typename F, size_t N>
constexpr auto map(const std::array<T, N> &v, F func) {
  using R = std::invoke_result_t<F, T>;
  std::array<R, N> out{};

  for (size_t i = 0; i < N; ++i) {
    out[i] = func(v[i]);
  }

  return out;
}

struct Range {
  int start_range, end_range, step_range;

  struct Iterator {
    int value;
    int step;

    LIBC_INLINE constexpr auto &operator*() const { return value; }

    LIBC_INLINE constexpr auto &operator++() {
      value += step;
      return *this;
    }

    LIBC_INLINE constexpr bool operator!=(const Iterator &other) const {
      return step > 0 ? value < other.value : value > other.value;
    }
  };

  LIBC_INLINE constexpr Iterator begin() const {
    return {start_range, step_range};
  }
  LIBC_INLINE constexpr Iterator end() const { return {end_range, step_range}; }

  LIBC_INLINE constexpr auto reverse() const {
    int count = (end_range - start_range + step_range - 1) / step_range;
    int new_start = start_range + (count - 1) * step_range;
    int new_end = start_range - step_range;
    return Range(new_start, new_end, -step_range);
  }

  LIBC_INLINE constexpr Range(int start, int end, int step = 1)
      : start_range(start), end_range(end), step_range(step) {}

  LIBC_INLINE constexpr Range(int end)
      : start_range(0), end_range(end), step_range(1) {}

  LIBC_INLINE constexpr size_t size() const {
    if (step_range > 0)
      return (end_range - start_range + step_range - 1) / step_range;
    return (start_range - end_range - step_range - 1) / (-step_range);
  }
};

template <typename T, size_t N> class ChunksMut {
public:
  struct Chunk {
    std::array<T, N> &arr;
    size_t chunk_begin;
    size_t chunk_end;

    LIBC_INLINE constexpr T &operator[](size_t i) const {
      return arr[chunk_begin + i];
    }

    LIBC_INLINE constexpr T &at(size_t i) const { return arr[chunk_begin + i]; }

    LIBC_INLINE constexpr T &front() const { return arr[chunk_begin]; }

    LIBC_INLINE constexpr T &back() const { return arr[chunk_end - 1]; }

    LIBC_INLINE constexpr size_t size() const noexcept {
      return chunk_end - chunk_begin;
    }

    LIBC_INLINE constexpr bool empty() const noexcept {
      return chunk_begin == chunk_end;
    }

    LIBC_INLINE constexpr auto begin_it() const {
      return arr.begin() + static_cast<ptrdiff_t>(chunk_begin);
    }
    LIBC_INLINE constexpr auto end_it() const {
      return arr.begin() + static_cast<ptrdiff_t>(chunk_end);
    }

    LIBC_INLINE constexpr auto begin() const { return begin_it(); }

    LIBC_INLINE constexpr auto end() const { return end_it(); }

    LIBC_INLINE constexpr T *data() const { return arr.data() + chunk_begin; }
  };

  class Iterator {
  public:
    LIBC_INLINE constexpr Iterator(std::array<T, N> &arr, size_t pos,
                                   size_t chunk)
        : arr(arr), index(pos), chunk_size(chunk) {}

    LIBC_INLINE constexpr Chunk operator*() const {
      size_t end = std::min(index + chunk_size, arr.size());
      return Chunk{arr, index, end};
    }

    LIBC_INLINE constexpr const Iterator &operator++() const {
      index += chunk_size;
      return *this;
    }

    LIBC_INLINE constexpr bool operator!=(const Iterator &other) const {
      return index != other.index;
    }

  private:
    std::array<T, N> &arr;
    mutable size_t index;
    mutable size_t chunk_size;
  };

  LIBC_INLINE constexpr ChunksMut(std::array<T, N> &v, size_t chunk)
      : arr(v), chunk_size(chunk) {}

  // number of chunks
  LIBC_INLINE constexpr size_t size() const {
    return (arr.size() + chunk_size - 1) / chunk_size;
  }

  LIBC_INLINE constexpr bool empty() const { return arr.empty(); }

  LIBC_INLINE constexpr Chunk operator[](size_t chunk_index) const {
    size_t begin = chunk_index * chunk_size;
    size_t end = std::min(begin + chunk_size, arr.size());

    return Chunk{arr, begin, end};
  }

  LIBC_INLINE constexpr Iterator begin() const {
    return Iterator(arr, 0, chunk_size);
  }
  LIBC_INLINE constexpr Iterator end() const {
    return Iterator(arr, arr.size(), chunk_size);
  }

private:
  std::array<T, N> &arr;
  size_t chunk_size;
};

template <typename T, size_t N>
LIBC_INLINE constexpr ChunksMut<T, N> chunks_mut(std::array<T, N> &arr,
                                                 size_t chunk_size) {
  return ChunksMut<T, N>(arr, chunk_size);
}

template <size_t n_, size_t parts_, size_t parts_per_shard_,
          size_t slots_total_, size_t buckets_total_, size_t slots_,
          size_t buckets_, const FastReduce &rem_shards_,
          const FastReduce &rem_parts_, const FastReduce &rem_buckets_,
          const FastReduce &rem_buckets_total_, const RemSlotsFM32 &rem_slots_,
          typename Key = uint64_t,
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

      ChunksMut<uint8_t, buckets_total_> shard_pilots =
          chunks_mut(pilots, std::max(buckets_ * parts_per_shard_,
                                      static_cast<size_t>(1)));
      ChunksMut<std::array<bool, slots_>, parts_> shard_taken =
          chunks_mut(taken, parts_per_shard_);

      for (size_t shard = 0;
           shard < std::min({shard_hashes.size(), shard_pilots.size(),
                             shard_taken.size()});
           shard++) {
        std::array<uint64_t, n_> hashes = shard_hashes[shard];
        typename ChunksMut<uint8_t, buckets_total_>::Chunk pilots =
            shard_pilots[shard];
        typename ChunksMut<std::array<bool, slots_>, parts_>::Chunk taken =
            shard_taken[shard];
        std::optional<std::pair<std::array<uint64_t, n_>,
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

    return {{this->seed_, this->pilots_, this->remap_}};
  }

  constexpr cpp::expected<std::monostate, std::nullopt_t>
  remap_free_slots(std::array<std::array<bool, slots_>, parts_> &taken) {
    auto val = map(taken, [&](auto container) {
      size_t counter = 0;

      for (auto element : container) {
        if (!element) {
          counter++;
        }
      }

      return counter;
    });

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
    return {map(keys, [&](Key key) { return this->hash_key(key); })};
  }

  constexpr uint64_t hash_key(Key x) const {
    uint64_t value = 0;
    constexpr uint64_t bits = sizeof(uint64_t) * 8;
    value = ((value << 5) | (value >> (bits - 5))) ^ x;
    value *= 0x517cc1b727220a95;
    return value ^ this->seed_;
  }

  constexpr size_t shard(uint64_t hx) const { return rem_shards_.reduce(hx); }

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

    for (auto part_in_shard : Range(1, parts_per_shard_ + 1)) {
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

  constexpr bool build_shard(
      size_t shard, std::array<uint64_t, n_> &hashes,
      std::array<uint32_t, parts_per_shard_ + 1> &part_starts,
      typename ChunksMut<uint8_t, buckets_total_>::Chunk pilots,
      typename ChunksMut<std::array<bool, slots_>, parts_>::Chunk taken) const {

    for (auto &&[part_in_shard, taken] : enumerate(taken)) {
      const auto num_chunks = pilots.size() / buckets_;
      for (size_t i = 0; i < num_chunks; ++i) {
        size_t begin = pilots.chunk_begin + i * buckets_;
        size_t end = std::min(begin + buckets_, pilots.arr.size());
        auto target_pilots = typename ChunksMut<uint8_t, buckets_total_>::Chunk{
            pilots.arr, begin, end};
        auto part = shard * parts_per_shard_ + part_in_shard;

        auto _cnt = this->build_part(
            part,
            cpp::span<uint64_t>(hashes).subspan(part_starts[part_in_shard],
                                                part_starts[part_in_shard + 1] -
                                                    part_starts[part_in_shard]),
            cpp::span(target_pilots.data(), target_pilots.size()), taken);
        if (!_cnt) {
          return false;
        }
      }
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

    auto kmax = 256;

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

      for (auto const &[i, e1] : enumerate(hashes_range)) {
        auto hx = this->slot_in_part_hp(e1, hp);
        for (auto e2 : hashes_range.subspan(i + 1)) {
          auto hy = this->slot_in_part_hp(e2, hp);
          if (hx == hy) {
            return true;
          }
        }
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

    for (auto const &[iter_num, new_b] : enumerate(bucket_order)) {
      auto const new_bucket =
          hashes.subspan(starts[new_b], starts[new_b + 1] - starts[new_b]);
      if (new_bucket.empty()) {
        pilots[new_b] = 0;
        continue;
      }
      auto const new_b_len = new_bucket.size();
      size_t evictions = 0;

      heap.push({new_b_len, new_b});
      for (auto const &[i, _] : enumerate(recent)) {
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
        for (auto delat : Range(kmax)) {
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

  constexpr std::optional<std::pair<uint64_t, uint64_t>>
  find_pilot(uint64_t kmax, cpp::span<uint64_t> bucket,
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
  constexpr std::optional<std::pair<uint64_t, uint64_t>>
  find_pilot_array(uint64_t kmax, cpp::span<uint64_t> bucket,
                   std::array<bool, slots_> &taken) const {
    auto cpy = cpp::span(bucket.data(), L);
    return this->find_pilot_slice(kmax, cpy, taken);
  }

  LIBC_INLINE constexpr std::optional<std::pair<uint64_t, uint64_t>>
  find_pilot_slice(uint64_t kmax, cpp::span<uint64_t> bucket,
                   std::array<bool, slots_> &taken) const {
    auto const r = bucket.size() / 4 * 4;
    for (auto p : Range(kmax)) {
      bool find_pilot_continue = false;
      auto const hp = this->hash_pilot(p);
      auto const check = [&](uint64_t hx) {
        return taken[this->slot_in_part_hp(hx, hp)];
      };
      auto bad = false;
      for (auto i : Range(0, r, 4)) {
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
    for (auto [i, hx] : enumerate(bucket)) {
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
    return rem_slots_.reduce(hx ^ hp);
  }

  constexpr std::pair<std::array<uint32_t, buckets_ + 1>,
                      std::array<uint32_t, buckets_>>
  sort_buckets(size_t part, cpp::span<uint64_t> hashes) const {
    std::array<uint32_t, buckets_ + 1> bucket_starts{};
    size_t bucket_starts_idx = 0;
    std::array<uint32_t, buckets_> order{};
    for (auto const &[i, _] : enumerate(order)) {
      order[i] = BucketIdxNONE;
    }
    std::array<size_t, 32> bucket_len_cnt = {0};

    size_t end = 0;
    bucket_starts[bucket_starts_idx++] = end;

    for (auto b : Range(buckets_)) {
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
    for (auto i : Range(max_bucket_size + 1).reverse()) {
      auto tmp = bucket_len_cnt[i];
      bucket_len_cnt[i] = acc;
      acc += tmp;
    }
    for (auto &b : Range(buckets_)) {
      size_t l = bucket_starts[b + 1] - bucket_starts[b];
      order[bucket_len_cnt[l]] = b;
      bucket_len_cnt[l] += 1;
    }

    return {bucket_starts, order};
  }

  constexpr size_t part(uint64_t hx) const { return rem_parts_.reduce(hx); }

  constexpr size_t bucket_in_part(uint64_t x) const {
    return rem_buckets_.reduce(x);
  }

  constexpr size_t bucket(uint64_t hx) const {
    return rem_buckets_total_.reduce(hx);
  }
};

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

LIBC_INLINE_VAR constexpr size_t shards = 1;

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
  LIBC_INLINE_VAR static constexpr FastReduce rem_shards = shards;
  LIBC_INLINE_VAR static constexpr FastReduce rem_parts = parts;
  LIBC_INLINE_VAR static constexpr FastReduce rem_buckets_per_part =
      buckets_per_part;
  LIBC_INLINE_VAR static constexpr FastReduce rem_buckets_total = buckets_total;
  LIBC_INLINE_VAR static constexpr RemSlotsFM32 rem_slots_per_part =
      std::max(slots_per_part, static_cast<size_t>(1));
};

template <size_t n, typename Key = uint64_t>
LIBC_INLINE constexpr auto init_hasher(const std::array<Key, n> &keys) {
  using F = std::array<uint32_t, ptrhash_config<n>::slots_total - n>;
  using PilotsTypeV = std::array<uint8_t, ptrhash_config<n>::buckets_total>;

  auto p =
      PtrHash<n, ptrhash_config<n>::parts, ptrhash_config<n>::parts_per_shard,
              ptrhash_config<n>::slots_total, ptrhash_config<n>::buckets_total,
              ptrhash_config<n>::slots_per_part,
              ptrhash_config<n>::buckets_per_part,
              ptrhash_config<n>::rem_shards, ptrhash_config<n>::rem_parts,
              ptrhash_config<n>::rem_buckets_per_part,
              ptrhash_config<n>::rem_buckets_total,
              ptrhash_config<n>::rem_slots_per_part, Key, F, PilotsTypeV>(
          0, PilotsTypeV(), F());
  auto result = p.compute_pilots(keys);

  assert(result && "Unable to construct PtrHash after 10 tries. Try using a "
                   "better hash or decreasing lambda.\n");

  auto &[seed, pilots, remap] = result.value();

  return PtrHash<
      n, ptrhash_config<n>::parts, ptrhash_config<n>::parts_per_shard,
      ptrhash_config<n>::slots_total, ptrhash_config<n>::buckets_total,
      ptrhash_config<n>::slots_per_part, ptrhash_config<n>::buckets_per_part,
      ptrhash_config<n>::rem_shards, ptrhash_config<n>::rem_parts,
      ptrhash_config<n>::rem_buckets_per_part,
      ptrhash_config<n>::rem_buckets_total,
      ptrhash_config<n>::rem_slots_per_part, Key, F, PilotsTypeV>(seed, pilots,
                                                                  remap);
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
