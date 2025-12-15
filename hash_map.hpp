#ifndef HM_HASH_MAP_HPP_
#define HM_HASH_MAP_HPP_

#include "ptr_hash.hpp"
#include <cstddef>
#include <wctype.h>

template <size_t Capacity, class Hasher = PtrHash<wint_t>> class PerfectHashMap {
public:
  struct Entry {
    wint_t key;
    wint_t value;
  };

  PerfectHashMap(
      std::array<std::array<wint_t, 2>, Capacity> pairs) {
    std::array<wint_t, Capacity> keys{};
    for (size_t i = 0; i < Capacity; i++) {
      keys[i] = pairs[i][0];
    }
    this->hasher_ = Hasher::New(Slice<wint_t>{keys.data(), keys.size()});

    for (auto &[key, value] : pairs) {
      auto const idx = this->hasher_.index(key);
      assert(idx < Capacity && "Index out of bounds");
      this->entries_[idx] = Entry{key, value};
    }
  }

  std::optional<wint_t> find(const wint_t key) {
    std::size_t idx = this->hasher_.index(key);
    if (idx >= Capacity)
      return std::nullopt;

    Entry &e = entries_[idx];
    if (e.key != key)
      return std::nullopt;

    return e.value;
  }

  bool contains(const wint_t key) { return find(key).has_value(); }

  static constexpr std::size_t size() { return Capacity; }

private:
  Entry entries_[Capacity];
  Hasher hasher_;
};

#endif // HM_HASH_MAP_HPP_
