#ifndef HM_HASH_MAP_HPP_
#define HM_HASH_MAP_HPP_

#include "ptr_hash.hpp"
#include <array>
#include <cstddef>
#include <optional>
#include <wctype.h>

template <size_t Capacity, class Hasher = PtrHash<wint_t>>
class PerfectHashMap {
public:
  struct Entry {
    wint_t key;
    wint_t value;
  };

  constexpr PerfectHashMap(std::array<std::array<wint_t, 2>, Capacity> pairs,
                           Hasher &hasher)
      : hasher_(hasher) {
    for (auto &[key, value] : pairs) {
      auto const idx = hasher.index(key);
      // static_assert(idx < Capacity, "Index out of bounds");
      this->entries_[idx] = Entry{key, value};
    }
  }

  constexpr std::optional<wint_t> find(const wint_t key) const {
    size_t idx = this->hasher_.index(key);
    if (idx >= Capacity)
      return std::nullopt;

    const Entry &e = entries_[idx];
    if (e.key != key)
      return std::nullopt;

    return e.value;
  }

  constexpr bool contains(const wint_t key) const {
    return this->find(key).has_value();
  }

  static constexpr std::size_t size() { return Capacity; }

private:
  Entry entries_[Capacity];
  Hasher &hasher_;
};

#endif // HM_HASH_MAP_HPP_
