#ifndef HM_HASH_MAP_HPP_
#define HM_HASH_MAP_HPP_

#include <array>
#include <cstddef>
#include <optional>
#include <wctype.h>

template <size_t Capacity, class Hasher> class PerfectHashMap {
public:
  struct Entry {
    wint_t key{};
    wint_t value{};
    constexpr Entry() = default;
    constexpr Entry(wint_t key, wint_t value) : key(key), value(value) {}
  };

  constexpr PerfectHashMap(
      const std::array<std::array<wint_t, 2>, Capacity> &pairs,
      const Hasher &hasher_)
      : hasher_(hasher_) {
    for (auto &[key, value] : pairs) {
      auto const idx = hasher_.index(key);
      // static_assert(idx < Capacity, "Index out of bounds");
      this->entries_[idx] = Entry{key, value};
    }
  }

  constexpr std::optional<wint_t> find(const wint_t key) const {
    size_t idx = hasher_.index(key);
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
  const Hasher &hasher_;
};

#endif // HM_HASH_MAP_HPP_
