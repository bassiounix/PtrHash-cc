#ifndef ENUMERATE_HPP_
#define ENUMERATE_HPP_

#include <cstddef>
#include <utility>

template <typename Iterable> struct Enumerate {
  Iterable &iterable;

  struct Iterator {
    size_t index;
    decltype(std::begin(iterable)) it;

    constexpr auto operator*() const {
      return std::pair<size_t, decltype(*it)>(index, *it);
    }

    constexpr Iterator &operator++() {
      ++index;
      ++it;
      return *this;
    }

    constexpr bool operator!=(const Iterator &other) const {
      return it != other.it;
    }
  };

  constexpr Iterator begin() const { return {0, std::begin(iterable)}; }

  constexpr Iterator end() const { return {0, std::end(iterable)}; }
};

template <typename Iterable>
constexpr Enumerate<Iterable> enumerate(Iterable &iterable) {
  return {iterable};
}

template <typename Iterable>
constexpr Enumerate<Iterable> enumerate(Iterable &&iterable) {
  return {iterable};
}

#endif // ENUMERATE_HPP_
