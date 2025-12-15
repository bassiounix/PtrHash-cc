#ifndef PORT_GENERATE_KEYS_HPP_
#define PORT_GENERATE_KEYS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>

template <size_t n> constexpr std::array<uint64_t, n> generate_keys() {
  srand(time(0));
  while (true) {
    std::array<uint64_t, n> keys;
    for (size_t i = 0; i < n; ++i) {
      keys[i] = rand();
    }
    auto keys2 = keys;

    std::sort(keys2.begin(), keys2.end());

    bool distinct = true;
    for (size_t i = 0; i + 1 < keys2.size(); ++i) {
      if (keys2[i] >= keys2[i + 1]) {
        distinct = false;
        break;
      }
    }

    if (distinct) {
      return keys;
    }
  };
}

#endif // PORT_GENERATE_KEYS_HPP_
