#include "generate_keys.hpp"
#include "hash_map.hpp"
#include <print>

void test_hasher() {
  auto keys = generate_keys<1000>();
  // std::array<size_t, 4> keys = {10, 2, 3, 4};
  constexpr auto n = keys.size();
  PtrHash mphf = PtrHash<>::New(Slice<uint64_t>{keys.data(), keys.size()},
               PtrHashParams<Linear>());
  auto key = 10;
  auto const idx = mphf.index(key);
  assert(idx < n);
  // auto indices = mphf.index_stream<32, true, _>(&keys);
  // assert(indices.sum::<size_t>() == (n * (n - 1)) / 2);
  auto taken = std::array<bool, n>{false};
  for (auto &key : keys) {
    auto idx = mphf.index(key);
    assert(!taken[idx]);
    taken[idx] = true;
  }
}

int main() {
  PerfectHashMap<2> phm =
      std::array<std::array<wint_t, 2>, 2>{{{1, 2}, {3, 4}}};

  std::println("phm.find 1 {}", phm.find(1).value_or(-1));
  std::println("phm.find 3 {}", phm.find(3).value_or(-1));
  std::println("phm.find 5 (none) {}", phm.find(5).value_or(0));
  std::println("phm.contains 2 {}", phm.contains(2));
  std::println("phm.size {}", phm.size());

  // test_hasher();
  return 0;
}
