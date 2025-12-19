#ifndef FASTRAND_HPP_
#define FASTRAND_HPP_

#include "fxhash.hpp"
#include <cstdint>
#include <ctime>
// #include <optional>
#include <pthread.h>

namespace fastrand {

constexpr uint64_t DEFAULT_RNG_SEED = 0xEF6F79ED30BA75A;

inline constexpr uint64_t random_seed() {
  constexpr auto state = FxHasherDecl::FxHasher64();
  state.write((uint64_t)time(0));
  state.write(pthread_self());
  return state.finish();
}

template <typename Rng> class RestoreOnDrop {
public:
  Rng &rng;
  Rng &current;

  ~RestoreOnDrop() { rng = Rng(this->current.seed_); }
};

class Rng {
public:
  mutable uint64_t seed_;

  // backup seed
  // constexpr Rng() : Rng(try_with_rng().value_or(Rng(0x4d595df4d0f33173))) {}
  constexpr Rng() : Rng(DEFAULT_RNG_SEED) {}

  constexpr Rng(uint64_t seed) : seed_(seed) {}
  constexpr Rng(const Rng &) = default;
  constexpr Rng(Rng &&) = default;

  constexpr Rng &operator=(const Rng &) = default;
  constexpr Rng &operator=(Rng &&) = default;

  inline constexpr uint64_t gen_u64() const {
    constexpr uint64_t WY_CONST_0 = 0x2d35'8dcc'aa6c'78a5;
    constexpr uint64_t WY_CONST_1 = 0x8bb8'4b93'962e'acc9;

    auto s = utility::wrapping_add(seed_, WY_CONST_0);
    seed_ = s;
    auto const t =
        static_cast<__uint128_t>(s) * static_cast<__uint128_t>(s ^ WY_CONST_1);
    return static_cast<uint64_t>(t) ^ static_cast<uint64_t>(t >> 64);
  }

  constexpr uint8_t u8() const {
    return static_cast<uint32_t>(this->gen_u64());
  }

  constexpr Rng fork() const { return Rng(this->gen_u64()); }
  constexpr void set(uint64_t i) const { this->seed_ = i; }

  // constexpr std::optional<Rng> try_with_rng() const;

  constexpr Rng replace(Rng &&n) const {
    auto ret = seed_;
    seed_ = n.seed_;
    return ret;
  }
};

// static Rng RNG = DEFAULT_RNG_SEED;

// constexpr std::optional<Rng> Rng::try_with_rng() const {
//   auto current = RNG;
//   RNG.set(0);
//   auto restore = RestoreOnDrop{RNG, current};
//   return restore.current.fork();
// }

} // namespace fastrand

#endif // FASTRAND_HPP_
