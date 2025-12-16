#ifndef PORT_PTR_HASH_PARAMS_HPP_
#define PORT_PTR_HASH_PARAMS_HPP_

#include "bucket_fn.hpp"
#include <cstddef>

enum class ShardingType { None, Memory, /*Disk, Hybrid*/ };

struct Sharding {
  ShardingType type = ShardingType::None;
  size_t mem = 0;
};

template <typename BF> struct PtrHashParams {
  bool remap : 1;
  double alpha;
  double lambda;
  BF bucket_fn;
  size_t keys_per_shard;
  Sharding sharding;
  bool single_part : 1;

  constexpr PtrHashParams<Linear>()
      : PtrHashParams(PtrHashParams::default_fast()) {}

  constexpr PtrHashParams(bool remap, double alpha, double lambda, BF bucket_fn,
                          size_t keys_per_shard, Sharding sharding,
                          bool single_part)
      : remap(remap), alpha(alpha), lambda(lambda), bucket_fn(bucket_fn),
        keys_per_shard(keys_per_shard), sharding(sharding),
        single_part(single_part) {}

  static constexpr PtrHashParams default_fast() {
    return PtrHashParams(true, 0.99, 3.0, Linear(), (size_t)1 << 31,
                         Sharding{ShardingType::None}, false);
  };

  static constexpr PtrHashParams default_square() {
    return PtrHashParams(true, 0.99, 3.5, SquareEps(), (size_t)1 << 31,
                         Sharding{ShardingType::None}, false);
  }

  static constexpr PtrHashParams default_balanced() {
    return PtrHashParams(true, 0.99, 3.5, CubicEps(), (size_t)1 << 31,
                         Sharding{ShardingType::None}, false);
  }

  static constexpr PtrHashParams default_compact() {
    return PtrHashParams(true, 0.99, 3.9, CubicEps(), (size_t)1 << 31,
                         Sharding{ShardingType::None}, false);
  }
};

#endif // PORT_PTR_HASH_PARAMS_HPP_
