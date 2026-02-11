#include <benchmark/benchmark.h>

static void BM_PtrHash(benchmark::State& state) {
  for (auto _ : state) {
    // This code gets timed
    phm.find(0x65);
  }
}
// Register the function as a benchmark
BENCHMARK(BM_PtrHash);
// Run the benchmark
BENCHMARK_MAIN();
