#include "benchmarks/benchmark_fast64.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/fast.h"
#include "competitors/fast64_search.h"

template <template <typename> typename Searcher>
void benchmark_32_fast(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                       bool pareto) {
  benchmark.template Run<FAST32<1>>();
  if (pareto) {
    benchmark.template Run<FAST32<16>>();
    benchmark.template Run<FAST32<256>>();
    benchmark.template Run<FAST32<512>>();
    benchmark.template Run<FAST32<768>>();
    benchmark.template Run<FAST32<1024>>();
    benchmark.template Run<FAST32<1526>>();
    benchmark.template Run<FAST32<2048>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<FAST32<4096>>();
      benchmark.template Run<FAST32<8192>>();
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_64_fast(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                       bool pareto) {
  benchmark.template Run<FAST64<1>>();
  if (pareto) {
    benchmark.template Run<FAST64<16>>();
    benchmark.template Run<FAST64<256>>();
    benchmark.template Run<FAST64<512>>();
    benchmark.template Run<FAST64<768>>();
    benchmark.template Run<FAST64<1024>>();
    benchmark.template Run<FAST64<1526>>();
    benchmark.template Run<FAST64<2048>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<FAST64<4096>>();
      benchmark.template Run<FAST64<8192>>();
    }
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_fast, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_fast, uint64_t);
