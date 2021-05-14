#include "benchmarks/benchmark_wormhole.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/wormhole.h"

template <template <typename> typename Searcher>
void benchmark_32_wormhole(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                           bool pareto) {
  benchmark.template Run<Wormhole<uint32_t, 512>>();
  if (pareto) {
    benchmark.template Run<Wormhole<uint32_t, 1024>>();
    benchmark.template Run<Wormhole<uint32_t, 2048>>();
    benchmark.template Run<Wormhole<uint32_t, 4096>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<Wormhole<uint32_t, 6384>>();
      benchmark.template Run<Wormhole<uint32_t, 5536>>();
      benchmark.template Run<Wormhole<uint32_t, 62144>>();
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_64_wormhole(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                           bool pareto) {
  benchmark.template Run<Wormhole<uint64_t, 512>>();
  if (pareto) {
    benchmark.template Run<Wormhole<uint64_t, 1024>>();
    benchmark.template Run<Wormhole<uint64_t, 2048>>();
    benchmark.template Run<Wormhole<uint64_t, 4096>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<Wormhole<uint64_t, 6384>>();
      benchmark.template Run<Wormhole<uint64_t, 5536>>();
      benchmark.template Run<Wormhole<uint64_t, 62144>>();
    }
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_wormhole, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_wormhole, uint64_t);
