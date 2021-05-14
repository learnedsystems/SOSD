#include "benchmarks/benchmark_alex.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/alex.h"

template <template <typename> typename Searcher>
void benchmark_32_alex(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                       bool pareto) {
  benchmark.template Run<Alex<uint32_t, 1>>();
  if (pareto) {
    benchmark.template Run<Alex<uint32_t, 4>>();
    benchmark.template Run<Alex<uint32_t, 8>>();
    benchmark.template Run<Alex<uint32_t, 16>>();
    benchmark.template Run<Alex<uint32_t, 32>>();
    benchmark.template Run<Alex<uint32_t, 64>>();
    benchmark.template Run<Alex<uint32_t, 128>>();
    benchmark.template Run<Alex<uint32_t, 256>>();
    benchmark.template Run<Alex<uint32_t, 512>>();
    benchmark.template Run<Alex<uint32_t, 1024>>();
    benchmark.template Run<Alex<uint32_t, 2048>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<Alex<uint32_t, 4096>>();
      benchmark.template Run<Alex<uint32_t, 8192>>();
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_64_alex(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                       bool pareto) {
  benchmark.template Run<Alex<uint64_t, 1>>();
  if (pareto) {
    benchmark.template Run<Alex<uint64_t, 4>>();
    benchmark.template Run<Alex<uint64_t, 8>>();
    benchmark.template Run<Alex<uint64_t, 16>>();
    benchmark.template Run<Alex<uint64_t, 32>>();
    benchmark.template Run<Alex<uint64_t, 64>>();
    benchmark.template Run<Alex<uint64_t, 128>>();
    benchmark.template Run<Alex<uint64_t, 256>>();
    benchmark.template Run<Alex<uint64_t, 512>>();
    benchmark.template Run<Alex<uint64_t, 1024>>();
    benchmark.template Run<Alex<uint64_t, 2048>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<Alex<uint64_t, 4096>>();
      benchmark.template Run<Alex<uint64_t, 8192>>();
    }
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_alex, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_alex, uint64_t);
