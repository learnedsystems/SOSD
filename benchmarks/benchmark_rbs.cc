#include "benchmarks/benchmark_rbs.h"

#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/radix_binary_search.h"

template <template <typename> typename Searcher>
void benchmark_32_rbs(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                      bool pareto) {
  benchmark.template Run<RadixBinarySearch<uint32_t, 28>>();
  if (pareto) {
    benchmark.template Run<RadixBinarySearch<uint32_t, 26>>();
    benchmark.template Run<RadixBinarySearch<uint32_t, 24>>();
    benchmark.template Run<RadixBinarySearch<uint32_t, 22>>();
    benchmark.template Run<RadixBinarySearch<uint32_t, 20>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<RadixBinarySearch<uint32_t, 18>>();
      benchmark.template Run<RadixBinarySearch<uint32_t, 16>>();
      benchmark.template Run<RadixBinarySearch<uint32_t, 14>>();
      benchmark.template Run<RadixBinarySearch<uint32_t, 12>>();
      benchmark.template Run<RadixBinarySearch<uint32_t, 10>>();
      benchmark.template Run<RadixBinarySearch<uint32_t, 8>>();
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_64_rbs(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                      bool pareto) {
  benchmark.template Run<RadixBinarySearch<uint64_t, 28>>();
  if (pareto) {
    benchmark.template Run<RadixBinarySearch<uint64_t, 26>>();
    benchmark.template Run<RadixBinarySearch<uint64_t, 24>>();
    benchmark.template Run<RadixBinarySearch<uint64_t, 22>>();
    benchmark.template Run<RadixBinarySearch<uint64_t, 20>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<RadixBinarySearch<uint64_t, 18>>();
      benchmark.template Run<RadixBinarySearch<uint64_t, 16>>();
      benchmark.template Run<RadixBinarySearch<uint64_t, 14>>();
      benchmark.template Run<RadixBinarySearch<uint64_t, 12>>();
      benchmark.template Run<RadixBinarySearch<uint64_t, 10>>();
      benchmark.template Run<RadixBinarySearch<uint64_t, 8>>();
    }
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_rbs, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_rbs, uint64_t);
