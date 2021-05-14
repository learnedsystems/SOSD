#include "benchmark.h"
#include "benchmarks/benchmark_btree.h"
#include "common.h"
#include "competitors/interpolation_btree.h"

template <template <typename> typename Searcher>
void benchmark_32_ibtree(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                         bool pareto) {
  benchmark.template Run<InterpolationBTree<uint32_t, 1>>();
  if (pareto) {
    benchmark.template Run<InterpolationBTree<uint32_t, 32>>();
    benchmark.template Run<InterpolationBTree<uint32_t, 64>>();
    benchmark.template Run<InterpolationBTree<uint32_t, 128>>();
    benchmark.template Run<InterpolationBTree<uint32_t, 256>>();
    benchmark.template Run<InterpolationBTree<uint32_t, 512>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<InterpolationBTree<uint32_t, 1024>>();
      benchmark.template Run<InterpolationBTree<uint32_t, 2048>>();
      benchmark.template Run<InterpolationBTree<uint32_t, 4096>>();
      benchmark.template Run<InterpolationBTree<uint32_t, 8192>>();
    }
  }
}

template <template <typename> typename Searcher>
void benchmark_64_ibtree(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                         bool pareto) {
  benchmark.template Run<InterpolationBTree<uint64_t, 1>>();
  if (pareto) {
    benchmark.template Run<InterpolationBTree<uint64_t, 32>>();
    benchmark.template Run<InterpolationBTree<uint64_t, 64>>();
    benchmark.template Run<InterpolationBTree<uint64_t, 128>>();
    benchmark.template Run<InterpolationBTree<uint64_t, 256>>();
    benchmark.template Run<InterpolationBTree<uint64_t, 512>>();
    if (benchmark.uses_binary_search()) {
      benchmark.template Run<InterpolationBTree<uint64_t, 1024>>();
      benchmark.template Run<InterpolationBTree<uint64_t, 2048>>();
      benchmark.template Run<InterpolationBTree<uint64_t, 4096>>();
      benchmark.template Run<InterpolationBTree<uint64_t, 8192>>();
    }
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_ibtree, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_ibtree, uint64_t);
