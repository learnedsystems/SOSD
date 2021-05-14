#include "benchmarks/benchmark_art.h"

#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/art_primary_lb.h"

template <template <typename> typename Searcher>
void benchmark_64_art(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                      bool pareto) {
  benchmark.template Run<ARTPrimaryLB<1>>();
  if (pareto) {
    benchmark.template Run<ARTPrimaryLB<32>>();
    benchmark.template Run<ARTPrimaryLB<64>>();
    benchmark.template Run<ARTPrimaryLB<128>>();
    benchmark.template Run<ARTPrimaryLB<256>>();
    benchmark.template Run<ARTPrimaryLB<512>>();

    if (benchmark.uses_binary_search()) {
      benchmark.template Run<ARTPrimaryLB<1024>>();
      benchmark.template Run<ARTPrimaryLB<2048>>();
      benchmark.template Run<ARTPrimaryLB<4096>>();
      benchmark.template Run<ARTPrimaryLB<8192>>();
    }
  }
}

INSTANTIATE_TEMPLATES(benchmark_64_art, uint64_t);
