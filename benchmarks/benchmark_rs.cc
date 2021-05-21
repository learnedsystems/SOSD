#include "benchmarks/benchmark_rs.h"

#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/rs.h"

template <template <typename> typename Searcher>
void benchmark_32_rs(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                     bool pareto) {
  benchmark.template Run<RS<uint32_t, 10>>();
  if (pareto) {
    benchmark.template Run<RS<uint32_t, 9>>();
    benchmark.template Run<RS<uint32_t, 8>>();
    benchmark.template Run<RS<uint32_t, 7>>();
    benchmark.template Run<RS<uint32_t, 6>>();
    benchmark.template Run<RS<uint32_t, 5>>();
    benchmark.template Run<RS<uint32_t, 4>>();
    benchmark.template Run<RS<uint32_t, 3>>();
    benchmark.template Run<RS<uint32_t, 2>>();
    benchmark.template Run<RS<uint32_t, 1>>();
  }
}

template <template <typename> typename Searcher>
void benchmark_64_rs(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                     bool pareto) {
  benchmark.template Run<RS<uint64_t, 10>>();
  if (pareto) {
    benchmark.template Run<RS<uint64_t, 9>>();
    benchmark.template Run<RS<uint64_t, 8>>();
    benchmark.template Run<RS<uint64_t, 7>>();
    benchmark.template Run<RS<uint64_t, 6>>();
    benchmark.template Run<RS<uint64_t, 5>>();
    benchmark.template Run<RS<uint64_t, 4>>();
    benchmark.template Run<RS<uint64_t, 3>>();
    benchmark.template Run<RS<uint64_t, 2>>();
    benchmark.template Run<RS<uint64_t, 1>>();
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_rs, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_rs, uint64_t);
