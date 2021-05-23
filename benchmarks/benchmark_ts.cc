#include "benchmarks/benchmark_ts.h"

#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/ts.h"

template <template <typename> typename Searcher>
void benchmark_32_ts(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                     bool pareto) {
  benchmark.template Run<TS<uint32_t, 10>>();
  if (pareto) {
    benchmark.template Run<TS<uint32_t, 9>>();
    benchmark.template Run<TS<uint32_t, 8>>();
    benchmark.template Run<TS<uint32_t, 7>>();
    benchmark.template Run<TS<uint32_t, 6>>();
    benchmark.template Run<TS<uint32_t, 5>>();
    benchmark.template Run<TS<uint32_t, 4>>();
    benchmark.template Run<TS<uint32_t, 3>>();
    benchmark.template Run<TS<uint32_t, 2>>();
    benchmark.template Run<TS<uint32_t, 1>>();
  }
}

template <template <typename> typename Searcher>
void benchmark_64_ts(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                     bool pareto) {
  benchmark.template Run<TS<uint64_t, 10>>();
  if (pareto) {
    benchmark.template Run<TS<uint64_t, 9>>();
    benchmark.template Run<TS<uint64_t, 8>>();
    benchmark.template Run<TS<uint64_t, 7>>();
    benchmark.template Run<TS<uint64_t, 6>>();
    benchmark.template Run<TS<uint64_t, 5>>();
    benchmark.template Run<TS<uint64_t, 4>>();
    benchmark.template Run<TS<uint64_t, 3>>();
    benchmark.template Run<TS<uint64_t, 2>>();
    benchmark.template Run<TS<uint64_t, 1>>();
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_ts, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_ts, uint64_t);
