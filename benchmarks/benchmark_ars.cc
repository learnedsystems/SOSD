#include "benchmarks/benchmark_ars.h"
#include "benchmark.h"
#include "competitors/ars/rs.h"

void benchmark_64_ars(sosd::Benchmark<uint64_t>& benchmark, bool pareto) {
  benchmark.Run<AdaptiveRadixSpline<uint64_t, 100>>();
  if (pareto) {
    benchmark.Run<AdaptiveRadixSpline<uint64_t, 10>>();
    benchmark.Run<AdaptiveRadixSpline<uint64_t, 50>>();
    benchmark.Run<AdaptiveRadixSpline<uint64_t, 500>>();
  }
}
