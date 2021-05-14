#pragma once
#include "benchmark.h"

template <template <typename> typename Searcher>
void benchmark_32_rbs(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                      bool pareto);

template <template <typename> typename Searcher>
void benchmark_64_rbs(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                      bool pareto);
