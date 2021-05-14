#pragma once
#include "benchmark.h"

template <template <typename> typename Searcher>
void benchmark_64_art(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                      bool pareto);
