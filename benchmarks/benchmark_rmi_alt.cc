#pragma once
#include <string>

#include "benchmark.h"
#include "competitors/rmi_alt.h"
#include "competitors/analysis-rmi/include/rmi/models.hpp"
#include "competitors/analysis-rmi/include/rmi/rmi.hpp"

template <template <typename> typename Searcher>
void benchmark_32_rmi_alt(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                          bool pareto, const std::string& filename) {
    benchmark.template Run<RMIAlternate<uint32_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 128, 1>>();
    if (pareto) {
      benchmark.template Run<RMIAlternate<uint32_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 256, 1>>();
      benchmark.template Run<RMIAlternate<uint32_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 512, 1>>();
      benchmark.template Run<RMIAlternate<uint32_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 1024, 1>>();
      benchmark.template Run<RMIAlternate<uint32_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 2048, 1>>();
      benchmark.template Run<RMIAlternate<uint32_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 4096, 1>>();
      benchmark.template Run<RMIAlternate<uint32_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 8192, 1>>();
      benchmark.template Run<RMIAlternate<uint32_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 16384, 1>>();
    }
}

template <template <typename> typename Searcher>
void benchmark_64_rmi_alt(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                          bool pareto, const std::string& filename) {
  benchmark.template Run<RMIAlternate<uint64_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 128, 1>>();
  if (pareto) {
    benchmark.template Run<RMIAlternate<uint64_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 256, 1>>();
    benchmark.template Run<RMIAlternate<uint64_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 512, 1>>();
    benchmark.template Run<RMIAlternate<uint64_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 1024, 1>>();
    benchmark.template Run<RMIAlternate<uint64_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 2048, 1>>();
    benchmark.template Run<RMIAlternate<uint64_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 4096, 1>>();
    benchmark.template Run<RMIAlternate<uint64_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 8192, 1>>();
    benchmark.template Run<RMIAlternate<uint64_t, rmi::LinearSpline, rmi::LinearRegression, rmi::RmiLAbs, 16384, 1>>();
  }
}
