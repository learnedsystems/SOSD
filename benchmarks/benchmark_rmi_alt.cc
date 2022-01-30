#include "benchmarks/benchmark_rmi_alt.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/analysis-rmi/include/rmi/models.hpp"
#include "competitors/analysis-rmi/include/rmi/rmi.hpp"
#include "competitors/rmi_cpp.h"

template <template <typename> typename Searcher>
void benchmark_32_rmi_cpp(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                          bool pareto) {
  benchmark.template Run<RMICpp<uint32_t, rmi::LinearSpline, rmi::LinearRegression,
                   rmi::RmiLAbs, 128, 1>>();
  if (pareto) {
    benchmark.template Run<RMICpp<uint32_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 256, 2>>();
    benchmark.template Run<RMICpp<uint32_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 512, 3>>();
    benchmark.template Run<RMICpp<uint32_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 1024, 4>>();
    benchmark.template Run<RMICpp<uint32_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 2048, 5>>();
    benchmark.template Run<RMICpp<uint32_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 4096, 6>>();
    benchmark.template Run<RMICpp<uint32_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 8192, 7>>();
    benchmark.template Run<RMICpp<uint32_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 16384, 8>>();
  }
}

template <template <typename> typename Searcher>
void benchmark_64_rmi_cpp(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                          bool pareto) {
  benchmark.template Run<RMICpp<uint64_t, rmi::LinearSpline, rmi::LinearRegression,
                   rmi::RmiLAbs, 128, 1>>();
  if (pareto) {
    benchmark.template Run<RMICpp<uint64_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 256, 2>>();
    benchmark.template Run<RMICpp<uint64_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 512, 3>>();
    benchmark.template Run<RMICpp<uint64_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 1024, 4>>();
    benchmark.template Run<RMICpp<uint64_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 2048, 5>>();
    benchmark.template Run<RMICpp<uint64_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 4096, 6>>();
    benchmark.template Run<RMICpp<uint64_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 8192, 7>>();
    benchmark.template Run<RMICpp<uint64_t, rmi::LinearSpline, rmi::LinearRegression,
                     rmi::RmiLAbs, 16384, 8>>();
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_rmi_cpp, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_rmi_cpp, uint64_t);
