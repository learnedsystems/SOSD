#include "benchmarks/benchmark_rmi_alt.h"

#include "benchmark.h"
#include "common.h"
#include "competitors/analysis-rmi/include/rmi/models.hpp"
#include "competitors/analysis-rmi/include/rmi/rmi_robust.hpp"
#include "competitors/rmi_cpp_robust.h"

template <template <typename> typename Searcher>
void benchmark_32_rmi_cpp(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                          bool pareto) {
  benchmark.template Run<
      RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                   rmi::RmiLAbsRobust, 128, 1>>();
  if (pareto) {
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 256, 2>>();
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 512, 3>>();
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 1024, 4>>();
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 2048, 5>>();
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 4096, 6>>();
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 8192, 7>>();
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 16384, 8>>();
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 32768, 9>>();
    benchmark.template Run<
        RMICppRobust<uint32_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 65536, 10>>();
  }
}

template <template <typename> typename Searcher>
void benchmark_64_rmi_cpp(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                          bool pareto) {
  benchmark.template Run<
      RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
                   rmi::RmiLAbsRobust, 128, 1>>();
  if (pareto) {
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 256, 2>>();
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 512, 3>>();
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 1024, 4>>();
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 2048, 5>>();
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 4096, 6>>();
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 8192, 7>>();
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
                     rmi::RmiLAbsRobust, 16384, 8>>();
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
            rmi::RmiLAbsRobust, 32768, 9>>();
    benchmark.template Run<
        RMICppRobust<uint64_t, rmi::LinearRegression, rmi::LinearRegression,
            rmi::RmiLAbsRobust, 65536, 10>>();
  }
}

INSTANTIATE_TEMPLATES(benchmark_32_rmi_cpp, uint32_t);
INSTANTIATE_TEMPLATES(benchmark_64_rmi_cpp, uint64_t);
