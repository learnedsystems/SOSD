#include "benchmarks/benchmark_rmi.h"

#include <string>

#include "benchmark.h"
#include "benchmarks/common.h"
#include "competitors/rmi_search.h"

#define NAME2(a, b) NAME2_HIDDEN(a, b)
#define NAME2_HIDDEN(a, b) a##b

#define NAME3(a, b, c) NAME3_HIDDEN(a, b, c)
#define NAME3_HIDDEN(a, b, c) a##b##c

#define NAME5(a, b, c, d, e) NAME5_HIDDEN(a, b, c, d, e)
#define NAME5_HIDDEN(a, b, c, d, e) a##b##c##d##e

#define run_rmi_binary(dtype, name, suffix, variant)                         \
  if (filename.find("/" #name "_" #dtype) != std::string::npos) {            \
    benchmark                                                                \
        .template Run<RMI_B<NAME2(dtype, _t), variant,                       \
                            NAME5(name, _, dtype, _, suffix)::BUILD_TIME_NS, \
                            NAME5(name, _, dtype, _, suffix)::RMI_SIZE,      \
                            NAME5(name, _, dtype, _, suffix)::NAME,          \
                            NAME5(name, _, dtype, _, suffix)::lookup,        \
                            NAME5(name, _, dtype, _, suffix)::load,          \
                            NAME5(name, _, dtype, _, suffix)::cleanup>>();   \
  }

#define run_rmi_binary_pareto(dtype, name)  \
  {                                         \
    run_rmi_binary(dtype, name, 0, 0);      \
    if (pareto) {                           \
      run_rmi_binary(dtype, name, 1, 1);    \
      run_rmi_binary(dtype, name, 2, 2);    \
      run_rmi_binary(dtype, name, 3, 3);    \
      run_rmi_binary(dtype, name, 4, 4);    \
      if (benchmark.uses_binary_search()) { \
        run_rmi_binary(dtype, name, 5, 5);  \
        run_rmi_binary(dtype, name, 6, 6);  \
        run_rmi_binary(dtype, name, 7, 7);  \
        run_rmi_binary(dtype, name, 8, 8);  \
        run_rmi_binary(dtype, name, 9, 9);  \
      }                                     \
    }                                       \
  }

template <template <typename> typename Searcher>
void benchmark_32_rmi(sosd::Benchmark<uint32_t, Searcher>& benchmark,
                      bool pareto, const std::string& filename) {
  run_rmi_binary_pareto(uint32, books_200M);
  run_rmi_binary_pareto(uint32, normal_200M);
  run_rmi_binary_pareto(uint32, lognormal_200M);
  run_rmi_binary_pareto(uint32, uniform_dense_200M);
  run_rmi_binary_pareto(uint32, uniform_sparse_200M);
}

template <template <typename> typename Searcher>
void benchmark_64_rmi(sosd::Benchmark<uint64_t, Searcher>& benchmark,
                      bool pareto, const std::string& filename) {
  run_rmi_binary_pareto(uint64, fb_200M);
  run_rmi_binary_pareto(uint64, osm_cellids_200M);
  run_rmi_binary_pareto(uint64, wiki_ts_200M);
  run_rmi_binary_pareto(uint64, books_200M);

  run_rmi_binary_pareto(uint64, osm_cellids_400M);
  run_rmi_binary_pareto(uint64, osm_cellids_600M);
  run_rmi_binary_pareto(uint64, osm_cellids_800M);

  run_rmi_binary_pareto(uint64, books_400M);
  run_rmi_binary_pareto(uint64, books_600M);
  run_rmi_binary_pareto(uint64, books_800M);

  run_rmi_binary_pareto(uint64, normal_200M);
  run_rmi_binary_pareto(uint64, lognormal_200M);
  run_rmi_binary_pareto(uint64, uniform_dense_200M);
  run_rmi_binary_pareto(uint64, uniform_sparse_200M);
}

INSTANTIATE_TEMPLATES_RMI(benchmark_32_rmi, uint32_t);
INSTANTIATE_TEMPLATES_RMI(benchmark_64_rmi, uint64_t);
