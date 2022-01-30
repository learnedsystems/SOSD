#include "benchmark.h"

#include <cstdlib>

#include "benchmarks/benchmark_alex.h"
#include "benchmarks/benchmark_art.h"
#include "benchmarks/benchmark_btree.h"
#include "benchmarks/benchmark_cht.h"
#include "benchmarks/benchmark_fast64.h"
#include "benchmarks/benchmark_fst.h"
#include "benchmarks/benchmark_ibtree.h"
#include "benchmarks/benchmark_pgm.h"
#include "benchmarks/benchmark_rbs.h"
#include "benchmarks/benchmark_rmi.h"
#include "benchmarks/benchmark_rmi_alt.h"
#include "benchmarks/benchmark_rs.h"
#include "benchmarks/benchmark_ts.h"
#include "benchmarks/benchmark_wormhole.h"
#include "competitors/binary_search.h"
#include "competitors/hash.h"
#include "competitors/stanford_hash.h"
#include "config.h"
#include "searches/branching_binary_search.h"
#include "searches/branchless_binary_search.h"
#include "searches/interpolation_search.h"
#include "searches/linear_search.h"
#include "util.h"
#include "utils/cxxopts.hpp"
using namespace std;

#define check_only(tag, code)        \
  if ((!only_mode) || only == tag) { \
    code;                            \
  }
#define add_search_type(name, func, type, search_class)                   \
  {                                                                       \
    if (search_type == (name)) {                                          \
      auto search = search_class<type>();                                 \
      sosd::Benchmark<type, search_class> benchmark(                      \
          filename, lookups, num_repeats, perf, build, fence, cold_cache, \
          track_errors, csv, num_threads, search);                        \
      func(benchmark, pareto, only_mode, only, filename);                 \
      found_search_type = true;                                           \
      break;                                                              \
    }                                                                     \
  }

template <class Benchmark>
void execute_32_bit(Benchmark benchmark, bool pareto, bool only_mode,
                    std::string only, std::string filename) {
  // Build and probe individual indexes.
  check_only("RMI", benchmark_32_rmi(benchmark, pareto, filename));
  check_only("RS", benchmark_32_rs(benchmark, pareto));
  check_only("TS", benchmark_32_ts(benchmark, pareto));
  check_only("PGM", benchmark_32_pgm(benchmark, pareto));
  check_only("CHT", benchmark_32_cht(benchmark, pareto));
  check_only("BTree", benchmark_32_btree(benchmark, pareto));
  check_only("IBTree", benchmark_32_ibtree(benchmark, pareto));
  check_only("FAST", benchmark_32_fast(benchmark, pareto));
  check_only("ALEX", benchmark_32_alex(benchmark, pareto));
  check_only("RMI_CPP", benchmark_32_rmi_cpp(benchmark, pareto));
#ifndef __APPLE__
#ifndef DISABLE_FST
  check_only("FST", benchmark_32_fst(benchmark, pareto));
#endif
  check_only("Wormhole", benchmark_32_wormhole(benchmark, pareto));
#endif

  if (benchmark.uses_binary_search()) {
    check_only("RBS", benchmark_32_rbs(benchmark, pareto));
    check_only("CuckooMap", benchmark.template Run<CuckooHash>());
    check_only("RobinHash", benchmark.template Run<RobinHash<uint32_t>>());
    check_only("BS", benchmark.template Run<BinarySearch<uint32_t>>());
  }
}

template <class Benchmark>
void execute_64_bit(Benchmark benchmark, bool pareto, bool only_mode,
                    std::string only, std::string filename) {
  // Build and probe individual indexes.
  check_only("RMI", benchmark_64_rmi(benchmark, pareto, filename));
  check_only("RS", benchmark_64_rs(benchmark, pareto));
  check_only("TS", benchmark_64_ts(benchmark, pareto));
  check_only("PGM", benchmark_64_pgm(benchmark, pareto));
  check_only("CHT", benchmark_64_cht(benchmark, pareto));
  check_only("ART", benchmark_64_art(benchmark, pareto));
  check_only("BTree", benchmark_64_btree(benchmark, pareto));
  check_only("IBTree", benchmark_64_ibtree(benchmark, pareto));
  check_only("FAST", benchmark_64_fast(benchmark, pareto));
  check_only("ALEX", benchmark_64_alex(benchmark, pareto));
  check_only("RMI_CPP", benchmark_64_rmi_cpp(benchmark, pareto));
#ifndef __APPLE__
#ifndef DISABLE_FST
  check_only("FST", benchmark_64_fst(benchmark, pareto));
#endif
  check_only("Wormhole", benchmark_64_wormhole(benchmark, pareto));
#endif

  if (benchmark.uses_binary_search()) {
    check_only("RBS", benchmark_64_rbs(benchmark, pareto));
    check_only("RobinHash", benchmark.template Run<RobinHash<uint64_t>>());
    check_only("BS", benchmark.template Run<BinarySearch<uint64_t>>());
  }
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("benchmark", "Searching on sorted data benchmark");
  options.positional_help("<data> <lookups>");
  options.add_options()("data", "Data file with keys",
                        cxxopts::value<std::string>())(
      "lookups", "Lookup key (query) file", cxxopts::value<std::string>())(
      "help", "Displays help")("r,repeats", "Number of repeats",
                               cxxopts::value<int>()->default_value("1"))(
      "t,threads", "Number of lookup threads",
      cxxopts::value<int>()->default_value("1"))("p,perf",
                                                 "Track performance counters")(
      "b,build", "Only measure and report build times")(
      "only", "Only run the specified index",
      cxxopts::value<std::string>()->default_value(""))(
      "cold-cache", "Clear the CPU cache between each lookup")(
      "pareto", "Run with multiple different sizes for each competitor")(
      "fence", "Execute a memory barrier between each lookup")(
      "errors",
      "Tracks index errors, and report those instead of lookup times")(
      "csv", "Output a CSV of results in addition to a text file")(
      "search",
      "Specify a search type, one of: binary, branchless_binary, linear, "
      "interpolation",
      cxxopts::value<std::string>()->default_value("binary"))(
      "positional", "extra positional arguments",
      cxxopts::value<std::vector<std::string>>());

  options.parse_positional({"data", "lookups", "positional"});

  const auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({}) << "\n";
    exit(0);
  }

  const size_t num_repeats = result["repeats"].as<int>();
  cout << "Repeating lookup code " << num_repeats << " time(s)." << endl;

  const size_t num_threads = result["threads"].as<int>();
  cout << "Using " << num_threads << " thread(s)." << endl;

  const bool perf = result.count("perf");
  const bool build = result.count("build");
  const bool fence = result.count("fence");
  const bool track_errors = result.count("errors");
  const bool cold_cache = result.count("cold-cache");
  const bool csv = result.count("csv");
  const bool pareto = result.count("pareto");
  const std::string filename = result["data"].as<std::string>();
  const std::string lookups = result["lookups"].as<std::string>();
  const std::string search_type = result["search"].as<std::string>();
  const bool only_mode = result.count("only") || std::getenv("SOSD_ONLY");
  std::string only;

  if (result.count("only")) {
    only = result["only"].as<std::string>();
  } else if (std::getenv("SOSD_ONLY")) {
    only = std::string(std::getenv("SOSD_ONLY"));
  } else {
    only = "";
  }

  const DataType type = util::resolve_type(filename);

  if (lookups.find("lookups") == std::string::npos) {
    cerr << "Warning: lookups file seems misnamed. Did you specify the right "
            "one?\n";
  }

  if (only_mode)
    cout << "Only executing indexes matching " << only << std::endl;

  // Pin main thread to core 0.
  util::set_cpu_affinity(0);
  bool found_search_type = false;

  switch (type) {
    case DataType::UINT32: {
      // Create benchmark.
      if constexpr (sosd_config::fast_mode) {
        util::fail("32-bit is not supported when SOSD is built with fast mode");
      } else {
        add_search_type("binary", execute_32_bit, uint32_t,
                        BranchingBinarySearch);
        add_search_type("branchless_binary", execute_32_bit, uint32_t,
                        BranchlessBinarySearch);
        add_search_type("linear", execute_32_bit, uint32_t, LinearSearch);
        add_search_type("interpolation", execute_32_bit, uint32_t,
                        InterpolationSearch);
      }

      break;
    }
    case DataType::UINT64: {
      // Create benchmark.
      if constexpr (sosd_config::fast_mode) {
        add_search_type("binary", execute_64_bit, uint64_t,
                        BranchingBinarySearch);
      } else {
        add_search_type("binary", execute_64_bit, uint64_t,
                        BranchingBinarySearch);
        add_search_type("branchless_binary", execute_64_bit, uint64_t,
                        BranchlessBinarySearch);
        add_search_type("linear", execute_64_bit, uint64_t, LinearSearch);
        add_search_type("interpolation", execute_64_bit, uint64_t,
                        InterpolationSearch);
      }
      break;
    }
  }

  if (!found_search_type) {
    std::cerr << "Specified search type is not implemented in this build. "
                 "Disable fast mode for other search types."
              << std::endl;
  }

  return 0;
}
