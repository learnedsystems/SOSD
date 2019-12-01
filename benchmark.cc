#include "benchmark.h"
#include "util.h"
#include "utils/cxxopts.hpp"

#include "competitors/oracle.h"
#include "competitors/binary_search.h"
#include "competitors/interpolation_search.h"
#include "competitors/rmi_search.h"
#include "competitors/radix_binary_search.h"
#include "competitors/spline/radix_spline.h"
#include "competitors/art.h"
#include "competitors/art32.h"
#include "competitors/fast.h"
#include "competitors/stx_btree.h"
#include "competitors/rm_tip.h"

using namespace std;

#define NAME2(a, b)         NAME2_HIDDEN(a,b)
#define NAME2_HIDDEN(a, b)  a ## b

#define NAME3(a, b, c)         NAME3_HIDDEN(a,b,c)
#define NAME3_HIDDEN(a, b, c)  a ## b ## c

#define NAME5(a, b, c, d, e)         NAME5_HIDDEN(a,b,c,d,e)
#define NAME5_HIDDEN(a, b, c, d, e)  a ## b ## c ## d ## e

#define run_rmi_linear(dtype, name, suffix)  if (filename.find("/" #name "_" #dtype) != std::string::npos) { benchmark.Run<RMI_L<NAME2(dtype, _t), NAME5(name, _, dtype, _, suffix)::BUILD_TIME_NS, NAME5(name, _, dtype, _,suffix)::RMI_SIZE, NAME5(name, _, dtype, _,suffix)::NAME, NAME5(name, _, dtype, _, suffix)::lookup>>(); }

#define run_rmi_binary(dtype, name, suffix)  if (filename.find("/" #name "_" #dtype) != std::string::npos) { benchmark.Run<RMI_B<NAME2(dtype, _t), NAME5(name, _, dtype, _, suffix)::BUILD_TIME_NS, NAME5(name, _, dtype, _, suffix)::RMI_SIZE, NAME5(name, _, dtype, _,suffix)::NAME, NAME5(name, _, dtype, _, suffix)::lookup>>(); }

int main(int argc, char* argv[]) {
  cxxopts::Options options("benchmark", "Searching on sorted data benchmark");
  options.positional_help("<data> <lookups>");
  options.add_options()
      ("data", "Data file with keys", cxxopts::value<std::string>())
      ("lookups", "Lookup key (query) file", cxxopts::value<std::string>())
      ("help", "Displays help")
      ("r,repeats",
       "Number of repeats",
       cxxopts::value<int>()->default_value("1"))
      ("p,perf", "Track performance counters")
      ("b,build", "Only measure and report build times")
      ("histogram", "Measure each lookup and output histogram data")
      ("positional",
       "extra positional arguments",
       cxxopts::value<std::vector<std::string>>());

  options.parse_positional({"data", "lookups", "positional"});

  const auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({}) << "\n";
    exit(0);
  }

  const size_t num_repeats = result["repeats"].as<int>();
  cout << "Repeating lookup code " << num_repeats << " time(s)." << endl;

  const bool perf = result.count("perf");
  const bool build = result.count("build");
  const bool histogram = result.count("histogram");
  const std::string filename = result["data"].as<std::string>();
  const std::string lookups = result["lookups"].as<std::string>();

  const DataType type = util::resolve_type(filename);

  if (lookups.find("lookups")==std::string::npos) {
    cerr
        << "Warning: lookups file seems misnamed. Did you specify the right one?\n";
  }

  // Pin main thread to core 0.
  util::set_cpu_affinity(0);

  switch (type) {
    case DataType::UINT32: {
      // Create benchmark.
      sosd::Benchmark<uint32_t>
          benchmark(filename, lookups, num_repeats, perf, build, histogram);

      // Build and probe individual indexes.
      benchmark.Run<OracleSearch<uint32_t>, true>();

      // RMIs
      run_rmi_linear(uint32, normal_200M, rmi);
      run_rmi_linear(uint32, lognormal_200M, rmi);
      run_rmi_binary(uint32, books_200M, rmi);
      run_rmi_binary(uint32, fb_200M, rmi);
      run_rmi_linear(uint32, uniform_dense_200M, rmi);
      run_rmi_linear(uint32, uniform_sparse_200M, rmi);

      benchmark.Run<RadixSpline<uint32_t>>();
      benchmark.Run<BinarySearch<uint32_t>>();
      benchmark.Run<InterpolationSearch<uint32_t>>();
      benchmark.Run<RadixBinarySearch<uint32_t>>();
      benchmark.Run<Fast>();
      benchmark.Run<ART32>();
      benchmark.Run<RMThreePointInterpolationSearch<uint32_t>>();
      benchmark.Run<STXBTree<uint32_t>>();

      break;
    }
    case DataType::UINT64: {
      // Create benchmark.
      sosd::Benchmark<uint64_t>
          benchmark(filename, lookups, num_repeats, perf, build, histogram);

      // Build and probe individual indexes.
      benchmark.Run<OracleSearch<uint64_t>, true>();

      // RMIs
      run_rmi_binary(uint64, wiki_ts_200M, rmi);
      run_rmi_binary(uint64, osm_cellids_200M, rmi);
      run_rmi_linear(uint64, normal_200M, rmi);
      run_rmi_binary(uint64, lognormal_200M, rmi);
      run_rmi_binary(uint64, books_200M, rmi);
      run_rmi_binary(uint64, fb_200M, rmi);
      run_rmi_linear(uint64, uniform_dense_200M, rmi);
      run_rmi_linear(uint64, uniform_sparse_200M, rmi);

      benchmark.Run<RadixSpline<uint64_t>>();
      benchmark.Run<RadixBinarySearch<uint64_t>>();
      benchmark.Run<ART>();
      benchmark.Run<BinarySearch<uint64_t>>();
      benchmark.Run<InterpolationSearch<uint64_t>>();
      benchmark.Run<RMThreePointInterpolationSearch<uint64_t>>();
      benchmark.Run<STXBTree<uint64_t>>();

      break;
    }
  }

  return 0;
}
