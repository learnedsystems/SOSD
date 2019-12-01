#pragma once

#include "util.h"
#include "utils/perf_event.h"

#include <algorithm>
#include <sstream>

#ifdef __linux__
#define checkLinux(x) (x)
#else
#define checkLinux(x) { util::fail("Only supported on Linux."); }
#endif

namespace sosd {

// KeyType: Controls the type of the key (the value will always be uint64_t)
//          Use uint64_t for 64 bit types and uint32_t for 32 bit types
//          KeyType must implement operator<
template<typename KeyType = uint64_t>
class Benchmark {
 public:
  Benchmark(const std::string& data_filename,
            const std::string& lookups_filename,
            const size_t num_repeats,
            const bool perf, const bool build,
            const bool measure_each)
      : data_filename_(data_filename),
        lookups_filename_(lookups_filename),
        num_repeats_(num_repeats),
        first_run_(true), perf(perf), build(build),
        measure_each(measure_each) {
    // Load data.
    std::vector<KeyType> keys = util::load_data<KeyType>(data_filename_);

    if (!is_sorted(keys.begin(), keys.end()))
      util::fail("keys have to be sorted");
    // Check whether keys are unique.
    unique_keys_ = util::is_unique(keys);
    if (unique_keys_)
      std::cout << "data is unique" << std::endl;
    else
      std::cout << "data contains duplicates" << std::endl;
    // Add artificial values to keys.
    data_ = util::add_values(keys);
    // Load lookups.
    lookups_ = util::load_data<EqualityLookup<KeyType>>(lookups_filename_);
  }

  template<class Index, bool ignore_errors = false>
  void Run() {
    // Build index.
    Index index;

    each_timing.clear();

    if (!index.applicable(unique_keys_, data_filename_)) {
      std::cout << "index " << index.name() << " is not applicable"
                << std::endl;
      return;
    }

    build_ns_ = util::timing([&] {
      index.Build(data_);
    });

    // RMIs have additional, external build time
    build_ns_ += index.additional_build_time();

    if (measure_each && perf) {
      util::fail("Can only specify one of measure each or perf counters.");
    }

    // Do equality lookups.
    if (perf) {
      checkLinux(({
        BenchmarkParameters params;
        params.setParam("index", index.name());
        PerfEventBlock e(lookups_.size(), params,/*printHeader=*/first_run_);
        DoEqualityLookups<Index, ignore_errors, false>(index);
      }));
    } else {
      if (measure_each) {
        DoEqualityLookups<Index, ignore_errors, true>(index);
      } else {
        DoEqualityLookups<Index, ignore_errors, false>(index);
      }
      PrintResult(index);
    }
    first_run_ = false;
  }

 private:
  template<class Index, bool ignore_errors, bool time_each>
  void DoEqualityLookups(Index& index) {
    if (build) return;

    size_t repeats = num_repeats_;
    if (index.name()=="InterpolationSearch")
      repeats = 1;

    runs_.resize(repeats);
    for (unsigned int i = 0; i < repeats; ++i) {
      bool run_failed = false;
      runs_[i] = util::timing([&] {
        for (const auto& lookup : lookups_) {
          // Compute the actual index for debugging.
          const volatile uint64_t lookup_key = lookup.key;
          const volatile uint64_t expected = lookup.result;
          volatile uint64_t actual;
          if (time_each) {
            const auto start = std::chrono::high_resolution_clock::now();
            actual = index.EqualityLookup(lookup_key);
            const auto end = std::chrono::high_resolution_clock::now();
            each_timing.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start).count());
          } else {
            actual = index.EqualityLookup(lookup_key);
          }

          if (actual!=expected && !ignore_errors) {
            const auto pos = std::find_if(
                data_.begin(),
                data_.end(),
                [lookup_key](const auto& kv) { return kv.key==lookup_key; }
            );

            const auto idx = std::distance(data_.begin(), pos);

            std::cerr << "equality lookup returned wrong result:" << std::endl;
            std::cerr << "lookup key: " << lookup_key << std::endl;
            std::cerr << "actual: " << actual << ", expected: " << expected
                      << std::endl
                      << "correct index is: " << idx << " with "
                      << data_[idx].value
                      << " " << index.name()
                      << std::endl;

            run_failed = true;
            break;
          }
        }
      });
      if (run_failed) {
        runs_ = std::vector<uint64_t>(repeats, 0);
        return;
      }
    }
  }

  template<class Index>
  void PrintResult(const Index& index) {
    if (measure_each) {
      const std::string filename = index.name() + ".dat";
      util::write_data(each_timing, filename);
      return;
    }

    std::ostringstream all_times;
    for (unsigned int i = 0; i < runs_.size(); ++i) {
      const double
          ns_per_lookup = static_cast<double>(runs_[i])/lookups_.size();
      all_times << "," << ns_per_lookup;
    }

    if (build) {
      std::cout << "RESULT: " << index.name() << "," << build_ns_
                << "," << index.size()
                << std::endl;
      return;
    }

    std::cout << "RESULT: " << index.name() << all_times.str()
              << "," << index.size()
              << std::endl;
  }

  const std::string data_filename_;
  const std::string lookups_filename_;
  std::vector<KeyValue<KeyType>> data_;
  bool unique_keys_;
  std::vector<EqualityLookup<KeyType>> lookups_;
  uint64_t build_ns_;
  // Run times.
  std::vector<uint64_t> runs_;
  // Number of times we repeat the lookup code.
  size_t num_repeats_;
  // Used to only print profiling header information for first run.
  bool first_run_;
  bool perf;
  bool build;
  bool measure_each;

  std::vector<uint64_t> each_timing;
};

} // namespace sosd
