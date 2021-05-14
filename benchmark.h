#pragma once

#include <immintrin.h>
#include <math.h>

#include <algorithm>
#include <dtl/thread.hpp>
#include <fstream>
#include <iostream>
#include <sstream>

#include "config.h"
#include "searches/branching_binary_search.h"
#include "util.h"
#include "utils/perf_event.h"

#ifdef __linux__
#define checkLinux(x) (x)
#else
#define checkLinux(x) \
  { util::fail("Only supported on Linux."); }
#endif

// Get the CPU affinity for the process.
static const auto cpu_mask = dtl::this_thread::get_cpu_affinity();

// Batch size in number of lookups.
static constexpr std::size_t batch_size = 1u << 16;

namespace sosd {

// KeyType: Controls the type of the key (the value will always be uint64_t)
//          Use uint64_t for 64 bit types and uint32_t for 32 bit types
//          KeyType must implement operator<
template <typename KeyType = uint64_t,
          template <typename> typename SearchClass = BranchingBinarySearch>
class Benchmark {
 public:
  Benchmark(const std::string& data_filename,
            const std::string& lookups_filename, const size_t num_repeats,
            const bool perf, const bool build, const bool fence,
            const bool cold_cache, const bool track_errors, const bool csv,
            const size_t num_threads, const SearchClass<KeyType> searcher)
      : data_filename_(data_filename),
        lookups_filename_(lookups_filename),
        num_repeats_(num_repeats),
        first_run_(true),
        perf_(perf),
        build_(build),
        fence_(fence),
        cold_cache_(cold_cache),
        track_errors_(track_errors),
        csv_(csv),
        num_threads_(num_threads),
        searcher_(searcher) {
    if ((int)cold_cache + (int)perf + (int)fence > 1) {
      util::fail(
          "Can only specify one of cold cache, perf counters, or fence.");
    }

    static constexpr const char* prefix = "data/";
    dataset_name_ = data_filename.data();
    dataset_name_.erase(
        dataset_name_.begin(),
        dataset_name_.begin() + dataset_name_.find(prefix) + strlen(prefix));

    // Load data.
    std::vector<KeyType> keys = util::load_data<KeyType>(data_filename_);

    log_sum_search_bound_ = 0.0;
    l1_sum_search_bound_ = 0.0;
    l2_sum_search_bound_ = 0.0;
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

    // Create the data for the index (key -> position).
    for (uint64_t pos = 0; pos < data_.size(); pos++) {
      index_data_.push_back((KeyValue<KeyType>){data_[pos].key, pos});
    }

    if (cold_cache) {
      memory_.resize(26e6 / 8);  // NOTE: L3 size of the machine
      util::FastRandom ranny(8128);
      for (uint64_t& iter : memory_) {
        iter = ranny.RandUint32();
      }
    }
  }

  template <class Index>
  void Run() {
    // Build index.
    Index index;

    if (!index.applicable(unique_keys_, data_filename_)) {
      std::cout << "index " << index.name() << " is not applicable"
                << std::endl;
      return;
    }

    build_ns_ = index.Build(index_data_);

    // Do equality lookups.
    if constexpr (!sosd_config::fast_mode) {
      if (track_errors_) {
        return DoLookupsWithErrorTracking(index);
      }
      if (perf_) {
        checkLinux(({
          BenchmarkParameters params;
          params.setParam("index", index.name());
          params.setParam("variant", index.variant());
          PerfEventBlock e(lookups_.size(), params, /*printHeader=*/first_run_);
          DoEqualityLookups<Index, false, false, false>(index);
        }));
      } else if (cold_cache_) {
        if (num_threads_ > 1)
          util::fail("cold cache not supported with multiple threads");
        DoEqualityLookups<Index, true, false, true>(index);
        PrintResult(index);
      } else if (fence_) {
        DoEqualityLookups<Index, false, true, false>(index);
        PrintResult(index);
      } else {
        DoEqualityLookups<Index, false, false, false>(index);
        PrintResult(index);
      }
    } else {
      if (perf_ || cold_cache_ || fence_) {
        util::fail(
            "Perf, cold cache, and fence mode require full builds. Disable "
            "fast mode.");
      }
      DoEqualityLookups<Index, false, false, false>(index);
      PrintResult(index);
    }

    first_run_ = false;
  }

  bool uses_binary_search() const {
    return (searcher_.name() == "BinarySearch") ||
           (searcher_.name() == "BranchlessBinarySearch");
  }

  bool uses_lienar_search() const { return searcher_.name() == "LinearSearch"; }

 private:
  bool CheckResults(uint64_t actual, uint64_t expected, KeyType lookup_key,
                    SearchBound bound) {
    if (actual != expected) {
      const auto pos = std::find_if(
          data_.begin(), data_.end(),
          [lookup_key](const auto& kv) { return kv.key == lookup_key; });

      const auto idx = std::distance(data_.begin(), pos);

      std::cerr << "equality lookup returned wrong result:" << std::endl;
      std::cerr << "lookup key: " << lookup_key << std::endl;
      std::cerr << "actual: " << actual << ", expected: " << expected
                << std::endl
                << "correct index is: " << idx << std::endl
                << "index start: " << bound.start << " stop: " << bound.stop
                << std::endl;

      return false;
    }

    return true;
  }

  template <class Index, bool time_each, bool fence, bool clear_cache>
  void DoEqualityLookups(Index& index) {
    if (build_) return;

    // Atomic counter used to assign work to threads.
    std::atomic<std::size_t> cntr(0);

    bool run_failed = false;

    if (clear_cache) std::cout << "rsum was: " << random_sum_ << std::endl;

    runs_.resize(num_repeats_);
    for (unsigned int i = 0; i < num_repeats_; ++i) {
      random_sum_ = 0;
      individual_ns_sum_ = 0;

      uint64_t ms;
      if (num_threads_ == 1) {
        ms = util::timing([&] {
          DoEqualityLookupsCoreLoop<Index, time_each, fence, clear_cache>(
              index, 0, lookups_.size(), run_failed);
        });
      } else {
        // Reset atomic counter.
        cntr.store(0);

        ms = util::timing([&] {
          while (true) {
            const size_t begin = cntr.fetch_add(batch_size);
            if (begin >= lookups_.size()) break;
            unsigned int limit = std::min(begin + batch_size, lookups_.size());
            DoEqualityLookupsCoreLoop<Index, time_each, fence, clear_cache>(
                index, begin, limit, run_failed);
          }
        });
      }

      runs_[i] = ms;
      if (run_failed) {
        runs_ = std::vector<uint64_t>(num_repeats_, 0);
        return;
      }
    }
  }

  template <class Index, bool time_each, bool fence, bool clear_cache>
  void DoEqualityLookupsCoreLoop(Index& index, unsigned int start,
                                 unsigned int limit, bool& run_failed) {
    SearchBound bound = {};
    size_t qualifying;
    uint64_t result;
    typename std::vector<Row<KeyType>>::iterator iter;

    for (unsigned int idx = start; idx < limit; ++idx) {
      // Compute the actual index for debugging.
      const volatile uint64_t lookup_key = lookups_[idx].key;
      const volatile uint64_t expected = lookups_[idx].result;

      if constexpr (clear_cache) {
        // Make sure that all cache lines from large buffer are loaded
        for (uint64_t& iter : memory_) {
          random_sum_ += iter;
        }
        _mm_mfence();

        const auto start = std::chrono::high_resolution_clock::now();
        bound = index.EqualityLookup(lookup_key);
        uint64_t actual = searcher_.search(data_, lookup_key, &qualifying,
                                           bound.start, bound.stop);
        if (!CheckResults(actual, expected, lookup_key, bound)) {
          run_failed = true;
          return;
        }
        const auto end = std::chrono::high_resolution_clock::now();

        const auto timing =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        individual_ns_sum_ += timing;

      } else {
        // not tracking errors, measure the lookup time.
        bound = index.EqualityLookup(lookup_key);
        iter = std::lower_bound(
            data_.begin() + bound.start, data_.begin() + bound.stop, lookup_key,
            [](const Row<KeyType>& lhs, const KeyType lookup_key) {
              return lhs.key < lookup_key;
            });
        result = 0;
        while (iter != data_.end() && iter->key == lookup_key) {
          result += iter->data[0];
          ++iter;
        }
        if (result != expected) {
          run_failed = true;
          return;
        }
      }

      if constexpr (fence) __sync_synchronize();
    }
  }

  template <class Index>
  void DoLookupsWithErrorTracking(Index& index) {
    assert(track_errors_);
    if (num_threads_ > 1 || perf_ || cold_cache_ || fence_) {
      util::fail(
          "error tracking can not be used in combination with: num_threads_ > "
          "1 || perf || cold_cache || fence");
    }

    SearchBound bound = {};
    for (unsigned int idx = 0; idx < lookups_.size(); ++idx) {
      const volatile uint64_t lookup_key = lookups_[idx].key;

      bound = index.EqualityLookup(lookup_key);
      if (bound.start != bound.stop) {
        log_sum_search_bound_ += log2((double)(bound.stop - bound.start));
        l1_sum_search_bound_ += abs((double)(bound.stop - bound.start));
        l2_sum_search_bound_ += pow((double)(bound.stop - bound.start), 2);
      }
    }

    log_sum_search_bound_ /= static_cast<double>(lookups_.size());
    l1_sum_search_bound_ /= static_cast<double>(lookups_.size());
    l2_sum_search_bound_ /= static_cast<double>(lookups_.size());
  }

  template <class Index>
  void PrintResult(const Index& index) {
    if (track_errors_) {
      std::cout << "RESULT: " << index.name() << "," << index.variant() << ","
                << log_sum_search_bound_ << "," << l1_sum_search_bound_ << ","
                << l2_sum_search_bound_ << std::endl;
      return;
    }

    if (build_) {
      std::cout << "RESULT: " << index.name() << "," << index.variant() << ","
                << build_ns_ << "," << index.size() << std::endl;
      return;
    }

    if (cold_cache_) {
      const double ns_per = (static_cast<double>(individual_ns_sum_)) /
                            (static_cast<double>(lookups_.size()));
      std::cout << "RESULT: " << index.name() << "," << index.variant() << ","
                << ns_per << "," << index.size() << "," << build_ns_ << ","
                << searcher_.name() << std::endl;
      return;
    }

    // print main results
    std::ostringstream all_times;
    for (unsigned int i = 0; i < runs_.size(); ++i) {
      const double ns_per_lookup =
          static_cast<double>(runs_[i]) / lookups_.size();
      all_times << "," << ns_per_lookup;
    }

    // don't print a line if (the first) run failed
    if (runs_[0] != 0) {
      std::cout << "RESULT: " << index.name() << "," << index.variant()
                << all_times.str()  // has a leading comma
                << "," << index.size() << "," << build_ns_ << ","
                << searcher_.name() << std::endl;
    }
    if (csv_) {
      PrintResultCSV(index);
    }
  }

  template <class Index>
  void PrintResultCSV(const Index& index) {
    const std::string filename =
        "./results/" + dataset_name_ + "_results_table.csv";

    std::ofstream fout(filename, std::ofstream::out | std::ofstream::app);

    if (!fout.is_open()) {
      std::cerr << "Failure to print CSV on " << filename << std::endl;
      return;
    }

    if (track_errors_) {
      fout << index.name() << "," << index.variant() << ","
           << log_sum_search_bound_ << "," << l1_sum_search_bound_ << ","
           << l2_sum_search_bound_ << std::endl;
      return;
    }

    if (build_) {
      fout << index.name() << "," << index.variant() << "," << build_ns_ << ","
           << index.size() << std::endl;
      return;
    }

    if (cold_cache_) {
      const double ns_per = (static_cast<double>(individual_ns_sum_)) /
                            (static_cast<double>(lookups_.size()));
      fout << index.name() << "," << index.variant() << "," << ns_per << ","
           << index.size() << "," << build_ns_ << "," << searcher_.name()
           << std::endl;
      return;
    }

    // compute median time
    std::vector<double> times;
    double median_time;
    for (unsigned int i = 0; i < runs_.size(); ++i) {
      const double ns_per_lookup =
          static_cast<double>(runs_[i]) / lookups_.size();
      times.push_back(ns_per_lookup);
    }
    std::sort(times.begin(), times.end());
    if (times.size() % 2 == 0) {
      median_time =
          0.5 * (times[times.size() / 2 - 1] + times[times.size() / 2]);
    } else {
      median_time = times[times.size() / 2];
    }

    // don't print a line if (the first) run failed
    if (runs_[0] != 0) {
      fout << index.name() << "," << index.variant() << "," << median_time
           << "," << index.size() << "," << build_ns_ << "," << searcher_.name()
           << "," << dataset_name_ << std::endl;
    }

    fout.close();
    return;
  }

  uint64_t random_sum_ = 0;
  uint64_t individual_ns_sum_ = 0;
  const std::string data_filename_;
  const std::string lookups_filename_;
  std::string dataset_name_;
  std::vector<Row<KeyType>> data_;
  std::vector<KeyValue<KeyType>> index_data_;
  bool unique_keys_;
  std::vector<EqualityLookup<KeyType>> lookups_;
  uint64_t build_ns_;
  double log_sum_search_bound_;
  double l1_sum_search_bound_;
  double l2_sum_search_bound_;
  // Run times.
  std::vector<uint64_t> runs_;
  // Number of times we repeat the lookup code.
  size_t num_repeats_;
  // Used to only print profiling header information for first run.
  bool first_run_;
  bool perf_;
  bool build_;
  bool fence_;
  bool measure_each_;
  bool cold_cache_;
  bool track_errors_;
  bool csv_;
  // Number of lookup threads.
  const size_t num_threads_;
  std::vector<uint64_t> memory_;  // Some memory we can read to flush the cache
  SearchClass<KeyType> searcher_;
};

}  // namespace sosd
