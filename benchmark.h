#pragma once

#include "util.h"
#include "utils/perf_event.h"
#include "searches/branching_binary_search.h"
#include "config.h"

#include <algorithm>
#include <sstream>
#include <math.h>
#include <immintrin.h>
#include <iostream>
#include <fstream>

#include <dtl/thread.hpp>

#ifdef __linux__
#define checkLinux(x) (x)
#else
#define checkLinux(x) { util::fail("Only supported on Linux."); }
#endif


// Get the CPU affinity for the process.
static const auto cpu_mask = dtl::this_thread::get_cpu_affinity();

// Batch size in number of lookups.
static constexpr std::size_t batch_size = 1u << 16;

namespace sosd {

// KeyType: Controls the type of the key (the value will always be uint64_t)
//          Use uint64_t for 64 bit types and uint32_t for 32 bit types
//          KeyType must implement operator<
template<typename KeyType = uint64_t,
         template<typename> typename SearchClass = BranchingBinarySearch>
class Benchmark {
 public:
  Benchmark(const std::string& data_filename,
            const std::string& lookups_filename,
	    const std::string& dataset_name,
            const size_t num_repeats,
            const bool perf, const bool build, const bool fence,
            const bool cold_cache, const bool track_errors,
            const bool csv, const size_t num_threads,
            const SearchClass<KeyType> searcher)
      : data_filename_(data_filename),
        lookups_filename_(lookups_filename),
	dataset_name_(dataset_name),
        num_repeats_(num_repeats),
        first_run_(true), perf(perf), build(build), fence(fence),
        cold_cache(cold_cache), track_errors(track_errors),
        csv_(csv), num_threads_(num_threads),
        searcher(searcher) {
    
    if ((int)cold_cache + (int)perf + (int)fence > 1) {
      util::fail("Can only specify one of cold cache, perf counters, or fence.");
    }
    
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
      index_data_.push_back((KeyValue<KeyType>) {data_[pos].key, pos});
    }
  }

  template<class Index>
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
      if (perf) {
        checkLinux(({
              BenchmarkParameters params;
              params.setParam("index", index.name());
              params.setParam("variant", index.variant());
              PerfEventBlock e(lookups_.size(), params,/*printHeader=*/first_run_);
              DoEqualityLookups<Index, false, false, false>(index);
            }));
      } else if (cold_cache) {
        if (num_threads_ > 1)
          util::fail("cold cache not supported with multiple threads");
        DoEqualityLookups<Index, true, false, true>(index);
        PrintResult(index);
      } else if (fence) {
        DoEqualityLookups<Index, false, true, false>(index);
        PrintResult(index);
      } else {
        DoEqualityLookups<Index, false, false, false>(index);
        PrintResult(index);
      }
    } else {
      if (perf || cold_cache || fence) {
        util::fail("Perf, cold cache, and fence mode require full builds. Disable fast mode.");
      }
      DoEqualityLookups<Index, false, false, false>(index);
      PrintResult(index);
    }
    
    
    first_run_ = false;
  }

  bool uses_binary_search() const {
    return (searcher.name() == "BinarySearch")
      || (searcher.name() == "BranchlessBinarySearch");
  }

  bool uses_lienar_search() const {
    return searcher.name() == "LinearSearch";
  }
  
private:
  bool CheckResults(uint64_t actual, uint64_t expected,
                    KeyType lookup_key, SearchBound bound) {
    if (actual!=expected) {
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
                << "correct index is: " << idx << std::endl
                << "index start: " << bound.start << " stop: "
                << bound.stop << std::endl;
      
      return false;
    }

    return true;
  }
  
  template<class Index, bool time_each, bool fence, bool clear_cache>
  void DoEqualityLookups(Index& index) {
    if (build) return;
    
    size_t repeats = num_repeats_;
    
    // Atomic counter used to assign work to threads.
    std::atomic<std::size_t> cntr(0);
    
    bool run_failed = false;
    

    std::vector<uint64_t> memory(26e6 / 8); // NOTE: L3 size of the machine
    if (clear_cache) {
      util::FastRandom ranny(8128);
      for(uint64_t& iter : memory) {
        iter = ranny.RandUint32();
      }
    }

    
    // Define function that contains lookup code.
    auto f = [&](const size_t thread_id) {
      while (true) {
        const size_t begin = cntr.fetch_add(batch_size);
        if (begin >= lookups_.size()) break;
        for (unsigned int idx = begin; idx < begin + batch_size && idx < lookups_.size();
             ++idx) {
          // Compute the actual index for debugging.
          const volatile uint64_t lookup_key = lookups_[idx].key;
          const volatile uint64_t expected = lookups_[idx].result;
          
          SearchBound bound;

          if (track_errors) {
            bound = index.EqualityLookup(lookup_key);
            if (bound.start != bound.stop) {
              log_sum_search_bound_ += log2((double)(bound.stop - bound.start));
              l1_sum_search_bound_ += abs((double)(bound.stop - bound.start));
              l2_sum_search_bound_ += pow((double)(bound.stop - bound.start), 2);
            }
          } else {
            uint64_t actual;
            size_t qualifying;
            
            if (clear_cache) {
              // Make sure that all cache lines from large buffer are loaded
              for(uint64_t& iter : memory) {
                random_sum += iter;
              }
              _mm_mfence();

              const auto start = std::chrono::high_resolution_clock::now();
              bound = index.EqualityLookup(lookup_key);
              actual = searcher.search(
                data_, lookup_key,
                &qualifying,
                bound.start, bound.stop);
              if (!CheckResults(actual, expected, lookup_key, bound)) {
                run_failed = true;
                return;
              }
              const auto end = std::chrono::high_resolution_clock::now();
            
              const auto timing = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
              individual_ns_sum += timing;
              
            } else {
              // not tracking errors, measure the lookup time.
              bound = index.EqualityLookup(lookup_key);
              actual = searcher.search(
                data_, lookup_key,
                &qualifying,
                bound.start, bound.stop);
              if (!CheckResults(actual, expected, lookup_key, bound)) {
                run_failed = true;
                return;
              }
            }  
          }
          if (fence) __sync_synchronize();
        }
      }
    };

    if (clear_cache)
      std::cout << "rsum was: " << random_sum << std::endl;

    runs_.resize(repeats);
    for (unsigned int i = 0; i < repeats; ++i) {
      random_sum = 0;
      individual_ns_sum = 0;

      // Reset atomic counter.
      cntr.store(0);
      const auto ms = util::timing([&] {
        dtl::run_in_parallel(f, cpu_mask, num_threads_);
      });
      log_sum_search_bound_ /= static_cast<double>(lookups_.size());
      l1_sum_search_bound_ /= static_cast<double>(lookups_.size());
      l2_sum_search_bound_ /= static_cast<double>(lookups_.size());

      runs_[i] = ms;
      if (run_failed) {
        runs_ = std::vector<uint64_t>(repeats, 0);
        return;
      }
    }
  }

  template<class Index>
  void PrintResult(const Index& index) {
    if (track_errors) {
      std::cout << "RESULT: " << index.name()
                << "," << index.variant()
                << "," << log_sum_search_bound_
                << "," << l1_sum_search_bound_
                << "," << l2_sum_search_bound_
                << std::endl;
      return;
    }

    if (build) {
      std::cout << "RESULT: " << index.name()
                << "," << index.variant()
                << "," << build_ns_
                << "," << index.size()
                << std::endl;
      return;
    }
    
    
    if (cold_cache) {
      double ns_per = ((double)individual_ns_sum) / ((double)lookups_.size());
      std::cout << "RESULT: " << index.name()
                << "," << index.variant()
                << "," << ns_per
                << "," << index.size() << "," << build_ns_
                << "," << searcher.name()
                << std::endl;
      return;
    }
    
    // print main results
    std::ostringstream all_times;
    for (unsigned int i = 0; i < runs_.size(); ++i) {
      const double ns_per_lookup = static_cast<double>(runs_[i])
        /lookups_.size();
      all_times << "," << ns_per_lookup;
    }
    
    
    // don't print a line if (the first) run failed
    if (runs_[0]!=0) {
      std::cout << "RESULT: " << index.name()
                << "," << index.variant()
                << all_times.str() // has a leading comma
                << "," << index.size() << "," << build_ns_
                << "," << searcher.name()
                << std::endl;
    }
    if (csv_) {
      PrintResultCSV(index);
    }
  }

  template<class Index>
  void PrintResultCSV(const Index& index) {
    const std::string filename = "./results/" + dataset_name_ + "_results_table.csv";

    std::ofstream fout(filename, std::ofstream::out | std::ofstream::app);

    if (!fout.is_open()) {
      std::cerr << "Failure to print CSV on " << filename << std::endl;
      return;
    }

    if (track_errors) {
      fout << index.name()
           << "," << index.variant()
           << "," << log_sum_search_bound_
           << "," << l1_sum_search_bound_
           << "," << l2_sum_search_bound_
           << std::endl;
      return;
    }

    if (build) {
      fout << index.name()
           << "," << index.variant()
           << "," << build_ns_
           << "," << index.size()
           << std::endl;
      return;
    }

    if (cold_cache) {
      double ns_per = ((double)individual_ns_sum) / ((double)lookups_.size());
      fout << index.name()
           << "," << index.variant()
           << "," << ns_per
           << "," << index.size()
           << "," << build_ns_
           << "," << searcher.name()
           << std::endl;
      return;
    }

    // print main results
    std::ostringstream all_times;
    for (unsigned int i = 0; i < runs_.size(); ++i) {
      const double ns_per_lookup = static_cast<double>(runs_[i])
        /lookups_.size();
      all_times << "," << ns_per_lookup;
    }

    // don't print a line if (the first) run failed
    if (runs_[0]!=0) {
      fout << index.name()
           << "," << index.variant()
           << all_times.str() // has a leading comma
           << "," << index.size()
           << "," << build_ns_
           << "," << searcher.name()
	   << "," << dataset_name_
           << std::endl;
    }
    
    fout.close();
    return;
  }

  uint64_t random_sum = 0;
  uint64_t individual_ns_sum = 0;
  const std::string data_filename_;
  const std::string lookups_filename_;
  const std::string dataset_name_;
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
  bool perf;
  bool build;
  bool fence;
  bool measure_each_;
  bool cold_cache;
  bool track_errors;
  bool csv_;
  // Number of lookup threads.
  const size_t num_threads_;

  SearchClass<KeyType> searcher;
};

} // namespace sosd
