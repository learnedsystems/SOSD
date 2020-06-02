#include "gtest/gtest.h"

#include "../adept.hpp"
#include "bloomfilter/old/bloomfilter_h1.hpp"
#include "bloomfilter/old/bloomfilter_h1_vec.hpp"
#include "../hash.hpp"
#include "../mem.hpp"
#include "../simd.hpp"
#include "../thread.hpp"
#include "../env.hpp"

#include <atomic>

#include <chrono>


using namespace dtl;


struct xorshift32 {
  $u32 x32;
  xorshift32() : x32(314159265) { };
  xorshift32(u32 seed) : x32(seed) { };

  inline u32
  operator()() {
    x32 ^= x32 << 13;
    x32 ^= x32 >> 17;
    x32 ^= x32 << 5;
    return x32;
  }

  template<typename T>
  static inline void
  next(T& x32) {
    x32 ^= x32 << 13;
    x32 ^= x32 >> 17;
    x32 ^= x32 << 5;
  }

};

inline auto timing(std::function<void()> fn) {
  auto start = std::chrono::high_resolution_clock::now();
  fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static constexpr std::chrono::seconds sec(1);
static constexpr double nano_to_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(sec).count();

TEST(bloom, prng_performance) {
  u64 repeat_cnt = 1u << 20;
  xorshift32 prng;
  f64 duration = timing([&] {
    for ($u64 i = 0; i < repeat_cnt; i++) {
      prng();
    }
  });
  u64 perf = (repeat_cnt) / (duration / nano_to_sec);
  std::cout << perf << " [prn/sec]    (" << prng.x32 << ")" << std::endl;
}


template<typename hash_fn_t>
void run_hash_benchmark(hash_fn_t hash_fn,
                        const std::size_t input_size = 1ull << 10 /* prevent auto-vectorization*/ ) {
  // prepare input
  std::vector<$u32> input;
  xorshift32 prng;
  for (std::size_t i = 0; i < input_size; i++) {
    input.push_back(prng());
  }

  // run benchmark
  u64 repeat_cnt = 100000;
  $u64 chksum = 0;
  f64 duration = timing([&] {
    for ($u64 r = 0; r != repeat_cnt; r++) {
      for ($u64 i = 0; i != input_size; i++) {
        chksum += hash_fn.hash(input[i]);
      }
    }
  });
  u64 perf = (input_size * repeat_cnt) / (duration / nano_to_sec);
  std::cout << "scalar:    " << perf << " [hashes/sec]    (chksum: " << chksum << ")" << std::endl;
}


template<typename hash_fn_t>
void run_hash_benchmark_autovec(hash_fn_t hash_fn) {
  // prepare input
  const std::size_t input_size = 1ull << 10;
  std::vector<$u32> input;
  xorshift32 prng;
  for (std::size_t i = 0; i < input_size; i++) {
    input.push_back(prng());
  }

  // run benchmark
  u64 repeat_cnt = 100000;
  $u64 chksum = 0;
  u64 duration = timing([&] {
    for ($u64 r = 0; r != repeat_cnt; r++) {
      for ($u64 i = 0; i != input_size; i++) {
        chksum += hash_fn.hash(input[i]);
      }
    }
  });
  u64 perf = (input_size * repeat_cnt) / (duration / nano_to_sec);
  std::cout << "auto-vec.: " << perf << " [hashes/sec]    (chksum: " << chksum << ")" << std::endl;
}


TEST(bloom, hash_performance) {
  std::cout.setf(std::ios::fixed);
  std::cout << "xorshift   " << std::endl;
  dtl::hash::xorshift_64<u32> xorshift_64;
  run_hash_benchmark(xorshift_64);
  run_hash_benchmark_autovec(xorshift_64);
  std::cout << "murmur1_32 " << std::endl;
  dtl::hash::murmur1_32<u32> murmur1_32;
  run_hash_benchmark(murmur1_32);
  run_hash_benchmark_autovec(murmur1_32);
  std::cout << "crc32      " << std::endl;
  dtl::hash::crc32<u32> crc32;
  run_hash_benchmark(crc32);
  run_hash_benchmark_autovec(crc32);
  std::cout << "knuth      " << std::endl;
  dtl::hash::knuth<u32> knuth;
  run_hash_benchmark(knuth);
  run_hash_benchmark_autovec(knuth);
}

// --- compile-time settings ---


struct bf {
  using key_t = $u32;
  using word_t = $u32;

  using key_alloc = dtl::mem::numa_allocator<key_t>;
  using word_alloc = dtl::mem::numa_allocator<word_t>;

};

//template<typename Alloc = bf::word_alloc>
using bf_t = dtl::bloomfilter_h1<bf::key_t, dtl::hash::knuth, bf::word_t, bf::word_alloc>;

//template<typename Alloc = bf::word_alloc>
using bf_vt = dtl::bloomfilter_h1_vec<bf::key_t, dtl::hash::knuth, bf::word_t, bf::word_alloc>;

static const u64 vec_unroll_factor = 4;

// --- runtime settings ---

// the grain size for parallel experiments
static u64 preferred_grain_size = 1ull << dtl::env<$i32>::get("GRAIN_SIZE", 16);

// set the bloomfilter_h1 size: m in [2^lo, 2^hi]
static i32 bf_size_lo_exp = dtl::env<$i32>::get("BF_SIZE_LO", 11);
static i32 bf_size_hi_exp = dtl::env<$i32>::get("BF_SIZE_HI", 31);

// repeats the benchmark with different concurrency settings
static i32 thread_cnt_lo = dtl::env<$i32>::get("THREAD_CNT_LO", 1);
static i32 thread_cnt_hi = dtl::env<$i32>::get("THREAD_CNT_HI", std::thread::hardware_concurrency());

// 1 = linear, 2 = exponential
static i32 thread_step_mode = dtl::env<$i32>::get("THREAD_STEP_MODE", 1);
static i32 thread_step = dtl::env<$i32>::get("THREAD_STEP", 1);

// the number of keys to probe per thread
static u64 key_cnt_per_thread = 1ull << dtl::env<$i32>::get("KEY_CNT", 24);

// the number of repetitions
static u64 repeat_cnt = dtl::env<$i32>::get("REPEAT_CNT", 16);;


// place bloomfilter_h1 in HBM?
static u1 use_hbm = dtl::env<$i32>::get("HBM", 1);
// replicate bloomfilter_h1 in HBM?
static u1 replicate_bloomfilter = dtl::env<$i32>::get("REPL", 1);


static auto inc_thread_cnt = [&](u64 i) {
  if (thread_step_mode == 1) {
    // linear
    return i + thread_step;
  }
  else {
    // exponential
    auto step = thread_step > 1 ? thread_step : 2;
    return i * step;
  }
};


void run_filter_benchmark(u64 bf_size) {
  dtl::thread_affinitize(0);
  u64 repeat_cnt = 1u << 28;
  bf_t bf(bf_size);
  {
    f64 duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
        bf.insert(dtl::hash::crc32<u32>::hash(i));
      }
    });
    u64 perf = (repeat_cnt) / (duration / nano_to_sec);
    std::cout << perf << " [inserts/sec]" << std::endl;
  }
  {
    $u64 found = 0;
    f64 duration = timing([&] {
      for ($u64 i = 0; i < repeat_cnt; i++) {
        found += bf.contains(dtl::hash::crc32<u32, 7331>::hash(i));
      }
    });
    u64 perf = (repeat_cnt) / (duration / nano_to_sec);
    std::cout << perf << " [probes/sec]    (matchcnt: " << found << ")" << std::endl;
  }

}


TEST(bloom, filter_performance) {
  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;
  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 2) {
    std::cout << "size " << (bf_size / 8) << " [bytes]" << std::endl;
    run_filter_benchmark(bf_size);
  }
}


void run_filter_benchmark_in_parallel(u64 bf_size, u64 thread_cnt) {
  dtl::thread_affinitize(std::thread::hardware_concurrency() - 1);
  u64 key_cnt = key_cnt_per_thread * thread_cnt;

  bf::word_alloc bf_cpu_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  bf::word_alloc bf_hbm_interleaved_alloc(dtl::mem::allocator_config::interleave_hbm());

  if (use_hbm) {
    std::cout << "Using HBM for bloomfilter_h1" << std::endl;
  }
  bf_t bf(bf_size, use_hbm ? bf_hbm_interleaved_alloc : bf_cpu_interleaved_alloc);
  {
    f64 duration = timing([&] {
      for ($u64 i = 0; i < bf_size >> 4; i++) {
        bf.insert(dtl::hash::crc32<u32>::hash(i));
      }
    });
    u64 perf = (key_cnt) / (duration / nano_to_sec);
  }

  // prepare the input (interleaved)
  bf::key_alloc input_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  std::vector<bf::key_t, bf::key_alloc> keys(input_interleaved_alloc);
  keys.resize(key_cnt);
  for ($u64 i = 0; i < key_cnt; i++) {
    keys[i] = dtl::hash::crc32<u32, 7331>::hash(i);
  }

  // size of a work item (dispatched to a thread)
  u64 preferred_grain_size = 1ull << 16;
  u64 grain_size = std::min(preferred_grain_size, key_cnt);

  std::vector<$u64> matches_found;
  matches_found.resize(thread_cnt, 0);
  std::atomic<$u64> grain_cntr(0);

  // create replicas is requested (see env 'HBM' and 'HBM_REPL')
  std::vector<bf_t> bloomfilter_replicas;
  // maps node_id -> replica_id
  std::vector<$u64> bloomfilter_node_map;
  // insert the already existing bloomfilter_h1 (as a fallback when numa/hbm is not available)
  bloomfilter_replicas.push_back(bf);
  // initially, let all nodes refer to the first replica
  bloomfilter_node_map.resize(dtl::mem::get_node_count(), 0);

  if (replicate_bloomfilter) {
    // replicate the bloomfilter_h1 to all HBM nodes
    if (dtl::mem::hbm_available()) {
      for (auto hbm_node_id : dtl::mem::get_hbm_nodes()) {
        // make a copy
        std::cout << "replicate bloomfilter_h1 to HBM node " << hbm_node_id << std::endl;
        bf::word_alloc on_node_alloc(dtl::mem::allocator_config::on_node(hbm_node_id));
        bf_t replica = bf.make_copy(on_node_alloc);
        bloomfilter_replicas.push_back(std::move(replica));
        // update mapping
        bloomfilter_node_map[hbm_node_id] = bloomfilter_replicas.size() - 1;
        dtl::mem::get_node_of_address(&bloomfilter_replicas[bloomfilter_node_map[hbm_node_id]].word_array[0]);
      }
    }
  }

  auto worker_fn = [&](u32 thread_id) {
    // determine NUMA node id
    const auto cpu_mask = dtl::this_thread::get_cpu_affinity();
    const auto cpu_id = cpu_mask.find_first(); // handwaving
    const auto numa_node_id = dtl::mem::get_node_of_cpu(cpu_id);

    // determine nearest HBM node (returns numa_node_id if HBM is not available)
    const auto hbm_node_id = dtl::mem::get_nearest_hbm_node(numa_node_id);

    // obtain the local bloomfilter_h1 instance
//    std::cout << "thread " << thread_id << " using BF instance #" << bloomfilter_node_map[hbm_node_id] << std::endl;
    const bf_t& _bf = bloomfilter_replicas[bloomfilter_node_map[hbm_node_id]];
//    dtl::mem::get_node_of_address(&_bf.word_array[0]);

    $u64 found = 0;
    while (true) {
      u64 cntr = grain_cntr.fetch_add(grain_size);
      u64 read_from = cntr % key_cnt;
      u64 read_to = std::min(key_cnt, read_from + grain_size);
      if (cntr >= key_cnt * repeat_cnt) break;
      for ($u64 i = read_from; i < read_to; i++) {
        found += _bf.contains(keys[i]);
      }
    }
    matches_found[thread_id] = found;
  };


  $f64 duration = timing([&] {
    dtl::run_in_parallel(worker_fn, thread_cnt);
  });

  duration /= repeat_cnt;

  $u64 found = 0;
  for ($u64 i = 0; i < thread_cnt; i++) {
    found += matches_found[i];
  }
  found /= repeat_cnt;
  u64 perf = (key_cnt) / (duration / nano_to_sec);
  std::cout << "bf_size: " << (bf_size / 8) << " [bytes], "
            << "thread_cnt: " << thread_cnt << ", "
            << "key_cnt: " << key_cnt << ", "
            << "grain_size: " << grain_size << ", "
            << "performance: " << perf << " [1/s]  (matchcnt: " << found << ")" << std::endl;
}


TEST(bloom, filter_performance_parallel) {
  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;

  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 2) {
    for ($u64 t = thread_cnt_lo; t <= thread_cnt_hi; t = inc_thread_cnt(t)) {
      run_filter_benchmark_in_parallel(bf_size, t);
    }
  }
}


void run_filter_benchmark_in_parallel_vec(u64 bf_size, u64 thread_cnt) {
  dtl::thread_affinitize(std::thread::hardware_concurrency() - 1);
  u64 key_cnt = key_cnt_per_thread * thread_cnt;

  bf::word_alloc bf_cpu_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  bf::word_alloc bf_hbm_interleaved_alloc(dtl::mem::allocator_config::interleave_hbm());

  if (use_hbm) {
    std::cout << "Using HBM for bloomfilter_h1" << std::endl;
  }
  bf_t bf(bf_size, use_hbm ? bf_hbm_interleaved_alloc : bf_cpu_interleaved_alloc);
  {
    f64 duration = timing([&] {
      for ($u64 i = 0; i < bf_size >> 4; i++) {
        bf.insert(dtl::hash::crc32<u32>::hash(i));
      }
    });
    u64 perf = (key_cnt) / (duration / nano_to_sec);
  }

  // prepare the input (interleaved)
  bf::key_alloc input_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  std::vector<bf::key_t, bf::key_alloc> keys(input_interleaved_alloc);
  keys.resize(key_cnt);
  for ($u64 i = 0; i < key_cnt; i++) {
    keys[i] = dtl::hash::crc32<u32, 7331>::hash(i);
  }

  // create replicas is requested (see env 'HBM' and 'HBM_REPL')
  std::vector<bf_t> bloomfilter_replicas;
  // maps node_id -> replica_id
  std::vector<$u64> bloomfilter_node_map;
  // insert the already existing bloomfilter_h1 (as a fallback when numa/hbm is not available)
  bloomfilter_replicas.push_back(bf);
  // initially, let all nodes refer to the first replica
  bloomfilter_node_map.resize(dtl::mem::get_node_count(), 0);

  if (replicate_bloomfilter) {
    // replicate the bloomfilter_h1 to all HBM nodes
    auto replica_nodes = (use_hbm && dtl::mem::hbm_available())
                         ? dtl::mem::get_hbm_nodes()
                         : dtl::mem::get_cpu_nodes();

    for (auto dst_node_id : replica_nodes) {
      // make a copy
      std::cout << "replicate bloomfilter_h1 to node " << dst_node_id << std::endl;
      bf::word_alloc on_node_alloc(dtl::mem::allocator_config::on_node(dst_node_id));
      bf_t replica = bf.make_copy(on_node_alloc);
      bloomfilter_replicas.push_back(std::move(replica));
      // update mapping
      bloomfilter_node_map[dst_node_id] = bloomfilter_replicas.size() - 1;
      dtl::mem::get_node_of_address(&bloomfilter_replicas[bloomfilter_node_map[dst_node_id]].word_array[0]);
    }
  }

  // size of a work item (dispatched to a thread)
  u64 grain_size = std::min(preferred_grain_size, key_cnt);

  std::vector<$u64> matches_found;
  matches_found.resize(thread_cnt, 0);
  std::atomic<$u64> grain_cntr(0);

  auto worker_fn = [&](u32 thread_id) {
    // determine NUMA node id
    const auto cpu_mask = dtl::this_thread::get_cpu_affinity();
    const auto cpu_id = cpu_mask.find_first(); // handwaving
    const auto numa_node_id = dtl::mem::get_node_of_cpu(cpu_id);

    // determine nearest HBM node (returns numa_node_id if HBM is not available)
    const auto hbm_node_id = dtl::mem::get_nearest_hbm_node(numa_node_id);

    // obtain the local bloomfilter_h1 instance
//    std::cout << "thread " << thread_id << " using BF instance #" << bloomfilter_node_map[hbm_node_id] << std::endl;
    const bf_t& _bf = bloomfilter_replicas[bloomfilter_node_map[hbm_node_id]];
//    dtl::mem::get_node_of_address(&_bf.word_array[0]);

    // SIMD extension
    bf_vt bf_vec { _bf };

    u64 vlen = dtl::simd::lane_count<bf::key_t> * vec_unroll_factor;
    using key_vt = dtl::vec<bf::key_t, vlen>;

    key_vt found_vec = 0;
    while (true) {
      u64 cntr = grain_cntr.fetch_add(grain_size);
      u64 read_from = cntr % key_cnt;
      u64 read_to = std::min(key_cnt, read_from + grain_size);
      if (cntr >= key_cnt * repeat_cnt) break;
      for ($u64 i = read_from; i < read_to; i += vlen) {
        const key_vt* k = reinterpret_cast<const key_vt*>(&keys[i]);
        auto mask = bf_vec.contains<vlen>(*k);
        found_vec[mask] += 1;
      }
    }
    $u64 found = 0;
    for (std::size_t i = 0; i < vlen; i++) {
      found += found_vec[i];
    }
    matches_found[thread_id] = found;
  };


  $f64 duration = timing([&] {
    dtl::run_in_parallel(worker_fn, thread_cnt);
  });

  duration /= repeat_cnt;

  $u64 found = 0;
  for ($u64 i = 0; i < thread_cnt; i++) {
    found += matches_found[i];
  }
  found /= repeat_cnt;
  u64 perf = (key_cnt) / (duration / nano_to_sec);
  std::cout << "bf_size: " << (bf_size / 8) << " [bytes], "
            << "thread_cnt: " << thread_cnt << ", "
            << "key_cnt: " << key_cnt << ", "
            << "grain_size: " << grain_size << ", "
            << "performance: " << perf << " [1/s]  (matchcnt: " << found << ")" << std::endl;
}


TEST(bloom, filter_performance_parallel_vec) {
  std::cout << "native degree of data parallelism: " << dtl::simd::lane_count<bf::key_t> << std::endl;
  std::cout << "                 unrolling factor: " << vec_unroll_factor << std::endl;

  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;

  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 2) {
    for ($u64 t = thread_cnt_lo; t <= thread_cnt_hi; t = inc_thread_cnt(t)) {
      run_filter_benchmark_in_parallel_vec(bf_size, t);
    }
  }
}
