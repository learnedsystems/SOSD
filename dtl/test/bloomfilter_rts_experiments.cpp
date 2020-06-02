#include "gtest/gtest.h"

#include <atomic>
#include <bitset>
#include <chrono>
#include <functional>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include <boost/tokenizer.hpp>

#include <dtl/dtl.hpp>
#include <bloomfilter/old/bloomfilter_runtime.hpp>
#include <dtl/mem.hpp>
#include <dtl/thread.hpp>
#include <dtl/env.hpp>

#include "immintrin.h"

using namespace dtl;


inline auto timing(std::function<void()> fn) {
  auto start = std::chrono::high_resolution_clock::now();
  fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static constexpr std::chrono::seconds sec(1);
static constexpr double nano_to_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(sec).count();


// --- compile-time settings ---
struct bf {
  using key_t = $u32;
  using word_t = $u32;

  using key_alloc = dtl::mem::numa_allocator<key_t>;
  using word_alloc = dtl::mem::numa_allocator<word_t>;
};


// --- runtime settings ---

// the grain size for parallel experiments
static u64 preferred_grain_size = 1ull << dtl::env<$i32>::get("GRAIN_SIZE", 16);

// set the bloomfilter_h1 size: m in [2^lo, 2^hi]
static i32 bf_size_lo_exp = dtl::env<$i32>::get("BF_SIZE_LO", 11);
static i32 bf_size_hi_exp = dtl::env<$i32>::get("BF_SIZE_HI", 31);

// the number of hash functions to use
static i32 bf_k = dtl::env<$i32>::get("BF_K", 1);

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

static void
print_env_settings() {
  std::cout
      << "Configuration:\n"
      << "  BF_K=" << bf_k
      << ", BF_SIZE_LO=" << bf_size_lo_exp
      << ", BF_SIZE_HI=" << bf_size_hi_exp
      << "\n  GRAIN_SIZE=" << dtl::env<$i32>::get("GRAIN_SIZE", 16)
      << ", THREAD_CNT_LO=" << thread_cnt_lo
      << ", THREAD_CNT_HI=" << thread_cnt_hi
      << ", THREAD_STEP=" << thread_step
      << ", THREAD_STEP_MODE=" << thread_step_mode << " (1=linear, 2=exponential)"
      << ", KEY_CNT=" << dtl::env<$i32>::get("KEY_CNT", 24) << " (per thread)"
      << ", REPEAT_CNT=" << repeat_cnt
      << "\n  HBM=" << static_cast<u32>(use_hbm) << " (0=no, 1=yes)"
      << ", REPL=" << static_cast<u32>(replicate_bloomfilter) << " (0=interleaved, 1=replicate)"
      << std::endl;
}

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



void run_filter_benchmark_in_parallel_vec(u32 k, u32 m, u64 thread_cnt) {
  dtl::thread_affinitize(std::thread::hardware_concurrency() - 1);
  u64 key_cnt = key_cnt_per_thread * thread_cnt;

  using bf_rts_t = dtl::bloomfilter_runtime;
  bf::word_alloc bf_cpu_interleaved_alloc(dtl::mem::allocator_config::interleave_cpu());
  bf::word_alloc bf_hbm_interleaved_alloc(dtl::mem::allocator_config::interleave_hbm());

//  if (use_hbm) {
//    std::cout << "Using HBM for bloomfilter_h1" << std::endl;
//  }
  // TODO  bf_rts_t bf(bf_size, use_hbm ? bf_hbm_interleaved_alloc : bf_cpu_interleaved_alloc);
  auto bf = bf_rts_t::construct(k, m);

  {
    f64 duration = timing([&] {
      for ($u64 i = 0; i < m >> 4; i++) {
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

  // create replicas as requested (see env 'HBM' and 'HBM_REPL')
  std::vector<bf_rts_t> bloomfilter_replicas;
  // maps node_id -> replica_id
  std::vector<$u64> bloomfilter_node_map;
  // insert the already existing bloomfilter_h1 (as a fallback when numa/hbm is not available)
  bloomfilter_replicas.push_back(std::move(bf));
  // initially, let all nodes refer to the first replica
  bloomfilter_node_map.resize(dtl::mem::get_node_count(), 0);

  bloomfilter_replicas[0].print_info();

  if (replicate_bloomfilter) {
    // replicate the bloomfilter_h1 to all HBM nodes
    auto replica_nodes = (use_hbm && dtl::mem::hbm_available())
                         ? dtl::mem::get_hbm_nodes()
                         : dtl::mem::get_cpu_nodes();

    for (auto dst_node_id : replica_nodes) {
      // make a copy
      std::cout << "replicate bloomfilter_h1 to node " << dst_node_id << std::endl;
      auto alloc_config = dtl::mem::allocator_config::on_node(dst_node_id);
      dtl::mem::numa_allocator<bf::word_t> allocator(alloc_config);
      bf_rts_t replica = bloomfilter_replicas[0].make_copy(alloc_config);
//      bf_rts_t replica(bloomfilter_replicas[0], allocator);
      bloomfilter_replicas.push_back(std::move(replica));
      // update mapping
      bloomfilter_node_map[dst_node_id] = bloomfilter_replicas.size() - 1;
    }
  }

  // size of a work item (dispatched to a thread)
  u64 grain_size = std::min(preferred_grain_size, key_cnt);

  std::vector<$f64> avg_cycles_per_probe;
  avg_cycles_per_probe.resize(thread_cnt, 0.0);

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
    const bf_rts_t& _bf = bloomfilter_replicas[bloomfilter_node_map[hbm_node_id]];

    // allocate a match vector
    std::vector<$u32> match_pos;
    match_pos.resize(grain_size, 0);

    $u64 tsc = 0;
    $u64 found = 0;
    $u64 probe_cntr = 0;
    while (true) {
      u64 cntr = grain_cntr.fetch_add(grain_size);
      u64 read_from = cntr % key_cnt;
      u64 read_to = std::min(key_cnt, read_from + grain_size);
      if (cntr >= key_cnt * repeat_cnt) break;
      u64 cnt = read_to - read_from;
      probe_cntr += cnt;
      __sync_synchronize();
      u64 tsc_begin = _rdtsc();
      auto match_cnt = _bf.batch_contains(&keys[read_from], cnt, &match_pos[0], 0);
      __sync_synchronize();
      u64 tsc_end = _rdtsc();
      tsc += tsc_end - tsc_begin;
      found += match_cnt;
    }
    matches_found[thread_id] = found;
    avg_cycles_per_probe[thread_id] = (tsc * 1.0) / probe_cntr;
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

  f64 cycles_per_probe = std::accumulate(avg_cycles_per_probe.begin(), avg_cycles_per_probe.end(), 0.0) / thread_cnt;

  u64 perf = (key_cnt) / (duration / nano_to_sec);
  std::cout << "bf_size: " << (m / 8) << " [bytes], "
            << "thread_cnt: " << thread_cnt << ", "
            << "key_cnt: " << key_cnt << ", "
            << "grain_size: " << grain_size << ", "
            << "performance: " << perf << " [1/s], "
            << "cycles/probe: " << cycles_per_probe
            << " (matchcnt: " << found << ")"
            << std::endl;

}


TEST(bloom, filter_performance_parallel_vec) {
  print_env_settings();
  u64 bf_size_lo = 1ull << bf_size_lo_exp;
  u64 bf_size_hi = 1ull << bf_size_hi_exp;

  for ($u64 bf_size = bf_size_lo; bf_size <= bf_size_hi; bf_size <<= 1) {
    for ($u64 t = thread_cnt_lo; t <= thread_cnt_hi; t = inc_thread_cnt(t)) {
      run_filter_benchmark_in_parallel_vec(bf_k, bf_size, t);
    }
  }
}



// ---------------------------------------------------------------------------------------------------------------------


TEST(bloom, accuracy) {
  // the max. size of the Bloom filter
  const std::size_t m_max = 128ull * 1024ull * 1024ull * 8ull;
  // the number of queries per run
  const std::size_t probe_cnt = 10 * 1024ull * 1024ull;

  const std::size_t n_max = 2 * m_max + 1024ull;

  std::cout << "# generating random data: " << std::flush;

  std::vector<$u32> data;
  data.reserve(n_max + probe_cnt);
  {
    u32 worker_cnt = dtl::next_power_of_two(std::min(std::thread::hardware_concurrency(), 64u));

    std::vector<std::vector<$u32>*> worker_data;
    worker_data.resize(worker_cnt);

    auto data_gen_worker_fn = [&](u32 thread_id) {
      const std::size_t limit = n_max / worker_cnt;

      auto bit_field = std::make_unique<std::bitset<1ull << 32>>();


      worker_data[thread_id] = new std::vector<$u32>;
      worker_data[thread_id]->reserve(limit);
      std::vector<$u32>& local_data = *worker_data[thread_id];

      std::random_device rd;
      std::mt19937 gen(rd());
      u32 range_span = std::numeric_limits<$u32>::max() / worker_cnt;
      u32 range_begin = thread_id * range_span;
      u32 range_end = range_begin +  range_span - 1;
      std::uniform_int_distribution<$u32> dis(range_begin, range_end);
      std::stringstream str;
      str << "# worker_id=" << thread_id << ", begin=" << range_begin << ", end=" << range_end << std::endl;
      std::cout << str.str() << std::flush;

      while (local_data.size() < limit) {
        auto v = dis(gen);
        if (!(*bit_field)[v]) {
          (*bit_field)[v] = true;
          local_data.push_back(v);
        }
      }
      (*bit_field).reset();
    };

    dtl::run_in_parallel(data_gen_worker_fn, worker_cnt);

    // concat vectors
    std::cout << "# concatenating data: " << std::flush;
    for (std::size_t i = 0; i < worker_cnt; i++) {
      data.insert(std::end(data), std::begin(*worker_data[i]), std::end(*worker_data[i]));
      worker_data[i]->reserve(16);
      worker_data[i]->clear();
      delete worker_data[i];
    }
    std::cout << "done" << std::endl;

    std::cout << "# shuffling data: " << std::flush;
    std::random_shuffle(data.begin(), data.end());
    std::cout << "done" << std::endl;

    std::cout << "# unique check: " << std::flush;
    auto bit_field = std::make_unique<std::bitset<1ull << 32>>();
    std::for_each(data.begin(), data.end(), [&](auto i) {
      if ((*bit_field)[i]) {
        std::cout << "Validation failed!" << std::endl;
      }
      (*bit_field)[i] = true;
    });
    std::cout << "done" << std::endl;
  }


  // the values for m
  std::vector<std::size_t> ms {
      32ull * 1024 * 8,            // L1 cache size
      64ull * 1024 * 8,
      128ull * 1024 * 8,
      256ull * 1024 * 8,           // L2 cache size (Intel)
      512ull * 1024 * 8,           // L2 cache size (AMD)
      8ull * 1024 * 1024 * 8,      // L3 cache sizes
      16ull * 1024 * 1024 * 8,
      32ull * 1024 * 1024 * 8,     // ~L3 cache size (Xeon E5-2680 v4)
      64ull * 1024 * 1024 * 8,
      128ull * 1024 * 1024 * 8,    // m_max
  };

  struct launch_params {
    $u64 k;
    $u64 m;
    $u64 n;
  };

  u64 data_points_per_test = 64;
  $u64 current_m_idx = 0;
  $u64 current_n = 0;
  $u64 current_k = 1;
  u64 k_max = 7;

  std::mutex generator_mutex;
  auto generate_test_cases = [&](launch_params& p) {
    std::lock_guard<std::mutex> lock(generator_mutex);
    if (current_n < ms[current_m_idx]) {
      // increment n
      auto m = ms[current_m_idx];
      auto m_low = (m / 100);
      if (current_n < m_low) {
        current_n += m_low / data_points_per_test;
        if (current_n > m_low) current_n = m_low;
      }
      else {
        current_n += ms[current_m_idx] / data_points_per_test;
      }
    }
    else if (current_m_idx < ms.size() - 1) {
      // increment m, reset n
      current_m_idx++;
      current_n = 0;
    }
    else {
      // increment k, reset m and n
      current_k++;
      current_m_idx = 0;
      current_n = 0;
    }

    u1 continue_tests = current_k <= k_max && current_m_idx < ms.size() && current_n <= m_max;
    if (continue_tests) {
      p.k = current_k;
      p.m = ms[current_m_idx];
      p.n = current_n;
    }
    return continue_tests;
  };

  using bf_t = dtl::bloomfilter_runtime;
  auto worker_fn = [&](u32 thread_id) {
    constexpr u32 batch_size = 1024;
    $u32 match_pos[batch_size];
    launch_params p;
    while (generate_test_cases(p)) {

      // create and populate Bloom filter
      bf_t bf = bf_t::construct(p.k, p.m);
      // use keys from lower half of data
      for (std::size_t i = 0; i < p.n; i++) {
        bf.insert(data[i]);
      }

      // query the Bloom filter
      $u64 match_cntr = 0;

      // upper half of data are keys that should result in negative queries
      // every match is a false positive
      for (std::size_t i = p.n; i < (p.n + probe_cnt); i += batch_size) {
        match_cntr += bf.batch_contains(&data[i], batch_size, &match_pos[0], 0);
      }

      std::stringstream result;
      result << p.k << "," << p.m << "," << p.n
             << "," << bf.hash_function_count()
             << "," << probe_cnt
             << "," << match_cntr
             << "," << (match_cntr * 1.0) / probe_cnt
             << "," << bf.pop_count()
             << "," << bf.load_factor()
             << std::endl;
      std::cout << result.str();
      bf.destruct();
    }
  };

  std::cout << "k,m,n,h,key_cnt,match_cnt,fpr,pop_count,load_factor" << std::endl;

  dtl::run_in_parallel(worker_fn, std::min(std::thread::hardware_concurrency(), 64u));
//  dtl::run_in_parallel(worker_fn, std::min(std::thread::hardware_concurrency(), 1u));
}



static void parse_csv(std::ifstream& csv_input,
                      std::function<void(std::vector<std::string>&)> callback) {

  using namespace boost;
  boost::char_separator<char> sep(",");
//  typedef tokenizer<escaped_list_separator<char>> Tokenizer;
  typedef boost::tokenizer<boost::char_separator<char> >
      Tokenizer;
  std::vector<std::string> tokens;
  std::string line;

  while (getline(csv_input, line)) {
    Tokenizer tok(line,sep);
    tokens.assign(tok.begin(), tok.end());
    callback(tokens);
  }
}


TEST(foo, DISABLED_bar) {
  double probe_perf[8][5];
  size_t m_values[100];
  std::string data("./probe_performance.csv");
  std::ifstream in(data.c_str());
  ASSERT_TRUE(in.is_open());
  int i = -1;
  $u64 current_m = 0;
  parse_csv(in, [&](std::vector<std::string>& fields) {
    auto k = stoul(fields[0]);
    auto m = stoul(fields[1]);
    if (current_m != m) {
      current_m = m;
      i++;
    }
    auto p = stod(fields[2]);
    std::cout << fields[0] << "|"
              << fields[1] << "|"
              << fields[2]
              << std::endl;
    m_values[i] = m;
    probe_perf[k][i] = p;
  });
  for (size_t k = 1; k < 8; k++) {
    for (size_t m = 0; m < 5; m++) {
      std::cout << probe_perf[k][m] << ",";
    }
    std::cout << std::endl;
  }


  std::vector<$f64> fpr_load_factor;
  std::vector<std::vector<$f64>> fpr_k;
  fpr_k.resize(8);
  std::string fpr_data("./false_positive_rate.csv");
  std::ifstream fpr_in(fpr_data.c_str());
  ASSERT_TRUE(fpr_in.is_open());
  parse_csv(fpr_in, [&](std::vector<std::string>& fields) {
    fpr_load_factor.push_back(stod(fields[0]));
    fpr_k[1].push_back(stod(fields[1]));
    fpr_k[2].push_back(stod(fields[2]));
    fpr_k[3].push_back(stod(fields[3]));
    fpr_k[4].push_back(stod(fields[4]));
    fpr_k[5].push_back(stod(fields[5]));
    fpr_k[6].push_back(stod(fields[6]));
    fpr_k[7].push_back(stod(fields[7]));
  });


  u64 card_r = 600037902u;
  u64 card_l = 20000000u;
  f64 sel = 0.01;
  u64 n = card_l * sel;
  f64 c_tuple_pre = 10;
  double c[8][5];
  double picked_f[8][5];
  for (size_t k = 1; k < 8; k++) {
    for (size_t m = 0; m < 5; m++) {

      // lookup fpr
      auto lf = (n * 1.0) / m_values[m];
      auto lf_it = std::lower_bound(fpr_load_factor.begin(), fpr_load_factor.end(), lf);
      auto lf_idx = std::distance(fpr_load_factor.begin(), lf_it);
      std::vector<$f64>& fpr = fpr_k[k];
      auto f = fpr[lf_idx];
      picked_f[k][m] = f;
      // lookup c_tuple_bloom
      auto c_tuple_bloom = probe_perf[k][m];

      // c_t,dis= f(m,n,k)٠(1-sel)٠c_t,pre  + c_t,bloom (m,k)

      c[k][m] = f * (1 - sel) * c_tuple_pre + c_tuple_bloom;
    }
  }

  std::cout << "costs: ------------------" << std::endl;
  std::cout << "k";
  for (size_t m = 0; m < 5; m++) {
    std::cout << "  " << m_values[m];
  }
  std::cout << std::endl;
  for (size_t k = 1; k < 8; k++) {
    std::cout << k ;
    for (size_t m = 0; m < 5; m++) {
      std::cout << "  " << c[k][m];
    }
    std::cout << std::endl;
  }

  std::cout << "costs: ------------------" << std::endl;
  std::cout << "k m costs";
  std::cout << std::endl;
  for (size_t k = 1; k < 8; k++) {
    for (size_t m = 0; m < 5; m++) {
      std::cout << m_values[m] << "  " <<k << "  " << c[k][m] << std::endl;
    }
    std::cout << std::endl;
  }
}