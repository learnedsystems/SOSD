#include "gtest/gtest.h"

#include <functional>
#include <numeric>
#include <iostream>
#include <string>
#include <sstream>

#include <dtl/dtl.hpp>
#include <dtl/color.hpp>
#include <dtl/env.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>
#include <dtl/thread.hpp>

#include "immintrin.h"

namespace dtl {
namespace mem {


} // namespace mem
} // namespace dtl


TEST(mem, numa) {
  auto node_cnt = dtl::mem::get_node_count();
  std::cout << "node cnt: " << node_cnt << std::endl;
  auto cpu_node_cnt = dtl::mem::get_cpu_nodes().size();
  std::cout << "cpu node cnt: " << cpu_node_cnt << std::endl;
  std::cout << "cpu node ids:";
  for (auto node_id : dtl::mem::get_cpu_nodes()) {
    std::cout << " " << node_id;
  }
  std::cout << std::endl;
  auto hbm_node_cnt = dtl::mem::get_hbm_nodes().size();
  std::cout << "hbm node cnt: " << hbm_node_cnt << std::endl;
  if (hbm_node_cnt > 0) {
    std::cout << "hbm node ids:";
    for (auto node_id : dtl::mem::get_hbm_nodes()) {
      std::cout << " " << node_id;
    }
    std::cout << std::endl;
  }
}

inline auto timing(std::function<void()> fn) {
  auto start = std::chrono::high_resolution_clock::now();
  fn();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

static constexpr std::chrono::seconds sec(1);
static constexpr double nano_to_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(sec).count();


class busy_barrier {
 private:
  const std::size_t threadCount;
  std::atomic<std::size_t> cntr;
  std::atomic<bool> gogo;

 public:
  explicit busy_barrier(std::size_t threadCount) :
      threadCount(threadCount), cntr(threadCount), gogo(false) {
  }
  void wait() {
    while (gogo) {
      // wait until barrier is ready for re-use
    }
    cntr--;
    if (cntr == 0) {
      cntr++;
      gogo = true;
    }
    else {
      while (!gogo) {
        // busy wait until all threads arrived
      }
      std::size_t prevCntr = cntr.fetch_add(1);
      if (prevCntr + 1 == threadCount) {
        gogo = false;
      }
    }
  }
};


// the grain size for parallel experiments
static u64 preferred_grain_size = 1ull << dtl::env<$i32>::get("GRAIN_SIZE", 16);

// repeats the benchmark with different concurrency settings
static i32 thread_cnt = dtl::env<$i32>::get("THREAD_CNT", 4);

// the number of repetitions
static u64 repeat_cnt = dtl::env<$i32>::get("REPEAT_CNT", 64);;


TEST(mem, bandwidth_test_hbm) {
  std::cout << "thread cnt: " << thread_cnt << std::endl;

  // settings
  u64 n_gb = 2ull;
  using T = $u64;
  using vec_t = std::vector<T, dtl::mem::numa_allocator<T>>;
  using barrier_t = busy_barrier;

  std::vector<$u64> accus;
  accus.resize(thread_cnt, 0);

  struct node_data_t {
    std::atomic<$u64> grain_cntr {0};
    vec_t* data {nullptr};
  };

  std::vector<node_data_t> node_data(dtl::mem::get_node_count());

  u64 len = n_gb * 1024 * 1024 * 1024 / sizeof(T);
  for (auto node_id : dtl::mem::get_nodes()) {
    std::cout << "preparing data on node " << node_id << std::endl;
//    dtl::mem::numa_allocator<T> alloc(dtl::mem::allocator_config::on_node(node_id));
    dtl::mem::numa_allocator<T> alloc(dtl::mem::allocator_config::interleave_cpu());
    node_data[node_id].data = new vec_t(len, 0, alloc);
    for (std::size_t i = 0; i < node_data[node_id].data->size(); i++) {
      (*node_data[node_id].data)[i] = i;
    }
  }


  u64 grain_size = std::min(preferred_grain_size, len);
  barrier_t kick_off(thread_cnt);

  auto worker_fn = [&](u32 thread_id) -> void {
    // determine NUMA node id
    const auto cpu_mask = dtl::this_thread::get_cpu_affinity();
    const auto cpu_id = cpu_mask.find_first(); // handwaving
    const auto numa_node_id = dtl::mem::get_node_of_cpu(cpu_id);

    // determine nearest HBM node (returns numa_node_id if HBM is not available)
//    const auto node_id = dtl::mem::get_nearest_hbm_node(numa_node_id);
    const auto node_id = numa_node_id;

    std::stringstream out;
    out << "thread id: " << thread_id << ", cpu node id: " << numa_node_id << ", data node id: " << node_id << std::endl;
    std::cout << out.str();

    vec_t& v = *node_data[node_id].data;

    kick_off.wait();

    // allocate a match vector
    $u64 accu;

    while (true) {
      u64 cntr = node_data[node_id].grain_cntr.fetch_add(grain_size);
      u64 read_from = cntr % len;
      u64 read_to = std::min(len, read_from + grain_size);
      if (cntr >= len * repeat_cnt) break;
      u64 cnt = read_to - read_from;
//      accu += std::accumulate(v.begin() + read_from, v.begin() + read_to, 0);
      for (std::size_t i = read_from; i < read_to; i+=8) {
        accu += v[i];
      }
    }
    accus[thread_id] = accu;
  };



  $u64 total_sum;
  $f64 duration = timing([&] {
    dtl::run_in_parallel(worker_fn, thread_cnt);
    total_sum = std::accumulate(accus.begin(), accus.end(), 0);
  });
  std::cout << total_sum << std::endl;
  std::cout  << (n_gb * dtl::mem::get_cpu_nodes().size() * repeat_cnt * 1.0) / (duration / nano_to_sec) << " GiB/s" << std::endl;
  std::cout  << (n_gb * dtl::mem::get_cpu_nodes().size() * repeat_cnt * 1024.0 * 1024.0 * 1024.0 / 1000.0 / 1000.0) / (duration / nano_to_sec) << " MB/s" << std::endl;
}

TEST(mem, bandwidth_test_hbm_rw) {
  std::cout << "thread cnt: " << thread_cnt << std::endl;

  // settings
  u64 n_gb = 1ull;
  using T = $u64;
  using vec_t = std::vector<T, dtl::mem::numa_allocator<T>>;
  using barrier_t = busy_barrier;

  std::vector<$u64> accus;
  accus.resize(thread_cnt, 0);

  struct node_data_t {
    std::atomic<$u64> grain_cntr {0};
    vec_t* data {nullptr};
    vec_t* out {nullptr};
  };

  std::vector<node_data_t> node_data(dtl::mem::get_node_count());

  u64 len = n_gb * 1024 * 1024 * 1024 / sizeof(T);
  for (auto node_id : dtl::mem::get_nodes()) {
    std::cout << "preparing data on node " << node_id << std::endl;
    dtl::mem::numa_allocator<T> alloc(dtl::mem::allocator_config::on_node(node_id));
//    dtl::mem::numa_allocator<T> alloc_out(dtl::mem::allocator_config::on_node(node_id > 3 ? node_id - 4 : node_id));
    dtl::mem::numa_allocator<T> alloc_out(dtl::mem::allocator_config::on_node(node_id));
    node_data[node_id].data = new vec_t(len, 0, alloc);
    node_data[node_id].out = new vec_t(len, 0, alloc_out);
    for (std::size_t i = 0; i < node_data[node_id].data->size(); i++) {
      (*node_data[node_id].data)[i] = i;
      (*node_data[node_id].out)[i] = 0;
    }
  }


  u64 grain_size = std::min(preferred_grain_size, len);
  barrier_t kick_off(thread_cnt);

  auto worker_fn = [&](u32 thread_id) -> void {
    // determine NUMA node id
    const auto cpu_mask = dtl::this_thread::get_cpu_affinity();
    const auto cpu_id = cpu_mask.find_first(); // handwaving
    const auto numa_node_id = dtl::mem::get_node_of_cpu(cpu_id);

    // determine nearest HBM node (returns numa_node_id if HBM is not available)
    const auto node_id = dtl::mem::get_nearest_hbm_node(numa_node_id);

    std::stringstream out;
    out << "thread id: " << thread_id << ", cpu node id: " << numa_node_id << ", data node id: " << node_id << std::endl;
    std::cout << out.str();

    vec_t& v = *node_data[node_id].data;
    vec_t& v_out = *node_data[node_id].out;

//    kick_off.wait();

    // allocate a match vector
    $u64 accu;

    while (true) {
      u64 cntr = node_data[node_id].grain_cntr.fetch_add(grain_size);
      u64 read_from = cntr % len;
      u64 read_to = std::min(len, read_from + grain_size);
      if (cntr >= len * repeat_cnt) break;
      u64 cnt = read_to - read_from;
//      accu += std::accumulate(v.begin() + read_from, v.begin() + read_to, 0);
      for (std::size_t i = read_from; i < read_to; i+=8) {
        v_out[i] = v[i];
      }
    }
    accus[thread_id] = accu;
  };



  $u64 total_sum;
  $f64 duration = timing([&] {
    dtl::run_in_parallel(worker_fn, thread_cnt);
    total_sum = std::accumulate(accus.begin(), accus.end(), 0);
  });
  std::cout << total_sum << std::endl;
  std::cout  << (2 * n_gb * dtl::mem::get_cpu_nodes().size() * repeat_cnt * 1.0) / (duration / nano_to_sec) << " GiB/s" << std::endl;
  std::cout  << (2 * n_gb * dtl::mem::get_cpu_nodes().size() * repeat_cnt * 1024.0 * 1024.0 * 1024.0 / 1000.0 / 1000.0) / (duration / nano_to_sec) << " MB/s" << std::endl;

}

TEST(foo, bar) {
  uint64_t prev = 0;
  for (uint32_t i = 0; i < std::numeric_limits<uint32_t>::max(); i++) {
    if (dtl::bits::pop_count(i) < 3) {
      uint64_t size = i; //(i + 64) / 1024.0 / 1024.0;
      std::cout << std::bitset<32>(i) << " " << size << " - " << (size - prev) << std::endl;
      prev = size;
    }
  }

}
