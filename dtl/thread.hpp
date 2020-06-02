#pragma once

#include <atomic>
#include <bitset>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <random>
#include <sched.h>
#include <thread>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/bitset.hpp>
#include <dtl/thread.hpp>
#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach_types.h>
#include <mach/thread_act.h>
#endif

namespace dtl {

#ifdef __APPLE__
// Adapted from https://yyshen.github.io/2015/01/18/binding_threads_to_cores_osx.html
#define CPU_SETSIZE 32 // TODO should be 1024

typedef struct cpu_set {
  $u32 count;
} cpu_set_t;

static inline void CPU_ZERO(cpu_set_t* cs) { cs->count = 0; }
static inline void CPU_SET(int num, cpu_set_t* cs) { cs->count |= (1 << num); }
static inline int CPU_ISSET(int num, cpu_set_t* cs) { return (cs->count & (1 << num)); }

inline int sched_getaffinity(pid_t pid, std::size_t cpu_size, cpu_set_t* cpu_set) {
  $i32 core_count = 0;
  std::size_t len = sizeof(core_count);
  i32 ret = sysctlbyname("machdep.cpu.core_count", &core_count, &len, 0, 0);
  if (ret) {
    printf("error getting core count %d\n", ret);
    return -1;
  }
  cpu_set->count = 0;
  for ($i32 i = 0; i < core_count; i++) {
    cpu_set->count |= (1 << i);
  }
  return 0;
}

inline int pthread_setaffinity_np(pthread_t thread, std::size_t cpu_size, cpu_set_t* cpu_set) {
  thread_port_t mach_thread;
  $i32 core = 0;
  for (core = 0; core < 8*cpu_size; core++) {
    if (CPU_ISSET(core, cpu_set)) break;
  }
  thread_affinity_policy_data_t policy = {core};
  mach_thread = pthread_mach_thread_np(thread);
  thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, (thread_policy_t) &policy, 1);
  return 0;
}
#endif

using cpu_mask = dtl::bitset<CPU_SETSIZE>;

namespace this_thread {

static auto
random_seed() {
  static std::random_device rd;
  return rd();
}

// Pseudo random number generator (per thread, to avoid synchronization)
static thread_local std::mt19937 rand32(random_seed());
static thread_local std::mt19937_64 rand64(random_seed());


/// pins the current thread to the specified CPU(s)
static void
set_cpu_affinity(const dtl::cpu_mask& cpu_mask) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  // convert bitset to CPU mask
  for (auto it = dtl::on_bits_begin(cpu_mask);
       it != dtl::on_bits_end(cpu_mask);
       it++) {
    CPU_SET(*it, &mask);
  }
  i32 result = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
  if (result != 0) {
    std::cout << "Failed to set CPU affinity." << std::endl;
  }
}

/// pins the current thread to a specific CPU
static void
set_cpu_affinity(u32 cpu_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu_id % std::thread::hardware_concurrency(), &mask);
  i32 result = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
  if (result != 0) {
    std::cout << "Failed to set CPU affinity." << std::endl;
  }
}

/// reset the CPU affinity of the current thread
[[maybe_unused]] static void
reset_cpu_affinity() {
  dtl::cpu_mask m;
  for ($u64 i = 0; i < std::thread::hardware_concurrency(); i++) {
    m[i] = 1;
  }
  set_cpu_affinity(m);
}

/// returns the CPU affinity of the current thread
static dtl::cpu_mask
get_cpu_affinity() {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  i32 result = sched_getaffinity(0, sizeof(mask), &mask);
  if (result != 0) {
    std::cout << "Failed to determine CPU affinity." << std::endl;
  }
  // convert c-style mask to std::bitset
  u64 bit_width = sizeof(cpu_set_t)*8;
  dtl::cpu_mask bm = dtl::to_bitset<bit_width>(&mask);
  return bm;
}


namespace detail {


// used to assign unique ids
static std::atomic<$u64> thread_cntr(0);
static thread_local $u64 uid = ~0ull;
static thread_local $u64 id = ~0ull;


static void
init(u32 thread_id, std::function<void()> fn) {
  // affinitize thread
  dtl::this_thread::set_cpu_affinity(thread_id);
  // set the given thread id
  dtl::this_thread::detail::id = thread_id;
  // set the unique id
  dtl::this_thread::detail::uid = dtl::this_thread::detail::thread_cntr.fetch_add(1);
  // run the given function in the current thread
  fn();
};


} // namespace detail


/// returns the given id of the current thread
[[maybe_unused]] static u64
get_id() {
  return detail::id;
}


/// returns the unique id of the current thread
[[maybe_unused]] static u64
get_uid() {
  return detail::uid;
}


} // namespace this_thread


/// spawn a new thread
[[maybe_unused]] static std::thread
thread(u32 thread_id, std::function<void()> fn) {
  return std::thread(dtl::this_thread::detail::init, thread_id, fn);
}

/// spawn a new thread
template<typename Fn, typename... Args>
static std::thread
thread(u32 thread_id, Fn&& fn, Args&&... args) {
  return std::thread(dtl::this_thread::detail::init, thread_id,
                     std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...));
}

/// thread affinitizer
/// @deprecated
static void
thread_affinitize(u32 thread_id) {
  dtl::this_thread::set_cpu_affinity(thread_id);
}

[[maybe_unused]] static void
run_in_parallel(std::function<void()> fn,
                u32 thread_cnt = std::thread::hardware_concurrency()) {

  auto thread_fn = [](u32 thread_id, std::function<void()> fn) {
    thread_affinitize(thread_id);
    fn();
  };

  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < thread_cnt - 1; i++) {
    workers.push_back(std::move(std::thread(thread_fn, i, fn)));
  }
  std::thread(thread_fn, thread_cnt - 1, fn).join();
  for (auto& worker : workers) {
    worker.join();
  }
}

static auto identity(u32 thread_id) -> u32 {
  return thread_id;
}

[[maybe_unused]] static void
run_in_parallel(std::function<void(u32 thread_id)> fn,
                u32 thread_cnt = std::thread::hardware_concurrency(),
                std::function<u32(u32)> cpu_map = identity) {

  auto thread_fn = [&cpu_map](u32 thread_id, std::function<void(u32 thread_id)> fn) {
    thread_affinitize(cpu_map(thread_id));
    fn(thread_id);
  };

  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < thread_cnt - 1; i++) {
    workers.push_back(std::move(std::thread(thread_fn, i, fn)));
  }
  std::thread(thread_fn, thread_cnt - 1, fn).join();
  for (auto& worker : workers) {
    worker.join();
  }
}

[[maybe_unused]] static void
run_in_parallel(std::function<void(u32 thread_id)> fn,
                const dtl::cpu_mask& cpu_mask,
                u32 thread_cnt) {

  const auto affinity_saved = dtl::this_thread::get_cpu_affinity();

  std::vector<$u32> map;
  for (auto it = cpu_mask.on_bits_begin(); it != cpu_mask.on_bits_end(); it++) {
    map.push_back(*it);
  }

  auto cpu_map = [&](u32 thread_id) {
    return map[thread_id % map.size()];
  };

  auto thread_fn = [&cpu_map](u32 thread_id, std::function<void(u32 thread_id)> fn) {
    thread_affinitize(cpu_map(thread_id));
    fn(thread_id);
  };

  std::vector<std::thread> workers;
  for (std::size_t i = 0; i < thread_cnt - 1; i++) {
    workers.push_back(std::move(std::thread(thread_fn, i, fn)));
  }
  std::thread(thread_fn, thread_cnt - 1, fn).join();
  for (auto& worker : workers) {
    worker.join();
  }
  dtl::this_thread::set_cpu_affinity(affinity_saved);
}

[[maybe_unused]] static void
run_in_parallel_async(std::function<void()> fn,
                      std::vector<std::thread>& workers,
                      u32 thread_cnt = std::thread::hardware_concurrency()) {
  workers.reserve(thread_cnt);

  auto thread_fn = [](u32 thread_id, std::function<void()> fn) {
    thread_affinitize(thread_id);
    fn();
  };

  for (std::size_t i = 0; i < thread_cnt; i++) {
    workers.push_back(std::move(std::thread(thread_fn, i, fn)));
  }
}

[[maybe_unused]] static void
wait_for_threads(std::vector<std::thread>& workers) {
  for (auto& worker : workers) {
    worker.join();
  }
  workers.clear();
}

} // namespace dtl
