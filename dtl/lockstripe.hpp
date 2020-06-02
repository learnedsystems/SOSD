#pragma once

#include <algorithm>
#include <mutex>
#include <type_traits>

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>


namespace dtl {


template<u64 N>
struct lockstripe {
  static_assert(dtl::is_power_of_two(N), "Template parameter 'N' must be a power of two.");
  static constexpr u64 mask = N - 1;
  static constexpr u64 page_size = 4096;

  struct entry {
    alignas(dtl::mem::cacheline_size) std::mutex mutex;
    std::array<$u8, page_size - sizeof(mutex)> padding; // at least on cache-line, but we are very pessimistic
  };

  const std::array<entry, N> entries;

  inline void
  lock(u64 n) {
    entries[n & mask].mutex.lock();
  };

  inline u1
  try_lock(u64 n) {
    return entries[n & mask].mutex.try_lock();
  };

  inline void
  unlock(u64 n) {
    entries[n & mask].mutex.unlock();
  };

};


} // namespace dtl