#pragma once

#include <chrono>
#include <map>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter/block_addressing_logic.hpp>

#include "immintrin.h"

namespace dtl {

namespace {

struct config {
  u32 k;
  u32 word_size;
  u32 word_cnt_per_block;
  u32 sector_cnt;
  dtl::block_addressing addr_mode;

  bool
  operator<(const config &o) const {
    return k < o.k
        || (k == o.k && word_size  < o.word_size)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block  < o.word_cnt_per_block)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt < o.sector_cnt)
        || (k == o.k && word_size == o.word_size && word_cnt_per_block == o.word_cnt_per_block && sector_cnt == o.sector_cnt && addr_mode < o.addr_mode);
  }

};

struct tuning_params {
  $u32 unroll_factor = 1;

  tuning_params() = default;
  ~tuning_params() = default;
  tuning_params(const tuning_params& other) = default;
  tuning_params(tuning_params&& other) = default;

  tuning_params& operator=(const tuning_params& rhs) = default;
  tuning_params& operator=(tuning_params&& rhs) = default;
};

} // anonymous namespace


//===----------------------------------------------------------------------===//
/// Provides tuning parameters to the Bloom filter instance.
struct blocked_bloomfilter_tune {

  /// Sets the SIMD unrolling factor for the given blocked Bloom filter config.
  /// Note: unrolling by 0 means -> scalar code (no SIMD)
  virtual void
  set_unroll_factor(u32 k,
                    u32 word_size,
                    u32 word_cnt_per_block,
                    u32 sector_cnt,
                    dtl::block_addressing addr_mode,
                    u32 unroll_factor) {
    throw std::runtime_error("Not supported");
  }


  /// Returns the SIMD unrolling factor for the given blocked Bloom filter config.
  /// Note: unrolling by 0 means -> scalar code (no SIMD)
  virtual $u32
  get_unroll_factor(u32 k,
                    u32 word_size,
                    u32 word_cnt_per_block,
                    u32 sector_cnt,
                    dtl::block_addressing addr_mode) const {
    return 1; // default
  }


  /// Determines the best performing SIMD unrolling factor for the given
  /// blocked Bloom filter config.
  virtual $u32
  tune_unroll_factor(u32 k,
                     u32 word_size,
                     u32 word_cnt_per_block,
                     u32 sector_cnt,
                     dtl::block_addressing addr_mode) {
    throw std::runtime_error("Not supported");
  }


  /// Determines the best performing SIMD unrolling factor for all valid
  /// blocked Bloom filter configs.
  virtual void
  tune_unroll_factor() {
    throw std::runtime_error("Not supported");
  }

};
//===----------------------------------------------------------------------===//

} // namespace dtl