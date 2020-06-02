#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/math.hpp>


namespace dtl {
namespace bloomfilter_dynamic {


//===----------------------------------------------------------------------===//
// Block addressing base class.
//===----------------------------------------------------------------------===//
struct block_addressing_logic {

  using size_t = $u32;
  using hash_value_t = $u32;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// The number of blocks.
  const size_t block_cnt;
  /// The number of bits required to address the individual blocks.
  const size_t block_cnt_log2;
  //===----------------------------------------------------------------------===//

  block_addressing_logic() noexcept = default;

  block_addressing_logic(const size_t block_cnt, const size_t block_cnt_log2)
      : block_cnt(block_cnt), block_cnt_log2(block_cnt_log2) { }


  virtual ~block_addressing_logic() = default;


  /// Returns the index of the block the hash value maps to.
  virtual size_t get_block_idx(hash_value_t hash_value) const noexcept = 0;

};


//===----------------------------------------------------------------------===//
// Block addressing - Magic modulus.
//===----------------------------------------------------------------------===//
struct bloomfilter_addressing_logic_magic : block_addressing_logic {

  using size_t = $u32;
  using hash_value_t = $u32;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const dtl::fast_divisor_u32_t fast_divisor;
  //===----------------------------------------------------------------------===//


  static
  size_t
  determine_block_cnt(const size_t desired_block_cnt) {
    u32 actual_block_cnt = dtl::next_cheap_magic(desired_block_cnt).divisor;
    return actual_block_cnt;
  }
  //===----------------------------------------------------------------------===//


 public:

  explicit
  bloomfilter_addressing_logic_magic(const size_t desired_block_cnt) noexcept
      : block_addressing_logic(determine_block_cnt(desired_block_cnt),
                                     dtl::log_2(dtl::next_power_of_two(determine_block_cnt(desired_block_cnt)))),
        fast_divisor(dtl::next_cheap_magic(block_cnt)) { }


  bloomfilter_addressing_logic_magic(const bloomfilter_addressing_logic_magic&) noexcept = default;


  bloomfilter_addressing_logic_magic(bloomfilter_addressing_logic_magic&&) noexcept = default;


  ~bloomfilter_addressing_logic_magic() = default;
  //===----------------------------------------------------------------------===//


  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  size_t
  get_block_idx(const hash_value_t hash_value) const noexcept override {
    const size_t block_idx = dtl::fast_mod_u32(hash_value, fast_divisor);
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Block addressing - Power of two.
//===----------------------------------------------------------------------===//
struct bloomfilter_addressing_logic_pow2 : block_addressing_logic {

  using size_t = $u32;
  using hash_value_t = $u32;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const size_t block_cnt_mask;
  //===----------------------------------------------------------------------===//


  static constexpr
  size_t
  determine_block_cnt(const size_t desired_block_cnt) {
    u32 actual_block_cnt = dtl::next_power_of_two(desired_block_cnt);
    return actual_block_cnt;
  }
  //===----------------------------------------------------------------------===//


 public:

  explicit
  bloomfilter_addressing_logic_pow2(const size_t desired_block_cnt) noexcept
      : block_addressing_logic(determine_block_cnt(desired_block_cnt),dtl::log_2(determine_block_cnt(desired_block_cnt))),
        block_cnt_mask(block_cnt - 1) { }


  bloomfilter_addressing_logic_pow2(const bloomfilter_addressing_logic_pow2&) noexcept = default;


  bloomfilter_addressing_logic_pow2(bloomfilter_addressing_logic_pow2&&) noexcept = default;


  ~bloomfilter_addressing_logic_pow2() = default;
  //===----------------------------------------------------------------------===//


  /// Returns the index of the block the hash value maps to.
  __forceinline__ __host__ __device__
  size_t
  get_block_idx(const hash_value_t hash_value) const noexcept override {
    const auto block_idx = (hash_value >> (sizeof(hash_value_t) * 8 - block_cnt_log2)) & block_cnt_mask;
    return block_idx;
  }
  //===----------------------------------------------------------------------===//


};

} // namespace bloomfilter_dynamic
} // namespace dtl