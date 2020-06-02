#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>

#include "hash.hpp"

namespace dtl {
namespace bloomfilter_dynamic {


/// A multi-word block. The k bits are distributed among all words of the block (optionally in a sectorized manner).
template<
    typename Tw,           // the word type to use for the bitset
    typename Th,           // the hash value type
    typename HashFn        // the hash function (family) to use
>
struct blocked_bloomfilter_block_logic {

  using key_t = $u32;
  using word_t = typename std::remove_cv<Tw>::type;
  using size_t = $u32;

  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");


  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_log2_mask = (1u << word_bitlength_log2) - 1;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

//  // The number of bits needed to address the individual words within a block.
//  static constexpr u32 word_cnt_log2 = dtl::ct::log_2_u32<word_cnt>::value;
//  static constexpr u32 word_cnt_mask = word_cnt - 1;
//
//

//  // The (static) length of the Bloom filter block.
//  static constexpr size_t m = word_cnt * word_bitlength;

  using hash_value_t = typename std::remove_cv<Th>::type;
  static constexpr size_t hash_value_bitlength = sizeof(hash_value_t) * 8;

  using hasher = HashFn;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  // The number of bits to set per element per sector.
  u32 k;
  // The number of words per block.
  u32 word_cnt;

  // The length of the Bloom filter block.
  u32 block_bitlength;
//  static constexpr u32 block_bitlength_log2 = dtl::ct::log_2_u32<block_bitlength>::value;
//  static constexpr u32 block_bitlength_mask = word_cnt - 1;

  u32 sector_cnt;
  u32 sector_cnt_mask;
  u32 sector_bitlength;
  // The number of bits needed to address the individual bits within a sector.
  u32 sector_bitlength_log2;
  word_t sector_mask;

  // The number of hash bits required per k.
  size_t required_hash_bits_per_k = sector_bitlength_log2;

  // The number of hash bits required per element.
  size_t required_hash_bits_per_element = k * required_hash_bits_per_k * sector_cnt;

  // When do we have to hash again
  u32 hash_mod = hash_value_bitlength / required_hash_bits_per_k;
  //===----------------------------------------------------------------------===//

 private:

  template<typename T>
  __forceinline__ __unroll_loops__ __host__ __device__
  static T load(T const *ptr) {
    return *ptr;
  }

 public:

  //===----------------------------------------------------------------------===//

  blocked_bloomfilter_block_logic(u32 B,    // the block size in bytes
                          u32 S,    // the sector size in bytes
                          u32 K)    // the number of bits to set per sector
    : k(K),
      word_cnt(B / sizeof(word_t)),
      block_bitlength(word_bitlength * word_cnt),
      sector_cnt(B / S),
      sector_cnt_mask(sector_cnt - 1),
      sector_bitlength(B * 8 / sector_cnt),
      sector_bitlength_log2(dtl::log_2(sector_bitlength)),
      sector_mask(sector_bitlength - 1)
  {
    if (k == 0) throw std::invalid_argument("Parameter 'k' must be at least '1'.");
    if (B < sizeof(word_t)) throw std::invalid_argument("The block size must be greater or equal to the word size.");
    if (S > B) throw std::invalid_argument("The sector size must not exceed the block size.");
    if (!is_power_of_two(B)) throw std::invalid_argument("The block size must be a power of two.");
    if (!is_power_of_two(S)) throw std::invalid_argument("The sector size must be a power of two.");
    if (!is_power_of_two(word_cnt)) throw std::invalid_argument("The number of words per block must be a power of two.");
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  void
  insert(const key_t& key, word_t* __restrict block) const noexcept {
    $u32 current_k = 0;
    hash_value_t hash_val = 0;
    // A very straight forward implementation (without any optimizations).
    // In each sector, set K bits.
    for ($u32 sec_idx = 0; sec_idx < sector_cnt; sec_idx++) {
      for ($u32 k_idx = 0; k_idx < k; k_idx++) {
        if (current_k % hash_mod == 0) {
          current_k++;
          hash_val = hasher::hash(key, current_k);
        }
        u32 sector_offset = sector_bitlength * sec_idx;
        u32 sector_bit_idx = hash_val >> (hash_value_bitlength - sector_bitlength_log2);
        u32 block_bit_idx = sector_offset + sector_bit_idx;

        u32 word_idx = block_bit_idx / word_bitlength;
        u32 word_bit_idx = block_bit_idx & word_bitlength_log2_mask;

        block[word_idx] |= word_t(1) << word_bit_idx;

        hash_val <<= required_hash_bits_per_k;
      }
    }
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  u1
  contains(const key_t& key, const word_t* __restrict block) const noexcept {
    auto ret_val = true;
    $u32 current_k = 0;
    hash_value_t hash_val = 0;
    // A very straight forward implementation (without any optimizations).
    // In each sector, set K bits.
    for ($u32 sec_idx = 0; sec_idx < sector_cnt; sec_idx++) {
      for ($u32 k_idx = 0; k_idx < k; k_idx++) {
        if (current_k % hash_mod == 0) {
          current_k++;
          hash_val = hasher::hash(key, current_k);
        }
        u32 sector_offset = sector_bitlength * sec_idx;
        u32 sector_bit_idx = hash_val >> (hash_value_bitlength - sector_bitlength_log2);
        u32 block_bit_idx = sector_offset + sector_bit_idx;

        u32 word_idx = block_bit_idx / word_bitlength;
        u32 word_bit_idx = block_bit_idx & word_bitlength_log2_mask;

        ret_val &= dtl::bits::bit_test(block[word_idx], word_bit_idx);

        hash_val <<= required_hash_bits_per_k;
      }
    }
    return ret_val;
  }
  //===----------------------------------------------------------------------===//


};

} // namespace bloomfilter_dynamic
} // namespace dtl