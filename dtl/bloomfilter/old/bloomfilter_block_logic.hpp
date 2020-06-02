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

#if defined(__CUDA_ARCH__)
#include <cub/cub.cuh>
#endif // defined(__CUDA_ARCH__)

#include "dtl/bloomfilter/hash_family.hpp"

namespace dtl {

/// A multi-word block. The k bits are distributed among all words of the block (optionally in a sectorized manner).
template<
    typename Tk,           // the key type
    typename Tw = u32,     // the word type to use for the bitset
    typename Th = u32,     // the hash value type
    u32 K = 2,             // the number of bits to set per sector
    u32 B = 4,             // the block size in bytes
    u32 S = 4              // the sector size in bytes
>
struct bloomfilter_block_logic {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  using size_t = $u32;

  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");

  static_assert(B >= sizeof(word_t), "The block size must be greater or equal to the word size.");
  static_assert(S <= B, "The sector size must not exceed the block size.");
  static_assert(is_power_of_two(S), "The sector size must be a power of two.");

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_log2_mask = (1u << word_bitlength_log2) - 1;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  // The number of words per block.
  static constexpr u32 word_cnt = B / sizeof(word_t);
  static_assert(is_power_of_two(word_cnt), "The number of words per block must be a power of two.");
  // The number of bits needed to address the individual words within a block.
  static constexpr u32 word_cnt_log2 = dtl::ct::log_2_u32<word_cnt>::value;
  static constexpr u32 word_cnt_mask = word_cnt - 1;

  static constexpr u32 block_bitlength = word_bitlength * word_cnt;
  static constexpr u32 block_bitlength_log2 = dtl::ct::log_2_u32<block_bitlength>::value;
  static constexpr u32 block_bitlength_mask = word_cnt - 1;


  // The number of bits to set per element per sector.
  static constexpr u32 k = K;
  static_assert(k > 0, "Parameter 'k' must be at least '1'.");


  // Split the block into multiple sectors (or sub-blocks) with a length of a power of two.
  // Note that sectorization is a specialization. Having only one sector = no sectorization.
  static constexpr u1 sectorized = S < B;
  static constexpr u32 sector_cnt = B / S;
  static constexpr u32 sector_cnt_mask = sector_cnt - 1;
  static constexpr u32 sector_bitlength = block_bitlength / sector_cnt;
  // The number of bits needed to address the individual bits within a sector.
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
//  static constexpr u32 sector_bitlength_log2_mask = (1u << sector_bitlength_log2) - 1;
  static constexpr word_t sector_mask = sector_bitlength - 1;

  // The (static) length of the Bloom filter block.
  static constexpr size_t m = word_cnt * word_bitlength;

  using hash_value_t = Th;
  static constexpr size_t hash_value_bitlength = sizeof(hash_value_t) * 8;

  // The number of hash bits required per k.
  static constexpr size_t required_hash_bits_per_k = sector_bitlength_log2;

  // The number of hash bits required per element.
  static constexpr size_t required_hash_bits_per_element = k * required_hash_bits_per_k * sector_cnt;

  // When do we have to hash again
  static constexpr u32 hash_mod = hash_value_bitlength / required_hash_bits_per_k;


  static constexpr u32 shift = sector_cnt >= word_cnt
                               ? dtl::ct::log_2_u32<sector_cnt / word_cnt>::value
                               : dtl::ct::log_2_u32<word_cnt / sector_cnt>::value;



private:

  template<typename T>
  __forceinline__ __unroll_loops__ __host__ __device__
  static T load(T const *ptr) {
#if defined(__CUDA_ARCH__)
    static constexpr cub::CacheLoadModifier cache_load_modifier = cub::LOAD_CG;
    return cub::ThreadLoad<cache_load_modifier>(word_array);
#else
    return *ptr;
#endif // defined(__CUDA_ARCH__)
  }

 public:

  //===----------------------------------------------------------------------===//

  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  insert(const key_t& key, word_t* __restrict block) noexcept {
    $u32 current_k = 0;
    hash_value_t hash_val = 0;
    // A very straight forward implementation (without any optimizations).
    // In each sector, set K bits.
    for ($u32 sec_idx = 0; sec_idx < sector_cnt; sec_idx++) {
      for ($u32 k_idx = 0; k_idx < K; k_idx++) {
        if (current_k % hash_mod == 0) {
          hash_val = dtl::hash::dyn::mul32::hash(key, current_k + 1);
        }
        u32 sector_offset = sector_bitlength * sec_idx;
        u32 sector_bit_idx = hash_val >> (hash_value_bitlength - sector_bitlength_log2);
        u32 block_bit_idx = sector_offset + sector_bit_idx;

        u32 word_idx = block_bit_idx / word_bitlength;
        u32 word_bit_idx = block_bit_idx & word_bitlength_log2_mask;

        block[word_idx] |= word_t(1) << word_bit_idx;

        hash_val <<= required_hash_bits_per_k;
        current_k++;
      }
    }
  }
  //===----------------------------------------------------------------------===//

  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const key_t& key, const word_t* __restrict block) noexcept {
    auto ret_val = true;
    $u32 current_k = 0;
    hash_value_t hash_val = 0;
    // A very straight forward implementation (without any optimizations).
    // In each sector, set K bits.
    for ($u32 sec_idx = 0; sec_idx < sector_cnt; sec_idx++) {
      for ($u32 k_idx = 0; k_idx < K; k_idx++) {
        if (current_k % hash_mod == 0) {
          hash_val = dtl::hash::dyn::mul32::hash(key, current_k + 1);
        }
        u32 sector_offset = sector_bitlength * sec_idx;
        u32 sector_bit_idx = hash_val >> (hash_value_bitlength - sector_bitlength_log2);
        u32 block_bit_idx = sector_offset + sector_bit_idx;

        u32 word_idx = block_bit_idx / word_bitlength;
        u32 word_bit_idx = block_bit_idx & word_bitlength_log2_mask;

        ret_val &= dtl::bits::bit_test(block[word_idx], word_bit_idx);

        hash_val <<= required_hash_bits_per_k;
        current_k++;
      }
    }
    return ret_val;
  }
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
