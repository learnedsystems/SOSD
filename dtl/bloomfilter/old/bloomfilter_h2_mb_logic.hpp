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

#include <cub/cub.cuh>

#include "immintrin.h"

namespace dtl {

template<typename Tk,      // the key type
    template<typename Ty> class HashFn,     // the first hash function to use
    template<typename Ty> class HashFn2,    // the second hash function to use
    typename Tw = u64,     // the word type to use for the bitset
    u32 K = 5,             // the number of hash functions to use
    u32 B = 2,             // the number of blocks (multi-blocking)
    u1 Sectorized = false
>
struct bloomfilter_h2_mb_logic {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  using size_t = $u32;

  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");


  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;


  // Inspect the given hash function
  static_assert(
      std::is_same<decltype(HashFn<key_t>::hash(0)), decltype(HashFn2<key_t>::hash(0))>::value,
      "The two hash functions must return the same type.");
  using hash_value_t = $u32; //decltype(HashFn<key_t>::hash(0)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  static constexpr u32 hash_fn_cnt = 2;


  // The number of hash functions to use.
  static constexpr u32 k = K;
  static_assert(k > 1, "Parameter 'k' must be at least '2'.");

  // Split each word into multiple sectors (sub words, with a length of a power of two).
  // Note that sectorization is a specialization. Having only one sector = no sectorization.
  static constexpr u1 sectorized = Sectorized;


  static constexpr u32 compute_sector_cnt() {
    static_assert(!sectorized || ((word_bitlength / dtl::next_power_of_two(k)) != 0), "The number of sectors must be greater than zero. Probably the given number of hash functions is set to high.");
    return (!sectorized) ? 1
                         : word_bitlength / (word_bitlength / dtl::next_power_of_two(k));
  }
  static constexpr u32 sector_cnt = compute_sector_cnt();
  static constexpr u32 sector_bitlength = word_bitlength / sector_cnt;
  // the number of bits needed to address the individual bits within a sector
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
  static constexpr word_t sector_mask = sector_bitlength - 1;

  // the number of remaining bits of the FIRST hash value (used to identify the word)
  static constexpr i32 remaining_hash_bit_cnt = static_cast<i32>(hash_value_bitlength) - (sectorized ? sector_bitlength_log2 : word_bitlength_log2);
  static constexpr u64 min_m = 2 * word_bitlength; // Using only one word would cause undefined behaviour in bit shifts later on.
  static constexpr u64 max_m = (1ull << remaining_hash_bit_cnt) * word_bitlength;

  // The number of blocks to use.
  static constexpr u32 block_cnt = B;
  static_assert(B > 0, "Parameter 'B' must be at least '1'.");

  // ---- Members ----
  size_t length_mask; // the length of the bitvector (length_mask + 1) is not stored explicitly
  size_t word_cnt_log2; // the number of bits to address the individual words of the bitvector
  // ----


  static constexpr
  size_t
  determine_actual_length(const size_t length) {
    // round up to the next power of two
    return std::max(
        static_cast<size_t>(next_power_of_two(length)),
        static_cast<size_t>(min_m)
    );
  }


  /// C'tor
  explicit
  bloomfilter_h2_mb_logic(const size_t length)
      : length_mask(determine_actual_length(length) - 1),
        word_cnt_log2(dtl::log_2((length_mask + 1) / word_bitlength)) {
    if (((length_mask + 1)) > max_m) throw std::invalid_argument("Length must not exceed 'max_m'.");
  }

  /// Copy c'tor
  bloomfilter_h2_mb_logic(const bloomfilter_h2_mb_logic&) = default;


  __forceinline__ __host__ __device__
  size_t
  which_word(const hash_value_t hash_val) const noexcept {
    const size_t word_idx = hash_val >> (hash_value_bitlength - word_cnt_log2);
    return word_idx;
  }


  __forceinline__ __unroll_loops__ __host__ __device__
  word_t
  which_bits(const hash_value_t first_hash_val,
             const hash_value_t second_hash_val) const noexcept {
    u32 first_bit_idx = (first_hash_val >> (hash_value_bitlength - word_cnt_log2 - sector_bitlength_log2)) & sector_mask;
    word_t word = word_t(1) << first_bit_idx;
    for (size_t i = 1; i < k; i++) {
      u32 shift = (hash_value_bitlength - 2) - (i * sector_bitlength_log2);
      u32 bit_idx = (second_hash_val >> shift) & sector_mask;
      u32 sector_offset = (i * sector_bitlength) & word_bitlength_mask;
      word |= word_t(1) << (bit_idx + sector_offset);
    }
    return word;
  }


  __forceinline__ __host__ __device__
  u1
  contains(const key_t& key, const word_t* word_array) const noexcept {
    const hash_value_t first_hash_val = HashFn<key_t>::hash(key);
    const hash_value_t second_hash_val = HashFn2<key_t>::hash(key);
    u32 word_idx = which_word(first_hash_val);
    const word_t search_mask = which_bits(first_hash_val, second_hash_val);
#if defined(__CUDA_ARCH__)
    const word_t word = cub::ThreadLoad<cub::LOAD_CS>(word_array + word_idx);
#else
    const word_t word = word_array[word_idx];
#endif // defined(__CUDA_ARCH__)
    return (word & search_mask) == search_mask;
  }


  __forceinline__ __unroll_loops__
  void
  insert(const key_t& key, word_t* word_array) noexcept {
    const hash_value_t first_hash_val = HashFn<key_t>::hash(key);
    const hash_value_t second_hash_val = HashFn2<key_t>::hash(key);
    u32 word_idx = which_word(first_hash_val) * block_cnt;
    for (size_t i = 0; i < block_cnt; i++) {
      word_t word = word_array[word_idx + i];
      word |= which_bits(first_hash_val, second_hash_val);
      word_array[word_idx] = word;
    }
  }


  __forceinline__
  size_t
  length() const noexcept {
    return length_mask + 1;
  }


  __forceinline__ __host__ __device__
  size_t
  word_cnt() const noexcept {
    return (length_mask + 1) / sizeof(word_t);
  }


};

} // namespace dtl
