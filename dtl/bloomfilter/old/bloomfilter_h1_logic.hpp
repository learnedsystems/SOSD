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
    template<typename Ty> class HashFn,     // the hash function to use
    typename Tw/* = u64*/,     // the word type to use for the bit array. Note: one word = one block.
    u32 K = 2,             // the number of bits set per inserted element
    u1 Sectorized = false  //
>
struct bloomfilter_h1_logic {

  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  using size_t = $u64;
//  using size_t = $u32;

  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");


  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;


  // Inspect the given hash function
  using hash_value_t = $u32; //decltype(HashFn<key_t>::hash(0)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;
  static constexpr u32 hash_fn_cnt = 1;


  // The number of hash functions to use.
  static constexpr u32 k = K;
  static_assert(k > 0, "Parameter 'k' must be in [1, 6].");
  static_assert(k < 7, "Parameter 'k' must be in [1, 6].");

  // Split each word into multiple sectors (sub words, with a length of a power of two).
  // Note that sectorization is a specialization. Having only one sector = no sectorization.
  static constexpr u1 sectorized = Sectorized;

  static constexpr u32 compute_sector_cnt() {
    static_assert(Sectorized ? (word_bitlength / dtl::next_power_of_two(k)) != 0 : true,
                  "The number of sectors must be greater than zero. Probably the given 'k' is set to high.");
    return Sectorized ? static_cast<u32>(word_bitlength / (word_bitlength / dtl::next_power_of_two(k)))
                      : 1;
  }

  static constexpr u32 sector_cnt = compute_sector_cnt();
  static constexpr u32 sector_bitlength = word_bitlength / sector_cnt;
  // the number of bits needed to address the individual bits within a sector
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
  static constexpr word_t sector_mask = sector_bitlength - 1;
  static constexpr u32 bit_cnt_per_k = sector_bitlength_log2;

  static constexpr i32 remaining_hash_bit_cnt = static_cast<i32>(hash_value_bitlength) - (sectorized ? k * sector_bitlength_log2 : k * word_bitlength_log2);
  static constexpr u64 min_m = 2 * word_bitlength; // Using only one word would cause undefined behaviour in bit shifts later on.
  static constexpr u64 max_m = (1ull << remaining_hash_bit_cnt) * word_bitlength;

  // ---- Members ----
  const size_t bitvector_length; // the length of the bitvector
  const hash_value_t length_mask; // the length mask (same type as the hash values)
  const hash_value_t word_cnt_log2; // The number of bits to address the individual words of the bitvector
  // ----


  static
  size_t
  determine_actual_length(const size_t length) {
    // round up to the next power of two
    const auto m = static_cast<size_t>(dtl::next_power_of_two(length));
    const auto min = static_cast<size_t>(min_m);
    return std::max(m, min);
  }


  /// C'tor
  explicit
  bloomfilter_h1_logic(const size_t length)
      : bitvector_length(determine_actual_length(length)),
        length_mask(static_cast<hash_value_t>(bitvector_length - 1)),
        word_cnt_log2(static_cast<hash_value_t>(dtl::log_2(bitvector_length / word_bitlength))) {
    if (bitvector_length > max_m) throw std::invalid_argument("Length must not exceed 'max_m'.");
  }

  /// Copy c'tor
  bloomfilter_h1_logic(const bloomfilter_h1_logic&) = default;


  __forceinline__ __host__ __device__
  hash_value_t
  hash(const key_t key) const noexcept {
    return HashFn<key_t>::hash(key);
  }


  __forceinline__ __host__ __device__
  hash_value_t
  which_word(const hash_value_t hash_val) const noexcept {
    const auto word_idx = hash_val >> (hash_value_bitlength - word_cnt_log2);
    return word_idx;
  }


  __forceinline__ __unroll_loops__ __host__ __device__
  word_t
  which_bits(const hash_value_t hash_val) const noexcept {
    word_t word = 0;
    for ($u32 i = 0; i < k; i++) {
      const u32 bit_idx = (hash_val >> (((hash_value_bitlength - word_cnt_log2) - ((i + 1) * sector_bitlength_log2)))) & sector_mask;
      const u32 sector_offset = (i * sector_bitlength) & word_bitlength_mask;
      word |= word_t(1) << (bit_idx + sector_offset);
    }
    return word;
  }


  __forceinline__ __host__ __device__
  u1
  contains(const key_t& key, const word_t* word_array) const noexcept {
    const hash_value_t hash_val = hash(key);
    const hash_value_t word_idx = which_word(hash_val);
    const word_t search_mask = which_bits(hash_val);
#if defined(__CUDA_ARCH__)
    const word_t word = cub::ThreadLoad<cub::LOAD_CS>(word_array + word_idx);
#else
    const word_t word = word_array[word_idx];
#endif // defined(__CUDA_ARCH__)
    return (word & search_mask) == search_mask;
  }


  __forceinline__
  void
  insert(const key_t& key, word_t* word_array) noexcept {
    const hash_value_t hash_val = HashFn<key_t>::hash(key);
    const hash_value_t word_idx = which_word(hash_val);
    word_t word = word_array[word_idx];
    word |= which_bits(hash_val);
    word_array[word_idx] = word;
  }


  __forceinline__ __host__ __device__
  size_t
  length() const noexcept {
    return bitvector_length;
  }


  __forceinline__ __host__ __device__
  size_t
  word_cnt() const noexcept {
    return length() / sizeof(word_t);
  }


};

} // namespace dtl
