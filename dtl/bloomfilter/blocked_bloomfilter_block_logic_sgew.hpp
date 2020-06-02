#pragma once

#include <atomic>
#include <bitset>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/math.hpp>
#include <dtl/simd.hpp>
#include <dtl/bloomfilter/vector_helper.hpp>

#include "immintrin.h"

#include <boost/integer/static_min_max.hpp>


namespace dtl {

namespace {

//===----------------------------------------------------------------------===//
// Recursive template to compute a search mask with k bits set.
// Used in case of sector count >= word_cnt, which allows to set/test multiple
// bits in a word in one go.
//===----------------------------------------------------------------------===//
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 s,                        // the numbers of sectors (must be a power of two)
    u32 k,                        // the number of bits to set
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)
    u32 remaining_k_cnt           // current k (used for recursion)

>
struct word_block {

  //===----------------------------------------------------------------------===//
  // Static part
  //===----------------------------------------------------------------------===//
  static_assert(dtl::is_power_of_two(s), "Parameter 's' must be a power of two.");

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static constexpr u32 sector_bitlength = word_bitlength / s;
  static constexpr u32 sector_bitlength_log2 = dtl::ct::log_2_u32<sector_bitlength>::value;
  static constexpr word_t sector_mask() { return static_cast<word_t>(sector_bitlength) - 1; }

  static_assert(sector_bitlength >= 8, "A sector must be at least one byte in size.");

  static constexpr u32 hash_bit_cnt_per_k = sector_bitlength_log2;
  static constexpr u32 k_cnt_per_hash_value = ((sizeof(hash_value_t) * 8) / hash_bit_cnt_per_k) ; // consider -1 to respect hash fn weakness in the low order bits
  static constexpr u32 k_cnt_per_sector = k / s;

  static constexpr u32 current_k = k - remaining_k_cnt;

  static constexpr u1 rehash = remaining_hash_bit_cnt < hash_bit_cnt_per_k;
  static constexpr u32 remaining_hash_bit_cnt_after_rehash = rehash ? hash_value_bitlength : remaining_hash_bit_cnt;
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const key_t key, hash_value_t& hash_val, word_t& word) noexcept {

    hash_val = rehash ? hasher<key_t, hash_fn_idx>::hash(key) : hash_val;

    // Set one bit in the given word; rehash if necessary
    constexpr u32 sector_idx = current_k / k_cnt_per_sector;
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k;
    $u32 bit_idx = ((hash_val >> shift) & sector_mask()) + (sector_idx * sector_bitlength);
    word |= word_t(1) << bit_idx;

    // Recurse
    word_block<key_t, word_t, s, k,
        hasher, hash_value_t,
        (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
        remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0, // the number of remaining hash bits
        remaining_k_cnt - 1> // decrement the remaining k counter
      ::which_bits(key, hash_val, word);
  }
  //===----------------------------------------------------------------------===//


  template<u64 n>
  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const vec<key_t, n>& keys,
             vec<hash_value_t, n>& hash_vals,
             vec<word_t, n>& words) noexcept {

    // Typedef vector types
    using key_vt = vec<key_t, n>;
    using hash_value_vt = vec<hash_value_t, n>;
    using word_vt = vec<word_t, n>;

    hash_vals = rehash ? hasher<key_vt, hash_fn_idx>::hash(keys) : hash_vals;
    const hash_value_t remaining_hash_bit_cnt_after_rehash = rehash ? hash_value_bitlength : remaining_hash_bit_cnt;

    // Set one bit in the given word; rehash if necessary
    constexpr u32 sector_idx = current_k / k_cnt_per_sector;
    constexpr u32 shift = remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k;
    hash_value_vt bit_idxs = ((hash_vals >> shift) & sector_mask()) + (sector_idx * sector_bitlength);
    words |= word_vt(1) << internal::vector_convert<hash_value_t, word_t, n>::convert(bit_idxs);

    // Recurse
    word_block<key_t, word_t, s, k,
        hasher, hash_value_t,
        (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
        remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0, // the number of remaining hash bits
        remaining_k_cnt - 1> // decrement the remaining k counter
      ::which_bits(keys, hash_vals, words);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The number of required hash functions.
  //===----------------------------------------------------------------------===//
  static constexpr u32 hash_fn_idx_end =
  word_block<key_t, word_t, s, k,
      hasher, hash_value_t,
      (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
      remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0, // the number of remaining hash bits
      remaining_k_cnt - 1> // decrement the remaining k counter
    ::hash_fn_idx_end;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The number of required hash bits.
  //===----------------------------------------------------------------------===//
  static constexpr u32 remaining_hash_bits =
  word_block<key_t, word_t, s, k,
      hasher, hash_value_t,
      (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
      remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0, // the number of remaining hash bits
      remaining_k_cnt - 1> // decrement the remaining k counter
    ::remaining_hash_bits;
  //===----------------------------------------------------------------------===//


};


template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 s,                        // the numbers of sectors (must be a power of two)
    u32 k,                        // the number of bits to set
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt    // the number of remaining hash bits (used for recursion)
>
struct word_block<key_t, word_t, s, k, hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt, 0 /* no remaining k's */> {

  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const key_t key, hash_value_t& hash_value, word_t& word) noexcept {
    // End of recursion
  }
  //===----------------------------------------------------------------------===//


  template<u64 n>
  __forceinline__ __unroll_loops__ __host__ __device__
  static void
  which_bits(const vec<key_t, n>& keys,
             vec<hash_value_t, n>& hash_values,
             vec<word_t, n>& words) noexcept {
    // End of recursion
  }
  //===----------------------------------------------------------------------===//

  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static constexpr u32 hash_fn_idx_end = hash_fn_idx;

  static constexpr u32 remaining_hash_bits = remaining_hash_bit_cnt;

};
//===----------------------------------------------------------------------===//

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Recursive template to work with multi-word blocks.
//
// Used in case of sector count >= word count, which allows to sequentially
// iterate over the words of a block (sequential block access pattern).
//===----------------------------------------------------------------------===//
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per block
    u32 s,                        // the numbers of sectors (must be a power of two and greater or equal to word_cnt))
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)
    u32 remaining_word_cnt,       // the remaining number of words to process in the block (used for recursion)

    u1 early_out = false          // allows for branching out during lookups (before the next word is loaded)
>
struct multiword_block {

  //===----------------------------------------------------------------------===//
  // Static part
  //===----------------------------------------------------------------------===//
  static constexpr u32 word_cnt_log2 = dtl::ct::log_2<word_cnt>::value;
  static_assert(dtl::is_power_of_two(word_cnt), "Parameter 'word_cnt' must be a power of two.");
  static_assert(k >= word_cnt, "Parameter 'k' must be greater or equal to 'word_cnt'.");
  static_assert(k % word_cnt == 0, "Parameter 'k' must be dividable by 'word_cnt'.");

  static constexpr u32 sector_cnt = s;
  static_assert(dtl::is_power_of_two(sector_cnt), "Parameter 'sector_cnt' must be a power of two.");

  static constexpr u32 sector_cnt_per_word = s / word_cnt;
  static_assert(sector_cnt_per_word > 0, "The number of sectors must be at least 'word_cnt'.");

  static constexpr u32 k_cnt_per_word = k / word_cnt;

  static constexpr u32 current_word_idx() { return word_cnt - remaining_word_cnt; }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__
  static void
  insert(word_t* __restrict block_ptr, const key_t key) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bits, remaining_word_cnt, early_out>
//        ::insert(block_ptr, key, hash_val);
        ::insert_atomic(block_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__
  static void
  insert(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {

    // Load the word of interest
    word_t word = block_ptr[current_word_idx()];

    word_t bit_mask = 0;

    using word_block_t =
      word_block<key_t, word_t, sector_cnt_per_word, k_cnt_per_word,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
        k_cnt_per_word>;
    word_block_t::which_bits(key, hash_val, bit_mask);

    // Update the bit vector
    word |= bit_mask;
    block_ptr[current_word_idx()] = word;

    // Process remaining words recursively, if any
    multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, word_block_t::hash_fn_idx_end, word_block_t::remaining_hash_bits, remaining_word_cnt - 1, early_out>
        ::insert(block_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//

  __forceinline__ __unroll_loops__
  static void
  insert_atomic(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {

    word_t bit_mask = 0;

    using word_block_t =
      word_block<key_t, word_t, sector_cnt_per_word, k_cnt_per_word,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
        k_cnt_per_word>;
    word_block_t::which_bits(key, hash_val, bit_mask);

    word_t* word_ptr = &block_ptr[current_word_idx()];
    std::atomic<word_t>* atomic_word_ptr = reinterpret_cast<std::atomic<word_t>*>(word_ptr);
    $u1 success = false;
    do {
      // Load the word of interest
      word_t word = atomic_word_ptr->load();
      // Update the bit vector (atomically)
      word_t updated_word = word | bit_mask;
      success = atomic_word_ptr->compare_exchange_weak(word, updated_word);
    } while (!success);

    // Process remaining words recursively, if any
    multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, word_block_t::hash_fn_idx_end, word_block_t::remaining_hash_bits, remaining_word_cnt - 1, early_out>
        ::insert_atomic(block_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const word_t* __restrict block_ptr, const key_t key) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bits, remaining_word_cnt, early_out>
        ::contains(block_ptr, key, hash_val, true);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val, u1 is_contained_in_block) noexcept {

    // Load the word of interest
    word_t word = block_ptr[current_word_idx()];

    word_t bit_mask = 0;

    // Compute the search mask
    using word_block_t =
      word_block<key_t, word_t, sector_cnt_per_word, k_cnt_per_word,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
        k_cnt_per_word>;
    word_block_t::which_bits(key, hash_val, bit_mask);

    // Bit testing
    u1 found_in_word = (word & bit_mask) == bit_mask;

    // Early out
    if (early_out) {
      if (likely(!found_in_word)) return false;
    }


    // Process remaining words recursively, if any
    return multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, word_block_t::hash_fn_idx_end, word_block_t::remaining_hash_bits, remaining_word_cnt - 1, early_out>
        ::contains(block_ptr, key, hash_val, found_in_word & is_contained_in_block);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static typename vec<word_t,n>::mask
  contains(const vec<key_t,n>& keys,
           const word_t* __restrict bitvector_base_address,
           const vec<key_t,n>& block_start_word_idxs) noexcept {

    vec<hash_value_t, n> hash_vals(0);
    auto is_contained_in_block_mask = vec<word_t,n>::mask::make_all_mask();

    // Call recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
                           hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                           remaining_word_cnt,
                           early_out>
        ::contains(keys, hash_vals, bitvector_base_address, block_start_word_idxs, is_contained_in_block_mask);
  }
  //===----------------------------------------------------------------------===//


  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           vec<hash_value_t,n>& hash_vals,
           const word_t* __restrict bitvector_base_address,
           const vec<key_t,n>& block_start_word_idxs,
           const typename vec<word_t,n>::mask& is_contained_in_block_mask) noexcept {

    // Typedef the vector types
    using word_vt = vec<word_t, n>;

    // Load the words of interest
    auto word_idxs = block_start_word_idxs + current_word_idx();
    const word_vt words = internal::vector_gather<word_t, hash_value_t, n>::gather(bitvector_base_address, word_idxs);

    // Compute the search mask
    word_vt bit_masks = 0;
    using word_block_t =
      word_block<key_t, word_t, sector_cnt_per_word, k_cnt_per_word,
        hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
        k_cnt_per_word>;
    word_block_t::which_bits(keys, hash_vals, bit_masks);

    // Update the bit vector
    auto found_in_word = (words & bit_masks) == bit_masks;

    // Early out
    if (early_out) {
      if (likely(found_in_word.none())) return found_in_word;
    }

    // Process remaining words recursively, if any
    return multiword_block<key_t, word_t, word_cnt, sector_cnt, k,
        hasher, hash_value_t, word_block_t::hash_fn_idx_end, word_block_t::remaining_hash_bits, remaining_word_cnt - 1,
        early_out>
        ::contains(keys, hash_vals, bitvector_base_address, block_start_word_idxs, found_in_word & is_contained_in_block_mask);
  }
  //===----------------------------------------------------------------------===//


};


template<
    typename key_t,               // the word type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per block
    u32 s,                        // the numbers of sectors (must be a power of two and greater or equal to word_cnt))
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)

    u1 early_out                  // allows for branching out during lookups (before the next sector is tested)
>
struct multiword_block<key_t, word_t, word_cnt, s, k,
    hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
    0 /* no more words remaining */, early_out> {

  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__
  static void
  insert(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {
    // End of recursion
  }
  __forceinline__ __unroll_loops__
  static void
  insert_atomic(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {
    // End of recursion
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val, u1 is_contained) noexcept {
    // End of recursion
    return is_contained;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           vec<hash_value_t,n>& hash_vals,
           const word_t* __restrict bitvector_base_address,
           const vec<key_t,n>& block_idxs,
           const typename vec<word_t,n>::mask& is_contained_mask) noexcept {
    // End of recursion
    return is_contained_mask;
  }
  //===----------------------------------------------------------------------===//


};
//===----------------------------------------------------------------------===//


} // namespace dtl
