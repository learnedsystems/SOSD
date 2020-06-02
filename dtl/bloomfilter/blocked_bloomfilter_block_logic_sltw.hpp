#pragma once

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
// Recursive template to work with multi-word sectors.
//
// Specialization for the case where sector count < word_cnt ('sltw'),
// which results in a random access pattern within the block (and sector).
// I.e., for every bit to set, the corresponding word needs to be determined
// and loaded.
//
// Same as in the 'sgew' case, we process the block sector by sector,
// however, a sector consists of more than one word.
//===----------------------------------------------------------------------===//
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per sector
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)
    u32 remaining_k_cnt           // the remaining number of bits to set in the sector (used for recursion)
>
struct multiword_sector {

  //===----------------------------------------------------------------------===//
  // Static part
  //===----------------------------------------------------------------------===//

  static constexpr u32 word_cnt_log2 = dtl::ct::log_2<word_cnt>::value;
  static constexpr u32 word_cnt_log2_mask = (1u << word_cnt_log2) - 1;
  static_assert(dtl::is_power_of_two(word_cnt), "Parameter 'word_cnt' must be a power of two.");
  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2<word_bitlength>::value;
  static constexpr u32 word_bitlength_log2_mask = (1u << word_bitlength_log2) - 1;


  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;

  static constexpr u32 k_cnt_per_sector = k;

  static constexpr u32 current_k_idx() { return k - remaining_k_cnt; }

  static constexpr u32 hash_bit_cnt_per_k = word_cnt_log2 + word_bitlength_log2;

  static constexpr u1 rehash = remaining_hash_bit_cnt < hash_bit_cnt_per_k;
  static constexpr u32 remaining_hash_bit_cnt_after_rehash = rehash ? hash_value_bitlength : remaining_hash_bit_cnt;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict sector_ptr, const key_t key) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    multiword_sector<key_t, word_t, word_cnt, k,
                     hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                     remaining_k_cnt>
                     ::insert_atomic(sector_ptr, key, hash_val);
//                     ::insert(sector_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert (Recursive)
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict sector_ptr, const key_t key, hash_value_t& hash_val) noexcept {

    hash_val = rehash ? hasher<key_t, hash_fn_idx>::hash(key) : hash_val;

    // Determine the word of interest
    constexpr u32 word_idx_shift = remaining_hash_bit_cnt_after_rehash - word_cnt_log2;
    $u32 word_idx = ((hash_val >> word_idx_shift) & word_cnt_log2_mask);

    // Load the word of interest
    word_t word = sector_ptr[word_idx];

    // Set a bit in the given word
    constexpr u32 bit_idx_shift = remaining_hash_bit_cnt_after_rehash - word_cnt_log2 - word_bitlength_log2;
    $u32 bit_idx = ((hash_val >> bit_idx_shift) & word_bitlength_log2_mask);
    word |= word_t(1) << bit_idx;

    // Update the bit vector
    sector_ptr[word_idx] = word;

    // Process remaining k's recursively, if any
    multiword_sector<key_t, word_t, word_cnt, k,
                     hasher, hash_value_t,
                     rehash ? hash_fn_idx + 1 : hash_fn_idx,
                     remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0,
                     remaining_k_cnt - 1>
      ::insert(sector_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert Atomic (Recursive)
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert_atomic(word_t* __restrict sector_ptr, const key_t key, hash_value_t& hash_val) noexcept {

    hash_val = rehash ? hasher<key_t, hash_fn_idx>::hash(key) : hash_val;

    // Determine the word of interest
    constexpr u32 word_idx_shift = remaining_hash_bit_cnt_after_rehash - word_cnt_log2;
    u32 word_idx = ((hash_val >> word_idx_shift) & word_cnt_log2_mask);

    // Set a bit in the given word
    constexpr u32 bit_idx_shift = remaining_hash_bit_cnt_after_rehash - word_cnt_log2 - word_bitlength_log2;
    u32 bit_idx = ((hash_val >> bit_idx_shift) & word_bitlength_log2_mask);
    const word_t which_bit = word_t(1) << bit_idx;

    word_t* word_ptr = &sector_ptr[word_idx];
    std::atomic<word_t>* atomic_word_ptr = reinterpret_cast<std::atomic<word_t>*>(word_ptr);
    $u1 success = false;
    do {
      // Load the word of interest
      word_t word = atomic_word_ptr->load();
      // Update the bit vector
      word_t updated_word = word | which_bit;
      success = atomic_word_ptr->compare_exchange_weak(word, updated_word);
    } while (!success);

    // Process remaining k's recursively, if any
    multiword_sector<key_t, word_t, word_cnt, k,
                     hasher, hash_value_t,
                     rehash ? hash_fn_idx + 1 : hash_fn_idx,
                     remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0,
                     remaining_k_cnt - 1>
      ::insert_atomic(sector_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const word_t* __restrict sector_ptr, const key_t key) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multiword_sector<key_t, word_t, word_cnt, k,
                            hasher, hash_value_t,
                            hash_fn_idx,
                            remaining_hash_bits,
                            remaining_k_cnt>
      ::contains(sector_ptr, key, hash_val, true);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (Recursive)
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const word_t* __restrict sector_ptr, const key_t key, hash_value_t& hash_val, u1 is_contained_in_sector) noexcept {

    hash_val = rehash ? hasher<key_t, hash_fn_idx>::hash(key) : hash_val;

    // Determine the word of interest
    constexpr u32 word_idx_shift = remaining_hash_bit_cnt_after_rehash - word_cnt_log2;
    $u32 word_idx = ((hash_val >> word_idx_shift) & word_cnt_log2_mask);

    // Load the word of interest
    word_t word = sector_ptr[word_idx];

    // Test a bit in the given word
    constexpr u32 bit_idx_shift = remaining_hash_bit_cnt_after_rehash - word_cnt_log2 - word_bitlength_log2;
    $u32 bit_idx = ((hash_val >> bit_idx_shift) & word_bitlength_log2_mask);
    u1 found_in_word = word & (word_t(1) << bit_idx);

    // Process remaining k's recursively, if any
    return multiword_sector<key_t, word_t, word_cnt, k,
                            hasher, hash_value_t,
                            rehash ? hash_fn_idx + 1 : hash_fn_idx,
                            remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0,
                            remaining_k_cnt - 1>
                            ::contains(sector_ptr, key, hash_val, found_in_word & is_contained_in_sector);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           const word_t* __restrict bitvector_base_address,
           const vec<key_t,n>& sector_start_word_idxs) noexcept {

    vec<hash_value_t, n> hash_vals(0);
    auto is_contained_in_sector_mask = vec<word_t,n>::mask::make_all_mask();

    // Call recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multiword_sector<key_t, word_t, word_cnt, k,
                            hasher, hash_value_t,
                            hash_fn_idx,
                            remaining_hash_bits,
                            remaining_k_cnt>
      ::contains(keys, hash_vals, bitvector_base_address, sector_start_word_idxs, is_contained_in_sector_mask);
  }


  //===----------------------------------------------------------------------===//
  // Contains (SIMD, Recursive)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           vec<hash_value_t,n>& hash_vals,
           const word_t* __restrict bitvector_base_address,
           const vec<hash_value_t,n>& sector_start_word_idxs,
           const typename vec<word_t,n>::mask is_contained_in_sector_mask) noexcept {

    // Typedef the vector types
    using key_vt = vec<key_t, n>;
    using word_vt = vec<word_t, n>;

    hash_vals = rehash ? hasher<key_vt, hash_fn_idx>::hash(keys) : hash_vals;

    // Determine the word of interest
    constexpr u32 word_idx_shift = remaining_hash_bit_cnt_after_rehash - word_cnt_log2;
    const auto in_sector_word_idxs = (hash_vals >> word_idx_shift) & static_cast<hash_value_t>(word_cnt_log2_mask);
    const auto word_idxs = sector_start_word_idxs + in_sector_word_idxs;

    // Gather the words of interest
    const word_vt words = internal::vector_gather<word_t, hash_value_t, n>::gather(bitvector_base_address, word_idxs);

    // Test a bit in the given word
    constexpr u32 bit_idx_shift = remaining_hash_bit_cnt_after_rehash - word_cnt_log2 - word_bitlength_log2;
    const auto bit_idx = (hash_vals >> bit_idx_shift) & static_cast<hash_value_t>(word_bitlength_log2_mask);
    const word_vt bits_to_test = word_vt(1) << internal::vector_convert<hash_value_t, word_t, n>::convert(bit_idx);
    const auto found_in_word_mask = (words & bits_to_test) == bits_to_test;

    // Process remaining k's recursively, if any
    return multiword_sector<key_t, word_t, word_cnt, k,
                            hasher, hash_value_t,
                            rehash ? hash_fn_idx + 1 : hash_fn_idx,
                            remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k : 0,
                            remaining_k_cnt - 1>
      ::contains(keys, hash_vals, bitvector_base_address, sector_start_word_idxs, found_in_word_mask & is_contained_in_sector_mask);

  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The number of required hash functions.
  //===----------------------------------------------------------------------===//
  static constexpr u32 hash_fn_idx_end =
      multiword_sector<key_t, word_t, word_cnt, k,
                       hasher, hash_value_t,
                       (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
                       remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k
                         ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k
                         : 0, // the number of remaining hash bits
                       remaining_k_cnt - 1> // decrement the remaining k counter
      ::hash_fn_idx_end;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // The number of required hash bits.
  //===----------------------------------------------------------------------===//
  static constexpr u32 remaining_hash_bits =
      multiword_sector<key_t, word_t, word_cnt, k,
                 hasher, hash_value_t,
                 (rehash ? hash_fn_idx + 1 : hash_fn_idx), // increment the hash function index
                 remaining_hash_bit_cnt_after_rehash >= hash_bit_cnt_per_k
                   ? remaining_hash_bit_cnt_after_rehash - hash_bit_cnt_per_k
                   : 0, // the number of remaining hash bits
                 remaining_k_cnt - 1> // decrement the remaining k counter
      ::remaining_hash_bits;
  //===----------------------------------------------------------------------===//

};


template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per sector
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt    // the number of remaining hash bits (used for recursion)
>
struct multiword_sector<key_t, word_t, word_cnt, k, hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt, 0 /* no more k's */> {

  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict sector_ptr, const key_t key, hash_value_t& hash_val) noexcept {
    // End of recursion.
  }
  __forceinline__
  static void
  insert_atomic(word_t* __restrict sector_ptr, const key_t key, hash_value_t& hash_val) noexcept {
    // End of recursion.
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  static u1
  contains(const word_t* __restrict sector_ptr, const key_t key, hash_value_t& hash_val, u1 is_contained_in_sector) noexcept {
    // End of recursion.
    return is_contained_in_sector;
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
           const vec<hash_value_t,n>& sector_start_word_idxs,
           const typename vec<word_t,n>::mask is_contained_in_sector_mask) noexcept {
    // End of recursion.
    return is_contained_in_sector_mask;
  }
  //===----------------------------------------------------------------------===//


  static constexpr u32 hash_fn_idx_end = hash_fn_idx;

  static constexpr u32 remaining_hash_bits = remaining_hash_bit_cnt;

};


} // anonymous namespace

//===----------------------------------------------------------------------===//
// Recursive template to work with multiple multi-word sectors.
//
// Specialization for the case where sector count < word_cnt ('sltw'),
// which results in a random access pattern within the block (and sector).
// I.e., for every bit to set/test, the corresponding word needs to be determined
// and loaded.
//
// Same as in the 'sgew' case, we process the block sector by sector,
// however, a sector consists of more than one word.
//===----------------------------------------------------------------------===//
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per block
    u32 s,                        // the numbers of sectors (must be a power)
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)
    u32 remaining_sector_cnt,     // the remaining number of sector (used for recursion)

    u1 early_out = false          // allows for branching out during lookups (before the next sector is tested)
>
struct multisector_block {

  //===----------------------------------------------------------------------===//
  // Static part
  //===----------------------------------------------------------------------===//
  static constexpr u32 sector_cnt = s;
  static_assert(dtl::is_power_of_two(sector_cnt), "Parameter 'sector_cnt' must be a power of two.");
  static constexpr u32 sector_cnt_log2 = dtl::ct::log_2<sector_cnt>::value;

  static constexpr u32 current_sector_idx = s - remaining_sector_cnt;

  static_assert(dtl::is_power_of_two(word_cnt), "Parameter 'word_cnt' must be a power of two.");
  static constexpr u32 word_cnt_per_sector = word_cnt / s;
  static constexpr u32 k_cnt_per_sector = k / sector_cnt;
  static_assert(k % sector_cnt == 0, "Parameter 'k' must be dividable by 's'.");


  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict block_ptr, const key_t key) noexcept {

    hash_value_t hash_val = 0;

    // Call the recursive function
    static constexpr u32 remaining_hash_bits = 0;
    multisector_block<key_t, word_t, word_cnt, s, k,
                      hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                      remaining_sector_cnt, early_out>
      ::insert(block_ptr, key, hash_val);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Insert (Recursive)
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {

    // Sector pointer
    word_t* sector_ptr = block_ptr + (word_cnt_per_sector * current_sector_idx);

    // Process remaining k's recursively, if any
    using sector_t =
      multiword_sector<key_t, word_t, word_cnt_per_sector, k_cnt_per_sector,
                       hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
                       k_cnt_per_sector>;
    sector_t::insert_atomic(sector_ptr, key, hash_val);
//    sector_t::insert(sector_ptr, key, hash_val);

    multisector_block<key_t, word_t, word_cnt, sector_cnt, k,
                      hasher, hash_value_t, sector_t::hash_fn_idx_end, sector_t::remaining_hash_bits,
                      remaining_sector_cnt - 1, early_out>
      ::insert(block_ptr, key, hash_val);
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
    return multisector_block<key_t, word_t, word_cnt, sector_cnt, k,
                             hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                             remaining_sector_cnt, early_out>
      ::contains(block_ptr, key, hash_val, true);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (Recursive)
  //===----------------------------------------------------------------------===//
  __forceinline__ __unroll_loops__ __host__ __device__
  static u1
  contains(const word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val, u1 is_contained_in_block) noexcept {

    // Sector pointer
    const word_t* sector_ptr = block_ptr + (word_cnt_per_sector * current_sector_idx);

    // Process the current sector
    using sector_t =
      multiword_sector<key_t, word_t, word_cnt_per_sector, k_cnt_per_sector,
                       hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
                       k_cnt_per_sector>;
    auto found_in_sector = sector_t::contains(sector_ptr, key, hash_val, true);

    // Early out
    if (early_out) {
      if (likely(!found_in_sector)) return false;
    }

    // Process remaining sectors recursively, if any
    return multisector_block<key_t, word_t, word_cnt, s, k,
                      hasher, hash_value_t, sector_t::hash_fn_idx_end, sector_t::remaining_hash_bits,
                      remaining_sector_cnt - 1, early_out>
        ::contains(block_ptr, key, hash_val, found_in_sector & is_contained_in_block);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           const word_t* __restrict bitvector_base_address,
           const vec<hash_value_t,n>& block_start_word_idxs) noexcept {

    vec<hash_value_t, n> hash_vals(0);
    const auto is_contained_in_block_mask = vec<word_t,n>::mask::make_all_mask(); // true

    // Call recursive function
    static constexpr u32 remaining_hash_bits = 0;
    return multisector_block<key_t, word_t, word_cnt, s, k,
                             hasher, hash_value_t, hash_fn_idx, remaining_hash_bits,
                             remaining_sector_cnt, early_out>
      ::contains(keys, hash_vals, bitvector_base_address, block_start_word_idxs, is_contained_in_block_mask);

  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD, Recursive)
  //===----------------------------------------------------------------------===//
  template<u64 n>
  __forceinline__ __unroll_loops__
  static auto
  contains(const vec<key_t,n>& keys,
           vec<hash_value_t,n>& hash_vals,
           const word_t* __restrict bitvector_base_address,
           const vec<hash_value_t,n>& block_start_word_idxs,
           const typename vec<word_t,n>::mask is_contained_in_block_mask) noexcept {

    // Sector pointers
    auto sector_start_word_idxs = block_start_word_idxs + (word_cnt_per_sector * current_sector_idx);

    // Process the current sector
    using sector_t =
            multiword_sector<key_t, word_t, word_cnt_per_sector, k_cnt_per_sector,
                             hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
                             k_cnt_per_sector>;
    auto found_in_sector_mask = sector_t::contains(keys, hash_vals, bitvector_base_address, sector_start_word_idxs, vec<word_t,n>::mask::make_all_mask());

    // Early out
    if (early_out) {
      if (likely(found_in_sector_mask.none())) return found_in_sector_mask;
    }

    // Process remaining sectors recursively, if any
    return multisector_block<key_t, word_t, word_cnt, s, k,
                             hasher, hash_value_t, sector_t::hash_fn_idx_end, sector_t::remaining_hash_bits,
                             remaining_sector_cnt - 1, early_out>
      ::contains(keys, hash_vals, bitvector_base_address, block_start_word_idxs, found_in_sector_mask & is_contained_in_block_mask);
  }
  //===----------------------------------------------------------------------===//

};


//===----------------------------------------------------------------------===//
template<
    typename key_t,               // the key type
    typename word_t,              // the word type
    u32 word_cnt,                 // the number of words per block
    u32 s,                        // the numbers of sectors (must be a power)
    u32 k,                        // the number of bits to set/test
    template<typename Ty, u32 i> class hasher,      // the hash function family to use
    typename hash_value_t,        // the hash value type to use

    u32 hash_fn_idx,              // current hash function index (used for recursion)
    u32 remaining_hash_bit_cnt,   // the number of remaining hash bits (used for recursion)

    u1 early_out                  // allows for branching out during lookups (before the next sector is tested)
>
struct multisector_block<key_t, word_t, word_cnt, s, k, hasher, hash_value_t, hash_fn_idx, remaining_hash_bit_cnt,
                         0 /* no more remaining sectors */, early_out>  {

  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
  __forceinline__
  static void
  insert(word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val) noexcept {
    // End of recursion.
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  static u1
  contains(const word_t* __restrict block_ptr, const key_t key, hash_value_t& hash_val, u1 is_contained_in_block) noexcept {
    // End of recursion.
    return is_contained_in_block;
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
           const vec<key_t,n>& block_start_word_idxs,
           const typename vec<word_t,n>::mask is_contained_in_block_mask) noexcept {
    // End of recursion.
    return is_contained_in_block_mask;
  }
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
