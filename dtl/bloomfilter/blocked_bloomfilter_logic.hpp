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
#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_bloomfilter_block_logic.hpp>

#include <boost/integer/static_min_max.hpp>

#include "immintrin.h"


namespace dtl {
namespace internal { // TODO should be in bloom filter namespace

//===----------------------------------------------------------------------===//
// Batch-wise Contains (SIMD)
//===----------------------------------------------------------------------===//
template<
    typename filter_t,
    u64 vector_len
>
struct dispatch {

  // Typedefs
  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;
  using vec_t = vec<key_t, vector_len>;
  using mask_t = typename vec<key_t, vector_len>::mask;


  __attribute__ ((__noinline__))
  static $u64
  batch_contains(const filter_t& filter,
                 const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) {

    const key_t* reader = keys;
    $u32* match_writer = match_positions;

    // Determine the number of keys that need to be probed sequentially, due to alignment
    u64 required_alignment_bytes = vec_t::byte_alignment;
    u1 is_aligned = (reinterpret_cast<uintptr_t>(reader) % alignof(key_t)) == 0; // TODO use dtl instead
//    u1 is_aligned = dtl::mem::is_aligned(reader)
    u64 t = is_aligned  // should always be true
            ? (required_alignment_bytes - (reinterpret_cast<uintptr_t>(reader) % required_alignment_bytes)) / sizeof(key_t) // FIXME first elements are processed sequentially even if aligned
            : key_cnt;
    u64 unaligned_key_cnt = std::min(static_cast<$u64>(key_cnt), t);
    // process the unaligned keys sequentially
    $u64 read_pos = 0;
    for (; read_pos < unaligned_key_cnt; read_pos++) {
      u1 is_match = filter.contains(filter_data, *reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    // Process the aligned keys vectorized
    u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {
      const auto mask = filter.template contains_vec<vector_len>(filter_data, *reinterpret_cast<const vec_t*>(reader));
      u64 match_cnt = mask.to_positions(match_writer, read_pos + match_offset);
      match_writer += match_cnt;
      reader += vector_len;
    }
    // process remaining keys sequentially
    for (; read_pos < key_cnt; read_pos++) {
      u1 is_match = filter.contains(filter_data, *reader);
      *match_writer = static_cast<$u32>(read_pos) + match_offset;
      match_writer += is_match;
      reader++;
    }
    return match_writer - match_positions;
  }

};


//===----------------------------------------------------------------------===//
// Batch-wise Contains (no SIMD)
//===----------------------------------------------------------------------===//
template<
    typename filter_t
>
struct dispatch<filter_t, 0> {

  // Typedefs
  using key_t = typename filter_t::key_t;
  using word_t = typename filter_t::word_t;

//  __forceinline__
  static $u64
  batch_contains(const filter_t& filter,
                 const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) {
    $u32* match_writer = match_positions;
    $u32 i = 0;
    if (key_cnt >= 4) {
      for (; i < key_cnt - 4; i += 4) {
        u1 is_match_0 = filter.contains(filter_data, keys[i]);
        u1 is_match_1 = filter.contains(filter_data, keys[i + 1]);
        u1 is_match_2 = filter.contains(filter_data, keys[i + 2]);
        u1 is_match_3 = filter.contains(filter_data, keys[i + 3]);
        *match_writer = i + match_offset;
        match_writer += is_match_0;
        *match_writer = (i + 1) + match_offset;
        match_writer += is_match_1;
        *match_writer = (i + 2) + match_offset;
        match_writer += is_match_2;
        *match_writer = (i + 3) + match_offset;
        match_writer += is_match_3;
      }
    }
    for (; i < key_cnt; i++) {
      u1 is_match = filter.contains(filter_data, keys[i]);
      *match_writer = i + match_offset;
      match_writer += is_match;
    }
    return match_writer - match_positions;
  }

};

} // namespace internal


//===----------------------------------------------------------------------===//
// A high-performance blocked Bloom filter template.
//===----------------------------------------------------------------------===//
template<
    typename Tk,                  // the key type
    template<typename Ty, u32 i> class Hasher,      // the hash function family to use
    typename Tw,                  // the word type to use for the bitset
    u32 Wc = 2,                   // the number of words per block
    u32 s = Wc,                   // the number of sectors
    u32 K = 8,                    // the number of hash functions to use
    dtl::block_addressing block_addressing = dtl::block_addressing::POWER_OF_TWO,
    u1 early_out = false
>
struct blocked_bloomfilter_logic {

  //===----------------------------------------------------------------------===//
  // The static part.
  //===----------------------------------------------------------------------===//
  using key_t = typename std::remove_cv<Tk>::type;
  using word_t = typename std::remove_cv<Tw>::type;
  static_assert(std::is_integral<key_t>::value, "The key type must be an integral type.");
  static_assert(std::is_integral<word_t>::value, "The word type must be an integral type.");

  using size_t = $u64;

  static constexpr u32 word_cnt_per_block = Wc;
  static constexpr u32 word_cnt_per_block_log2 = dtl::ct::log_2<word_cnt_per_block>::value;
  static_assert(dtl::is_power_of_two(Wc), "Parameter 'Wc' must be a power of two.");

  static constexpr u32 sector_cnt = s;

  static constexpr u32 word_bitlength = sizeof(word_t) * 8;
  static constexpr u32 word_bitlength_log2 = dtl::ct::log_2_u32<word_bitlength>::value;
  static constexpr u32 word_bitlength_mask = word_bitlength - 1;

  static constexpr u32 block_bitlength = sizeof(word_t) * 8 * word_cnt_per_block;

  // Inspect the given hash function.
  using hash_value_t = decltype(Hasher<key_t, 0>::hash(42)); // TODO find out why NVCC complains
  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");
  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;


  // The number of hash functions to use.
  static constexpr u32 k = K;

  // The first hash function to use inside the block. Note: 0 is used for block addressing
  static constexpr u32 block_hash_fn_idx = 1;

  // The block type. Determined based on word and sector counts.
  using block_t = typename blocked_bloomfilter_block_logic<key_t, word_t, word_cnt_per_block, sector_cnt, k,
                                                           Hasher, hash_value_t, early_out, block_hash_fn_idx>::type;

  // The block addressing logic (either MAGIC or POWER_OF_TWO).
  using addr_t = block_addressing_logic<block_addressing>;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// Addressing logic instance.
  const addr_t addr;
  //===----------------------------------------------------------------------===//


  /// C'tor.
  /// Note, that the actual length might be (slightly) different to the
  /// desired length. The function get_length() returns the actual length.
  explicit
  blocked_bloomfilter_logic(const size_t desired_length)
      : addr((desired_length + (block_bitlength - 1)) / block_bitlength) { }

  /// Copy c'tor
  blocked_bloomfilter_logic(const blocked_bloomfilter_logic&) = default;

  ~blocked_bloomfilter_logic() = default;


  //===----------------------------------------------------------------------===//
  // Insert
  //===----------------------------------------------------------------------===//
//  __forceinline__ __host__
  void
  insert(word_t* __restrict filter_data,
         const key_t key) noexcept {
    const hash_value_t block_addressing_hash_val = Hasher<const key_t, 0>::hash(key);
    const hash_value_t block_idx = addr.get_block_idx(block_addressing_hash_val);
    const hash_value_t bitvector_word_idx = block_idx << word_cnt_per_block_log2;

    auto block_ptr = &filter_data[bitvector_word_idx];

    block_t::insert(block_ptr, key);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Batch Insert
  //===----------------------------------------------------------------------===//
  void
  batch_insert(word_t* __restrict filter_data,
               const key_t* keys, u32 key_cnt) noexcept {
    for (std::size_t i = 0; i < key_cnt; i++) {
      insert(filter_data, keys[i]);
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains
  //===----------------------------------------------------------------------===//
  u1
  contains(const word_t* __restrict filter_data,
           const key_t key) const noexcept {
    const hash_value_t block_addressing_hash_val = Hasher<const key_t, 0>::hash(key);
    const hash_value_t block_idx = addr.get_block_idx(block_addressing_hash_val);
    const hash_value_t bitvector_word_idx = block_idx << word_cnt_per_block_log2;

    const auto block_ptr = &filter_data[bitvector_word_idx];

    u1 found = block_t::contains(block_ptr, key);
    return found;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Contains (SIMD)
  //===----------------------------------------------------------------------===//
  template<u64 n> // the vector length
  __forceinline__ __host__
  typename vec<word_t, n>::mask
  contains_vec(const word_t* __restrict filter_data,
               const vec<key_t, n>& keys) const noexcept {
    // Typedef the vector types.
    using key_vt = vec<key_t, n>;
    using hash_value_vt = vec<hash_value_t, n>;

    const hash_value_vt block_addressing_hash_vals = Hasher<key_vt, 0>::hash(keys);
    const hash_value_vt block_idxs = addr.get_block_idxs(block_addressing_hash_vals);
    const hash_value_vt bitvector_word_idx = block_idxs << word_cnt_per_block_log2;

    auto found = block_t::contains(keys, filter_data, bitvector_word_idx);
    return found;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Batch-wise Contains
  //===----------------------------------------------------------------------===//
  template<u64 vector_len>
  $u64
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const {
    return internal::dispatch<blocked_bloomfilter_logic, vector_len>
             ::batch_contains(*this, filter_data,
                              keys, key_cnt,
                              match_positions, match_offset);
  }


  //===----------------------------------------------------------------------===//
  /// Returns (actual) length in bits.
  size_t
  get_length() const noexcept {
    return addr.get_block_cnt() * block_bitlength;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns (actual) length in number of words.
  size_t
  word_cnt() const noexcept {
    return addr.get_block_cnt() * word_cnt_per_block;
  }
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
