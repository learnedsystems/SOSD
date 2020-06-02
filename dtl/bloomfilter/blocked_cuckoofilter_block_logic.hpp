#pragma once

#include <cstdlib>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>

#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_multiword_table.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_util.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_word_table.hpp>


namespace dtl {
namespace cuckoofilter {

//===----------------------------------------------------------------------===//
// A statically sized cuckoo filter (used for blocking).
//===----------------------------------------------------------------------===//
template<
    typename __key_t = uint32_t,
    typename __table_t = blocked_cuckoofilter_multiword_table<uint64_t, 64, 16, 4>
>
struct blocked_cuckoofilter_block_logic {

  using key_t = __key_t;
  using table_t = __table_t;
  using word_t = typename table_t::word_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32<hash_value_t>;

  static constexpr uint32_t block_size = table_t::table_size_bytes;
  static constexpr uint32_t block_bitlength = table_t::table_size_bytes * 8;
  static constexpr uint32_t capacity = table_t::capacity;
  static constexpr uint32_t required_hash_bits = table_t::required_hash_bits;

//private:

  //===----------------------------------------------------------------------===//
  /// Returns the bucket index the given hash value maps to.
  __forceinline__
  static uint32_t
  get_bucket_idx(const hash_value_t hash_value) {
    return table_t::bucket_addressing_bits > 0 ? (hash_value >> (32 - table_t::bucket_addressing_bits))
                                               : 0;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the bucket index the given hash value maps to (SIMD).
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  static dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_bucket_idxs(const Tv& hash_values) {
    return table_t::bucket_addressing_bits > 0 ? (hash_values >> (32 - table_t::bucket_addressing_bits))
                                               : Tv::make(0);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the alternative bucket index base on the (first) bucket index and
  /// the given element tag.
  __forceinline__
  static uint32_t
  get_alternative_bucket_idx(const uint32_t bucket_idx, const uint32_t tag) {
    return table_t::bucket_addressing_bits > 0 ? (bucket_idx ^ ((tag * 0x5bd1e995u) >> (32 - table_t::bucket_addressing_bits)))
                                               : bucket_idx;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the alternative bucket index base on the (first) bucket index and
  /// the given element tag.
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  static dtl::vec<hash_value_t, dtl::vector_length<Tv>::value>
  get_alternative_bucket_idxs(const Tv& bucket_idxs, const Tv& tags) {
    return table_t::bucket_addressing_bits > 0 ? (bucket_idxs ^ ((tags * 0x5bd1e995u) >> (32 - table_t::bucket_addressing_bits)))
                                               : bucket_idxs;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Inserts the given tag into the specified bucket.
  /// Insertion may lead to (multiple) relocations inside the cuckoo table.
  /// If now free slot can be found, a bucket is marked as overflown.
  __forceinline__
  static void
  insert(word_t* __restrict block_ptr, const uint32_t bucket_idx, const uint32_t tag) {
    uint32_t current_idx = bucket_idx;
    uint32_t current_tag = tag;
    uint32_t old_tag;

    // Try to insert without kicking other tags out.
    old_tag = table_t::insert_tag(block_ptr, current_idx, current_tag);
    if (old_tag == table_t::null_tag) { return; } // successfully inserted
    if (old_tag == table_t::overflow_tag) { return; } // hit an overflowed bucket (always return true)

    // Re-try at the alternative bucket.
    current_idx = get_alternative_bucket_idx(current_idx, current_tag);

    for (uint32_t count = 0; count < 10; count++) {
      old_tag = table_t::insert_tag_relocate(block_ptr, current_idx, current_tag);
      if (old_tag == table_t::null_tag) { return; } // successfully inserted
      if (old_tag == table_t::overflow_tag) { return; } // hit an overflowed bucket (always return true)
      current_tag = old_tag;
      current_idx = get_alternative_bucket_idx(current_idx, current_tag);
    }
    // Failed to find a place for the current tag through partial-key cuckoo hashing.
    // Introduce an overflow bucket.
    table_t::mark_overflow(block_ptr, current_idx);
  }
  //===----------------------------------------------------------------------===//


public:

  //===----------------------------------------------------------------------===//
  /// Insert an element based on the given hash value.
  __forceinline__
  static void
  insert_hash(word_t* __restrict block_ptr, const hash_value_t& hash_value) {
    auto bucket_idx = get_bucket_idx(hash_value);
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    insert(block_ptr, bucket_idx, tag);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Insert an element.
  __forceinline__
  static void
  insert_key(word_t* __restrict block_ptr, const key_t& key) {
    auto hash_value = hasher::hash(key);
    insert_hash(block_ptr, hash_value);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Probe the table for a hash value.
  __forceinline__
  static bool
  contains_hash(const word_t* __restrict block_ptr, const hash_value_t& hash_value) {
    auto bucket_idx = get_bucket_idx(hash_value);
    auto tag = (hash_value >> (32 - table_t::bucket_addressing_bits - table_t::tag_size_bits)) & table_t::tag_mask;
    tag += (tag == 0); // tag must not be zero
    const auto alt_bucket_idx = get_alternative_bucket_idx(bucket_idx, tag);
    return table_t::find_tag_in_buckets(block_ptr, bucket_idx, alt_bucket_idx, tag);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Probe the table for an element.
  __forceinline__
  static bool
  contains_key(const word_t* __restrict block_ptr, const key_t& key) {
    const auto hash_value = hasher::hash(key);
    return contains_hash(block_ptr, hash_value);
  }
  //===----------------------------------------------------------------------===//

};
//===----------------------------------------------------------------------===//

} // namespace cuckoofilter
} // namespace dtl