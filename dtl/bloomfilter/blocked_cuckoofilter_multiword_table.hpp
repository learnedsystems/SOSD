#pragma once

#include <cstdlib>
#include <cstring>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>

#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_util.hpp>
#include <dtl/thread.hpp>

namespace dtl {
namespace cuckoofilter {


//===----------------------------------------------------------------------===//
namespace internal {


//===----------------------------------------------------------------------===//
// Specializations for finding tags in buckets.
//===----------------------------------------------------------------------===//

// Optimized for the case where at least two buckets fit into a single word.
template<typename table_t, uint32_t _bucket_cnt_per_word>
struct find_tag {

  __forceinline__ static bool
  find_tag_in_buckets(const typename table_t::word_t* __restrict block_ptr,
                      uint32_t bucket_idx, uint32_t alternative_bucket_idx, uint32_t tag) {
    const auto bucket = table_t::read_bucket(block_ptr, bucket_idx);
    const auto alternative_bucket = table_t::read_bucket(block_ptr, alternative_bucket_idx);
    bool found = false;
    found |= bucket == table_t::overflow_bucket;
    found |= alternative_bucket == table_t::overflow_bucket;
    // Merge both buckets into one word -> do only one search
    const typename table_t::word_t merged_buckets = (static_cast<typename table_t::word_t>(bucket) << table_t::bucket_size_bits) | alternative_bucket;
    found |= packed_value<typename table_t::word_t, table_t::tag_size_bits>::contains(merged_buckets, tag);
    return found;
  };

};


// Optimized for the case where only one bucket fits into a word.
// Both candidate buckets need to be checked one after another.
template<typename table_t>
struct find_tag<table_t, 1> {
//  template<typename table_t>
  __forceinline__ static bool
  find_tag_in_buckets(const typename table_t::word_t* __restrict block_ptr,
                      uint32_t bucket_idx, uint32_t alternative_bucket_idx, uint32_t tag) {
    const auto bucket = table_t::read_bucket(block_ptr, bucket_idx);
    const auto alternative_bucket = table_t::read_bucket(block_ptr, alternative_bucket_idx);
    bool found = false;
    found |= bucket == table_t::overflow_bucket;
    found |= alternative_bucket == table_t::overflow_bucket;
    found |= packed_value<typename table_t::word_t, table_t::tag_size_bits>::contains(bucket, tag);
    found |= packed_value<typename table_t::word_t, table_t::tag_size_bits>::contains(alternative_bucket, tag);
    return found;
  };

};


} // namespace internal
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
/// A statically sized Cuckoo filter table.
/// The table has the following restrictions/properties:
///   - the bucket size must not exceed the size of a processor word
///   - buckets are aligned, so that a single bucket cannot wrap around word boundaries
template<
    typename __word_t,               // The fundamental type used to store the table.
    uint32_t __table_size_bytes,     // Typically the size of a cache line (64 bytes).
    uint32_t __tag_size_bits,        // The number of bits per tag (aka fingerprints).
    uint32_t __tags_per_bucket       // Aka associativity (should be at least 4).
>
struct blocked_cuckoofilter_multiword_table {

  using word_t = __word_t;
  static constexpr uint32_t word_size_bits = sizeof(word_t) * 8;
  static constexpr uint32_t table_size_bytes = __table_size_bytes;

  static constexpr uint32_t tag_size_bits = __tag_size_bits;
  static constexpr uint32_t tag_mask = (1u << tag_size_bits) - 1;
  static constexpr uint32_t tags_per_bucket = __tags_per_bucket;
  static constexpr uint32_t bucket_size_bits = tag_size_bits * tags_per_bucket;
  static_assert(bucket_size_bits <= (sizeof(word_t)*8), "The bucket size must not exceed a word.");

  static constexpr uint32_t word_cnt = table_size_bytes / sizeof(word_t);
  static constexpr uint32_t word_cnt_log2 = dtl::ct::log_2_u32<word_cnt>::value;
  static constexpr uint32_t bucket_cnt_per_word = word_size_bits / bucket_size_bits;

  static constexpr word_t bucket_mask = (bucket_size_bits == word_size_bits) ? word_t(-1) : (word_t(1) << bucket_size_bits) - 1;
  static constexpr uint32_t bucket_count = bucket_cnt_per_word * word_cnt;
  static_assert(dtl::is_power_of_two(bucket_count), "Bucket count must be a power of two.");
  static constexpr uint32_t bucket_addressing_bits = dtl::ct::log_2_u32<dtl::next_power_of_two(bucket_count)>::value;

  static constexpr uint32_t capacity = bucket_count * tags_per_bucket;
  static constexpr uint32_t required_hash_bits = tag_size_bits + bucket_addressing_bits;
  static_assert(required_hash_bits <= 32, "The required hash bits must not exceed 32 bits.");

  static constexpr uint32_t null_tag = 0;
  static constexpr uint32_t overflow_tag = uint32_t(-1);
  static constexpr word_t overflow_bucket = bucket_mask;


  __forceinline__
  static word_t
  read_bucket(const word_t* __restrict block_ptr, const uint32_t bucket_idx) {
    const auto word_idx = bucket_idx & ((1u << word_cnt_log2) - 1);
    word_t word = block_ptr[word_idx];
    const auto in_word_bucket_idx = bucket_idx >> word_cnt_log2;
    const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);
    return bucket;
  }


  __forceinline__
  static void
  write_bucket(word_t* __restrict block_ptr, const uint32_t bucket_idx, const word_t bucket_content) {
    const auto word_idx = bucket_idx & ((1u << word_cnt_log2) - 1);
    word_t word = block_ptr[word_idx];
    const auto in_word_bucket_idx = bucket_idx >> word_cnt_log2;
    const auto shift_amount = bucket_size_bits * in_word_bucket_idx;
    const auto to_write = word ^ (bucket_content << shift_amount);
    word ^= to_write;
    block_ptr[word_idx] = word;
  }


  __forceinline__
  static void
  mark_overflow(word_t* __restrict block_ptr, const uint32_t bucket_idx) {
    write_bucket(block_ptr, bucket_idx, overflow_bucket);
  }


  __forceinline__
  static uint32_t
  read_tag_from_bucket(const word_t bucket, const uint32_t tag_idx) {
    auto tag = (bucket >> (tag_size_bits * tag_idx)) & tag_mask;
    return static_cast<uint32_t>(tag);
  }


  __forceinline__
  static uint32_t
  read_tag(const word_t* __restrict block_ptr, const uint32_t bucket_idx, const uint32_t tag_idx) {
    auto bucket = read_bucket(block_ptr, bucket_idx);
    auto tag = read_tag_from_bucket(bucket, tag_idx);
    return tag;
  }


  __forceinline__
  static uint32_t
  write_tag(word_t* __restrict block_ptr, const uint32_t bucket_idx, const uint32_t tag_idx, const uint32_t tag_content) {
    auto bucket = read_bucket(block_ptr, bucket_idx);
    auto existing_tag = read_tag(block_ptr, bucket_idx, tag_idx);
    const auto to_write = existing_tag ^ tag_content;
    bucket ^= word_t(to_write) << (tag_size_bits * tag_idx);
    write_bucket(block_ptr, bucket_idx, bucket);
    return existing_tag;
  }


  __forceinline__
  static uint32_t
  insert_tag_relocate(word_t* __restrict block_ptr,
                      const uint32_t bucket_idx, const uint32_t tag) {
    // Check whether this is an overflow bucket.
    auto bucket = read_bucket(block_ptr, bucket_idx);
    if (bucket == overflow_bucket) {
      return overflow_tag;
    }
    // Check the buckets' entries for free space.
    for (uint32_t tag_idx = 0; tag_idx < tags_per_bucket; tag_idx++) {
      auto t = read_tag_from_bucket(bucket, tag_idx);
      if (t == tag) {
        return null_tag;
      }
      else if (t == 0) {
        write_tag(block_ptr, bucket_idx, tag_idx, tag);
        return null_tag;
      }
    }
    // Couldn't find an empty place.
    // Relocate existing tag.
    uint32_t rnd_tag_idx = static_cast<uint32_t>(dtl::this_thread::rand32()) % tags_per_bucket;
    return write_tag(block_ptr, bucket_idx, rnd_tag_idx, tag);
  }


  __forceinline__
  static uint32_t
  insert_tag(word_t* __restrict block_ptr, const uint32_t bucket_idx, const uint32_t tag) {
    // Check whether this is an overflow bucket.
    auto bucket = read_bucket(block_ptr, bucket_idx);
    if (bucket == overflow_bucket) {
      return overflow_tag;
    }
    // Check the buckets' entries for free space. // TODO optimize!
    for (uint32_t tag_idx = 0; tag_idx < tags_per_bucket; tag_idx++) {
      auto t = read_tag_from_bucket(bucket, tag_idx);
      if (t == tag) {
        // Tag is already stored in bucket.
        return null_tag;
      }
      else if (t == null_tag) {
        // Found an empty slot.
        write_tag(block_ptr, bucket_idx, tag_idx, tag);
        return null_tag;
      }
    }
    return tag;
  }


  __forceinline__
  static bool
  find_tag_in_buckets(const word_t* __restrict block_ptr,
                      const uint32_t bucket_idx, const uint32_t alternative_bucket_idx, const uint32_t tag) {
    return internal::find_tag<blocked_cuckoofilter_multiword_table, bucket_cnt_per_word>
                   ::find_tag_in_buckets(block_ptr, bucket_idx, alternative_bucket_idx, tag);
  }

};
//===----------------------------------------------------------------------===//

} // namespace cuckoofilter
} // namespace dtl
