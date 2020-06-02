#pragma once

#include <cstdlib>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>

#include <dtl/bloomfilter/blocked_cuckoofilter_util.hpp>

namespace dtl {
namespace cuckoofilter {


//===----------------------------------------------------------------------===//
/// A a tiny statically sized Cuckoo filter table that fits into a single word.
template<
    typename __word_t,               // The fundamental type used to store the table.
    uint32_t __tag_size_bits,        // The number of bits per tag (aka fingerprints).
    uint32_t __tags_per_bucket       // Aka associativity (should be at least 4).
>
struct blocked_cuckoofilter_word_table {

  using word_t = uint64_t;

  static constexpr uint32_t tag_size_bits = __tag_size_bits;
  static constexpr uint32_t tags_per_bucket = __tags_per_bucket;

  static constexpr word_t tag_mask = (1u << tag_size_bits) - 1;
  static constexpr uint32_t bucket_size_bits = tag_size_bits * tags_per_bucket;
  static constexpr word_t bucket_mask = (1u << bucket_size_bits) - 1;
  static constexpr word_t bucket_count = sizeof(word_t) * 8 / bucket_size_bits;
  static constexpr uint32_t bucket_addressing_bits = dtl::ct::log_2_u32<dtl::next_power_of_two(bucket_count)>::value;

  static constexpr uint32_t capacity = bucket_count * tags_per_bucket;
  static constexpr uint32_t required_hash_bits = tag_size_bits + bucket_addressing_bits;
  static_assert(required_hash_bits <= 32, "The required hash bits must not exceed 32 bits.");


  static constexpr word_t null_tag = 0;
  static constexpr uint32_t overflow_tag = uint32_t(-1);
  static constexpr word_t overflow_bucket = bucket_mask;


  __forceinline__
  static word_t
  read_bucket(const word_t* __restrict block_ptr, const uint32_t bucket_idx) {
    auto bucket = block_ptr[0] >> (bucket_size_bits * bucket_idx);
    return static_cast<uint32_t>(bucket);
  }


  __forceinline__
  static void
  mark_overflow(word_t* __restrict block_ptr, const uint32_t bucket_idx) {
    auto existing_bucket = read_bucket(block_ptr, bucket_idx);
    auto b = word_t(overflow_bucket ^ existing_bucket) << (bucket_size_bits * bucket_idx);
    block_ptr[0] ^= b;
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
    auto tag = (bucket >> (tag_size_bits * tag_idx)) & tag_mask;
    return static_cast<uint32_t>(tag);
  }


  __forceinline__
  static uint32_t
  write_tag(word_t* __restrict block_ptr, const uint32_t bucket_idx, const uint32_t tag_idx, const uint32_t tag) {
    auto existing_tag = read_tag(block_ptr, bucket_idx, tag_idx);
    auto t = (word_t(tag ^ existing_tag) << (bucket_size_bits * bucket_idx)) << (tag_size_bits * tag_idx);
    block_ptr[0] ^= t;
    return existing_tag;
  }


  __forceinline__
  static uint32_t
  insert_tag_relocate(word_t* __restrict block_ptr, const uint32_t bucket_idx, const uint32_t tag) {
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
    // couldn't find an empty place
    // kick out existing tag
    uint32_t rnd_tag_idx = static_cast<uint32_t>(std::rand()) % tags_per_bucket;
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
    return tag;
  }


  __forceinline__
  static bool
  find_tag_in_buckets(const word_t* __restrict block_ptr,
                      const uint32_t bucket_idx, const uint32_t alternative_bucket_idx, const uint32_t tag) {
    word_t f = block_ptr[0];
    const word_t mask = (word_t(bucket_mask) << (bucket_idx * bucket_size_bits))
                      | (word_t(bucket_mask) << (alternative_bucket_idx * bucket_size_bits));
    const auto masked_buckets = f & mask;
    return packed_value<word_t, tag_size_bits>::contains(masked_buckets, tag)
         | packed_value<word_t, tag_size_bits>::contains(masked_buckets, bucket_mask); // overflow check
  }


};
//===----------------------------------------------------------------------===//


} // namespace cuckoofilter
} // namespace dtl
