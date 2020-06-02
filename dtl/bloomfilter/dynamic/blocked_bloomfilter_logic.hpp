#pragma once

#include <bitset>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <memory>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/math.hpp>

#include "block_addressing_logic.hpp"
#include "blocked_bloomfilter_block_logic.hpp"

namespace dtl {
namespace bloomfilter_dynamic {


template<
    typename HashFn        // the hash function (family) to use
>
struct blocked_bloomfilter_logic {

  using key_t = $u32;
  using word_t = $u8;
  using size_t = $u32;

  //===----------------------------------------------------------------------===//
  // Inspect the given hash functions
  //===----------------------------------------------------------------------===//

  using hash_value_t = $u32; //decltype(HashFn<key_t>::hash(0)); // TODO find out why NVCC complains

  static_assert(std::is_integral<hash_value_t>::value, "Hash function must return an integral type.");

  static constexpr u32 hash_value_bitlength = sizeof(hash_value_t) * 8;


  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  /// The block addressing scheme.
  std::unique_ptr<block_addressing_logic> addr;
  /// The block logic.
  blocked_bloomfilter_block_logic<word_t, key_t, HashFn> block;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Determine _addressing logic depending on the length.
  //===----------------------------------------------------------------------===//
  static std::unique_ptr<block_addressing_logic>
  determine_addressing_logic(std::size_t length /* in bits */,
                             u32 block_size_bytes) {
    u32 block_size_bits = block_size_bytes * 8;
    u32 desired_block_cnt = (length + block_size_bits - 1) / block_size_bits;
    if (is_power_of_two(desired_block_cnt)) {
      return std::make_unique<bloomfilter_addressing_logic_pow2>(desired_block_cnt);
    }
    else {
      const bloomfilter_addressing_logic_magic magic(desired_block_cnt);
      const bloomfilter_addressing_logic_pow2 pow2(desired_block_cnt);
      if (magic.block_cnt >= pow2.block_cnt) {
        return std::make_unique<bloomfilter_addressing_logic_pow2>(desired_block_cnt);
      }
      else {
        return std::make_unique<bloomfilter_addressing_logic_magic>(desired_block_cnt);
      }
    }
  }
  //===----------------------------------------------------------------------===//


 public:

  explicit
  blocked_bloomfilter_logic(const std::size_t length /* in bits */,
                    u32 block_size_bytes,
                    u32 sector_size_bytes,
                    u32 k)
      : addr(determine_addressing_logic(length, block_size_bytes)),
        block(block_size_bytes, sector_size_bytes, k) { }


  blocked_bloomfilter_logic(const blocked_bloomfilter_logic& src) noexcept
      : addr(src.addr.get()),
        block(src.block) {}


  blocked_bloomfilter_logic(blocked_bloomfilter_logic&& src) noexcept
      : addr(std::move(src.addr)),
        block(std::move(src.block)) {}


  blocked_bloomfilter_logic& operator=(blocked_bloomfilter_logic&& src) {
    addr = std::move(src.addr);
    block = std::move(src.block);
  };
  //===----------------------------------------------------------------------===//


  /// Returns the size of the Bloom filter (in number of bits).
  __forceinline__ __host__ __device__
  std::size_t
  length() const noexcept {
    return static_cast<std::size_t>(addr->block_cnt) * block.block_bitlength;
  }
  //===----------------------------------------------------------------------===//


  /// Returns the number of blocks the Bloom filter consists of.
  __forceinline__ __host__ __device__
  std::size_t
  block_cnt() const noexcept {
    return addr->block_cnt;
  }
  //===----------------------------------------------------------------------===//


  /// Returns the number of sectors the Bloom filter consists of.
  __forceinline__ __host__ __device__
  std::size_t
  sector_cnt() const noexcept {
    return block_cnt() * block.sector_cnt;
  }
  //===----------------------------------------------------------------------===//


  /// Returns the number of words the Bloom filter consists of.
  __forceinline__ __host__ __device__
  std::size_t
  word_cnt() const noexcept {
    return addr->block_cnt * (block.block_bitlength / (sizeof(word_t) * 8));
  }
  //===----------------------------------------------------------------------===//


  __forceinline__ __host__
  void
  insert(word_t* __restrict filter,
         const key_t& key) const noexcept {
    const hash_value_t hash_val = HashFn::hash(key, 0);
    const size_t word_idx = addr->get_block_idx(hash_val) * block.word_cnt;
    block.insert(key, &filter[word_idx]);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__
  void
  batch_insert(word_t* __restrict filter,
               const key_t* __restrict keys, const uint32_t key_cnt) const {
    for (uint32_t j = 0; j < key_cnt; j++) {
      insert(filter, keys[j]);
    }
  };
  //===----------------------------------------------------------------------===//


  __forceinline__ __host__ __device__
  u1
  contains(const word_t* __restrict filter,
           const key_t& key) const noexcept {
    const hash_value_t hash_val = HashFn::hash(key, 0);
    const size_t word_idx = addr->get_block_idx(hash_val) * block.word_cnt;
    return block.contains(key, &filter[word_idx]);
  }
  //===----------------------------------------------------------------------===//


  __forceinline__
  uint64_t
  batch_contains(const word_t* __restrict filter,
                 const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    constexpr u32 mini_batch_size = 16;
    const u32 mini_batch_cnt = key_cnt / mini_batch_size;

    $u32* match_writer = match_positions;
    for ($u32 mb = 0; mb < mini_batch_cnt; mb++) {
      for (uint32_t j = mb * mini_batch_size; j < ((mb + 1) * mini_batch_size); j++) {
        const auto is_contained = contains(filter, keys[j]);
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }
    for (uint32_t j = (mini_batch_cnt * mini_batch_size); j < key_cnt; j++) {
      const auto is_contained = contains(filter, keys[j]);
      *match_writer = j + match_offset;
      match_writer += is_contained;
    }
    return match_writer - match_positions;
  };
  //===----------------------------------------------------------------------===//


  std::string
  info() const noexcept {
    return "{\"name\":\"dynamic_blocked_bloom\",\"size\":" + std::to_string(length() / 8)
           + ",\"B\":" + std::to_string(block.block_bitlength / 8)
           + ",\"S\":" + std::to_string(block.sector_bitlength / 8)
           + ",\"K\":" + std::to_string(block.k)
           + ",\"k\":" + std::to_string(block.k * block.sector_cnt)
           + ",\"addr\":" + (dynamic_cast<bloomfilter_addressing_logic_pow2*>(addr.get()) ? "\"pow2\"" : "\"magic\"")
           + "}";
  }
  //===----------------------------------------------------------------------===//

};

} // namespace bloomfilter_dynamic
} // namespace dtl