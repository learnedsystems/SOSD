#pragma once

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>

#include <dtl/bloomfilter/dynamic/blocked_bloomfilter_logic.hpp>
#include <dtl/bloomfilter/dynamic/hash.hpp>


namespace dtl {
namespace bloomfilter_dynamic {


struct blocked_bloomfilter {
  using key_t = uint32_t;
  using filter_logic_t = dtl::bloomfilter_dynamic::blocked_bloomfilter_logic<dtl::bloomfilter_dynamic::hasher_mul32>;
  using word_t = typename filter_logic_t::word_t;

  //===----------------------------------------------------------------------===//
  const filter_logic_t filter_logic;
  std::vector<word_t> filter_data;
  //===----------------------------------------------------------------------===//


  blocked_bloomfilter(u64 m,
                      u32 block_size_bytes,
                      u32 sector_size_bytes,
                      u32 k)
      : filter_logic(m, block_size_bytes, sector_size_bytes, k), filter_data(filter_logic.word_cnt(), 0) {}


  blocked_bloomfilter(const blocked_bloomfilter&) noexcept = default;


  blocked_bloomfilter(blocked_bloomfilter&&) noexcept = default;
  //===----------------------------------------------------------------------===//


  __forceinline__
  void
  insert(const key_t& key) noexcept {
    filter_logic.insert(&filter_data[0], key);
  };
  //===----------------------------------------------------------------------===//


  __forceinline__
  bool
  contains(const key_t& key) const noexcept {
    return filter_logic.contains(&filter_data[0], key);
  }
  //===----------------------------------------------------------------------===//


  uint64_t
  batch_contains(const key_t* keys, u32 key_cnt,
                 $u32* match_positions, u32 match_offset) const {
    return filter_logic.batch_contains(&filter_data[0], keys, key_cnt, match_positions, match_offset);
  }
  //===----------------------------------------------------------------------===//


};
//===----------------------------------------------------------------------===//


} // namespace bloomfilter_dynamic
} // namespace dtl