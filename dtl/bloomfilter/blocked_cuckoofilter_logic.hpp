#pragma once

#include <cstdlib>

#include <dtl/dtl.hpp>
#include <dtl/hash.hpp>
#include <dtl/math.hpp>
#include <dtl/mem.hpp>

#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_block_logic.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_multiword_table.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_util.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_word_table.hpp>
#ifdef __AVX2__
#include <dtl/bloomfilter/blocked_cuckoofilter_simd.hpp>
#endif


namespace dtl {
namespace cuckoofilter {

//===----------------------------------------------------------------------===//
// A blocked cuckoo filter template.
//===----------------------------------------------------------------------===//
template<
    typename _key_t = uint32_t,
    typename _block_t = blocked_cuckoofilter_block_logic<_key_t>,
    block_addressing _block_addressing = block_addressing::POWER_OF_TWO
>
struct blocked_cuckoofilter {

  // TODO
//  template<
//      typename key_t,
//      $u32 hash_fn_no
//  >
//  using hasher = dtl::hash::stat::mul32<key_t, hash_fn_no>;


  using key_t = _key_t;
  using block_t = _block_t;
  using word_t = typename block_t::word_t;
  using hash_value_t = uint32_t;
  using hasher = dtl::hash::knuth_32_alt<hash_value_t>;
  using addr_t = block_addressing_logic<_block_addressing>;

  static constexpr u32 word_cnt_per_block = block_t::table_t::word_cnt;
  static constexpr u32 word_cnt_per_block_log2 = dtl::ct::log_2<word_cnt_per_block>::value;

  //===----------------------------------------------------------------------===//
  // Members
  //===----------------------------------------------------------------------===//
  const addr_t addr;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  explicit
  blocked_cuckoofilter(const std::size_t desired_length)
      : addr((desired_length + (block_t::block_bitlength - 1)) / block_t::block_bitlength) { }

  blocked_cuckoofilter(const blocked_cuckoofilter&) noexcept = default;

  blocked_cuckoofilter(blocked_cuckoofilter&&) noexcept = default;

  ~blocked_cuckoofilter() noexcept = default;
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  void
  insert(word_t* __restrict filter_data, const key_t& key) const {
    const auto hash_val = hasher::hash(key);
    const auto block_idx = addr.get_block_idx(hash_val);
    const auto word_idx = block_idx << word_cnt_per_block_log2;
    auto block_ptr = &filter_data[word_idx];
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
      block_t::insert_hash(block_ptr, hash_val << addr.get_required_addressing_bits());
    }
    else {
      block_t::insert_key(block_ptr, key);
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  void
  batch_insert(word_t* __restrict filter_data, const key_t* keys, const uint32_t key_cnt) const {
    for (uint32_t i = 0; i < key_cnt; i++) {
      insert(filter_data, keys[i]);
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__
  bool
  contains(const word_t* __restrict filter_data, const key_t& key) const {
    auto hash_val = hasher::hash(key);
    auto block_idx = addr.get_block_idx(hash_val);
    const auto word_idx = block_idx << word_cnt_per_block_log2;
    auto block_ptr = &filter_data[word_idx];
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
      return block_t::contains_hash(block_ptr, hash_val << addr.get_required_addressing_bits());
    }
    else {
      return block_t::contains_key(block_ptr, key);
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Performs a batch-probe
  __forceinline__ __unroll_loops__ __host__
  std::size_t
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, u32 key_cnt,
                 $u32* __restrict match_positions, u32 match_offset) const {
    constexpr u32 mini_batch_size = 16;
    const u32 mini_batch_cnt = key_cnt / mini_batch_size;

    $u32* match_writer = match_positions;
    if ((addr.get_required_addressing_bits() + block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {

      for ($u32 mb = 0; mb < mini_batch_cnt; mb++) {
        for (uint32_t j = mb * mini_batch_size; j < ((mb + 1) * mini_batch_size); j++) {
          auto h = hasher::hash(keys[j]);
          auto i = addr.get_block_idx(h);
          auto w = i << word_cnt_per_block_log2;
          auto p = &filter_data[w];
          auto is_contained = block_t::contains_hash(p, h << addr.get_required_addressing_bits());
          *match_writer = j + match_offset;
          match_writer += is_contained;
        }
      }
      for (uint32_t j = (mini_batch_cnt * mini_batch_size); j < key_cnt; j++) {
        auto h = hasher::hash(keys[j]);
        auto i = addr.get_block_idx(h);
        auto w = i << word_cnt_per_block_log2;
        auto p = &filter_data[w];
        auto is_contained = block_t::contains_hash(p, h << addr.get_required_addressing_bits());
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }

    else {
      for ($u32 mb = 0; mb < mini_batch_cnt; mb++) {
        for (uint32_t j = mb * mini_batch_size; j < ((mb + 1) * mini_batch_size); j++) {
          auto k = keys[j];
          auto h = hasher::hash(k);
          auto i = addr.get_block_idx(h);
          auto w = i << word_cnt_per_block_log2;
          auto p = &filter_data[w];
          auto is_contained = block_t::contains_key(p, k);
          *match_writer = j + match_offset;
          match_writer += is_contained;
        }
      }
      for (uint32_t j = (mini_batch_cnt * mini_batch_size); j < key_cnt; j++) {
        auto k = keys[j];
        auto h = hasher::hash(k);
        auto i = addr.get_block_idx(h);
        auto w = i << word_cnt_per_block_log2;
        auto p = &filter_data[w];
        auto is_contained = block_t::contains_key(p, k);
        *match_writer = j + match_offset;
        match_writer += is_contained;
      }
    }
    return match_writer - match_positions;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the number of words the filter consists of.
  __forceinline__ __host__ __device__
  std::size_t
  size() const noexcept {
    return addr.block_cnt * word_cnt_per_block;
  }
  //===----------------------------------------------------------------------===//


};
//===----------------------------------------------------------------------===//

} // namespace cuckoofilter


// TODO move somewhere else
static constexpr uint64_t cache_line_size = 64;


//===----------------------------------------------------------------------===//
// Cuckoo filter base class.
// using static polymorphism (CRTP)
//===----------------------------------------------------------------------===//
template<typename _key_t, typename _word_t, typename _derived>
struct blocked_cuckoofilter_logic_base {

  using key_t = _key_t;
  using word_t = _word_t;

  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  void
  insert(word_t* __restrict filter_data, const key_t& key) {
    return static_cast<_derived*>(this)->filter.insert(filter_data, key);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__ __host__
  void
  batch_insert(word_t* __restrict filter_data, const key_t* keys, const uint32_t key_cnt) {
    static_cast<_derived*>(this)->filter.batch_insert(filter_data, keys, key_cnt);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__ __host__ __device__
  bool
  contains(const word_t* __restrict filter_data, const key_t& key) const {
    return static_cast<const _derived*>(this)->filter.contains(filter_data, key);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  __forceinline__  __host__
  uint64_t
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return static_cast<const _derived*>(this)->filter.batch_contains(filter_data, keys, key_cnt, match_positions, match_offset);
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Returns the number of words the filter consists of.
  __forceinline__ __host__ __device__
  std::size_t
  size() const noexcept {
    return static_cast<const _derived*>(this)->filter.size();
  }
  //===----------------------------------------------------------------------===//

  /// Returns (actual) length in bits.
  __forceinline__
  std::size_t
  get_length() const noexcept {
    return static_cast<const _derived*>(this)->filter.size() * sizeof(word_t) * 8;
  }
  //===----------------------------------------------------------------------===//

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Instantiations of some reasonable cuckoo filters.
// Note, that not all instantiations are suitable for SIMD.
//===----------------------------------------------------------------------===//
template<uint32_t _block_size_bytes, uint32_t _bits_per_element, uint32_t _associativity, block_addressing _addressing>
struct blocked_cuckoofilter_logic {};


template<uint32_t _block_size_bytes, block_addressing _addressing>
struct blocked_cuckoofilter_logic<_block_size_bytes, 16, 4, _addressing>
    : blocked_cuckoofilter_logic_base<uint32_t, uint64_t, blocked_cuckoofilter_logic<_block_size_bytes, 16, 4, _addressing>> {

  static constexpr uint32_t block_size_bytes = _block_size_bytes;
  static constexpr uint32_t tag_size_bits = 16;
  static constexpr uint32_t associativity = 4;

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, tag_size_bits, associativity>;
  using block_t = cuckoofilter::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::blocked_cuckoofilter<uint32_t, block_t, _addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter_logic(const std::size_t length) : filter(length) { }

#ifdef __AVX2__
  // use SIMD implementation
  template<u64 vector_len = dtl::simd::lane_count<key_t>>
  __forceinline__ uint64_t
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return dtl::cuckoofilter::internal::simd_batch_contains_16_4<blocked_cuckoofilter_logic, vector_len>(
        *this, filter_data, keys, key_cnt, match_positions, match_offset);
  };
#endif

};


template<uint32_t _block_size_bytes, block_addressing _addressing>
struct blocked_cuckoofilter_logic<_block_size_bytes, 16, 2, _addressing>
    : blocked_cuckoofilter_logic_base<uint32_t, uint32_t, blocked_cuckoofilter_logic<_block_size_bytes, 16, 2, _addressing>> {

  static constexpr uint32_t block_size_bytes = _block_size_bytes;
  static constexpr uint32_t tag_size_bits = 16;
  static constexpr uint32_t associativity = 2;

  using key_t = uint32_t;
  using word_t = uint32_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, tag_size_bits, associativity>;
  using block_t = cuckoofilter::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::blocked_cuckoofilter<uint32_t, block_t, _addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter_logic(const std::size_t length) : filter(length) { }

  //TODO <- maybe -  SIMD implementation

};


template<uint32_t _block_size_bytes, block_addressing _addressing>
struct blocked_cuckoofilter_logic<_block_size_bytes, 12, 4, _addressing>
    : blocked_cuckoofilter_logic_base<uint32_t, uint64_t, blocked_cuckoofilter_logic<_block_size_bytes, 12, 4, _addressing>> {

  static constexpr uint32_t block_size_bytes = _block_size_bytes;
  static constexpr uint32_t tag_size_bits = 12;
  static constexpr uint32_t associativity = 4;

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, tag_size_bits, associativity>;
  using block_t = cuckoofilter::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::blocked_cuckoofilter<uint32_t, block_t, _addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter_logic(const std::size_t length) : filter(length) { }

};


template<uint32_t _block_size_bytes, block_addressing _addressing>
struct blocked_cuckoofilter_logic<_block_size_bytes, 10, 6, _addressing>
    : blocked_cuckoofilter_logic_base<uint32_t, uint64_t, blocked_cuckoofilter_logic<_block_size_bytes, 10, 6, _addressing>> {

  static constexpr uint32_t block_size_bytes = _block_size_bytes;
  static constexpr uint32_t tag_size_bits = 10;
  static constexpr uint32_t associativity = 6;

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, tag_size_bits, associativity>;
  using block_t = cuckoofilter::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::blocked_cuckoofilter<uint32_t, block_t, _addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter_logic(const std::size_t length) : filter(length) { }

};


template<uint32_t _block_size_bytes, block_addressing _addressing>
struct blocked_cuckoofilter_logic<_block_size_bytes, 8, 8, _addressing>
    : blocked_cuckoofilter_logic_base<uint32_t, uint64_t, blocked_cuckoofilter_logic<_block_size_bytes, 8, 8, _addressing>> {

  static constexpr uint32_t block_size_bytes = _block_size_bytes;
  static constexpr uint32_t tag_size_bits = 8;
  static constexpr uint32_t associativity = 8;

  using key_t = uint32_t;
  using word_t = uint64_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, tag_size_bits, associativity>;
  using block_t = cuckoofilter::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::blocked_cuckoofilter<uint32_t, block_t, _addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter_logic(const std::size_t length) : filter(length) { }

};


template<uint32_t _block_size_bytes, block_addressing _addressing>
struct blocked_cuckoofilter_logic<_block_size_bytes, 8, 4, _addressing>
    : blocked_cuckoofilter_logic_base<uint32_t, uint32_t, blocked_cuckoofilter_logic<_block_size_bytes, 8, 4, _addressing>> {

  static constexpr uint32_t block_size_bytes = _block_size_bytes;
  static constexpr uint32_t tag_size_bits = 8;
  static constexpr uint32_t associativity = 4;

  using key_t = uint32_t;
  using word_t = uint32_t;
  using table_t = cuckoofilter::blocked_cuckoofilter_multiword_table<word_t, block_size_bytes, tag_size_bits, associativity>;
  using block_t = cuckoofilter::blocked_cuckoofilter_block_logic<key_t, table_t>;
  using filter_t = cuckoofilter::blocked_cuckoofilter<uint32_t, block_t, _addressing>;

  filter_t filter; // the actual filter instance

  explicit blocked_cuckoofilter_logic(const std::size_t length) : filter(length) { }

#ifdef __AVX2__
  // use SIMD implementation
  template<u64 vector_len = dtl::simd::lane_count<key_t>>
  __forceinline__ uint64_t
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* __restrict keys, const uint32_t key_cnt,
                 uint32_t* __restrict match_positions, const uint32_t match_offset) const {
    return dtl::cuckoofilter::internal::simd_batch_contains_8_4<blocked_cuckoofilter_logic, vector_len>(
        *this, filter_data, keys, key_cnt, match_positions, match_offset);
  };
#endif // __AVX2__

};
//===----------------------------------------------------------------------===//


} // namespace dtl