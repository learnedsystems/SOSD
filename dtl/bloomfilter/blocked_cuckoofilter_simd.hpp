#pragma once

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>

#include "vector_helper.hpp"

namespace dtl {
namespace cuckoofilter {
namespace internal {


//===----------------------------------------------------------------------===//
// SIMDized implementations for batch_contains.
// Note: Only Cuckoo filter <8,4> and <16,4> are supported yet.
//       and AVX-2 only! // TODO implement for AVX512
//===----------------------------------------------------------------------===//

template<typename _filter_t, u64 vector_len>
__forceinline__ __unroll_loops__ __host__
static std::size_t
simd_batch_contains_8_4(const _filter_t& filter, const typename _filter_t::word_t* __restrict filter_data,
                        u32* __restrict keys, u32 key_cnt,
                        $u32* __restrict match_positions, u32 match_offset) {
  using namespace dtl;
  using filter_t = _filter_t;
  using key_t = $u32;
  using hash_value_t = $u32;

  const key_t* reader = keys;
  $u32* match_writer = match_positions;

  // determine the number of keys that need to be probed sequentially, due to alignment
  u64 required_alignment_bytes = 64;
  u64 t = dtl::mem::is_aligned(reader)  // should always be true
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

  // process the aligned keys vectorized
  using key_vt = vec<key_t, vector_len>;
  using ptr_vt = vec<$u64, vector_len>;

  constexpr u32 block_size_log2 = dtl::ct::log_2_u64<filter_t::block_t::block_size>::value;
  constexpr u32 word_size_log2 = dtl::ct::log_2_u64<sizeof(typename filter_t::table_t::word_t)>::value;

  r256 offset_vec = {.i = _mm256_set1_epi32(match_offset + read_pos) };
  const r256 overflow_tag = {.i = _mm256_set1_epi64x(-1) };
  u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;

  if ((filter.filter.addr.get_required_addressing_bits() + filter_t::block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {

    // --- contains hash ---
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {

      const key_vt& key_v = *reinterpret_cast<const key_vt*>(reader);
      const auto block_hash_v = dtl::hash::knuth_32_alt<key_vt>::hash(key_v);
      auto block_idx_v = filter.filter.addr.get_block_idxs(block_hash_v);

      // compute block address
      ptr_vt ptr_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(block_idx_v);
      ptr_v <<= block_size_log2;
      ptr_v += reinterpret_cast<std::uintptr_t>(&filter_data[0]);

      auto bucket_hash_v = block_hash_v << filter.filter.addr.get_required_addressing_bits();
      auto bucket_idx_v = filter_t::block_t::get_bucket_idxs(bucket_hash_v);
      auto tag_v = (bucket_hash_v >> (32 - filter_t::table_t::bucket_addressing_bits - filter_t::table_t::tag_size_bits))
                   & static_cast<uint32_t>(filter_t::table_t::tag_mask);
      tag_v[tag_v == 0] += 1; // tag must not be zero
      auto alternative_bucket_idx_v = filter_t::block_t::get_alternative_bucket_idxs(bucket_idx_v, tag_v);

      const auto word_idx_v = bucket_idx_v & ((1u << filter_t::table_t::word_cnt_log2) - 1);
      const auto alternative_word_idx_v = alternative_bucket_idx_v & ((1u << filter_t::table_t::word_cnt_log2) - 1);


//      const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);

//      const auto in_word_bucket_idx = bucket_idx_v >> filter_t::table_type::word_cnt_log2; // can either be 0 or 1
      const auto bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(
          word_idx_v << word_size_log2);

//      const auto in_word_alternative_bucket_idx = alternative_bucket_idx_v >> filter_t::table_type::word_cnt_log2; // can either be 0 or 1
      const auto alternative_bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(
          alternative_word_idx_v << word_size_log2);

      // load the buckets
      const auto bucket_v = dtl::internal::vector_gather<$u32, $u64, vector_len>::gather(bucket_ptr_v);
      const auto alternative_bucket_v = dtl::internal::vector_gather<$u32, $u64, vector_len>::gather(alternative_bucket_ptr_v);

      auto dup_tag_v = (tag_v | (tag_v << 8)) ;
      dup_tag_v |= dup_tag_v << 16;

      const auto b = reinterpret_cast<const r256*>(&bucket_v.data);
      const auto ba = reinterpret_cast<const r256*>(&alternative_bucket_v.data);
      auto t = reinterpret_cast<r256*>(&dup_tag_v.data);
      for (std::size_t i = 0; i < bucket_v.nested_vector_cnt; i++) {
        const r256 tags = t[i];
        const r256 bucket_content0 = b[i];
        const r256 bucket_content1 = ba[i];
        const r256 t0 = {.i = _mm256_cmpeq_epi8(bucket_content0.i, tags.i) };
        const r256 o0 = {.i = _mm256_cmpeq_epi8(bucket_content0.i, overflow_tag.i) };
        const r256 t1 = {.i = _mm256_cmpeq_epi8(bucket_content1.i, tags.i) };
        const r256 o1 = {.i = _mm256_cmpeq_epi8(bucket_content1.i, overflow_tag.i) };
        const r256 t2 = {.i = _mm256_or_si256(_mm256_or_si256(t0.i, o0.i), _mm256_or_si256(t1.i, o1.i)) };
        const r256 t3 = {.i = _mm256_cmpeq_epi32(t2.i, _mm256_setzero_si256()) };
        const auto mt = _mm256_movemask_ps(t3.s) ^ 0b11111111u;
        const r256 match_pos_vec = { .i = { _mm256_cvtepi16_epi32(dtl::simd::lut_match_pos[mt].i) } };
        const r256 pos_vec = {.i = _mm256_add_epi32(offset_vec.i, match_pos_vec.i) };
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(match_writer), pos_vec.i);
        match_writer += bits::pop_count(mt);
        offset_vec.i = _mm256_add_epi32(offset_vec.i, _mm256_set1_epi32(8));
      }

      reader += vector_len;
    }
  }
  else {
    // --- contains key ---
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {

      const key_vt& key_v = *reinterpret_cast<const key_vt*>(reader);
      const auto block_hash_v = dtl::hash::knuth_32_alt<key_vt>::hash(key_v);
      auto block_idx_v = filter.filter.addr.get_block_idxs(block_hash_v);

      // compute block address
      ptr_vt ptr_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(block_idx_v);
      ptr_v <<= block_size_log2;
      ptr_v += reinterpret_cast<std::uintptr_t>(&filter_data[0]);

      auto bucket_hash_v = dtl::hash::knuth_32<key_vt>::hash(key_v);
      auto bucket_idx_v = filter_t::block_t::get_bucket_idxs(bucket_hash_v);
      auto tag_v = (bucket_hash_v >> (32 - filter_t::table_t::bucket_addressing_bits - filter_t::table_t::tag_size_bits))
                   & static_cast<uint32_t>(filter_t::table_t::tag_mask);
      tag_v[tag_v == 0] += 1; // tag must not be zero
      auto alternative_bucket_idx_v = filter_t::block_t::get_alternative_bucket_idxs(bucket_idx_v, tag_v);

      const auto word_idx_v = bucket_idx_v & ((1u << filter_t::table_t::word_cnt_log2) - 1);
      const auto alternative_word_idx_v = alternative_bucket_idx_v & ((1u << filter_t::table_t::word_cnt_log2) - 1);


//      const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);

//      const auto in_word_bucket_idx = bucket_idx_v >> filter_t::table_type::word_cnt_log2; // can either be 0 or 1
      const auto bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(
          word_idx_v << word_size_log2);

//      const auto in_word_alternative_bucket_idx = alternative_bucket_idx_v >> filter_t::table_type::word_cnt_log2; // can either be 0 or 1
      const auto alternative_bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(
          alternative_word_idx_v << word_size_log2);

      // load the buckets
      const auto bucket_v = dtl::internal::vector_gather<$u32, $u64, vector_len>::gather(bucket_ptr_v);
      const auto alternative_bucket_v = dtl::internal::vector_gather<$u32, $u64, vector_len>::gather(alternative_bucket_ptr_v);

      auto dup_tag_v = (tag_v | (tag_v << 8)) ;
      dup_tag_v |= dup_tag_v << 16;

      const auto b = reinterpret_cast<const r256*>(&bucket_v.data);
      const auto ba = reinterpret_cast<const r256*>(&alternative_bucket_v.data);
      auto t = reinterpret_cast<r256*>(&dup_tag_v.data);
      for (std::size_t i = 0; i < bucket_v.nested_vector_cnt; i++) {
        const r256 tags = t[i];
        const r256 bucket_content0 = b[i];
        const r256 bucket_content1 = ba[i];
        const r256 t0 = {.i = _mm256_cmpeq_epi8(bucket_content0.i, tags.i) };
        const r256 o0 = {.i = _mm256_cmpeq_epi8(bucket_content0.i, overflow_tag.i) };
        const r256 t1 = {.i = _mm256_cmpeq_epi8(bucket_content1.i, tags.i) };
        const r256 o1 = {.i = _mm256_cmpeq_epi8(bucket_content1.i, overflow_tag.i) };
        const r256 t2 = {.i = _mm256_or_si256(_mm256_or_si256(t0.i, o0.i), _mm256_or_si256(t1.i, o1.i)) };
        const r256 t3 = {.i = _mm256_cmpeq_epi32(t2.i, _mm256_setzero_si256()) };
        const auto mt = _mm256_movemask_ps(t3.s) ^ 0b11111111u;
        const r256 match_pos_vec = { .i = { _mm256_cvtepi16_epi32(dtl::simd::lut_match_pos[mt].i) } };
        const r256 pos_vec = {.i = _mm256_add_epi32(offset_vec.i, match_pos_vec.i) };
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(match_writer), pos_vec.i);
        match_writer += bits::pop_count(mt);
        offset_vec.i = _mm256_add_epi32(offset_vec.i, _mm256_set1_epi32(8));
      }

      reader += vector_len;
    }

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


template<typename _filter_t, u64 vector_len>
__forceinline__ __unroll_loops__ __host__
static std::size_t
//batch_contains(const dtl::blocked_cuckoo_filter<16, 4, _addressing>& filter,
simd_batch_contains_16_4(const _filter_t& filter, const typename _filter_t::word_t* __restrict filter_data,
                         u32* __restrict keys, u32 key_cnt,
                         $u32* __restrict match_positions, u32 match_offset) {
  using namespace dtl;
  using filter_t = _filter_t;
  using key_t = $u32;
  using hash_value_t = $u32;

  const key_t* reader = keys;
  $u32* match_writer = match_positions;

  // determine the number of keys that need to be probed sequentially, due to alignment
  u64 required_alignment_bytes = 64;
  u64 t = dtl::mem::is_aligned(reader)  // should always be true
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

  // process the aligned keys vectorized
//  constexpr std::size_t vector_len = 32;
  using key_vt = vec<key_t, vector_len>;
  using ptr_vt = vec<$u64, vector_len>;

  constexpr u32 block_size_log2 = dtl::ct::log_2_u64<filter_t::block_t::block_size>::value;
  constexpr u32 word_size_log2 = dtl::ct::log_2_u64<sizeof(typename filter_t::table_t::word_t)>::value;

  r128 offset_vec = {.i = _mm_set1_epi32(match_offset + read_pos) };
  const r256 overflow_tag = {.i = _mm256_set1_epi64x(-1) };
  u64 aligned_key_cnt = ((key_cnt - unaligned_key_cnt) / vector_len) * vector_len;

  if ((filter.filter.addr.get_required_addressing_bits() + filter_t::block_t::required_hash_bits) <= (sizeof(hash_value_t) * 8)) {
    // --- contains hash ---
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {

      const key_vt& key_v = *reinterpret_cast<const key_vt*>(reader);
      const auto block_hash_v = dtl::hash::knuth_32_alt<key_vt>::hash(key_v);
      auto block_idx_v = filter.filter.addr.get_block_idxs(block_hash_v);

      // compute block address
      ptr_vt ptr_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(block_idx_v);
      ptr_v <<= block_size_log2;
      ptr_v += reinterpret_cast<std::uintptr_t>(&filter_data[0]);

      auto bucket_hash_v = block_hash_v << filter.filter.addr.get_required_addressing_bits();
      auto bucket_idx_v = filter_t::block_t::get_bucket_idxs(bucket_hash_v);
      auto tag_v = (bucket_hash_v >> (32 - filter_t::table_t::bucket_addressing_bits - filter_t::table_t::tag_size_bits))
                   & static_cast<uint32_t>(filter_t::table_t::tag_mask);
      tag_v[tag_v == 0] += 1; // tag must not be zero
      auto alternative_bucket_idx_v = filter_t::block_t::get_alternative_bucket_idxs(bucket_idx_v, tag_v);

      const auto word_idx_v = bucket_idx_v & ((1u << filter_t::table_t::word_cnt_log2) - 1);
      const auto alternative_word_idx_v = alternative_bucket_idx_v & ((1u << filter_t::table_t::word_cnt_log2) - 1);
//      const auto in_word_bucket_idx = bucket_idx >> word_cnt_log2;
//      const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);

      const auto bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(word_idx_v << word_size_log2);
      const auto alternative_bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(alternative_word_idx_v << word_size_log2);

      // load the buckets
      const auto bucket_v = dtl::internal::vector_gather<$u64, $u64, vector_len>::gather(bucket_ptr_v);
      const auto alternative_bucket_v = dtl::internal::vector_gather<$u64, $u64, vector_len>::gather(alternative_bucket_ptr_v);

      auto dup_tag_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(tag_v | (tag_v << 16)) ;
      dup_tag_v |= dup_tag_v << 32;

      const auto b = reinterpret_cast<const r256*>(&bucket_v.data);
      const auto ba = reinterpret_cast<const r256*>(&alternative_bucket_v.data);
      auto t = reinterpret_cast<r256*>(&dup_tag_v.data);
      for (std::size_t i = 0; i < bucket_v.nested_vector_cnt; i++) {
        const r256 tags = t[i];
        const r256 bucket_content0 = b[i];
        const r256 bucket_content1 = ba[i];
        const r256 t0 = {.i = _mm256_cmpeq_epi16(bucket_content0.i, tags.i) };
        const r256 o0 = {.i = _mm256_cmpeq_epi16(bucket_content0.i, overflow_tag.i) };
        const r256 t1 = {.i = _mm256_cmpeq_epi16(bucket_content1.i, tags.i) };
        const r256 o1 = {.i = _mm256_cmpeq_epi16(bucket_content1.i, overflow_tag.i) };
        const r256 t2 = {.i = _mm256_or_si256(_mm256_or_si256(t0.i, o0.i), _mm256_or_si256(t1.i, o1.i)) };
        const r256 t3 = {.i = _mm256_cmpeq_epi64(t2.i, _mm256_setzero_si256()) };
        const auto mt = _mm256_movemask_pd(t3.d) ^ 0b1111;
//        std::cout << std::bitset<4>(mt) << " ";
        const r128 match_pos_vec = { .i = dtl::simd::lut_match_pos_4bit[mt].i };
        const r128 pos_vec = {.i = _mm_add_epi32(offset_vec.i, match_pos_vec.i) };
        _mm_storeu_si128(reinterpret_cast<__m128i*>(match_writer), pos_vec.i);
        match_writer += bits::pop_count(mt);
        offset_vec.i = _mm_add_epi32(offset_vec.i, _mm_set1_epi32(4));
      }

      reader += vector_len;
    }
  }
  else {
    // --- contains key ---
    for (; read_pos < (unaligned_key_cnt + aligned_key_cnt); read_pos += vector_len) {

      const key_vt& key_v = *reinterpret_cast<const key_vt*>(reader);
      const auto block_hash_v = dtl::hash::knuth_32_alt<key_vt>::hash(key_v);
      auto block_idx_v = filter.filter.addr.get_block_idxs(block_hash_v);

      // compute block address
      ptr_vt ptr_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(block_idx_v);
      ptr_v <<= block_size_log2;
      ptr_v += reinterpret_cast<std::uintptr_t>(&filter_data[0]);

      // contains hash
      auto bucket_hash_v = dtl::hash::knuth_32<key_vt>::hash(key_v);
      auto bucket_idx_v = filter_t::block_t::get_bucket_idxs(bucket_hash_v);
      auto tag_v = (bucket_hash_v >> (32 - filter_t::table_t::bucket_addressing_bits - filter_t::table_t::tag_size_bits))
                   & static_cast<uint32_t>(filter_t::table_t::tag_mask);
      tag_v[tag_v == 0] += 1; // tag must not be zero
      auto alternative_bucket_idx_v = filter_t::block_t::get_alternative_bucket_idxs(bucket_idx_v, tag_v);

      const auto word_idx_v = bucket_idx_v & ((1u << filter_t::table_t::word_cnt_log2) - 1);
      const auto alternative_word_idx_v = alternative_bucket_idx_v & ((1u << filter_t::table_t::word_cnt_log2) - 1);
//      const auto in_word_bucket_idx = bucket_idx >> word_cnt_log2;
//      const auto bucket = word >> (bucket_size_bits * in_word_bucket_idx);

      const auto bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(word_idx_v << word_size_log2);
      const auto alternative_bucket_ptr_v = ptr_v + dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(alternative_word_idx_v << word_size_log2);

      // load the buckets
      const auto bucket_v = dtl::internal::vector_gather<$u64, $u64, vector_len>::gather(bucket_ptr_v);
      const auto alternative_bucket_v = dtl::internal::vector_gather<$u64, $u64, vector_len>::gather(alternative_bucket_ptr_v);

      auto dup_tag_v = dtl::internal::vector_convert<$u32, $u64, vector_len>::convert(tag_v | (tag_v << 16)) ;
      dup_tag_v |= dup_tag_v << 32;

      const auto b = reinterpret_cast<const r256*>(&bucket_v.data);
      const auto ba = reinterpret_cast<const r256*>(&alternative_bucket_v.data);
      auto t = reinterpret_cast<r256*>(&dup_tag_v.data);
      for (std::size_t i = 0; i < bucket_v.nested_vector_cnt; i++) {
        const r256 tags = t[i];
        const r256 bucket_content0 = b[i];
        const r256 bucket_content1 = ba[i];
        const r256 t0 = {.i = _mm256_cmpeq_epi16(bucket_content0.i, tags.i) };
        const r256 o0 = {.i = _mm256_cmpeq_epi16(bucket_content0.i, overflow_tag.i) };
        const r256 t1 = {.i = _mm256_cmpeq_epi16(bucket_content1.i, tags.i) };
        const r256 o1 = {.i = _mm256_cmpeq_epi16(bucket_content1.i, overflow_tag.i) };
        const r256 t2 = {.i = _mm256_or_si256(_mm256_or_si256(t0.i, o0.i), _mm256_or_si256(t1.i, o1.i)) };
        const r256 t3 = {.i = _mm256_cmpeq_epi64(t2.i, _mm256_setzero_si256()) };
        const auto mt = _mm256_movemask_pd(t3.d) ^ 0b1111;
        const r128 match_pos_vec = { .i = dtl::simd::lut_match_pos_4bit[mt].i };
        const r128 pos_vec = {.i = _mm_add_epi32(offset_vec.i, match_pos_vec.i) };
        _mm_storeu_si128(reinterpret_cast<__m128i*>(match_writer), pos_vec.i);
        match_writer += bits::pop_count(mt);
        offset_vec.i = _mm_add_epi32(offset_vec.i, _mm_set1_epi32(4));
      }

      reader += vector_len;
    }
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


} // namespace internal
} // namespace cuckoofilter
} // namespace dtl
