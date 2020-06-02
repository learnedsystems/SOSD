#pragma once

#include <limits>

#include <dtl/dtl.hpp>
//#include <dtl/simd.hpp>

//#include "immintrin.h"

// TODO remove dependency (libdivide is only used to find magic numbers)
#include "thirdparty/libdivide/libdivide.h"


namespace dtl {

static constexpr uint32_t max_divisor_u32 = std::numeric_limits<uint32_t>::max() - 2;

struct fast_divisor_u32_t {
  uint32_t magic;
  uint32_t shift_amount;
  uint32_t divisor;
};


/// Finds a magic number (which is not a power of two) to divide by n.
/// Thereby, only magic number are considered that do not require an 'addition' step. (cheap)
static fast_divisor_u32_t
next_cheap_magic(uint32_t n) {
  n = std::max(n, 2u);

  const uint32_t d_max = max_divisor_u32;
  assert(n <= d_max);

  // TODO remove redundancy
  enum {
    LIBDIVIDE_32_SHIFT_MASK = 0x1F,
    LIBDIVIDE_64_SHIFT_MASK = 0x3F,
    LIBDIVIDE_ADD_MARKER = 0x40,
    LIBDIVIDE_U32_SHIFT_PATH = 0x80,
    LIBDIVIDE_U64_SHIFT_PATH = 0x80,
    LIBDIVIDE_S32_SHIFT_PATH = 0x20,
    LIBDIVIDE_NEGATIVE_DIVISOR = 0x80
  };

  for (uint32_t d = n; d <= d_max; d++) {
    auto div = libdivide::libdivide_u32_gen(d);
    auto add_indicator = ((div.more & LIBDIVIDE_ADD_MARKER) != 0);
    if (add_indicator) { continue; }
    if (div.more & LIBDIVIDE_U32_SHIFT_PATH) { continue; }
    // only "algo 1"
    assert(d >= n);
    return {div.magic, (div.more & static_cast<uint32_t>(LIBDIVIDE_32_SHIFT_MASK)), d};
  }
  throw "Failed to find a cheap magic number.";
}

//===----------------------------------------------------------------------===//
// Scalar
//===----------------------------------------------------------------------===//

__forceinline__ __host__ __device__
static uint32_t
mulhi_u32(const uint32_t a, const uint32_t b) {
  const uint64_t prod = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
  return static_cast<uint32_t>(prod >> 32);
}

// works only for "cheap" magic numbers
__forceinline__ __host__ __device__
static uint32_t
fast_div_u32(const uint32_t dividend, const uint32_t magic, const uint32_t shift_amount) {
  const uint32_t quotient = mulhi_u32(dividend, magic);
  return quotient >> shift_amount;
}

__forceinline__ __host__ __device__
static uint32_t
fast_div_u32(const uint32_t dividend, const fast_divisor_u32_t& divisor) {
  return fast_div_u32(dividend, divisor.magic, divisor.shift_amount);
}

__forceinline__ __host__ __device__
static uint32_t
fast_mod_u32(const uint32_t dividend, const fast_divisor_u32_t& divisor) {
  return dividend - fast_div_u32(dividend, divisor.magic, divisor.shift_amount) * divisor.divisor;
}


//===----------------------------------------------------------------------===//
// AVX-512
//===----------------------------------------------------------------------===//

#if defined(__AVX512F__)

__forceinline__
static __m512i
mulhi_u32(const __m512i a, const uint32_t b) {
  const __m512i b_v = _mm512_set1_epi32(b);
  // shift 32-bit input to 64-bit lanes
  const __m512i a_odd_u64 = _mm512_srli_epi64(a, 32);
  const __m512i a_even_u64 = a;
  // multiply
  const __m512i p_odd_u64 = _mm512_mul_epu32(a_odd_u64, b_v);
  const __m512i p_even_u64 =_mm512_mul_epu32(a_even_u64, b_v);
  // merge the higher 32-bits of products back into a single ZMM register
  const __m512i p_even_hi_u64 = _mm512_srli_epi64(p_even_u64, 32);
  return _mm512_mask_mov_epi32(p_odd_u64, __mmask16(0b0101010101010101), p_even_hi_u64);
}

__forceinline__
static __m512i
mulhi_u32(const __m512i a, const __m512i b) {
  // shift 32-bit input to 64-bit lanes
  const __m512i a_odd_u64 = _mm512_srli_epi64(a, 32);
  const __m512i a_even_u64 = a;
  const __m512i b_odd_u64 = _mm512_srli_epi64(b, 32);
  const __m512i b_even_u64 = b;
  // multiply
  const __m512i p_odd_u64 = _mm512_mul_epu32(a_odd_u64, b_odd_u64);
  const __m512i p_even_u64 =_mm512_mul_epu32(a_even_u64, b_even_u64);
  // merge the higher 32-bits of products back into a single ZMM register
  const __m512i p_even_hi_u64 = _mm512_srli_epi64(p_even_u64, 32);
  return _mm512_mask_mov_epi32(p_odd_u64, __mmask16(0b0101010101010101), p_even_hi_u64);
}

// works only for "cheap" magic numbers
__forceinline__
static __m512i
fast_div_u32(const __m512i dividend, const uint32_t magic, const uint32_t shift_amount) {
  const __m512i quotient = mulhi_u32(dividend, magic);
  return _mm512_srli_epi32(quotient, shift_amount);
}

__forceinline__
static __m512i
fast_div_u32(const __m512i dividend, const __m512i magic, const __m512i shift_amount) {
  const __m512i quotient = mulhi_u32(dividend, magic);
  return _mm512_srlv_epi32(quotient, shift_amount);
}

__forceinline__
static __m512i
fast_div_u32(const __m512i dividend, const fast_divisor_u32_t& divisor) {
  return fast_div_u32(dividend, divisor.magic, divisor.shift_amount);
}

__forceinline__
static __m512i
fast_mod_u32(const __m512i dividend, const fast_divisor_u32_t& divisor) {
  const __m512i floor_div = fast_div_u32(dividend, divisor.magic, divisor.shift_amount);
  const __m512i t = _mm512_mullo_epi32(floor_div, _mm512_set1_epi32(divisor.divisor));
  return dividend - t ;
}

#endif // defined(__AVX512F__)


//===----------------------------------------------------------------------===//
// AVX2
//===----------------------------------------------------------------------===//

#if defined(__AVX2__)

__forceinline__
static __m256i
mulhi_u32(const __m256i a, const uint32_t b) {
  const __m256i b_v = _mm256_set1_epi32(b);
  // shift 32-bit input to 64-bit lanes
  const __m256i a_odd_u64 = _mm256_srli_epi64(a, 32);
  const __m256i a_even_u64 = a;
  // multiply
  const __m256i p_odd_u64 = _mm256_mul_epu32(a_odd_u64, b_v);
  const __m256i p_even_u64 =_mm256_mul_epu32(a_even_u64, b_v);
  // merge the higher 32-bits of products back into a single ZMM register
  const __m256i p_even_hi_u64 = _mm256_srli_epi64(p_even_u64, 32);
  return _mm256_blend_epi32(p_odd_u64, p_even_hi_u64, 0b01010101);
}

__forceinline__
static __m256i
mulhi_u32(const __m256i a, const __m256i b) {
  // shift 32-bit input to 64-bit lanes
  const __m256i a_odd_u64 = _mm256_srli_epi64(a, 32);
  const __m256i a_even_u64 = a;
  const __m256i b_odd_u64 = _mm256_srli_epi64(b, 32);
  const __m256i b_even_u64 = b;
  // multiply
  const __m256i p_odd_u64 = _mm256_mul_epu32(a_odd_u64, b_odd_u64);
  const __m256i p_even_u64 =_mm256_mul_epu32(a_even_u64, b_even_u64);
  // merge the higher 32-bits of products back into a single YMM register
  const __m256i p_even_hi_u64 = _mm256_srli_epi64(p_even_u64, 32);
  return _mm256_blend_epi32(p_odd_u64, p_even_hi_u64, 0b01010101);
}

// works only for "cheap" magic numbers
__forceinline__
static __m256i
fast_div_u32(const __m256i dividend, const uint32_t magic, const uint32_t shift_amount) {
  const __m256i quotient = mulhi_u32(dividend, magic);
  return _mm256_srli_epi32(quotient, shift_amount);
}

__forceinline__
static __m256i
fast_div_u32(const __m256i dividend, const __m256i magic, const __m256i shift_amount) {
  const __m256i quotient = mulhi_u32(dividend, magic);
  return _mm256_srlv_epi32(quotient, shift_amount);
}

__forceinline__
static __m256i
fast_div_u32(const __m256i dividend, const fast_divisor_u32_t& divisor) {
  return fast_div_u32(dividend, divisor.magic, divisor.shift_amount);
}

__forceinline__
static __m256i
fast_mod_u32(const __m256i dividend, const fast_divisor_u32_t& divisor) {
  const __m256i floor_div = fast_div_u32(dividend, divisor.magic, divisor.shift_amount);
  const __m256i t = _mm256_mullo_epi32(floor_div, _mm256_set1_epi32(divisor.divisor));
  return dividend - t;
}

#endif // defined(__AVX2__)


} // namespace dtl
