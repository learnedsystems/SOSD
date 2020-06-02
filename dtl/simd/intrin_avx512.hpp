#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/intrin_avx512.hpp> directly; include <dtl/simd.hpp> instead."
#endif

#include <dtl/dtl.hpp>
#include <dtl/bits.hpp>

#include "immintrin.h"

namespace dtl {
namespace simd {

namespace internal {
namespace avx512 {

struct mask16 {
  __mmask16 data;
  __forceinline__ u1 all() const { return data == __mmask16(-1); };
  __forceinline__ u1 any() const { return data != __mmask16(0); };
  __forceinline__ u1 none() const { return data == __mmask16(0); };
  __forceinline__ void set(u1 value) {
    data = __mmask16(0) - value;
  }
  __forceinline__ void set(u64 idx, u1 value) {
    data = __mmask16(1) << idx;
  }
  __forceinline__ u1 get(u64 idx) const {
    return (data & (__mmask16(1) << idx)) != __mmask16(0);
  }
  __forceinline__ mask16 bit_and(const mask16& o) const { return mask16 { _mm512_kand(data, o.data) }; }
  __forceinline__ mask16 bit_or(const mask16& o) const { return mask16 { _mm512_kor(data, o.data) }; }
  __forceinline__ mask16 bit_xor(const mask16& o) const { return mask16 { _mm512_kxor(data, o.data) }; }
  __forceinline__ mask16 bit_not() const { return mask16 { _mm512_knot(data) }; }

  __forceinline__ $u64
  to_positions($u32* positions, $u32 offset) const {
    if (data == 0) return 0;
    static const __m512i sequence = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i seq = _mm512_add_epi64(sequence, _mm512_set1_epi32(offset));
    _mm512_mask_compressstoreu_epi32(positions, data, seq);
    return dtl::bits::pop_count(static_cast<u32>(data));
  }
};

//    __mmask8

} // avx512 namespace
} // internal namespace


namespace {
using mask16 = internal::avx512::mask16;
}


// --- vector types

template<>
struct vs<$i32, 8> : base<$i32, 8> {
  using type = __m256i;
  using mask_type = mask16;
  type data;
};

template<>
struct vs<$u32, 8> : base<$u32, 8> {
  using type = __m256i;
  using mask_type = mask16;
  type data;
};

template<>
struct vs<$i32, 16> : base<$i32, 16> {
  using type = __m512i;
  using mask_type = mask16;
  type data;
};

template<>
struct vs<$u32, 16> : base<$u32, 16> {
  using type = __m512i;
  using mask_type = mask16;
  type data;
};

template<>
struct vs<$i64, 8> : base<$i64, 8> {
  using type = __m512i;
  using mask_type = mask16;
  type data;
};

template<>
struct vs<$u64, 8> : base<$u64, 8> {
  using type = __m512i;
  using mask_type = mask16;
  type data;
};


// --- broadcast / set

#define __GENERATE(Tp, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct broadcast<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> { \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::value_type& a) const noexcept { \
    return IntrinFn(a);                                \
  }                                                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::value_type& a,         \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, a);            \
  }                                                    \
};

__GENERATE($i32, __m512i, $i32, _mm512_set1_epi32, _mm512_mask_set1_epi32)
__GENERATE($u32, __m512i, $u32, _mm512_set1_epi32, _mm512_mask_set1_epi32)
__GENERATE($i64, __m512i, $i64, _mm512_set1_epi64, _mm512_mask_set1_epi64)
__GENERATE($u64, __m512i, $u64, _mm512_set1_epi64, _mm512_mask_set1_epi64)
#undef __GENERATE


#define __GENERATE_BLEND(Tp, Tv, Ta, IntrinFnMask)     \
template<>                                             \
struct blend<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> {     \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::vector_type& a,        \
             const typename fn::vector_type& b,        \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(mask.data, a, b);              \
  }                                                    \
};

__GENERATE_BLEND($i32, __m512i, __m512i, _mm512_mask_blend_epi32)
__GENERATE_BLEND($u32, __m512i, __m512i, _mm512_mask_blend_epi32)
__GENERATE_BLEND($i64, __m512i, __m512i, _mm512_mask_blend_epi64)
__GENERATE_BLEND($u64, __m512i, __m512i, _mm512_mask_blend_epi64)
#undef __GENERATE


// --- Gather

#define __GENERATE(Tp, Scale, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct gather<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta, Tv> {\
  using fn = vector_fn<Tp, Tv, Ta, Tv>;                \
  using ptr = std::conditional<sizeof(Tp) == 8, long long int, std::make_signed<Tp>::type>::type; \
  __forceinline__ typename fn::vector_type                      \
  operator()(const Tp* const base_addr, const typename fn::argument_type& idx) const noexcept { \
    int scale = base_addr ? Scale : 1;                 \
    switch(scale) {                                    \
      case 1: return IntrinFn(idx, reinterpret_cast<const ptr*>(base_addr), 1); \
      case 2: return IntrinFn(idx, reinterpret_cast<const ptr*>(base_addr), 2); \
      case 4: return IntrinFn(idx, reinterpret_cast<const ptr*>(base_addr), 4); \
      case 8: return IntrinFn(idx, reinterpret_cast<const ptr*>(base_addr), 8); \
    }                                                  \
  }                                                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const Tp* const base_addr,                \
             const typename fn::argument_type& idx,    \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    int scale = base_addr ? Scale : 1;                 \
    switch(scale) {                                    \
      case 1: return IntrinFnMask(src, mask.data, idx, reinterpret_cast<const ptr*>(base_addr), 1); \
      case 2: return IntrinFnMask(src, mask.data, idx, reinterpret_cast<const ptr*>(base_addr), 2); \
      case 4: return IntrinFnMask(src, mask.data, idx, reinterpret_cast<const ptr*>(base_addr), 4); \
      case 8: return IntrinFnMask(src, mask.data, idx, reinterpret_cast<const ptr*>(base_addr), 8); \
    }                                                  \
  }                                                    \
};

// TODO implement type-extending and type-narrowing gathers. E.g. gather i32 values from i64 indices and vice versa
__GENERATE($i32, 4, __m512i, __m512i, _mm512_i32gather_epi32, _mm512_mask_i32gather_epi32)
__GENERATE($u32, 4, __m512i, __m512i, _mm512_i32gather_epi32, _mm512_mask_i32gather_epi32)
__GENERATE($i64, 8, __m512i, __m512i, _mm512_i64gather_epi64, _mm512_mask_i64gather_epi64)
__GENERATE($u64, 8, __m512i, __m512i, _mm512_i64gather_epi64, _mm512_mask_i64gather_epi64)
#undef __GENERATE


// --- Store

#define __GENERATE(Tp, Scale, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct scatter<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> {   \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  using ptr = std::conditional<sizeof(Tp) == 8, long long int, std::make_signed<Tp>::type>::type; \
  __forceinline__ void                                          \
  operator()(Tp* base_addr,                            \
             const typename fn::vector_type& idx,      \
             const typename fn::vector_type& a) const noexcept { \
    IntrinFn(reinterpret_cast<ptr*>(base_addr), idx, a, Scale); \
  }                                                    \
  __forceinline__ void                                          \
  operator()(Tp* base_addr,                            \
             const typename fn::vector_type& idx,      \
             const typename fn::vector_type& a,        \
             const mask16 mask) const noexcept {       \
    IntrinFnMask(reinterpret_cast<ptr*>(base_addr), mask.data, idx, a, Scale); \
  }                                                    \
};

// TODO ???
__GENERATE($i32, 4, __m512i, __m512i, _mm512_i32scatter_epi32, _mm512_mask_i32scatter_epi32)
__GENERATE($u32, 4, __m512i, __m512i, _mm512_i32scatter_epi32, _mm512_mask_i32scatter_epi32)
__GENERATE($i64, 8, __m512i, __m512i, _mm512_i64scatter_epi64, _mm512_mask_i64scatter_epi64)
__GENERATE($u64, 8, __m512i, __m512i, _mm512_i64scatter_epi64, _mm512_mask_i64scatter_epi64)
#undef __GENERATE


// --- Arithmetic

#define __GENERATE_ARITH(Op, Tp, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct Op<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> {        \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs) const noexcept { \
    return IntrinFn(lhs, rhs);                         \
  }                                                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs,      \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, lhs, rhs);     \
  }                                                    \
};

inline __m512i
_mm512_mul_epi64(const __m512i& lhs, const __m512i& rhs) {
  const __m512i hi_lhs = _mm512_srli_epi64(lhs, 32);
  const __m512i hi_rhs = _mm512_srli_epi64(rhs, 32);
  const __m512i t1 = _mm512_mul_epu32(lhs, hi_rhs);
  const __m512i t2 = _mm512_mul_epu32(lhs, rhs);
  const __m512i t3 = _mm512_mul_epu32(hi_lhs, rhs);
  const __m512i t4 = _mm512_add_epi64(_mm512_slli_epi64(t3, 32), t2);
  const __m512i t5 = _mm512_add_epi64(_mm512_slli_epi64(t1, 32), t4);
  return t5;
}

inline __m512i
_mm512_mask_mul_epi64(const __m512i& src, const __mmask16 k,
                      const __m512i& lhs, const __m512i& rhs) {
  const __m512i hi_lhs = _mm512_srli_epi64(lhs, 32);
  const __m512i hi_rhs = _mm512_srli_epi64(rhs, 32);
  const __m512i t1 = _mm512_mul_epu32(lhs, hi_rhs);
  const __m512i t2 = _mm512_mul_epu32(lhs, rhs);
  const __m512i t3 = _mm512_mul_epu32(hi_lhs, rhs);
  const __m512i t4 = _mm512_add_epi64(_mm512_slli_epi64(t3, 32), t2);
  const __m512i t5 = _mm512_add_epi64(_mm512_slli_epi64(t1, 32), t4);
  return _mm512_mask_blend_epi64(k, src, t5);
}


__GENERATE_ARITH(plus, $i32, __m512i, $i32, _mm512_add_epi32, _mm512_mask_add_epi32)
__GENERATE_ARITH(plus, $u32, __m512i, $u32, _mm512_add_epi32, _mm512_mask_add_epi32)
__GENERATE_ARITH(plus, $i64, __m512i, $i64, _mm512_add_epi64, _mm512_mask_add_epi64)
__GENERATE_ARITH(plus, $u64, __m512i, $u64, _mm512_add_epi64, _mm512_mask_add_epi64)
__GENERATE_ARITH(minus, $i32, __m512i, $i32, _mm512_sub_epi32, _mm512_mask_sub_epi32)
__GENERATE_ARITH(minus, $u32, __m512i, $u32, _mm512_sub_epi32, _mm512_mask_sub_epi32)
__GENERATE_ARITH(minus, $i64, __m512i, $i64, _mm512_sub_epi64, _mm512_mask_sub_epi64)
__GENERATE_ARITH(minus, $u64, __m512i, $u64, _mm512_sub_epi64, _mm512_mask_sub_epi64)
__GENERATE_ARITH(multiplies, $i32, __m512i, $i32, _mm512_mullo_epi32, _mm512_mask_mullo_epi32)
__GENERATE_ARITH(multiplies, $u32, __m512i, $u32, _mm512_mullo_epi32, _mm512_mask_mullo_epi32)
__GENERATE_ARITH(multiplies, $i64, __m512i, $i64, _mm512_mul_epi64, _mm512_mask_mul_epi64) // TODO fix
__GENERATE_ARITH(multiplies, $u64, __m512i, $u64, _mm512_mul_epi64, _mm512_mask_mul_epi64) // TODO fix
#undef __GENERATE_ARITH


// --- Shift

#define __GENERATE_SHIFT(Op, Tp, Tv, Ta, IntrinFn, IntrinFnMask) \
template<>                                             \
struct Op<Tp, Tv, Ta> : vector_fn<Tp, Tv, Ta> {        \
  using fn = vector_fn<Tp, Tv, Ta>;                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::vector_type& a,        \
             const typename fn::argument_type& count) const noexcept { \
    return IntrinFn(a, count);                         \
  }                                                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::vector_type& a,        \
             const typename fn::argument_type& count,  \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, a, count);     \
  }                                                    \
};

__GENERATE_SHIFT(shift_left, $i32, __m512i, $i32, _mm512_slli_epi32, _mm512_mask_slli_epi32)
__GENERATE_SHIFT(shift_left, $u32, __m512i, $u32, _mm512_slli_epi32, _mm512_mask_slli_epi32)
__GENERATE_SHIFT(shift_right, $i32, __m512i, $i32, _mm512_srli_epi32, _mm512_mask_srli_epi32)
__GENERATE_SHIFT(shift_right, $u32, __m512i, $u32, _mm512_srli_epi32, _mm512_mask_srli_epi32)
__GENERATE_SHIFT(shift_left_var, $i32, __m512i, __m512i, _mm512_sllv_epi32, _mm512_mask_sllv_epi32)
__GENERATE_SHIFT(shift_left_var, $u32, __m512i, __m512i, _mm512_sllv_epi32, _mm512_mask_sllv_epi32)
__GENERATE_SHIFT(shift_right_var, $i32, __m512i, __m512i, _mm512_srlv_epi32, _mm512_mask_srlv_epi32)
__GENERATE_SHIFT(shift_right_var, $u32, __m512i, __m512i, _mm512_srlv_epi32, _mm512_mask_srlv_epi32)

__GENERATE_SHIFT(shift_left, $i64, __m512i, $i64, _mm512_slli_epi64, _mm512_mask_slli_epi64)
__GENERATE_SHIFT(shift_left, $u64, __m512i, $u64, _mm512_slli_epi64, _mm512_mask_slli_epi64)
__GENERATE_SHIFT(shift_right, $i64, __m512i, $i64, _mm512_srli_epi64, _mm512_mask_srli_epi64)
__GENERATE_SHIFT(shift_right, $u64, __m512i, $u64, _mm512_srli_epi64, _mm512_mask_srli_epi64)
__GENERATE_SHIFT(shift_left_var, $i64, __m512i, __m512i, _mm512_sllv_epi64, _mm512_mask_sllv_epi64)
__GENERATE_SHIFT(shift_left_var, $u64, __m512i, __m512i, _mm512_sllv_epi64, _mm512_mask_sllv_epi64)
__GENERATE_SHIFT(shift_right_var, $i64, __m512i, __m512i, _mm512_srlv_epi64, _mm512_mask_srlv_epi64)
__GENERATE_SHIFT(shift_right_var, $u64, __m512i, __m512i, _mm512_srlv_epi64, _mm512_mask_srlv_epi64)
#undef __GENERATE_SHIFT


// --- Bitwise operators

#define __GENERATE_BITWISE(Op, Tp, Tv, IntrinFn, IntrinFnMask) \
template<>                                             \
struct Op<Tp, Tv> : vector_fn<Tp, Tv> {                \
  using fn = vector_fn<Tp, Tv>;                        \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs) const noexcept { \
    return IntrinFn(lhs, rhs);                         \
  }                                                    \
  __forceinline__ typename fn::vector_type                      \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs,      \
             const typename fn::vector_type& src,      \
             const mask16 mask) const noexcept {       \
    return IntrinFnMask(src, mask.data, lhs, rhs);     \
  }                                                    \
};

// TODO find a better way to implement bitwise operations
__GENERATE_BITWISE(bit_and, $i32, __m512i, _mm512_and_epi32, _mm512_mask_and_epi32)
__GENERATE_BITWISE(bit_and, $u32, __m512i, _mm512_and_epi32, _mm512_mask_and_epi32)
__GENERATE_BITWISE(bit_or,  $i32, __m512i, _mm512_or_epi32,  _mm512_mask_or_epi32)
__GENERATE_BITWISE(bit_or,  $u32, __m512i, _mm512_or_epi32,  _mm512_mask_or_epi32)
__GENERATE_BITWISE(bit_xor, $i32, __m512i, _mm512_xor_epi32, _mm512_mask_xor_epi32)
__GENERATE_BITWISE(bit_xor, $u32, __m512i, _mm512_xor_epi32, _mm512_mask_xor_epi32)

__GENERATE_BITWISE(bit_and, $i64, __m512i, _mm512_and_epi64, _mm512_mask_and_epi64)
__GENERATE_BITWISE(bit_and, $u64, __m512i, _mm512_and_epi64, _mm512_mask_and_epi64)
__GENERATE_BITWISE(bit_or,  $i64, __m512i, _mm512_or_epi64,  _mm512_mask_or_epi64)
__GENERATE_BITWISE(bit_or,  $u64, __m512i, _mm512_or_epi64,  _mm512_mask_or_epi64)
__GENERATE_BITWISE(bit_xor, $i64, __m512i, _mm512_xor_epi64, _mm512_mask_xor_epi64)
__GENERATE_BITWISE(bit_xor, $u64, __m512i, _mm512_xor_epi64, _mm512_mask_xor_epi64)
#undef __GENERATE_BITWISE


// --- Comparison

#define __GENERATE_CMP(Op, Tp, Tv, IntrinFn, IntrinFnMask) \
template<>                                             \
struct Op<Tp, Tv, Tv, mask16> : vector_fn<Tp, Tv, Tv, mask16> { \
  using fn = vector_fn<Tp, Tv, Tv, mask16>;            \
  __forceinline__ mask16                                        \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs) const noexcept { \
    return mask16{ IntrinFn(lhs, rhs) };               \
  }                                                    \
  __forceinline__ mask16                                        \
  operator()(const typename fn::vector_type& lhs,      \
             const typename fn::vector_type& rhs,      \
             const mask16 mask) const noexcept {       \
    return mask16{ IntrinFnMask(mask.data, lhs, rhs) }; \
  }                                                    \
};

#define __GENERATE_NE(T) \
inline __mmask16 _mm512_cmpne_##T##_mask(__m512i a, __m512i b) { \
  const __mmask16 eq = _mm512_cmpeq_##T##_mask(a, b);            \
  return _mm512_knot(eq);                                        \
}                                                                \
inline __mmask16 _mm512_mask_cmpne_##T##_mask(__mmask16 k1, __m512i a, __m512i b) { \
  const __mmask16 eq = _mm512_mask_cmpeq_##T##_mask(k1, a, b);   \
  return _mm512_kand(k1, _mm512_knot(eq));                       \
}

__GENERATE_NE(epi32)
__GENERATE_NE(epu32)
__GENERATE_NE(epi64)
__GENERATE_NE(epu64)
#undef __GENERATE_NE

__GENERATE_CMP(less,          $i32, __m512i, _mm512_cmplt_epi32_mask, _mm512_mask_cmplt_epi32_mask)
__GENERATE_CMP(less_equal,    $i32, __m512i, _mm512_cmple_epi32_mask, _mm512_mask_cmple_epi32_mask)
__GENERATE_CMP(greater,       $i32, __m512i, _mm512_cmpgt_epi32_mask, _mm512_mask_cmpgt_epi32_mask)
__GENERATE_CMP(greater_equal, $i32, __m512i, _mm512_cmpge_epi32_mask, _mm512_mask_cmpge_epi32_mask)
__GENERATE_CMP(equal,         $i32, __m512i, _mm512_cmpeq_epi32_mask, _mm512_mask_cmpeq_epi32_mask)
__GENERATE_CMP(not_equal,     $i32, __m512i, _mm512_cmpne_epi32_mask, _mm512_mask_cmpne_epi32_mask)

__GENERATE_CMP(less,          $u32, __m512i, _mm512_cmplt_epu32_mask, _mm512_mask_cmplt_epu32_mask)
__GENERATE_CMP(less_equal,    $u32, __m512i, _mm512_cmple_epu32_mask, _mm512_mask_cmple_epu32_mask)
__GENERATE_CMP(greater,       $u32, __m512i, _mm512_cmpgt_epu32_mask, _mm512_mask_cmpgt_epu32_mask)
__GENERATE_CMP(greater_equal, $u32, __m512i, _mm512_cmpge_epu32_mask, _mm512_mask_cmpge_epu32_mask)
__GENERATE_CMP(equal,         $u32, __m512i, _mm512_cmpeq_epu32_mask, _mm512_mask_cmpeq_epu32_mask)
__GENERATE_CMP(not_equal,     $u32, __m512i, _mm512_cmpne_epu32_mask, _mm512_mask_cmpne_epu32_mask)

__GENERATE_CMP(less,          $i64, __m512i, _mm512_cmplt_epi64_mask, _mm512_mask_cmplt_epi64_mask)
__GENERATE_CMP(less_equal,    $i64, __m512i, _mm512_cmple_epi64_mask, _mm512_mask_cmple_epi64_mask)
__GENERATE_CMP(greater,       $i64, __m512i, _mm512_cmpgt_epi64_mask, _mm512_mask_cmpgt_epi64_mask)
__GENERATE_CMP(greater_equal, $i64, __m512i, _mm512_cmpge_epi64_mask, _mm512_mask_cmpge_epi64_mask)
__GENERATE_CMP(equal,         $i64, __m512i, _mm512_cmpeq_epi64_mask, _mm512_mask_cmpeq_epi64_mask)
__GENERATE_CMP(not_equal,     $i64, __m512i, _mm512_cmpne_epi64_mask, _mm512_mask_cmpne_epi64_mask)

__GENERATE_CMP(less,          $u64, __m512i, _mm512_cmplt_epu64_mask, _mm512_mask_cmplt_epu64_mask)
__GENERATE_CMP(less_equal,    $u64, __m512i, _mm512_cmple_epu64_mask, _mm512_mask_cmple_epu64_mask)
__GENERATE_CMP(greater,       $u64, __m512i, _mm512_cmpgt_epu64_mask, _mm512_mask_cmpgt_epu64_mask)
__GENERATE_CMP(greater_equal, $u64, __m512i, _mm512_cmpge_epu64_mask, _mm512_mask_cmpge_epu64_mask)
__GENERATE_CMP(equal,         $u64, __m512i, _mm512_cmpeq_epu64_mask, _mm512_mask_cmpeq_epu64_mask)
__GENERATE_CMP(not_equal,     $u64, __m512i, _mm512_cmpne_epu64_mask, _mm512_mask_cmpne_epu64_mask)

#undef __GENERATE_CMP


} // namespace simd
} // namespace dtl