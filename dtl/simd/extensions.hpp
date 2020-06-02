//===----------------------------------------------------------------------===//
// SIMD extensions
//===----------------------------------------------------------------------===//

#pragma once

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>


namespace dtl {


//===----------------------------------------------------------------------===//
// mulhi_u32
//===----------------------------------------------------------------------===//

namespace simd {

/// Multiply the packed 32-bit integers in a and b, producing intermediate 64-bit integers,
/// and returns the high 32 bits of the intermediate integers.
template<typename primitive_t, typename vector_t, typename argument_t>
struct mulhi_u32 : dtl::simd::vector_fn<primitive_t, vector_t, argument_t> {};

#if !defined(__AVX2__)
template<>
struct mulhi_u32<$u32, $u32, $u32> : dtl::simd::vector_fn<$u32, $u32, $u32> {
  __forceinline__
  $u32 operator()(const $u32& a, const $u32& b) const noexcept {
    return dtl::mulhi_u32(a, b);
  }
};
#endif

#if defined(__AVX2__)
template<>
struct mulhi_u32<$u32, __m256i, __m256i> : dtl::simd::vector_fn<$u32, __m256i, __m256i> {
  __forceinline__
  __m256i operator()(const __m256i& a, const __m256i& b) const noexcept {
    return dtl::mulhi_u32(a, b);
  }
};
#endif // defined(__AVX2__)

#if defined(__AVX512F__)
template<>
struct mulhi_u32<$u32, __m512i, __m512i> : dtl::simd::vector_fn<$u32, __m512i, __m512i> {
  __forceinline__
  __m512i operator()(const __m512i& a, const __m512i& b) const noexcept {
    return dtl::mulhi_u32(a, b);
  }
};
#endif // defined(__AVX512F__)

} // namespace simd

template<typename Tv>
__forceinline__
static Tv
mulhi_u32(const Tv& a, const Tv& b) {
  using Fn = simd::mulhi_u32<typename Tv::scalar_type, typename Tv::nested_type, typename Tv::nested_type>;
  const Tv c = a.template map<Fn>(b);
  return c;
}

template<typename Tv>
__forceinline__
static Tv
mulhi_u32(const Tv& a, const typename Tv::scalar_type& b) {
  using Fn = simd::mulhi_u32<typename Tv::scalar_type, typename Tv::nested_type, typename Tv::nested_type>;
  const Tv c = a.template map<Fn>(b);
  return c;
}


//===----------------------------------------------------------------------===//
// Fast modulus
//===----------------------------------------------------------------------===//

template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
__forceinline__
static Tv
fast_mod_u32(const Tv& dividend, const fast_divisor_u32_t& divisor) {
  using MulHiFn = simd::mulhi_u32<typename Tv::scalar_type, typename Tv::nested_type, typename Tv::nested_type>;
  const Tv t1 = dividend.template map<MulHiFn>(divisor.magic);
  const Tv t2 = t1 >> divisor.shift_amount;
  const Tv t3 = t2 * divisor.divisor;
  const Tv remainder = dividend - t3;
  return remainder;
}


//===----------------------------------------------------------------------===//

} // namespace dtl

