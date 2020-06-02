#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/intrinsics.hpp> directly; include <dtl/simd.h> instead."
#endif

#include "../adept.hpp"
#include "vec.hpp"

namespace dtl {
namespace simd {

// helper
namespace {

template<typename Tv, class enable = void>
struct deduce_mask {
  using type = bool;
};

template<typename Tv>
struct deduce_mask<Tv, typename std::enable_if<!std::is_integral<Tv>::value>::type> {
  using type = typename Tv::mask;
};

}

/// Base class for vectorized function objects. (= functions that operate on packed data)
template<typename Tp, typename Tv = Tp, typename Ta = Tp, typename Tr = Tv>
struct vector_fn {
  using value_type = Tp; // the primitive (scalar) type, e.g., u32.
  using vector_type = Tv; // the native vector type, e.g. __m256i
  using argument_type = Ta; // TODO
  using result_type = Tr; // the return type (defaults to the vector type) // TODO: remove?
  // Note: the mask is of type bool, iff the vector type is a fundamental type
  //using mask_type = typename deduce_mask<Tv>::type;
};


// Sets all vector components to the given scalar value (broadcasts a value to all SIMD lanes)
template<typename primitive_t, typename vector_t = primitive_t, typename argument_t = primitive_t>
struct broadcast : vector_fn<primitive_t, vector_t, argument_t> {};

// Sets the vectors' components to the given values (basically a copy, if no mask is applied)
template<typename primitive_t, typename vector_t = primitive_t, typename argument_t = vector_t>
struct set : vector_fn<primitive_t, vector_t, argument_t> {};

template<typename primitive_t, typename vector_t, typename argument_t = vector_t>
struct blend : vector_fn<primitive_t, vector_t, argument_t> {};


// Load
template<typename primitive_t, typename vector_t, typename argument_t>
struct gather : vector_fn<primitive_t, vector_t, argument_t> {};


// Store
template<typename primitive_t, typename vector_t, typename argument_t>
struct scatter : vector_fn<primitive_t, vector_t> {};


// Arithmetic
template<typename primitive_t, typename vector_t, typename argument_t = primitive_t>
struct plus : vector_fn<primitive_t, vector_t> {};

template<typename primitive_t, typename vector_t, typename argument_t = primitive_t>
struct minus : vector_fn<primitive_t, vector_t> {};

template<typename primitive_t, typename vector_t, typename argument_t = primitive_t>
struct multiplies : vector_fn<primitive_t, vector_t> {};


// Shift
template<typename primitive_t, typename vector_t, typename argument_t>
struct shift_left : vector_fn<primitive_t, vector_t, i32> {};

template<typename primitive_t, typename vector_t, typename argument_t>
struct shift_left_var : vector_fn<primitive_t, vector_t, vector_t> {};

template<typename primitive_t, typename vector_t, typename argument_t>
struct shift_right : vector_fn<primitive_t, vector_t, i32> {};

template<typename primitive_t, typename vector_t, typename argument_t>
struct shift_right_var : vector_fn<primitive_t, vector_t, vector_t> {};


// Bitwise
template<typename primitive_t, typename vector_t, typename argument_t = primitive_t>
struct bit_and : vector_fn<void, vector_t> {};

template<typename primitive_t, typename vector_t, typename argument_t = primitive_t>
struct bit_or : vector_fn<primitive_t, vector_t> {};

template<typename primitive_t, typename vector_t, typename argument_t = primitive_t>
struct bit_xor : vector_fn<primitive_t, vector_t> {};

template<typename primitive_t, typename vector_t, typename argument_t = primitive_t>
struct bit_not : vector_fn<primitive_t, vector_t> {};


// Comparison (Note: the return type of a comparison is a mask)
template<typename primitive_t, typename vector_t, typename argument_t, typename return_t>
struct less : vector_fn<primitive_t, vector_t, argument_t, return_t> {};

template<typename primitive_t, typename vector_t, typename argument_t, typename return_t>
struct less_equal : vector_fn<primitive_t, vector_t, argument_t, return_t> {};

template<typename primitive_t, typename vector_t, typename argument_t, typename return_t>
struct greater : vector_fn<primitive_t, vector_t, argument_t, return_t> {};

template<typename primitive_t, typename vector_t, typename argument_t, typename return_t>
struct greater_equal : vector_fn<primitive_t, vector_t, argument_t, return_t> {};

template<typename primitive_t, typename vector_t, typename argument_t, typename return_t>
struct equal : vector_fn<primitive_t, vector_t, argument_t, return_t> {};

template<typename primitive_t, typename vector_t, typename argument_t, typename return_t>
struct not_equal : vector_fn<primitive_t, vector_t, argument_t, return_t> {};

} // namespace simd
} // namespace dtl

#if defined(__AVX512F__)
#include "intrin_avx512.hpp"
#endif

#if defined(__AVX2__) && !defined(__AVX512F__)
#include "intrin_avx2.hpp"
#endif

// SSE not yet supported
//#if defined(__SSE2__) && !defined(__AVX2__) && !defined(__AVX512F__)
//#include "intrin_sse.hpp"
//#endif

#if !defined(__AVX2__) && !defined(__AVX512F__)
#include "intrin_x64.hpp"
#endif