#pragma once

#ifndef _DTL_SIMD_INCLUDED
#error "Never use <dtl/simd/vec.hpp> directly; include <dtl/simd.hpp> instead."
#endif

#include "../adept.hpp"
#include "../math.hpp"

#include <array>
#include <bitset>
#include <functional>
#include <type_traits>

namespace dtl {
namespace simd {

/// Base class for a vector consisting of N primitive values of type Tp
template<typename Tp, u64 N>
struct base {
  using type = Tp;
  static constexpr u64 value = N;
  static constexpr u64 length = N;
};

/// Recursive template to find the largest possible (native) vector implementation.
template<typename Tp, u64 N>
struct vs : vs<Tp, N / 2> {
  static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");
};

} // namespace simd
} // namespace dtl

// include architecture dependent implementations...
#include "intrinsics.hpp"

namespace dtl {
namespace simd {

/*
/// smallest vector type, consisting of 1 component (used when no specialization is found)
template<typename T>
struct vs<T, 1> {
  static constexpr u64 value = 1;
  using type = vec<T, 1>;
  type data;
};
*/

/*
template<typename T>
struct vs<T, 32> {
  static constexpr u64 value = 32;
  using type = vec<T, 32>;
  type data;
};
*/

struct v_base {};

template<class T>
struct is_vector {
  static constexpr bool value = std::is_base_of<v_base, T>::value;
};



/// The general vector class with N components of the (primitive) type Tp.
///
/// If there exists a native vector type that can hold N values of type Tp, e.g. __m256i,
/// then an instance makes direct use of it. If the N exceeds the size of the largest
/// available native vector type an instance will be a composition of multiple (smaller)
/// native vectors.
template<typename Tp, u64 N>
struct v : v_base {

  static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");
  // TODO assert fundamental type
  // TODO unroll loops in compound types - __attribute__((optimize("unroll-loops")))

  /// The scalar type
  using scalar_type = typename std::remove_cv<Tp>::type;

  /// The overall length of the vector, in terms of number of elements.
  static constexpr u64 length = N;

  /// The native vector wrapper that is used under the hood.
  /// Note: The wrapper determines the largest available native vector type.
  using nested_vector = vs<scalar_type, N>; // TODO (maybe) make it a template parameter. give the user the possibility to specify the native vector type

  /// The alignment of the vector.
  static constexpr std::size_t byte_alignment = std::alignment_of<typename nested_vector::type>::value;

  /// The length of the native vector, in terms of number of elements.
  static constexpr u64 nested_vector_length = nested_vector::value;

  /// The number of nested native vectors, if the vector is a composition of multiple smaller vectors, 1 otherwise.
  static constexpr u64 nested_vector_cnt = N / nested_vector_length;

  /// True, if the vector is a composition of multiple native vectors, false otherwise.
  static constexpr u1 is_compound = (nested_vector_cnt != 1);

  /// The native vector type (e.g., __m256i).
  using nested_type = typename nested_vector::type;

  /// Helper to typedef a compound type.
  template<typename T_inner, u64 Cnt>
  using make_compound = typename std::array<T_inner, Cnt>;

  /// The compound vector type. Note: Is the same as nested_type, if not compound.
  using compound_type = typename std::conditional<is_compound,
      make_compound<nested_type, nested_vector_cnt>,
      nested_type>::type;

  /// The actual vector data. (the one and only non-static member variable of this class).
  compound_type data;

  /// The native 'mask' type of the surrounding vector.
  using nested_mask_type = typename nested_vector::mask_type;

  /// The 'mask' type is a composition if the surrounding vector is also a composition.
  using compound_mask_type = typename std::conditional<is_compound,
      make_compound<nested_mask_type, nested_vector_cnt>,
      nested_mask_type>::type;

  //===----------------------------------------------------------------------===//

  /// The mask type (template) of the surrounding vector.
  ///
  /// As the vector can be a composition of multiple (smaller) native
  /// vectors, the same applies for the mask type.
  /// Note, that the mask implementations are architecture dependent.
  ///
  /// General rules for working with masks:
  ///  1) Mask are preferably created by comparison operations and used
  ///     with masked vector operations. Manual construction should be avoided.
  ///  2) Avoid materialization. Instances should have a very short lifetime
  ///     and are not supposed to be stored in main-memory. Use the 'to_int'
  ///     function to obtain a bitmask represented as an integer.
  ///  3) Avoid (costly) direct access through the set/get functions. On pre-KNL
  ///     architectures this has a severe performance impact.
  ///  4) The special functions 'all', 'any' and 'none' are supposed to be
  ///     fast and efficient on all architectures. - Semantics are equal to
  ///     the std::bitset implementations.
  ///  5) Bitwise operations are supposed to be fast and efficient on all
  ///     architectures.
  struct m {

    /// The actual mask data. (the one and only non-static member variable of this class)
    compound_mask_type data;

    m() {
      set<is_compound>(this->data, false);
    }
    m(u32 i) {
      set<is_compound>(this->data, i);
    }
    m(const m&) = default;
    m(m&&) = default;
    m(compound_mask_type&& d) : data { std::move(d) } {};
    m& operator=(const m&) = default;
    m& operator=(m&&) = default;

    struct all_fn {
      constexpr u1 operator()(const nested_mask_type& mask) const {
        return mask.all();
      }
      constexpr u1 aggr(u1 a, u1 b) const { return a & b; };
    };

    struct any_fn {
      constexpr u1 operator()(const nested_mask_type& mask) const {
        return mask.any();
      }
      constexpr u1 aggr(u1 a, u1 b) const { return a | b; };
    };

    struct none_fn {
      constexpr u1 operator()(const nested_mask_type& mask) const {
        return mask.none();
      }
      constexpr u1 aggr(u1 a, u1 b) const { return a & b; };
    };

    template<u1 Compound = false, typename Fn>
    static __forceinline__ u1
    op(Fn fn, const nested_mask_type& mask) {
      return fn(mask);
    }

    template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
    static __forceinline__ u1
    op(Fn fn, const compound_mask_type& masks) {
      $u1 result = op<!Compound>(fn, masks[0]);
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        result = fn.aggr(result, op<!Compound>(fn, masks[i]));
      }
      return result;
    }

    /// Returns true if all boolean values in the mask are true, false otherwise.
    u1 all() const { return op<is_compound>(all_fn(), data); }

    /// Returns true if at least one boolean value in the mask is true, false otherwise.
    u1 any() const { return op<is_compound>(any_fn(), data); }

    /// Returns true if all boolean values in the mask are false, false otherwise.
    u1 none() const { return op<is_compound>(none_fn(), data); }

    /// Sets the bit a position 'idx' to the given 'value'.
    template<u1 Compound = false>
    static __forceinline__ void
    set(nested_mask_type& mask, u64 idx, u1 value) {
      return mask.set(idx, value);
    }

    /// Sets the bit a position 'idx' to the given 'value'.
    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static __forceinline__ void
    set(compound_mask_type& masks, u64 idx, u1 value) {
      u64 m_idx = idx / nested_vector_length;
      u64 n_idx = idx % nested_vector_length;
      return set<!Compound>(masks[m_idx], n_idx, value);
    }

    /// Sets ALL bits to the given 'value'.
    template<u1 Compound = false>
    static __forceinline__ void
    set(nested_mask_type& mask, u1 value) {
      mask.set(value);
    }

    /// Sets ALL bits to the given 'value'.
    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static __forceinline__ void
    set(compound_mask_type& masks, u1 value) {
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        set<!Compound>(masks[i], value);
      }
    }

    template<u1 Compound = false>
    static __forceinline__ u1
    get(const nested_mask_type& mask, u64 idx) {
      return mask.get(idx);
    }

    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static __forceinline__ u1
    get(const compound_mask_type& masks, u64 idx) {
      u64 m_idx = idx / nested_vector_length;
      u64 n_idx = idx % nested_vector_length;
      return get<!Compound>(masks[m_idx], n_idx);
    }

    /// Sets the mask at position 'idx' to 'true'. Use with caution as this operation might be very expensive.
    __forceinline__ void
    set(u64 idx, u1 value) { set<is_compound>(data, idx, value); }

    /// Gets the boolean value from the mask at position 'idx'. Use with caution as this operation might be very expensive.
    __forceinline__ u1
    get(u64 idx) const { return get<is_compound>(data, idx); }


    struct bit_and_fn {
      constexpr nested_mask_type
      operator()(const nested_mask_type& a, const nested_mask_type& b) const {
        return a.bit_and(b);
      }
    };

    struct bit_or_fn {
      constexpr nested_mask_type
      operator()(const nested_mask_type& a, const nested_mask_type& b) const {
        return a.bit_or(b);
      }
    };

    struct bit_xor_fn {
      constexpr nested_mask_type
      operator()(const nested_mask_type& a, const nested_mask_type& b) const {
        return a.bit_xor(b);
      }
    };

    struct bit_not_fn {
      constexpr nested_mask_type
      operator()(const nested_mask_type& a) const {
        return a.bit_not();
      }
    };

    // binary functions
    template<u1 Compound = false, typename Fn>
    static __forceinline__ nested_mask_type
    bit_op(Fn fn, const nested_mask_type& a, const nested_mask_type& b) {
      return fn(a, b);
    }
    // binary functions
    template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
    static __forceinline__ compound_mask_type
    bit_op(Fn fn, const compound_mask_type& a, const compound_mask_type& b) {
      compound_mask_type result;
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        result[i] = bit_op<!Compound>(fn, a[i], b[i]);
      }
      return result;
    }

    // unary functions
    template<u1 Compound = false, typename Fn>
    static __forceinline__ nested_mask_type
    bit_op(Fn fn, const nested_mask_type& a) {
      return fn(a);
    }
    // unary functions
    template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
    static __forceinline__ compound_mask_type
    bit_op(Fn fn, const compound_mask_type& a) {
      compound_mask_type result;
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        result[i] = bit_op<!Compound>(fn, a[i]);
      }
      return result;
    }


    /// Performs a bitwise AND.
    __forceinline__ m operator&(const m& o) const { return m { bit_op<is_compound>(bit_and_fn(), data, o.data) }; }
    __forceinline__ m& operator&=(const m& o) { data = bit_op<is_compound>(bit_and_fn(), data, o.data); return (*this); }

    /// Performs a bitwise OR.
    __forceinline__ m operator|(const m& o) const { return m { bit_op<is_compound>(bit_or_fn(), data, o.data) }; }
    __forceinline__ m& operator|=(const m& o) { data = bit_op<is_compound>(bit_or_fn(), data, o.data); return (*this); }

    /// Performs a bitwise XOR.
    __forceinline__ m operator^(const m& o) const { return m { bit_op<is_compound>(bit_xor_fn(), data, o.data) }; }
    __forceinline__ m& operator^=(const m& o) { data = bit_op<is_compound>(bit_xor_fn(), data, o.data); return (*this); }

    /// Performs a bitwise negation.
    __forceinline__ m operator!() const { return m { bit_op<is_compound>(bit_not_fn(), data) }; }


    /// Returns a mask instance where all components are set to 'true'.
    static __forceinline__ m
    make_all_mask() {
      m result;
      set<is_compound>(result.data, true);
      return result;
    };

    /// Returns a mask instance where all components are set to 'false'.
    static __forceinline__ m
    make_none_mask() {
      m result;
      set<is_compound>(result.data, false);
      return result;
    };


    /// Converts the mask into a position list and returns the number of elements. (the size of the position list must be at least N)
    template<u1 Compound = false>
    static __forceinline__ $u64
    to_positions(const nested_mask_type& mask, $u32* position_list, u32 offset) {
      return mask.to_positions(position_list, offset);
    }

    /// Converts the mask into a position list and returns the number of elements. (the size of the position list must be at least N)
    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static __forceinline__ $u64
    to_positions(const compound_mask_type& compound_mask, $u32* position_list, u32 offset) {
      $u32 match_cnt = 0;
      $u32* match_writer = position_list;
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        u64 cnt = to_positions<!Compound>(compound_mask[i], match_writer, offset + (nested_vector_length * i));
        match_cnt += cnt;
        match_writer += cnt;
      }
      return match_cnt;
    }

    /// Converts the mask into a position list and returns the number of elements. (the size of the position list must be at least N)
    __forceinline__ $u64
    to_positions($u32* position_list, u32 offset = 0) const {
      return to_positions<is_compound>(data, position_list, offset);
    }


    /// Converts the mask into an integer.
    template<u1 Compound = false>
    static __forceinline__ $u64
    to_int(const nested_mask_type& mask) {
      return mask.to_int();
    }

    /// Converts the mask into an integer.
    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static __forceinline__ $u64
    to_int(const compound_mask_type& compound_mask) {
      $u64 int_bitmask = 0;
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        u64 t = to_int<!Compound>(compound_mask[i]);
        int_bitmask |= t << ((N/nested_vector_cnt) * i);
      }
      return int_bitmask;
    }

    /// Converts the mask into an integer.
    __forceinline__ $u64
    to_int() const {
      static_assert(N <= 64, "Mask to integer conversion requires the vector length to be less or equal to 64.");
      return to_int<is_compound>(data);
    }



    /// Initializes the mask according to the bits set in the integer.
    template<u1 Compound = false>
    static __forceinline__ void
    from_int(nested_mask_type& mask, u64 int_bitmask) {
      mask.set_from_int(int_bitmask);
    }

    /// Converts the mask into an integer.
    template<u1 Compound, typename = std::enable_if_t<Compound>>
    static __forceinline__ void
    from_int(compound_mask_type& compound_mask, u64 int_bitmask) {
      for ($u64 i = 0; i < nested_vector_cnt; i++) {
        from_int<!Compound>(compound_mask[i], int_bitmask >> ((N/nested_vector_cnt) * i));
      }
    }

    /// Creates a mask from an integer.
    static __forceinline__ m
    from_int(u64 int_bitmask) {
      m result;
      from_int<is_compound>(result.data, int_bitmask);
      return result;
    }

//    // Creates a mask instance from a bitset
//    template<std::size_t Nb>
//    static __forceinline__ m
//    make_from_bitset(const dtl::simd::bitset<Nb> bs) {
//      static_assert(Nb >= N, "The size of the bitset must greater or equal than the vector length.");
//      return from_bitset(bs);
//    }
//    template<u1 Compound = false, std::size_t Nb>
//    static __forceinline__ nested_mask_type
//    from_bitset(const dtl::simd::bitset<Nb>& bs, u64 offset) {
//      nested_mask_type mask;
//      mask.set(bs.get(offset, nested_vector_length));
//      return mask;
//    }
//    template<u1 Compound, std::size_t Nb, typename = std::enable_if_t<Compound>>
//    static __forceinline__ compound_mask_type
//    from_bitset(const dtl::simd::bitset<Nb>& bs, u64 offset = 0ull) {
//      compound_mask_type result;
//      for ($u64 i = 0; i < length; i += nested_vector_length) {
//        result[i] = from_bitset<!Compound>(bs, i);
//      }
//      return result;
//    }


  };

  // Public mask API

  /// The mask type of the surrounding vector.
  using mask_t = m;
  using mask = m; // alias for those who don't like '_t's

  /// Returns a mask instance where all components are set to 'true'.
  static mask_t make_all_mask() { return mask_t::make_all_mask(); };

  /// Returns a mask instance where all components are set to 'false'.
  static mask_t make_none_mask() { return mask_t::make_none_mask(); };

  // --- end of mask


  //===----------------------------------------------------------------------===//


  // Specialize function objects for the current native vector type. (reduces verboseness later on)
  struct op {

    // template parameters are
    //   1) primitive type
    //   2) native vector type
    //   3) argument type
    //   4) return type (defaults to vector type)

    using broadcast = dtl::simd::broadcast<scalar_type, nested_type, scalar_type>;
    using set = dtl::simd::set<scalar_type, nested_type, nested_type>;
    using blend = dtl::simd::blend<scalar_type, nested_type, nested_type>;

    using plus = dtl::simd::plus<scalar_type, nested_type>;
    using minus = dtl::simd::minus<scalar_type, nested_type>;
    using multiplies = dtl::simd::multiplies<scalar_type, nested_type>;

    using shift_left = dtl::simd::shift_left<scalar_type, nested_type, i32>;
    using shift_left_var = dtl::simd::shift_left_var<scalar_type, nested_type, nested_type>;
    using shift_right = dtl::simd::shift_right<scalar_type, nested_type, i32>;
    using shift_right_var = dtl::simd::shift_right_var<scalar_type, nested_type, nested_type>;

    using bit_and = dtl::simd::bit_and<scalar_type, nested_type>;
    using bit_or = dtl::simd::bit_or<scalar_type, nested_type>;
    using bit_xor = dtl::simd::bit_xor<scalar_type, nested_type>;
    using bit_not = dtl::simd::bit_not<scalar_type, nested_type>;

    using less = dtl::simd::less<scalar_type, nested_type, nested_type, nested_mask_type>;
    using equal = dtl::simd::equal<scalar_type, nested_type, nested_type, nested_mask_type>;
    using not_equal = dtl::simd::not_equal<scalar_type, nested_type, nested_type, nested_mask_type>;
    using greater = dtl::simd::greater<scalar_type, nested_type, nested_type, nested_mask_type>; // TODO remove

  };


  //===----------------------------------------------------------------------===//
  // C'tors
  //===----------------------------------------------------------------------===//

  v() = default;

  __forceinline__
  v(const scalar_type scalar_value) {
    *this = make(scalar_value);
  }

  v(compound_type&& d) : data { std::move(d) } {};

  v(const v& other) = default;
  v(v&& other) = default;
//  template<typename Tp_other>
//  explicit
//  v(const v<Tp_other, N>& other) {
//    for (auto )
//  }

  // brace-initializer list c'tor
//  template<typename ...T>
//  explicit
//  v(T&&... t) : data { std::forward<T>(t)... } { }
//
//  explicit
//  v(v&& other) : data(std::move(other.data)) { }


  //===----------------------------------------------------------------------===//


  /// Assignment
  __forceinline__ v&
  operator=(const v& other) = default;

  __forceinline__ v&
  operator=(v&& other) = default;


  /// Assigns the given scalar value to all vector components.
  __forceinline__ v&
  operator=(const scalar_type& scalar_value) noexcept {
    data = unary_op<is_compound>(typename op::broadcast(), data, scalar_value);
    return *this;
  }

  /// Assigns the given scalar value to the vector components specified by the mask.
  __forceinline__ v&
  mask_assign(const scalar_type& scalar_value, const m& mask) noexcept {
    data = unary_op<is_compound>(typename op::blend(), /*data,*/ make(scalar_value).data, data, (!mask).data);
    return *this;
  }

  __forceinline__ v&
  mask_assign(const v& other, const m& mask) noexcept {
    data = unary_op<is_compound>(typename op::blend(), /*data,*/ other.data, data, (!mask).data);
    return *this;
  }


  //===----------------------------------------------------------------------===//


  /// Creates a vector where all components are set to the given scalar value.
  static __forceinline__ v
  make(const scalar_type& scalar_value) {
    v result;
    result = scalar_value;
    return std::move(result);
  }

  /// Creates a copy of the given vector.
  static __forceinline__ v
  make(const v& other) {
    v result;
    result.data = other.data;
    return result;
  }

  /// Creates a nested vector with all components set to the given scalar value.
  /// In other words, the given value is broadcasted to all vector components.
  static __forceinline__ nested_type
  make_nested(const scalar_type& scalar_value) {
    auto fn = typename op::broadcast();
    return fn(scalar_value);
  }


  //===----------------------------------------------------------------------===//
  // Unary functions
  //===----------------------------------------------------------------------===//

  template<u1 Compound = false, typename Fn>
  static __forceinline__ nested_type
  unary_op(Fn op, const nested_type& type_selector,
           const typename Fn::argument_type& a) noexcept {
    return op(a);
  }

  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static __forceinline__ compound_type
  unary_op(Fn op, const compound_type& type_selector,
           const typename Fn::argument_type& a) noexcept {
    compound_type result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = unary_op<!Compound>(op, result[i], a);
    }
    return result;
  }

  /// Unary operation: op(native vector)
  template<u1 Compound = false, typename Fn>
  static __forceinline__ nested_type
  unary_op(Fn op, // const nested_type& type_selector,
           const nested_type& a) noexcept {
    return op(a);
  }

  /// Unary operation (merge masked): op(native vector)
  template<u1 Compound = false, typename Fn>
  static __forceinline__ nested_type
  unary_op(Fn op, // const nested_type& type_selector,
           const nested_type& a,
           // merge masking
           const nested_type& src,
           const nested_mask_type& mask) noexcept {
    return op(a, src, mask);
  }

  /// Unary operation: op(compound vector)
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static __forceinline__ compound_type
  unary_op(Fn op, // const compound_type& type_selector,
           const compound_type& a) noexcept {
    compound_type result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = unary_op<!Compound>(op, a[i]);
    }
    return result;
  }

  /// Unary operation (merge masked): op(compound vector)
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static __forceinline__ compound_type
  unary_op(Fn op, // const compound_type& type_selector,
           const compound_type& a,
           // merge masking
           const compound_type& src,
           const compound_mask_type& mask) noexcept {
    compound_type result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = unary_op<!Compound>(op, a[i], src[i], mask[i]);
    }
    return result;
  }


//  // optimization
//  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
//  static __forceinline__ compound_type
//  unary_op(Fn op, // const compound_type& type_selector,
//           const nested_type& a) noexcept {
//    compound_type result;
//    result[0] = op(a);
//    for ($u64 i = 1; i < nested_vector_cnt; i++) {
//      result[i] = result[0];
//    }
//    return result;
//  }

//  // optimization
//  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
//  static __forceinline__ compound_type
//  unary_op(Fn op, // const compound_type& type_selector,
//           const nested_type& a,
//           // merge masking
//           const compound_type& src,
//           const m& mask) noexcept {
//    compound_type result;
//    for ($u64 i = 0; i < nested_vector_cnt; i++) {
//      result[i] = op(a, src[i], mask.data[i]);
//    }
//    return result;
//  }


  //===----------------------------------------------------------------------===//
  // Binary functions
  //===----------------------------------------------------------------------===//

  /// Applies a binary operation to a NON-compound (native) vector type.
  template<u1 Compound = false, typename Fn>
  static __forceinline__ typename Fn::result_type
  binary_op(Fn op, const typename Fn::vector_type& lhs,
                   const typename Fn::vector_type& rhs) noexcept {
    return op(lhs, rhs);
  }
  template<u1 Compound = false, typename Fn>
  static __forceinline__ typename Fn::result_type
  binary_op(Fn op, const typename Fn::vector_type& lhs,
                   const typename Fn::vector_type& rhs,
                   // merge masking
                   const typename Fn::vector_type& src,
                   const nested_mask_type& mask) noexcept {
    return op(lhs, rhs, src, mask);
  }


  /// Applies a binary operation to a compound vector type.
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static __forceinline__ make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const compound_type& lhs,
                   const compound_type& rhs) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op<!Compound>(op, lhs[i], rhs[i]);
    }
    return result;
  }
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
  static __forceinline__ make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const compound_type& lhs,
                   const compound_type& rhs,
                   // merge masking
                   const compound_type& src,
                   const compound_mask_type& mask) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op<!Compound>(op, lhs[i], rhs[i], src[i], mask[i]);
    }
    return result;
  }

//  // optimization
//  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
//  static __forceinline__ make_compound<typename Fn::result_type, nested_vector_cnt>
//  binary_op(Fn op, const nested_type& lhs,
//                   const compound_type& rhs) noexcept {
//    make_compound<typename Fn::result_type, nested_vector_cnt> result;
//    for ($u64 i = 0; i < nested_vector_cnt; i++) {
//      result[i] = binary_op(op, lhs, rhs[i]);
//    }
//    return result;
//  }
//  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
//  static __forceinline__ make_compound<typename Fn::result_type, nested_vector_cnt>
//  binary_op(Fn op, const nested_type& lhs,
//                   const compound_type& rhs,
//                   // merge masking
//                   const compound_type& src,
//                   const compound_mask_type& mask) noexcept {
//    make_compound<typename Fn::result_type, nested_vector_cnt> result;
//    for ($u64 i = 0; i < nested_vector_cnt; i++) {
//      result[i] = binary_op(op, lhs, rhs[i], src[i], mask[i]);
//    }
//    return result;
//  }

  /// Applies an operation of type: vector op scalar
  /// The scalar value needs to be broadcasted to all SIMD lanes first.
  /// Note: This is an optimization for compound vectors.
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>, typename = std::enable_if_t<Compound>> // TODO why twice?
  static __forceinline__ make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const compound_type& lhs,
                   const nested_type& rhs) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op(op, lhs[i], rhs);
    }
    return result;
  }
  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>, typename = std::enable_if_t<Compound>> // TODO why twice?
  static __forceinline__ make_compound<typename Fn::result_type, nested_vector_cnt>
  binary_op(Fn op, const compound_type& lhs,
                   const nested_type& rhs,
                   // merge masking
                   const compound_type& src,
                   const compound_mask_type& mask) noexcept {
    make_compound<typename Fn::result_type, nested_vector_cnt> result;
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      result[i] = binary_op(op, lhs[i], rhs, src[i], mask[i]);
    }
    return result;
  }


//  template<typename Fn>
//  static __forceinline__ nested_type
//  binary_op(Fn op, const nested_type& lhs, i32& rhs) noexcept {
//    return op(lhs, rhs);
//  }

//  /// Applies an operation of type: vector op scalar (w/o broadcasting the value to all SIMD lanes)
//  template<u1 Compound, typename Fn, typename = std::enable_if_t<Compound>>
//  static __forceinline__ compound_type
//  binary_op(Fn op, const compound_type& lhs, i32& rhs) noexcept {
//    compound_type result;
//    for ($u64 i = 0; i < nested_vector_cnt; i++) {
//      result[i] = binary_op(op, lhs[i], rhs);
//    }
//    return result;
//  }


  template<typename VECTOR_FN>
  __forceinline__ v map(const v& o) const noexcept { return v { binary_op<is_compound, VECTOR_FN>(VECTOR_FN(), data, o.data) }; }
  template<typename VECTOR_FN>
  __forceinline__ v map(const scalar_type& s) const noexcept { return v { binary_op<is_compound, VECTOR_FN>(VECTOR_FN(), data, make_nested(s)) }; }

  __forceinline__ v operator+(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, o.data) }; }
  __forceinline__ v operator+(const scalar_type& s) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, make_nested(s)) }; }
  __forceinline__ v operator+() const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, make_nested(0)) }; }
  __forceinline__ v& operator+=(const v& o) noexcept { data = binary_op<is_compound>(typename op::plus(), data, o.data); return *this; }
  __forceinline__ v& operator+=(const scalar_type& s) noexcept  { data = binary_op<is_compound>(typename op::plus(), data, make_nested(s)); return *this; }

  __forceinline__ v mask_plus(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, o.data, data, op_mask.data) }; }
  __forceinline__ v mask_plus(const scalar_type& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, make_nested(s), data, op_mask.data) }; }
  __forceinline__ v mask_plus(const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::plus(), data, make_nested(0), data, op_mask.data) }; }
  __forceinline__ v& mask_assign_plus(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::plus(), data, o.data, data, op_mask.data ); return *this; }
  __forceinline__ v& mask_assign_plus(const scalar_type& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::plus(), data, make_nested(s), data, op_mask.data ); return *this; }

  __forceinline__ v operator-(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, o.data) }; }
  __forceinline__ v operator-(const scalar_type& s) const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, make_nested(s)) }; }
  __forceinline__ v operator-() const noexcept { return v { binary_op<is_compound>(typename op::minus(), make_nested(0), data) }; }
  __forceinline__ v& operator-=(const v& o) noexcept { data = binary_op<is_compound>(typename op::minus(), data, o.data); return (*this); }
  __forceinline__ v& operator-=(const scalar_type& s) noexcept  { data = binary_op<is_compound>(typename op::minus(), data, make_nested(s)); return (*this); }

  __forceinline__ v mask_minus(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, o.data, data, op_mask.data) }; }
  __forceinline__ v mask_minus(const scalar_type& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::minus(), data, make_nested(s), data, op_mask.data) }; }
//  __forceinline__ v mask_minus(const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::minus(), make_nested(0), data, data, op_mask.data) }; }
  __forceinline__ v mask_minus(const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::minus(), make(0).data, data, data, op_mask.data) }; }
  __forceinline__ v& mask_assign_minus(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::minus(), data, o.data, data, op_mask.data ); return *this; }
  __forceinline__ v& mask_assign_minus(const scalar_type& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::minus(), data, make_nested(s), data, op_mask.data ); return *this; }

  __forceinline__ v operator*(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::multiplies(), data, o.data) }; }
  __forceinline__ v operator*(const scalar_type& s) const noexcept { return v { binary_op<is_compound>(typename op::multiplies(), data, make_nested(s)) }; }
  __forceinline__ v& operator*=(const v& o) noexcept { data = binary_op<is_compound>(typename op::multiplies(), data, o.data); return (*this); }
  __forceinline__ v& operator*=(const scalar_type& s) noexcept  { data = binary_op<is_compound>(typename op::multiplies(), data, make_nested(s)); return (*this); }

  __forceinline__ v mask_multiplies(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::multiplies(), data, o.data, data, op_mask.data) }; }
  __forceinline__ v mask_multiplies(const scalar_type& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::multiplies(), data, make_nested(s), data, op_mask.data) }; }
  __forceinline__ v& mask_assign_multiplies(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::multiplies(), data, o.data, data, op_mask.data ); return *this; }
  __forceinline__ v& mask_assign_multiplies(const scalar_type& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::multiplies(), data, make_nested(s), data, op_mask.data ); return *this; }

  __forceinline__ v operator<<(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::shift_left_var(), data, o.data) }; }
  __forceinline__ v operator<<(const i32& s) const noexcept { return v { binary_op<is_compound>(typename op::shift_left_var(), data, make_nested(s)) }; } // TODO optimize
  __forceinline__ v& operator<<=(const v& o) noexcept { data = binary_op<is_compound>(typename op::shift_left_var(), data, o.data); return (*this); }
  __forceinline__ v& operator<<=(const i32& s) noexcept  { data = binary_op<is_compound>(typename op::shift_left_var(), data, make_nested(s)); return (*this); }

  template<typename Trhs, typename = std::enable_if_t<is_vector<Trhs>::value>>
  __forceinline__ v operator<<(const Trhs& o) const noexcept {
    v rhs;
    for ($u64 i = 0; i < N; i++) {
      rhs.insert(o[i], i);
    }
    return v { binary_op<is_compound>(typename op::shift_left_var(), data, rhs.data) };
  }

  __forceinline__ v mask_shift_left(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::shift_left_var(), data, o.data, data, op_mask.data) }; }
  __forceinline__ v mask_shift_left(const i32& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::shift_left(), data, make_nested(s), data, op_mask.data) }; }
  __forceinline__ v& mask_assign_shift_left(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::shift_left_var(), data, o.data, data, op_mask.data ); return *this; }
  __forceinline__ v& mask_assign_shift_left(const i32& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::shift_left(), data, make_nested(s), data, op_mask.data ); return *this; }

  __forceinline__ v operator>>(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::shift_right_var(), data, o.data) }; }
  __forceinline__ v operator>>(const i32& s) const noexcept { return v { binary_op<is_compound>(typename op::shift_right_var(), data, make_nested(s)) }; } // TODO optimize
  __forceinline__ v& operator>>=(const v& o) noexcept { data = binary_op<is_compound>(typename op::shift_right_var(), data, o.data); return (*this); }
  __forceinline__ v& operator>>=(const i32& s) noexcept  { data = binary_op<is_compound>(typename op::shift_right(), data, make_nested(s)); return (*this); }

  __forceinline__ v mask_shift_right(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::shift_right_var(), data, o.data, data, op_mask.data) }; }
  __forceinline__ v mask_shift_right(const i32& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::shift_right(), data, make_nested(s), data, op_mask.data) }; }
  __forceinline__ v& mask_assign_shift_right(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::shift_right_var(), data, o.data, data, op_mask.data ); return *this; }
  __forceinline__ v& mask_assign_shift_right(const i32& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::shift_right(), data, make_nested(s), data, op_mask.data ); return *this; }

  __forceinline__ v operator&(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::bit_and(), data, o.data) }; }
  __forceinline__ v operator&(const scalar_type& s) const noexcept { return v { binary_op<is_compound>(typename op::bit_and(), data, make_nested(s)) }; }
  __forceinline__ v& operator&=(const v& o) noexcept { data = binary_op<is_compound>(typename op::bit_and(), data, o.data); return (*this); }
  __forceinline__ v& operator&=(const scalar_type& s) noexcept  { data = binary_op<is_compound>(typename op::bit_and(), data, make_nested(s)); return (*this); }

  __forceinline__ v mask_bit_and(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_and(), data, o.data, data, op_mask.data) }; }
  __forceinline__ v mask_bit_and(const scalar_type& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_and(), data, make_nested(s), data, op_mask.data) }; }
  __forceinline__ v& mask_assign_bit_and(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_and(), data, o.data, data, op_mask.data ); return *this; }
  __forceinline__ v& mask_assign_bit_and(const scalar_type& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_and(), data, make_nested(s), data, op_mask.data ); return *this; }

  __forceinline__ v operator|(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::bit_or(), data, o.data) }; }
  __forceinline__ v operator|(const scalar_type& s) const noexcept { return v { binary_op<is_compound>(typename op::bit_or(), data, make_nested(s)) }; }
  __forceinline__ v& operator|=(const v& o) noexcept { data = binary_op<is_compound>(typename op::bit_or(), data, o.data); return (*this); }
  __forceinline__ v& operator|=(const i32& s) noexcept  { data = binary_op<is_compound>(typename op::bit_or(), data, make_nested(s)); return (*this); }

  __forceinline__ v mask_bit_or(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_or(), data, o.data, data, op_mask.data) }; }
  __forceinline__ v mask_bit_or(const scalar_type& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_or(), data, make_nested(s), data, op_mask.data) }; }
  __forceinline__ v& mask_assign_bit_or(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_or(), data, o.data, data, op_mask.data ); return *this; }
  __forceinline__ v& mask_assign_bit_or(const scalar_type& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_or(), data, make_nested(s), data, op_mask.data ); return *this; }

  __forceinline__ v operator^(const v& o) const noexcept { return v { binary_op<is_compound>(typename op::bit_xor(), data, o.data) }; }
  __forceinline__ v operator^(const scalar_type& s) const noexcept { return v { binary_op<is_compound>(typename op::bit_xor(), data, make_nested(s)) }; }
  __forceinline__ v& operator^=(const v& o) noexcept { data = binary_op<is_compound>(typename op::bit_xor(), data, o.data); return (*this); }
  __forceinline__ v& operator^=(const scalar_type& s) noexcept { data = binary_op<is_compound>(typename op::bit_xor(), data, make_nested(s)); return (*this); }

  __forceinline__ v mask_bit_xor(const v& o, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_xor(), data, o.data, data, op_mask.data) }; }
  __forceinline__ v mask_bit_xor(const scalar_type& s, const m& op_mask) const noexcept { return v { binary_op<is_compound>(typename op::bit_xor(), data, make_nested(s), data, op_mask.data) }; }
  __forceinline__ v& mask_assign_bit_xor(const v& o, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_xor(), data, o.data, data, op_mask.data ); return *this; }
  __forceinline__ v& mask_assign_bit_xor(const scalar_type& s, const m& op_mask) noexcept { data = binary_op<is_compound>(typename op::bit_xor(), data, make_nested(s), data, op_mask.data ); return *this; }

  __forceinline__ v zero_mask(const m& mask) {
    return v {unary_op<is_compound>(typename op::blend(), /*data,*/ make(0).data, data, (!mask).data) };
  }

  __forceinline__ v operator~() const noexcept { return v { unary_op<is_compound>(typename op::bit_not(), data) }; }

  static __forceinline__ scalar_type
  extract(const nested_type& native_vector, u64 idx) noexcept {
    return reinterpret_cast<const scalar_type*>(&native_vector)[idx]; // TODO improve performance
  }

  template<typename T, typename = std::enable_if_t<(sizeof(T), is_compound)>>
  static __forceinline__ scalar_type
  extract(const T& compound_vector, u64 idx) noexcept {
    return extract(compound_vector[idx / nested_vector_length], idx % nested_vector_length);
  }

  template<u1 Compound = false>
  static __forceinline__ void
  insert(nested_type& native_vector, const scalar_type& value, u64 idx) noexcept {
    reinterpret_cast<scalar_type*>(&native_vector)[idx] = value; // TODO improve performance
  }

  template<u1 Compound, typename = std::enable_if_t<Compound>>
  static __forceinline__ void
  insert(compound_type& compound_vector, const scalar_type& value, u64 idx) noexcept {
    insert<!Compound>(compound_vector[idx / nested_vector_length], value, idx % nested_vector_length);
  }

  __forceinline__ void
  insert(const scalar_type& value, u64 idx) noexcept {
    insert<is_compound>(data, value, idx);
  }

  /// Read-only access to the individual vector components
  scalar_type operator[](u64 idx) const noexcept {
    return extract(data, idx);
  }
  // ---


  // Comparisons
  __forceinline__ m
  operator<(const v& o) const noexcept {
    return m { binary_op<is_compound>(typename op::less(), data, o.data) };
  }

  __forceinline__ m
  operator<(const scalar_type& s) const noexcept {
    return m { binary_op<is_compound>(typename op::less(), data, make_nested(s)) };
  }

  __forceinline__ m
  operator>(const v& o) const noexcept {
    return m { binary_op<is_compound>(typename op::greater(), data, o.data) };
  }

  __forceinline__ m
  operator>(const scalar_type& s) const noexcept {
    return m { binary_op<is_compound>(typename op::greater(), data, make_nested(s)) };
  }

  __forceinline__ m
  operator==(const v& o) const noexcept {
    return m { binary_op<is_compound>(typename op::equal(), o.data, data) };
  }

  __forceinline__ m
  operator==(const scalar_type& s) const noexcept {
    return m { binary_op<is_compound>(typename op::equal(), data, make_nested(s)) };
  }

  __forceinline__ m
  operator!=(const v& o) const noexcept {
    return m { binary_op<is_compound>(typename op::not_equal(), o.data, data) };
  }

  __forceinline__ m
  operator!=(const scalar_type& s) const noexcept {
    return m { binary_op<is_compound>(typename op::not_equal(), data, make_nested(s)) };
  }
  // ---


  // load
  // TODO rename to gather

//  template<u1 Compound, typename scalar_type>
//  static __forceinline__ nested_type
//  __gather__(const scalar_type* const base_addr,
//             const typename Tiv& idxs) {
//    return gather<scalar_type, typename Tiv::nested_type, typename Tiv::nested_type>()(base_addr, idxs.data);
//  }
//
//  template<u1 Compound, typename Tiv, typename = std::enable_if_t<Compound>>
//  static __forceinline__ compound_type
//  __gather__(const scalar_type* const base_addr,
//             const typename Tiv::compound_type& idxs) {
//    compound_type result;
//    for ($u64 i = 0; i < nested_vector_cnt; i++) {
//      result[i] = __gather<!Compound, Tiv>(base_addr, idxs.data[i]);
//    }
//    return result;
//  }
//
//  template<typename Trv, typename Tiv> // result vector type, index vector type
//  static __forceinline__ Trv
//  __gather(const typename Trv::scalar_type* const base_addr,
//           const typename Tiv& idxs) {
//    return Trv { Trv::__gather__<Tiv::is_compound, Trv::scalar_type>(base_addr, idxs.data) };
//  }
//
//
//  template<typename T>
//  v<T, N> load(const T* const base_addr) const {
//    using result_t = v<T, N>;
//    static_assert(result_t::nested_vector_length == nested_vector_length, "BAM");
//    static_assert(result_t::nested_vector_cnt == nested_vector_cnt, "BAM");
//    return result_t { load<is_compound, T>(base_addr, data) };
//  }
//
//  template<typename T>
////  v<T, N, vs<T, nested_vector_length>>
//  v<T, N> load() const {
//    using result_t = v<T, N>;
//    static_assert(result_t::nested_vector_length == nested_vector_length, "BAM");
//    static_assert(result_t::nested_vector_cnt == nested_vector_cnt, "BAM");
//    return result_t { load<is_compound, T>(nullptr, data) };
//  }
  // ---


  // store
  template<u1 Compound = false, typename T>
  static __forceinline__ typename v<T, N>::nested_type
  store(T* const base_addr,
        const nested_type& where_idxs,
        const typename v<T, N>::nested_type what) {
    return scatter<scalar_type, nested_type, T>()(base_addr, where_idxs, what);
  }

  template<u1 Compound = false, typename T>
  static __forceinline__ typename v<T, N>::nested_type
  store(T* const base_addr,
        const nested_type& where_idxs,
        const typename v<T, N>::nested_type what,
        const nested_mask_type& mask) {
    return scatter<scalar_type, nested_type, T>()(base_addr, where_idxs, what, mask);
  }

  template<u1 Compound, typename T, typename = std::enable_if_t<Compound>>
  static __forceinline__ void
  store(T* const base_addr,
        const compound_type& where_idxs,
        const typename v<T, N>::compound_type& what) {
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      store<is_compound>(base_addr, where_idxs[i], what[i]);
    }
  }

  template<u1 Compound, typename T, typename = std::enable_if_t<Compound>>
  static __forceinline__ void
  store(T* const base_addr,
        const compound_type& where_idxs,
        const typename v<T, N>::compound_type& what,
        const compound_mask_type& mask) {
    for ($u64 i = 0; i < nested_vector_cnt; i++) {
      store<is_compound>(base_addr, where_idxs[i], what[i], mask[i]);
    }
  }

  template<typename T>
  __forceinline__ void
  store(T* const base_address, const v<T, N>& what) {
    store<is_compound>(base_address, data, what.data);
  }

  template<typename T>
  __forceinline__ void
  store(T* const base_address, const v<T, N>& what, const m& mask) {
    store<is_compound>(base_address, data, what.data, mask.data);
  }
  // ---


  // helper
  void
  print(std::ostream& os) const {
    os << "[" << (*this)[0];
    for ($u64 i = 1; i < length; i++) {
      os << ", ";
      os << (*this)[i];
    }
    os << "]";
  }

  template<u64... Idxs>
  static constexpr v
  make_index_vector(std::array<scalar_type, N>* const arr, integer_sequence<Idxs...>) {
    *arr = { Idxs... };
  }

  static constexpr v
  make_index_vector() {
    v result = make(0);
    std::array<scalar_type, N>* const arr = reinterpret_cast<std::array<scalar_type, N>*>(&result.data);
    make_index_vector(arr, make_integer_sequence<N>());
    return result;
  };




  //TODO implement casts in SIMD
  template<typename Tp_target>
  __forceinline__ v<Tp_target, N>
  cast() const {
    v<Tp_target, N> result;
    for ($u64 i = 0; i < N; i++) {
      result.insert((*this)[i], i);
    }
    return result;
  }



  // --- syntactic sugar for masked operations

  struct masked_reference {
    v& vector;
    m mask;

//    m(u32 i) {
//      set<is_compound>(this->data, i);
//    }
//    masked_reference(const masked_reference&) = default;
//    masked_reference(masked_reference&&) = default;
//    masked_reference(compound_mask_type&& d) : data { std::move(d) } {};
//    masked_reference& operator=(const masked_reference&) = default;
//    masked_reference& operator=(masked_reference&&) = default;


    __forceinline__ v& operator=(const v& o) { vector.mask_assign(o, mask); return vector; }
    __forceinline__ v& operator=(const scalar_type& s) { vector.mask_assign(s, mask); return vector; }
    __forceinline__ v operator+(const v& o) const noexcept { return vector.mask_plus(o, mask); }
    __forceinline__ v operator+(const scalar_type& s) const noexcept { return vector.mask_plus(s, mask); }
    __forceinline__ v operator+() const noexcept { return vector.mask_plus(mask); }
    __forceinline__ v& operator+=(const v& o) noexcept { vector.mask_assign_plus(o,mask); return vector; }
    __forceinline__ v& operator+=(const scalar_type& s) noexcept { vector.mask_assign_plus(s,mask); return vector; }
    __forceinline__ v operator-(const v& o) const noexcept { return vector.mask_minus(o, mask); }
    __forceinline__ v operator-(const scalar_type& s) const noexcept { return vector.mask_minus(s, mask); }
    __forceinline__ v operator-() const noexcept { return vector.mask_minus(mask); }
    __forceinline__ v& operator-=(const v& o) noexcept { vector.mask_assign_minus(o, mask); return vector; }
    __forceinline__ v& operator-=(const scalar_type& s) noexcept { vector.mask_assign_minus(s, mask); return vector; }
    __forceinline__ v operator*(const v& o) const noexcept { return vector.mask_multiplies(o, mask); }
    __forceinline__ v operator*(const scalar_type& s) const noexcept { return vector.mask_multiplies(s, mask); }
    __forceinline__ v& operator*=(const v& o) noexcept { vector.mask_assign_multiplies(o, mask); return vector; }
    __forceinline__ v& operator*=(const scalar_type& s) noexcept { vector.mask_assign_multiplies(s, mask); return vector; }
    __forceinline__ v operator<<(const v& o) const noexcept { return vector.mask_shift_left(o, mask); }
    __forceinline__ v operator<<(const scalar_type& s) const noexcept { return vector.mask_shift_left(s, mask); }
    __forceinline__ v& operator<<=(const v& o) noexcept { vector.mask_assign_shift_left(o, mask); return vector; }
    __forceinline__ v& operator<<=(const scalar_type& s) noexcept { vector.mask_assign_shift_left(s, mask); return vector; }
    __forceinline__ v operator>>(const v& o) const noexcept { return vector.mask_shift_right(o, mask); }
    __forceinline__ v operator>>(const scalar_type& s) const noexcept { return vector.mask_shift_right(s, mask); }
    __forceinline__ v& operator>>=(const v& o) noexcept { vector.mask_assign_shift_right(o, mask); return vector; }
    __forceinline__ v& operator>>=(const scalar_type& s) noexcept { vector.mask_assign_shift_right(s, mask); return vector; }
    __forceinline__ v operator&(const v& o) const noexcept { return vector.mask_bit_and(o, mask); }
    __forceinline__ v operator&(const scalar_type& s) const noexcept { return vector.mask_bit_and(s, mask); }
    __forceinline__ v& operator&=(const v& o) noexcept { vector.mask_assign_bit_and(o, mask); return vector; }
    __forceinline__ v& operator&=(const scalar_type& s) noexcept { vector.mask_assign_bit_and(s, mask); return vector; }
    __forceinline__ v operator|(const v& o) const noexcept { return vector.mask_bit_or(o, mask); }
    __forceinline__ v operator|(const scalar_type& s) const noexcept { return vector.mask_bit_or(s, mask); }
    __forceinline__ v& operator|=(const v& o) noexcept { vector.mask_assign_bit_or(o, mask); return vector; }
    __forceinline__ v& operator|=(const scalar_type& s) noexcept { vector.mask_assign_bit_or(s, mask); return vector; }
    __forceinline__ v operator^(const v& o) const noexcept { return vector.mask_bit_xor(o, mask); }
    __forceinline__ v operator^(const scalar_type& s) const noexcept { return vector.mask_bit_xor(s, mask); }
    __forceinline__ v& operator^=(const v& o) noexcept { vector.mask_assign_bit_xor(o, mask); return vector; }
    __forceinline__ v& operator^=(const scalar_type& s) noexcept { vector.mask_assign_bit_xor(s, mask); return vector; }
  };

  __forceinline__ masked_reference
  operator[](const m& op_mask) noexcept {
//    return masked_reference{ *this, m{op_mask.data} };
    return masked_reference{ *this, op_mask };
  }

//  __forceinline__ masked_reference
//  operator[](const typename v<$u32, N>::m op_mask) noexcept {
//    //static_assert(std::is_same<typename v<$u32, N>::m::type, m::type>::value, "Mask is not compatible with this vector.");
//    return masked_reference{ *this, m{op_mask.data} };
//  }
//
//  __forceinline__ masked_reference
//  operator[](const typename v<$u64, N>::m op_mask) noexcept {
//    //static_assert(std::is_same<typename v<$u32, N>::m::type, m::type>::value, "Mask is not compatible with this vector.");
//    return masked_reference{ *this, m{op_mask.data} };
//  }
  // TODO specialize for all valid primitive types

  // ---


};


/// left shift of the form of: scalar << vector
template<typename T, u64 N>
v<T, N> operator<<(const T& lhs, const v<T, N>& rhs) {
  v<T, N> lhs_vec = v<T, N>::make(lhs);
  return lhs_vec << rhs;
}

template<typename Tlhs, typename Trhs, typename = std::enable_if_t<is_vector<Trhs>::value>>
__forceinline__ Tlhs
operator<<(const Tlhs& lhs, const Trhs& o) noexcept {
  Tlhs rhs;
  for ($u64 i = 0; i < Tlhs::length; i++) {
    rhs.insert(o[i], i);
  }
  return lhs << rhs;
}


// not sure if this is causing problems...
template<typename Tl, typename T, u64 N>
v<T, N> operator<<(const Tl& lhs, const v<T, N>& rhs) {
  v<T, N> lhs_vec = v<T, N>::make(Tl(lhs));
  return lhs_vec << rhs;
}


} // namespace simd

// --- Gather ---

namespace {
template<u1 Compound = false,
    typename Tp,       // the primitive data type
    typename Trv,      // the return vector type
    typename Tiv>      // the index vector type
static __forceinline__ typename Trv::nested_type
__gather(const Tp* const base_addr,
         const typename Tiv::nested_type& idxs) {
  return simd::gather<Tp, typename Tiv::nested_type, typename Tiv::nested_type>()(base_addr, idxs);
}

template<u1 Compound,
    typename Tp,       // the primitive data type
    typename Trv,      // the return vector type
    typename Tiv,      // the index vector type
    typename = std::enable_if_t<Compound>>
static __forceinline__ typename Trv::compound_type
__gather(const Tp* const base_addr,
         const typename Tiv::compound_type& idxs) {
  typename Trv::compound_type result;
  for ($u64 i = 0; i < Tiv::nested_vector_cnt; i++) {
    result[i] = __gather<!Compound, Tp, Trv, Tiv>(base_addr, idxs[i]);
  }
  return result;
}
} // anonymous namespace

/// Gathers values of primitive type Tp from the
/// base address + offsets stored in vector Tiv
template<typename Tp,  // primitive value type
         typename Tiv> // index vector
static __forceinline__ simd::v<Tp, Tiv::length>
gather(const Tp* const base_addr, const Tiv& idxs) {
  using return_vec_t = simd::v<Tp, Tiv::length>;
  using index_vec_t = Tiv;

  auto data = __gather<index_vec_t::is_compound, Tp, return_vec_t, index_vec_t>(base_addr, idxs.data);
  return return_vec_t { data };
}

/// Gathers values of primitive type Tp from the
/// absolute addresses stored in vector Tiv
template<typename Tp,  // primitive value type
         typename Tiv> // index vector
static __forceinline__ simd::v<Tp, Tiv::length>
gather(const Tiv& idxs) {
  using return_vec_t = simd::v<Tp, Tiv::length>;
  using index_vec_t = Tiv;

  auto data = __gather<index_vec_t::is_compound, Tp, return_vec_t, index_vec_t>(0, idxs.data);
  return return_vec_t { data };
}


// --- Scatter ---

namespace {
template<u1 Compound = false,
    typename Tp,       // the primitive data type
    typename Tiv,      // the index vector type
    typename Tvv>      // the value vector type
static __forceinline__ void
__scatter(Tp* base_addr,
          const typename Tiv::nested_type& idxs,
          const typename Tvv::nested_type& vals) {
  simd::scatter<Tp, typename Tiv::nested_type, typename Tvv::nested_type>()(base_addr, idxs, vals);
}

template<u1 Compound,
    typename Tp,       // the primitive data type
    typename Tiv,      // the index vector type
    typename Tvv,      // the value vector type
    typename = std::enable_if_t<Compound>>
static __forceinline__ void
__scatter(Tp* base_addr,
          const typename Tiv::compound_type& idxs,
          const typename Tvv::compound_type& vals) {
  for ($u64 i = 0; i < Tiv::nested_vector_cnt; i++) {
    __scatter<!Compound, Tp, Tiv, Tvv>(base_addr, idxs[i], vals[i]);
  }
}
} // anonymous namespace

/// Scatters values of primitive type Tp to
/// base address + offsets stored in vector Tiv
template<typename Tp,  // primitive value type
         typename Tiv, // index vector
         typename Tvv> // value vector
static __forceinline__ void
scatter(const Tvv& vals, Tp* base_addr, const Tiv& idxs) {
  using index_vec_t = Tiv;
  using value_vec_t = Tvv;
  __scatter<index_vec_t::is_compound, Tp, index_vec_t, value_vec_t>(base_addr, idxs.data, vals.data);
}


//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//
//TODO implement casts in SIMD
template<typename Tp_target, typename Tp_source, std::size_t N>
__forceinline__ dtl::simd::v<Tp_target, N>
cast(const dtl::simd::v<Tp_source, N> src) {
  dtl::simd::v<Tp_target, N> result;
  for ($u64 i = 0; i < N; i++) {
    result.insert(src[i], i);
  }
  return result;
}


template<typename Tm_dst, typename Tm_src>
__forceinline__ Tm_dst
cast_mask(const Tm_src& src_mask) {
  return Tm_dst::from_int(src_mask.to_int());
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Type support
//===----------------------------------------------------------------------===//

template<class T>
struct is_vector {
  static constexpr bool value = std::is_base_of<dtl::simd::v_base, T>::value;
};

namespace internal {

template<typename Tv, std::size_t _vector_length = Tv::length>
struct vector_len_helper {
  static constexpr std::size_t value = _vector_length;
};

} // namespace internal

template<typename Tv>
struct vector_length {
  static_assert(is_vector<Tv>::value, "The given type is not a vector.");
  static constexpr std::size_t value = internal::vector_len_helper<Tv>::value;
};
//===----------------------------------------------------------------------===//

} // namespace dtl
