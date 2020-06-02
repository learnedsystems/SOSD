#pragma once

#include "adept.hpp"
#include "math.hpp"

#include <array>
#include <bitset>
#include <functional>


/// determine the type of elements stored in an (plain old) array
/// note: see also 'std::tuple_element<std::array>', which is kind of similar
template<class T>
struct array_info {
  static constexpr u1 is_array = false;
  static constexpr u1 is_std_array = false;
  static constexpr u64 length = 0;
  using value_type = void;
};

template<class T>
struct array_info<T[]> {
  static constexpr u1 is_array = true;
  static constexpr u1 is_std_array = false;
  static constexpr u64 length = std::extent<T>::value;
  using value_type = T;
};

template<typename T, u64 N>
struct array_info<T[N]> {
  static constexpr u1 is_array = true;
  static constexpr u1 is_std_array = false;
  static constexpr u64 length = N;
  using value_type = T;
};

template<typename T, u64 N>
struct array_info<std::array<T, N>> {
  static constexpr u1 is_array = true;
  static constexpr u1 is_std_array = true;
  static constexpr u64 length = N;
  using value_type = T;
};

namespace simd {


template<typename T, u64 N, typename R>
struct vec {
  static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");
  static_assert(N * sizeof(T) == sizeof(R), "Size of the internal vector representation does not match the size of the vector components.");

  alignas(64) R data;

  using vec_t = vec<T, N, R>;
  using mask_t = std::bitset<N>;

  static mask_t make_all_mask() {
    mask_t m;
    return ~m;
  }

  static mask_t make_none_mask() {
    mask_t m;
    return m;
  }

  T& operator[](u64 index) {
    return data[index];
  }

  T operator[](u64 index) const {
    return data[index];
  }

  vec gather(const vec<$u64, N, R>& index) const {
    vec d;
    for (size_t i = 0; i < N; i++) {
      d.data[i] = data[index[i]];
    }
    return d;
  }

  /*
  template<typename Td>
  static vec<Td, N, std::array<Td, N>> gather(const Td* const ptr, const vec<$u64, N, std::array<$u64, N>>& index) {
    vec<Td, N, std::array<Td, N>> d;
    for (size_t i = 0; i < N; i++) {
      d.data[i] = ptr[index[i]];
    }
    return d;
  }
  template<typename Td>
  static vec<Td, N, std::array<Td, N>> gather(const Td* const ptr, const vec<$u32, N, std::array<$u32, N>>& index) {
    vec<Td, N, std::array<Td, N>> d;
    for (size_t i = 0; i < N; i++) {
      d.data[i] = ptr[index[i]];
    }
    return d;
  }
  */


  vec gather(const vec<$u64, N, R>& index, const mask_t& mask) const {
    vec d;
    for (size_t i = 0; i < N; i++) {
      if (!mask[i]) continue;
      d.data[i] = data[index[i]];
    }
    return d;
  }

  void scatter(const vec& what, const vec<$u64, N, R>& where) {
    for (size_t i = 0; i < N; i++) {
      data[where[i]] = what[i];
    }
  }

  void scatter(const vec& what, const vec<$u64, N, R>& where, const mask_t& mask) {
    for (size_t i = 0; i < N; i++) {
      if (!mask[i]) continue;
      data[where[i]] = what[i];
    }
  }

  vec conflict_detection(const vec& a) {
    // just a naive translation of the '_mm512_conflict_epi32'
    // intrinsic function.
    vec dst;
    for ($u64 j = 0; j < N; j++) {
      for ($u64 k = 0; k < j; k++) {
        u64 are_equal = a[j] == a[k];
        dst[j] |= are_equal << k;
      }
    }
    return dst;
  }


  // binary operators
  vec binary_operator(auto op, const vec& b) const  noexcept {
    vec d;
    for (size_t i = 0; i < N; i++) {
      d.data[i] = op(data[i], b.data[i]);
    }
    return d;
  }

  vec operator+(const vec& o) const noexcept { return binary_operator(std::plus<T>(), o); }
  vec operator-(const vec& o) const noexcept { return binary_operator(std::minus<T>(), o); }
  vec operator*(const vec& o) const noexcept { return binary_operator(std::multiplies<T>(), o); }
  vec operator/(const vec& o) const noexcept { return binary_operator(std::divides<T>(), o); }
  vec operator|(const vec& o) const noexcept { return binary_operator(std::bit_or<T>(), o); }
  vec operator^(const vec& o) const noexcept { return binary_operator(std::bit_xor<T>(), o); }
  vec operator&(const vec& o) const noexcept { return binary_operator(std::bit_and<T>(), o); }
  vec operator<<(const vec& o) const noexcept { return binary_operator(std::bit_shift_left<T>(), o); }
  vec operator>>(const vec& o) const noexcept { return binary_operator(std::bit_shift_right<T>(), o); }

  vec binary_operator(auto op, const T& b) const noexcept {
    vec d;
    for (size_t i = 0; i < N; i++) {
      d.data[i] = op(data[i], b);
    }
    return d;
  }

  vec operator+(const T& o) const noexcept { return binary_operator(std::plus<T>(), o); }
  vec operator-(const T& o) const noexcept { return binary_operator(std::minus<T>(), o); }
  vec operator*(const T& o) const noexcept { return binary_operator(std::multiplies<T>(), o); }
  vec operator/(const T& o) const noexcept { return binary_operator(std::divides<T>(), o); }
  vec operator|(const T& o) const noexcept { return binary_operator(std::bit_or<T>(), o); }
  vec operator^(const T& o) const noexcept { return binary_operator(std::bit_xor<T>(), o); }
  vec operator&(const T& o) const noexcept { return binary_operator(std::bit_and<T>(), o); }
  vec operator<<(const T& o) const noexcept { return binary_operator(std::bit_shift_left<T>(), o); }
  vec operator>>(const T& o) const noexcept { return binary_operator(std::bit_shift_right<T>(), o); }


  // unary operators
  vec unary_operator(auto op) const noexcept {
    vec d;
    for (size_t i = 0; i < N; i++) {
      d.data[i] = op(data[i]);
    }
    return d;
  }

  vec operator~() const noexcept { return unary_operator(std::bit_not<T>()); }

  // comparison operators / relational operators
  mask_t comparison_operator(auto op, const vec& b) const noexcept {
    mask_t mask;
    for (size_t i = 0; i < N; i++) {
      mask[i] = op(data[i], b.data[i]);
    }
    return mask;
  }
  mask_t operator<(const vec& o) const noexcept { return comparison_operator(std::less<T>(), o); }
  mask_t operator<=(const vec& o) const noexcept { return comparison_operator(std::less_equal<T>(), o); }
  mask_t operator==(const vec& o) const noexcept { return comparison_operator(std::equal_to<T>(), o); }
  mask_t operator!=(const vec& o) const noexcept { return comparison_operator(std::not_equal_to<T>(), o); }
  mask_t operator>=(const vec& o) const noexcept { return comparison_operator(std::greater_equal<T>(), o); }
  mask_t operator>(const vec& o) const noexcept { return comparison_operator(std::greater<T>(), o); }

  mask_t comparison_operator(auto op, const T& b) const noexcept {
    mask_t mask;
    for (size_t i = 0; i < N; i++) {
      mask[i] = op(data[i], b);
    }
    return mask;
  }
  mask_t operator<(const T& o) const noexcept { return comparison_operator(std::less<T>(), o); }
  mask_t operator<=(const T& o) const noexcept { return comparison_operator(std::less_equal<T>(), o); }
  mask_t operator==(const T& o) const noexcept { return comparison_operator(std::equal_to<T>(), o); }
  mask_t operator!=(const T& o) const noexcept { return comparison_operator(std::not_equal_to<T>(), o); }
  mask_t operator>=(const T& o) const noexcept { return comparison_operator(std::greater_equal<T>(), o); }
  mask_t operator>(const T& o) const noexcept { return comparison_operator(std::greater<T>(), o); }



  template<typename S>
  vec<S, N, R> cast() const noexcept {
    vec<S, N, R> d;
    for (size_t i = 0; i < N; i++) {
      d.data[i] = data[i];
    }
    return d;
  }

  template<size_t W>
  vec<T, W, R>* begin()  {
    return reinterpret_cast<vec<T, W, R>*>(data);
  }

  template<size_t W>
  vec<T, W, R>* end() const {
    return begin() + (N / W);
  }

  template<u64... Idxs>
  static constexpr vec make_index_vector(integer_sequence<Idxs...>) {
    return {{ Idxs... }};
  }

  static constexpr vec make_index_vector() {
    return make_index_vector(make_integer_sequence<N>());
  };


  // compound assignment operators
  vec& compound_assignment_operator(auto op, const vec& b) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] = op(data[i], b.data[i]);
    }
    return *this;
  }

  vec& compound_assignment_operator(auto op, const vec& b, const mask_t& m) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] = m[i] ? op(data[i], b.data[i]) : data[i];
    }
    return *this;
  }

  vec& operator+=(const vec& o) noexcept { return compound_assignment_operator(std::plus<T>(), o); }
  vec& assignment_plus(const vec& o, const mask_t& m) noexcept { return compound_assignment_operator(std::plus<T>(), o, m); }
  vec& operator-=(const vec& o) noexcept { return compound_assignment_operator(std::minus<T>(), o); }
  vec& operator|=(const vec& o) noexcept { return compound_assignment_operator(std::bit_or<T>(), o); }
  vec& operator^=(const vec& o) noexcept { return compound_assignment_operator(std::bit_xor<T>(), o); }
  vec& operator&=(const vec& o) noexcept { return compound_assignment_operator(std::bit_and<T>(), o); }


  vec& compound_assignment_operator(auto op, const T& b) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] = op(data[i], b);
    }
    return *this;
  }

  vec& compound_assignment_operator(auto op, const T& b, const mask_t& m) noexcept {
    for (size_t i = 0; i < N; i++) {
      data[i] = m[i] ? op(data[i], b) : data[i];
    }
    return *this;
  }

  vec& operator+=(const T& o) noexcept { return compound_assignment_operator(std::plus<T>(), o); }
  vec& assignment_plus(const T& o, const mask_t& m) noexcept { return compound_assignment_operator(std::plus<T>(), o, m); }
  vec& operator-=(const T& o) noexcept { return compound_assignment_operator(std::minus<T>(), o); }
  vec& operator|=(const T& o) noexcept { return compound_assignment_operator(std::bit_or<T>(), o); }
  vec& operator^=(const T& o) noexcept { return compound_assignment_operator(std::bit_xor<T>(), o); }
  vec& operator&=(const T& o) noexcept { return compound_assignment_operator(std::bit_and<T>(), o); }
  vec& assignment_bit_and(const T& o, const mask_t& m) noexcept { return compound_assignment_operator(std::bit_and<T>(), o, m); }



};

} // namespace simd

template<typename T, u64 N>
using vec = simd::vec<T, N, std::array<T, N>>;

template<typename Td, typename Ti, u64 N>
static vec<Td, N> gather(const Td* const ptr, const vec<Ti, N>& indices) {
  vec<Td, N> d;
  for (size_t i = 0; i < N; i++) {
    d.data[i] = ptr[indices[i]];
  }
  return d;
}


template<typename T, u64 N>
vec<T, N> operator<<(const T& lhs, const vec<T, N>& rhs) {
  vec<T, N> r;
  for ($u64 i = 0; i < N; i++) {
    r[i] = lhs << rhs[i];
  }
  return r;
}

/// not sure if this is causing problems...
template<typename Tl, typename T, u64 N>
vec<T, N> operator<<(const Tl& lhs, const vec<T, N>& rhs) {
  vec<T, N> r;
  for ($u64 i = 0; i < N; i++) {
    r[i] = lhs << rhs[i];
  }
  return r;
}