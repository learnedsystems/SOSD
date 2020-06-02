#pragma once

#include <iostream>
#include <cstdint>
#include <array>

#include <boost/align/aligned_allocator.hpp>


// type aliases for more concise code
using i1 = const bool;
using u1 = const bool;
using i8 = const int8_t;
using u8 = const uint8_t;
using i16 = const int16_t;
using u16 = const uint16_t;
using i32 = const int32_t;
using u32 = const uint32_t;
using i64 = const int64_t;
using u64 = const uint64_t;

using $i1 = bool;
using $u1 = bool;
using $i8 = int8_t;
using $u8 = uint8_t;
using $i16 = int16_t;
using $u16 = uint16_t;
using $i32 = int32_t;
using $u32 = uint32_t;
using $i64 = int64_t;
using $u64 = uint64_t;

using f32 = const float;
using f64 = const double;

using $f32 = float;
using $f64 = double;


// polyfill until C++14
template<u64...>
struct integer_sequence {};

template<u64 N, u64... Ints>
struct make_integer_sequence : make_integer_sequence<N - 1, N - 1, Ints...> {};

template<u64... Ints>
struct make_integer_sequence<0, Ints...> : integer_sequence<Ints...> {};


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


// Compiler hints
#ifndef assume_aligned
#ifndef __CUDA_ARCH__
#define assume_aligned(address, byte) __builtin_assume_aligned(address, byte)
#else
#define assume_aligned(address, byte)
#endif
#endif

//#ifndef unreachable
//#define unreachable() __builtin_unreachable();
//#endif

#ifndef likely
#define likely(expr) __builtin_expect(!!(expr), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect((x),0)
#endif

#if defined(NDEBUG)
  #if !defined(forceinline)
  #define forceinline inline __attribute__((always_inline))
  #endif
#else
  #if !defined(forceinline)
  #define forceinline
  #endif
#endif

#if defined(NDEBUG)
// Release build.
  #if !defined(__forceinline__) && !(defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__))
    #define __forceinline__ inline __attribute__((always_inline))
  #endif
#else
  // Debug build. __forceinline__ does nothing
  #if !defined(__forceinline__)
    #define __forceinline__
  #endif
#endif

#if defined(NDEBUG)
  #define unroll_loops __attribute__((optimize("unroll-loops")))
#else
  #define unroll_loops
#endif

#if defined(NDEBUG)
  #define __unroll_loops__ __attribute__((optimize("unroll-loops")))
#else
  #define __unroll_loops__
#endif

// add missing operator function objects
namespace std {

  template<typename T>
  struct bit_shift_left {
    constexpr T operator()(const T &lhs, const T &rhs) const {
      return lhs << rhs;
    }
  };

  template<typename T>
  struct bit_shift_right {
    constexpr T operator()(const T &lhs, const T &rhs) const {
      return lhs >> rhs;
    }
  };

  template<typename T>
  struct post_increment {
    constexpr T operator()(const T &lhs) const {
      return lhs++;
    }
  };

  template<typename T>
  struct post_decrement {
    constexpr T operator()(const T &lhs) const {
      return lhs--;
    }
  };

}

#include <bitset>
#include <vector>

template<class T>
auto operator<<(std::ostream& os, const T& t) -> decltype(t.print(os), os) {
  t.print(os);
  return os;
}

template<size_t n>
void print(const std::bitset<n>& b) {
  for (size_t i = 0; i < n; i++) {
    std::cout << b[i];
  }
  std::cout << std::endl;
}

template<size_t n>
void print(const std::bitset<n>& b, const size_t l) {
  const size_t x = 1 << l;
  for (size_t i = 0; i < n; i++) {
    if (i % x == 0) {
      std::cout << b[i];
    }
    else {
      std::cout << "_";
    }
  }
  std::cout << std::endl;
}

template<typename T>
void print(const std::vector<T>& v, const size_t l) {
  const size_t x = 1 << l;
  for (size_t i = 0; i < v.size(); i++) {
    if (i % x == 0) {
//      std::cout << v[v.size() - (i + 1)] << ", ";
      std::cout << v[i] << ", ";
    }
    else {
      std::cout << "_, ";
    }
  }
  std::cout << std::endl;
}

[[maybe_unused]] static void print(const std::vector<bool>& v, const size_t l) {
  const size_t x = 1 << l;
  for (size_t i = 0; i < v.size(); i++) {
    if (i % x == 0) {
      std::cout << v[i];
    }
    else {
      std::cout << "_, ";
    }
  }
  std::cout << std::endl;
}


[[maybe_unused]] static void
print(const std::vector<bool>& v) {
  if (v.size() == 0) return;
  std::cout << (v[0] ? "1" : "0");
  for (size_t i = 1; i < v.size(); i++) {
    std::cout << "" << (v[i] ? "1" : "0");
  }
  std::cout << std::endl;
}


// for CUDA portability
#if !defined(__host__)
#define __host__
#endif

#if !defined(__device__)
#define __device__
#endif

#if !defined(__restrict__)
#define __restrict__
#endif
// ---


namespace dtl {

template<typename T, std::size_t A = 64>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, A>>;

//template<typename T, std::size_t L>
//using aligned_array = alignas(64) std::array<T, L>; //FIXME

} // namespace dtl
