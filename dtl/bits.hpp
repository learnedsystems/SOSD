#pragma once

#include <dtl/dtl.hpp>

#if defined(__BMI__)
#include "immintrin.h"
#endif

namespace dtl {

namespace bits {


/// Counts the number of set bits.
#if defined(__CUDA_ARCH__)
__forceinline__ __device__
u32 pop_count(u32 a) { return __popc(a); }
__forceinline__ __device__
u32 pop_count(u64 a) { return __popcll(a); }
#else
//__forceinline__
//constexpr u32 pop_count(u8 a) { return __builtin_popcount(a); }
//__forceinline__
//constexpr u32 pop_count(u16 a) { return __builtin_popcount(a); }
//__forceinline__
//constexpr u32 pop_count(u32 a) { return __builtin_popcount(a); }
__forceinline__
constexpr u32 pop_count(u64 a) { return __builtin_popcountll(a); }
#endif


/// Counts the number of leading zeros.
#if defined(__CUDA_ARCH__)
__forceinline__ __device__
u64 lz_count(u32 a) { return __clz(a); }
#else
__forceinline__
constexpr u64 lz_count(u32 a) { return __builtin_clz(a); }
#endif

/// Counts the number of leading zeros.
#if defined(__CUDA_ARCH__)
__forceinline__ __device__
u64 lz_count(u64 a) { return __clzll(a); }
#else
__forceinline__
constexpr u64 lz_count(u64 a) { return __builtin_clzll(a); }
#endif


/// counts the number of tailing zeros
inline u64
tz_count(u32 a) { return __builtin_ctz(a); }

/// counts the number of tailing zeros
inline u64
tz_count(u64 a) { return __builtin_ctzll(a); }

#if defined(__BMI__)
inline u32
blsr_u32(u32 a) { return _blsr_u32(a); };
#else
inline u32
blsr_u32(u32 a) { return (a - 1) & a; };
#endif


/// extract contiguous bits
inline u32
extract(u32 a, u32 start, u32 len) {
#if defined(__BMI__)
  return _bextr_u32(a, start, len);
#else
  return (a >> start) & ((u32(1) << len) - 1);
#endif
}

/// extract contiguous bits
inline u64
extract(u64 a, u32 start, u32 len) {
#if defined(__BMI__)
  return _bextr_u64(a, start, len);
#else
  return (a >> start) & ((u64(1) << len) - 1);
#endif
}

/// return the i-th bit in a
__forceinline__ __host__ __device__
constexpr u1
bit_test(u8 a, u32 i) {
  return a & (u8(1) << i);
}

__forceinline__ __host__ __device__
constexpr u1
bit_test(u32 a, u32 i) {
  return a & (u32(1) << i);
}

__forceinline__ __host__ __device__
constexpr u1
bit_test(u64 a, u32 i) {
  return a & (u64(1) << i);
}


} // namespace bits

} // namespace dtl