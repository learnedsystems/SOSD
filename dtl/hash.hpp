#pragma once

#include "dtl.hpp"
#include "nmmintrin.h"

#include <stdint.h>

namespace dtl {
namespace hash {

template<typename T, u32 seed = 1337>
struct crc32 {
  using Ty = typename std::remove_cv<T>::type;

  static inline Ty
  hash(const Ty& key) {
    return _mm_crc32_u32(key, seed);
  }
};


template<typename T>
struct xorshift_32 {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    const u32 seed = 2654435769u;
    Ty h = key ^ seed;
    h ^= h << 13;
    h ^= h >> 17;
    h ^= h << 5;
    return h;
  }
};


template<typename T>
struct xorshift_64 {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    Ty h = key;
    h ^= h << 13;
    h ^= h >> 7;
    h ^= h << 17;
    return h;
  }
};


/// Knuth multiplicative hashing taken from TAOCP volume 3 (2nd edition), section 6.4, page 516.
/// see: http://stackoverflow.com/questions/11871245/knuth-multiplicative-hash
/// 0 <= p <= 32
template<typename T, u32 p = 32>
struct knuth_32 {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
//    Ty knuth = 2654435769u; // 0b10011110001101110111100110111001
    Ty knuth = 596572387u; // Peter 1
    return (key * knuth) >> (32 - p);
  }
};


template<typename T, u32 p = 32>
struct knuth_32_alt {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
//    Ty knuth = 1799596469u; // 0b01101011010000111010100110110101
    Ty knuth = 370248451u; // Peter 2
    return (key * knuth) >> (32 - p);
  }
};


template<typename T, u32 p = 32>
struct knuth_32_alt2 {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    Ty knuth = 2654435769u; // 0b10011110001101110111100110111001
//    Ty knuth = 596572387u; // Peter 1
    return (key * knuth) >> (32 - p);
  }
};



template<typename T>
struct knuth {
  using Ty = typename std::remove_cv<T>::type;
  using F = knuth_32<T, 32>;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return F::hash(key);
  }
};


template<typename T>
struct knuth_alt {
  using Ty = typename std::remove_cv<T>::type;
  using F = knuth_32_alt<T, 32>;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return F::hash(key);
  }
};


template<typename T>
struct knuth_alt2 {
  using Ty = typename std::remove_cv<T>::type;
  using F = knuth_32_alt2<T, 32>;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return F::hash(key);
  }
};


template<typename T>
struct knuth64 {
  using Ty = typename std::remove_cv<T>::type;
  using F = knuth_32<T, 64>;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return F::hash(key);
  }
};


template<typename T>
struct identity {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return key;
  }
};


template<typename T, u32 seed = 596572387u>
struct murmur1_32 {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    const Ty m = seed;
    const Ty hi = (m * 4u) ^ 0xc6a4a793u;
    Ty h = hi;
    h += key;
    h *= m;
    h ^= h >> 16;
    h *= m;
    h ^= h >> 10;
    h *= m;
    h ^= h >> 17;
    return h;
  }
};


template<typename T>
struct murmur_32 {
  using Ty = typename std::remove_cv<T>::type;
  using F = murmur1_32<T, 596572387u>;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return F::hash(key);
  }
};


template<typename T>
struct murmur_32_alt {
  using Ty = typename std::remove_cv<T>::type;
  using F = murmur1_32<T, 370248451u>;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return F::hash(key);
  }
};


template<typename T, u64 seed = 0xc6a4a7935bd1e995ull>
struct murmur64a_64 {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    const Ty m = seed;
    const Ty r = 47u;
    const Ty hi = 0x8445d61a4e774912ull ^ (8 * m);
    Ty h = hi;
    Ty k = key;
    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
  }
};

} // namespace hash
} // namespace dtl

#include "hash_fvn.hpp"