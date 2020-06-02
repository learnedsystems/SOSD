#pragma once

#include "dtl.hpp"
#include "nmmintrin.h"

#include <stdint.h>

namespace dtl {
namespace hash {

// fnv(x) = y
// peter1: 2084897189
// peter2: 2713107310
// knuth1: 2348177672
// knuth2: 931137737

constexpr uint32_t FNV1_32A_INIT = 0x811c9dc5u;

/// 32 bit FNV-1a
template<typename T, uint32_t init = FNV1_32A_INIT>
struct fnv_32a {
  using Ty = typename std::remove_cv<T>::type;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    const Ty byte_mask = 0b11111111;
    const Ty fnv_32_prime = 0x01000193u;
    Ty h = init;

    /*
     * FNV-1a hash each octet
     */
    const Ty oct_3 = (key >> 24) & byte_mask;
    h ^= oct_3;
    h *= fnv_32_prime;
    const Ty oct_2 = (key >> 16) & byte_mask;
    h ^= oct_2;
    h *= fnv_32_prime;
    const Ty oct_1 = (key >> 8) & byte_mask;
    h ^= oct_1;
    h *= fnv_32_prime;
    const Ty oct_0 = key & byte_mask;
    h ^= oct_0;
    h *= fnv_32_prime;

    /* return our new hash value */
    return h;
  }
};

template<typename T>
struct fnv_32 {
  using Ty = typename std::remove_cv<T>::type;
  using F = fnv_32a<T>;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return F::hash(key);
  }
};

template<typename T>
struct fnv_32_alt {
  using Ty = typename std::remove_cv<T>::type;
  using F = fnv_32a<T, 2084897189u>;

  __host__ __device__
  static inline Ty
  hash(const Ty& key) {
    return F::hash(key);
  }
};

} // namespace hash
} // namespace dtl
