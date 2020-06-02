#pragma once

#include <dtl/dtl.hpp>
#include <dtl/simd.hpp>

namespace dtl {
namespace hash {

namespace stat { // static

//===----------------------------------------------------------------------===//
// Multiplicative hashing of 32-bit keys (compile-time static).
//
// Note: The key type is a template parameter to support dtl::vec types (SIMD).
//===----------------------------------------------------------------------===//
template<
    typename key_t,
    $u32 hash_fn_no
>
struct mul32 { };

template<typename key_t>
struct mul32<key_t, 0> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 596572387u; } // Peter 1
};

template<typename key_t>
struct mul32<key_t, 1> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 370248451u; } // Peter 2
};

template<typename key_t>
struct mul32<key_t, 2> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 2654435769u; } // Knuth 1
};

template<typename key_t>
struct mul32<key_t, 3> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1799596469u; } // Knuth 2
};

template<typename key_t>
struct mul32<key_t, 4> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 0x9e3779b1u; } // https://lowrey.me/exploring-knuths-multiplicative-hash-2/
};

template<typename key_t>
struct mul32<key_t, 5> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 2284105051u; } // Impala 3
};

template<typename key_t>
struct mul32<key_t, 6> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1203114875u; } // Impala 1 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 7> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1150766481u; } // Impala 2 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 8> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 2729912477u; } // Impala 4 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 9> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1884591559u; } // Impala 5 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 10> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 770785867u; } // Impala 6 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 11> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 2667333959u; } // Impala 7 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 12> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 1550580529u; } // Impala 8 (odd, not prime)
};

template<typename key_t>
struct mul32<key_t, 13> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 0xcc9e2d51u; } // Murmur 3 (x86_32 c1)
};

template<typename key_t>
struct mul32<key_t, 14> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 0x1b873593u; } // Murmur 3 (x86_32 c2)
};

template<typename key_t>
struct mul32<key_t, 15> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 0x85ebca6bu; } // Murmur 3 (finalization mix constant)
};

template<typename key_t>
struct mul32<key_t, 16> {
  __forceinline__ __host__ __device__
  static key_t hash(const key_t& key) { return key * 0xc2b2ae35u; } // Murmur 3 (finalization mix constant)
};
//===----------------------------------------------------------------------===//


} // namespace stat(ic)


namespace dyn { // dynamic

//===----------------------------------------------------------------------===//
// Multiplicative hashing of 32-bit keys (dynamic).
//===----------------------------------------------------------------------===//
struct mul32 {

  __forceinline__ __host__ __device__
  static u32
  hash(u32& key, u32 hash_no) {
    static constexpr u32 primes[17] {
        596572387u,   // Peter 1
        370248451u,   // Peter 2
        2654435769u,  // Knuth 1
        1799596469u,  // Knuth 2
        0x9e3779b1u,  // https://lowrey.me/exploring-knuths-multiplicative-hash-2/
        2284105051u,  // Impala 3
        1203114875u,  // Impala 1 (odd, not prime)
        1150766481u,  // Impala 2 (odd, not prime)
        2729912477u,  // Impala 4 (odd, not prime)
        1884591559u,  // Impala 5 (odd, not prime)
        770785867u,   // Impala 6 (odd, not prime)
        2667333959u,  // Impala 7 (odd, not prime)
        1550580529u,  // Impala 8 (odd, not prime)
        0xcc9e2d51u,  // Murmur 3 (x86_32 c1)
        0x1b873593u,  // Murmur 3 (x86_32 c2)
        0x85ebca6bu,  // Murmur 3 (finalization mix constant)
        0xc2b2ae35u,  // Murmur 3 (finalization mix constant)
    };
    if (hash_no > 16) {
      throw std::invalid_argument("hash_no out of bounds: " + std::to_string(hash_no));
    }
    return key * primes[hash_no];
  }

  //===----------------------------------------------------------------------===//
  // Template for dtl::vec types (SIMD).
  //===----------------------------------------------------------------------===//
  template<typename Tv, typename = std::enable_if_t<dtl::is_vector<Tv>::value>>
  __forceinline__ __host__
  static dtl::vec<uint32_t, dtl::vector_length<Tv>::value>
  hash(const Tv& keys,
       const uint32_t hash_no) {
    static constexpr u32 primes[17] {
        596572387u,   // Peter 1
        370248451u,   // Peter 2
        2654435769u,  // Knuth 1
        1799596469u,  // Knuth 2
        0x9e3779b1u,  // https://lowrey.me/exploring-knuths-multiplicative-hash-2/
        2284105051u,  // Impala 3
        1203114875u,  // Impala 1 (odd, not prime)
        1150766481u,  // Impala 2 (odd, not prime)
        2729912477u,  // Impala 4 (odd, not prime)
        1884591559u,  // Impala 5 (odd, not prime)
        770785867u,   // Impala 6 (odd, not prime)
        2667333959u,  // Impala 7 (odd, not prime)
        1550580529u,  // Impala 8 (odd, not prime)
        0xcc9e2d51u,  // Murmur 3 (x86_32 c1)
        0x1b873593u,  // Murmur 3 (x86_32 c2)
        0x85ebca6bu,  // Murmur 3 (finalization mix constant)
        0xc2b2ae35u,  // Murmur 3 (finalization mix constant)
    };
    if (hash_no > 16) {
      throw std::invalid_argument("hash_no out of bounds: " + std::to_string(hash_no));
    }
    return keys * primes[hash_no];
  };

};
//===----------------------------------------------------------------------===//


} // namespace dyn
} // namespace hash


template<
    typename key_t,
    $u32 hash_fn_no
>
using hasher = dtl::hash::stat::mul32<key_t, hash_fn_no>;


} // namespace dtl