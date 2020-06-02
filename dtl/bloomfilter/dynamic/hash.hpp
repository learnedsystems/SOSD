#pragma once

#include <dtl/dtl.hpp>

namespace dtl {
namespace bloomfilter_dynamic {

struct hasher_mul32 {

  __forceinline__ __host__ __device__
  static u32 hash(u32& key, u32 hash_no) {
    static constexpr u32 primes[13] {
        596572387u,   // Peter 1
        370248451u,   // Peter 2
        2654435769u,  // Knuth 1
        1799596469u,  // Knuth 2
        0x9E3779B1u,  // https://lowrey.me/exploring-knuths-multiplicative-hash-2/
        2284105051u,  // Impala 3
        1203114875u,  // Impala 1 (odd, not prime)
        1150766481u,  // Impala 2 (odd, not prime)
        2729912477u,  // Impala 4 (odd, not prime)
        1884591559u,  // Impala 5 (odd, not prime)
        770785867u,   // Impala 6 (odd, not prime)
        2667333959u,  // Impala 7 (odd, not prime)
        1550580529u,  // Impala 8 (odd, not prime)
    };
    if (hash_no > 13) {
      std::cerr << "hash_no out of bounds: " << hash_no << std::endl;
      throw "BAM";
    }
    return key * primes[hash_no];
  }
};

} // namespace bloomfilter_dynamic
} // namespace dtl