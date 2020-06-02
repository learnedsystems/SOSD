#pragma once

#include "adept.hpp"
#include "math.hpp"
#include <bitset>

namespace dtl {

  template<u64 N, u64 M>
  class zone_mask {
    static_assert(is_power_of_two(N), "Template parameter 'N' must be a power of two.");
    static_assert(is_power_of_two(M), "Template parameter 'M' must be a power of two.");
    static_assert(M <= N, "Template parameter 'M' must be less or equal to 'N'.");

  public:

    static constexpr u64 zone_size = N / M;

    std::bitset<M> data;

    inline void
    reset() {
      data.reset();
    }

    inline void
    set(u64 i) {
      data.set(i / zone_size);
    }

    inline zone_mask
    operator|(const zone_mask& other) {
      return zone_mask { data | other.data };
    }

    inline zone_mask
    operator&(const zone_mask& other) {
      return zone_mask { data & other.data };
    }


    static std::bitset<M>
    compress(const std::bitset<N>& bitmask) {
      static_assert(is_power_of_two(M), "Template parameter 'M' must be a power of two.");
      u64 zone_size = N / M;
      u64 zone_cnt = N / zone_size;

      std::bitset<N> tmp_mask;
      for ($u64 i = 0; i < zone_size; i++) {
        tmp_mask |= bitmask >> i;
      }

      std::bitset<M> zone_mask;
      for ($u64 i = 0; i < zone_cnt; i++) {
        zone_mask[i] = tmp_mask[i * zone_size];
      }
      return zone_mask;
    }

    static std::bitset<N>
    decode(const std::bitset<M>& compressed_bitmask) {
      static_assert(is_power_of_two(M), "Template parameter 'M' must be a power of two.");
      u64 zone_size = N / M;
      u64 zone_cnt = N / zone_size;
      std::bitset<N> bitmask;
      for ($u64 i = 0; i < zone_cnt; i++) {
        if (!compressed_bitmask[i]) continue;
        for ($u64 j = 0; j < zone_size; j++) {
          bitmask[i * zone_size + j] = true;
        }
      }
      return bitmask;
    }

  };

}
