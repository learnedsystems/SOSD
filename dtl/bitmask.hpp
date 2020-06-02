#pragma once

#include "adept.hpp"
#include "math.hpp"
#include <array>
#include <bitset>
#include <functional>
#include <vector>
#include "immintrin.h"

#include "tree.hpp"

namespace dtl {

  template<u64 LEN>
  struct match_vector {
    $u32 match_positions[LEN];
    $u32 match_cnt;
  };

  /// Transforms a bitmask into a match vector.
  /// Note: Based on the implementation of Song and Chen described in the paper 'Exploiting SIMD for Complex
  /// Numerical Predicates'. This implementation works well for very selective queries where only few bits are set.
  template<u64 LEN>
  static void extract_match_positions($u32 bitmask, match_vector<LEN>& matches) {
    static_assert(LEN >= 32, "Match vector length must at least be equal to LEN");
    $u32* writer = matches.match_positions;
    for ($u32 m = _mm_popcnt_u32(bitmask); m > 0; m--) {
      $u32 bit_pos = __builtin_ctz(bitmask);
      *writer = bit_pos;
      bitmask = _blsr_u32(bitmask);
      writer++;
    }
    matches.match_cnt = writer - matches.match_positions;
  }

  template<u64 LEN>
  static void extract_match_positions($u64 bitmask, match_vector<LEN>& matches) {
    static_assert(LEN >= 64, "Match vector length must at least be equal to LEN");
    $u32* writer = matches.match_positions;
    for ($u32 m = _mm_popcnt_u64(bitmask); m > 0; m--) {
      $u32 bit_pos = __builtin_ctzll(bitmask);
      *writer = bit_pos;
      bitmask = _blsr_u64(bitmask);
      writer++;
    }
    matches.match_cnt = writer - matches.match_positions;
  }

} // namespace dtl
