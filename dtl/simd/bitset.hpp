#pragma once

#include <cstddef>

#include <dtl/dtl.hpp>
#include <dtl/bitset.hpp>
#include <dtl/simd.hpp>

namespace dtl {
namespace simd {

template<std::size_t Nb>
struct bitset : public dtl::bitset<Nb> {

  /// Returns the value of the bit at the position pos.
  template<std::size_t N>
  static inline typename v<$u64, N>::m
  test(const v<$u64, N>& bitset_addr, const v<$u64, N>& pos) {
    const auto word_idx = pos >> 6;
    const auto bit_idx = pos & 0b111111ull;
    const auto word_addr = bitset_addr + offsetof(bitset, _M_w) + (word_idx << 3);
    const auto word = dtl::gather<$u64>(word_addr);
    const auto test_mask = v<$u64, N>::make(1) << bit_idx;
    return (word & test_mask) != 0;
  }

};

} // namespace simd
} // namespace dtl