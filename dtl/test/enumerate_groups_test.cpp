#include "gtest/gtest.h"

#include <bitset>

// --- helper ---
inline uint32_t
lz_cnt_u16(const uint16_t a) {
  return a ? __builtin_clz(static_cast<uint32_t>(a)) - 16 : 0;
}
inline uint32_t
tz_cnt_u16(const uint16_t a) {
  return a ? __builtin_ctz(static_cast<uint32_t>(a)) : 0;
}

std::string
to_string_u32(const __m512i values) {
  auto a = reinterpret_cast<const uint32_t*>(&values);
  std::stringstream str;
  str << "[ ";
  str << a[15];
  for (std::size_t i = 15; i > 0; i--) {
    str << ", " << a[i - 1];
  }
  str << " ]";
  return str.str();
}

inline void
assert_eq(const __m512i& a, const __m512i& b) {
  bool is_eq = __mmask16(~0) == _mm512_cmpeq_epi32_mask(a, b);
  if (!is_eq) {
    std::cout << "a = " << to_string_u32(a) << std::endl;
    std::cout << "b = " << to_string_u32(b) << std::endl;
  }
  ASSERT_EQ(__mmask16(~0), _mm512_cmpeq_epi32_mask(a, b));
}

// --- ---


const __m512i ZERO = _mm512_setzero_si512();
const __m512i ONE = _mm512_set_epi32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
const __m512i SEQ = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
const __m512i SEQ1= _mm512_set_epi32(16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1);


/// Group the elements in the input vector and enumerate the elements
/// of each individual group starting with 0.
template<bool TwoPaths = true>
inline __m512i
enumerate_groups(const __m512i& elements) {
  // the return value
  __m512i enumeration = ZERO;
  // detect conflicts
  __m512i conflicts = _mm512_conflict_epi32(elements);
  // determine conflict free elements (the number of conflict free elements = the number of groups)
  __mmask16 conflict_free_mask = _mm512_cmpeq_epi32_mask(ZERO, conflicts);
  // remaining (conflicting) elements
  __mmask16 remaining_mask = ~conflict_free_mask;

  if (TwoPaths) {
//    const auto group_cnt = _popcnt32(conflict_free_mask);
//    if (group_cnt > 3 && group_cnt < 13) {
      // walk the chain of conflicts (in parallel) and solve them step-by-step.
      // in worst case the chain length is 16, however if the number
      // of groups is not in [4,12], we fall back to a different strategy (see below).
      while (remaining_mask) {
        // remove already handled conflicts
        conflicts = _mm512_and_epi32(conflicts, _mm512_set1_epi32(remaining_mask));
        // update the conflict free mask
        conflict_free_mask = _mm512_mask_cmpeq_epi32_mask(remaining_mask, ZERO, conflicts);
        // enumerate the conflict free elements of the current iteration
        enumeration = _mm512_mask_add_epi32(enumeration, remaining_mask, enumeration, ONE);
        remaining_mask ^= conflict_free_mask;
      }
      return enumeration;
//    }
  }

  // process remaining elements (in at most 8 iterations; or 4 iterations if 'TwoPaths' == true)
  while (remaining_mask) {
    // determine the left-most (conflicting) element with the most conflicts
    const auto idx = 15 - lz_cnt_u16(remaining_mask);
    // read the conflict mask of the left-most element
    const auto conflict_mask = reinterpret_cast<const uint32_t*>(&conflicts)[idx];
    // enumerate elements
    enumeration = _mm512_mask_expand_epi32(enumeration, conflict_mask, SEQ1);
    remaining_mask = (remaining_mask ^ (conflict_mask | 1u << idx)) & remaining_mask;
  }
  return enumeration;
}


TEST(enumerate_groups, single_group) {
  assert_eq(_mm512_set_epi32(15,14,13,12, 11,10,9,8, 7,6,5,4, 3,2,1,0),
            enumerate_groups(ZERO) );
}

TEST(enumerate_groups, one_element_per_group) {
  assert_eq(ZERO,
            enumerate_groups(SEQ) );
}

TEST(enumerate_groups, two_fourteen) {
  assert_eq(_mm512_set_epi32(13,1,12,11, 10,0,9,8, 7,6,5,4, 3,2,1,0),
            enumerate_groups(_mm512_set_epi32(0,1,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0)) );
}

TEST(enumerate_groups, four_groups) {
  assert_eq(_mm512_set_epi32(3,2,1,0, 3,2,1,0, 3,2,1,0, 3,2,1,0),
            enumerate_groups(_mm512_set_epi32(3,3,3,3, 2,2,2,2, 1,1,1,1, 0,0,0,0)) );
}

TEST(enumerate_groups, eight_groups) {
  assert_eq(_mm512_set_epi32(7,6,5,4, 7,6,5,4, 3,2,1,0, 3,2,1,0),
            enumerate_groups(_mm512_set_epi32(1,1,1,1, 0,0,0,0, 1,1,1,1, 0,0,0,0)) );
}

struct __m512i__mmask16 {
  __m512i vector;
  __mmask16 mask;
};

/// Group the elements in the input vector and enumerate the elements
/// of each individual group starting with 0.
template<bool TwoPaths = true>
inline __m512i__mmask16
group_count(const __m512i& elements) {
  // the return value
  __m512i__mmask16 counter;
  // detect conflicts
  __m512i conflicts = _mm512_conflict_epi32(elements);
  // determine conflict free elements (the number of conflict free elements = the number of groups)
  __mmask16 conflict_free_mask = _mm512_cmpeq_epi32_mask(ZERO, conflicts);
  // the aggregates are stored in the conflict free lanes
  counter.mask = conflict_free_mask;
  // remaining (conflicting) elements
  __mmask16 remaining_mask = ~conflict_free_mask;

  if (TwoPaths) {
//    const auto group_cnt = _popcnt32(conflict_free_mask);
//    if (group_cnt > 3 && group_cnt < 13) {
    // walk the chain of conflicts (in parallel) and solve them step-by-step.
    // in worst case the chain length is 16, however if the number
    // of groups is not in [4,12], we fall back to a different strategy (see below).
    while (remaining_mask) {
      // remove already handled conflicts
      conflicts = _mm512_and_epi32(conflicts, _mm512_set1_epi32(remaining_mask));
      // update the conflict free mask
      conflict_free_mask = _mm512_mask_cmpeq_epi32_mask(remaining_mask, ZERO, conflicts);
      // enumerate the conflict free elements of the current iteration
      counter = _mm512_mask_add_epi32(counter, remaining_mask, counter, ONE);
      remaining_mask ^= conflict_free_mask;
    }
    return
//    }
  }

  // process remaining elements (in at most 8 iterations; or 4 iterations if 'TwoPaths' == true)
  while (remaining_mask) {
    // determine the left-most (conflicting) element with the most conflicts
    const auto idx = 15 - lz_cnt_u16(remaining_mask);
    // read the conflict mask of the left-most element
    const auto conflict_mask = reinterpret_cast<const uint32_t*>(&conflicts)[idx];
    // enumerate elements
    counter = _mm512_mask_expand_epi32(counter, conflict_mask, SEQ1);
    remaining_mask = (remaining_mask ^ (conflict_mask | 1u << idx)) & remaining_mask;
  }
  return counter;
}
