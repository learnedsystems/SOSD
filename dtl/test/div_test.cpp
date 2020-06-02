#include "gtest/gtest.h"

#include <functional>

#include <dtl/dtl.hpp>
#include <dtl/div.hpp>
#include <dtl/simd.hpp>


TEST(fast_modulus, random_x86) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(1, std::numeric_limits<uint32_t>::max() - 2);

  uint32_t a = 0;
  uint32_t b = dtl::max_divisor_u32;
  auto b_fast = dtl::next_cheap_magic(dtl::max_divisor_u32);
  uint32_t r = a % b_fast.divisor;

  for (std::size_t repeat = 0; repeat < 1000000; repeat++) {
    ASSERT_EQ(a % b_fast.divisor, r)
                  << "Fast modulus computation failed for " << a << " mod " << b_fast.divisor
                  << " (magic=" << b_fast.magic << ", shift_amount=" << b_fast.shift_amount << ").";

    a = dis(gen);
    if (repeat % 10 == 0) {
      b = dis(gen);
      b_fast = dtl::next_cheap_magic(b);
    }
    r = a % b_fast.divisor;
  }
}

TEST(fast_modulus, random_avx2) {
  using dtl::r256;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(1, std::numeric_limits<uint32_t>::max() - 2);

  r256 a { .i = _mm256_set1_epi32(0) };
  uint32_t b = dtl::max_divisor_u32;
  auto b_fast = dtl::next_cheap_magic(dtl::max_divisor_u32);

  for (std::size_t repeat = 0; repeat < 100000; repeat++) {
    const r256 r { .i = dtl::fast_mod_u32(a.i, b_fast) };

    for (std::size_t i = 0; i < 8; i++) {
      ASSERT_EQ(a.u32[i] % b_fast.divisor, r.u32[i])
                << "Fast modulus computation failed for " << a.u32[i] << " mod " << b_fast.divisor
                << " (magic=" << b_fast.magic << ", shift_amount=" << b_fast.shift_amount << ").";
    }

    for (std::size_t i = 0; i < 8; i++) {
      a.u32[i] = dis(gen);
    }
    if (repeat % 10 == 0) {
      b = dis(gen);
      b_fast = dtl::next_cheap_magic(b);
    }
  }
}


#if defined(__AVX512F__)

TEST(fast_modulus, random_avx512) {
  using dtl::r512;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(1, std::numeric_limits<uint32_t>::max() - 2);

  r512 a { .i = _mm512_set1_epi32(0) };
  uint32_t b = dtl::max_divisor_u32;
  auto b_fast = dtl::next_cheap_magic(dtl::max_divisor_u32);

  for (std::size_t repeat = 0; repeat < 100000; repeat++) {
    const r512 r { .i = dtl::fast_mod_u32(a.i, b_fast) };

    for (std::size_t i = 0; i < 16; i++) {
      ASSERT_EQ(a.u32[i] % b_fast.divisor, r.u32[i])
                << "Fast modulus computation failed for " << a.u32[i] << " mod " << b_fast.divisor
                << " (magic=" << b_fast.magic << ", shift_amount=" << b_fast.shift_amount << ").";
    }

    for (std::size_t i = 0; i < 16; i++) {
      a.u32[i] = dis(gen);
    }
    if (repeat % 10 == 0) {
      b = dis(gen);
      b_fast = dtl::next_cheap_magic(b);
    }
  }
}

#endif // defined(__AVX512F__)

TEST(fast_modulus, random_vec) {

  u64 vec_len = dtl::simd::lane_count<u32>;
  using vec_t = dtl::vec<u32, vec_len>;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(1, std::numeric_limits<uint32_t>::max() - 2);

  vec_t a = 0;
  uint32_t b = dtl::max_divisor_u32;
  auto b_fast = dtl::next_cheap_magic(dtl::max_divisor_u32);

  for (std::size_t repeat = 0; repeat < 100000; repeat++) {
    const vec_t r = dtl::fast_mod_u32(a, b_fast);

    for (std::size_t i = 0; i < vec_len; i++) {
      ASSERT_EQ(a[i] % b_fast.divisor, r[i])
                    << "Fast modulus computation failed for " << a[i] << " mod " << b_fast.divisor
                    << " (magic=" << b_fast.magic << ", shift_amount=" << b_fast.shift_amount << ").";
    }

    for (std::size_t i = 0; i < vec_len; i++) {
      a.insert(dis(gen), i);
    }
    if (repeat % 10 == 0) {
      b = dis(gen);
      b_fast = dtl::next_cheap_magic(b);
    }
  }
}


