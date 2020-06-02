#include "gtest/gtest.h"

#include <dtl/dtl.hpp>
#include <dtl/bitpack.hpp>


template<typename T>
void run_test_unsigned(u64 k, u64 N) {
  static_assert(std::is_unsigned<T>::value, "unsigned integer type expected.");

  std::vector<T> in;
  for (std::size_t i = 0; i < N; i++) {
    in.push_back(i);
  }

  auto bitpacked = dtl::bitpack_horizontal(k, in);
  auto out = dtl::bitunpack_horizontal<T>(bitpacked);

  for (std::size_t i = 0; i < N; i++) {
    ASSERT_EQ(in[i] % (1ull << k), out[i]);
  }
}

TEST(bitpack, horizontal_unsigned) {
  for ($u64 k = 1; k < 32; k++) {
    run_test_unsigned<$u64>(k, 1ull << 16);
    run_test_unsigned<$u32>(k, 1ull << 16);
  }
}


TEST(bitpack, horizontal_sign_extend) {
  std::vector<$i32> in { -2, -1, 0, 1, 2};
  auto bitpacked = dtl::bitpack_horizontal(8, in);
  auto out = dtl::bitunpack_horizontal<$i32>(bitpacked);
  ASSERT_EQ(in, out);
}


template<typename T>
void run_test_signed(u64 k, u64 N) {
  static_assert(std::is_signed<T>::value, "signed integer type expected.");

  std::vector<T> in;
  $i64 x = N/2;
  for ($i64 i = -x; i < x; i++) {
    in.push_back(i);
  }

  auto bitpacked = dtl::bitpack_horizontal(k, in);
  auto out = dtl::bitunpack_horizontal<T>(bitpacked);

  for (std::size_t i = 0; i < N; i++) {
    ASSERT_EQ(in[i], out[i]);
  }
}

TEST(bitpack, horizontal_signed) {
  for ($u64 k = 8; k < 32; k++) {
    run_test_signed<$i64>(k, 1ull << 7);
    run_test_signed<$i32>(k, 1ull << 7);
  }
}
