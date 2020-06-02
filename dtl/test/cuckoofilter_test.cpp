#include "gtest/gtest.h"

#include <dtl/dtl.hpp>
#include <dtl/bloomfilter/blocked_cuckoofilter_util.hpp>

using namespace dtl;
using namespace dtl::cuckoo_filter;


template<typename data_t, uint32_t bits_per_value>
void contains_test() {

  const uint32_t data_bitlength = sizeof(data_t) * 8;

  const uint32_t min_value = 1;
  const uint32_t max_value = (1u << bits_per_value) - 1;

  const uint32_t max_elements = data_bitlength / bits_per_value;

  for (uint32_t value = min_value; value <= max_value; value++) {
    data_t data = value;
    uint32_t match_cntr = 0;
    for (uint32_t i = 0; i < data_bitlength; i++) {
      auto element_num = i / bits_per_value;
      bool is_contained = packed_value<data_t, bits_per_value>::contains(data, value);
      if (i % bits_per_value == 0
          && element_num < max_elements) {
        ASSERT_TRUE(is_contained) << "Failed with value=" << value << ", i=" << i << ", element_num=" << element_num;
        match_cntr++;
      }
      else {
        ASSERT_FALSE(is_contained) << "Failed with value=" << value << ", i=" << i << ", element_num=" << element_num;
      }
      data <<= 1;
    }
    ASSERT_EQ(match_cntr, max_elements);
  }


}

TEST(cuckoofilter, packed_value_contains_test) {
  contains_test<uint32_t, 2>();
  contains_test<uint32_t, 3>();
  contains_test<uint32_t, 4>();
  contains_test<uint32_t, 5>();
  contains_test<uint32_t, 6>();
  contains_test<uint32_t, 7>();
  contains_test<uint32_t, 8>();
  contains_test<uint32_t, 10>();
  contains_test<uint32_t, 12>();
//  contains_test<uint32_t, 15>();
  contains_test<uint32_t, 16>();
  contains_test<uint64_t, 2>();
  contains_test<uint64_t, 3>();
  contains_test<uint64_t, 4>();
  contains_test<uint64_t, 5>();
  contains_test<uint64_t, 6>();
  contains_test<uint64_t, 7>();
  contains_test<uint64_t, 8>();
  contains_test<uint64_t, 10>();
  contains_test<uint64_t, 12>();
//  contains_test<uint64_t, 15>();
  contains_test<uint64_t, 16>();
  SUCCEED();
}

