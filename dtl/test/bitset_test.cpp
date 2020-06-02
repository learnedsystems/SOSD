#include "gtest/gtest.h"

#include <sstream>
#include <string>

#include <dtl/dtl.hpp>
#include <dtl/bitset.hpp>

#include <bitset>

template<typename T>
class bitset_test : public ::testing::Test {
public:
  using bitset_t = T;
  bitset_t bits;
  using ref_bitset_t = std::bitset<bitset_t::length>;
  ref_bitset_t ref_bits;

  void
  assert_eq() {
    for ($u64 i = 0; i < bitset_t::length; i++) {
      ASSERT_EQ(bits[i], ref_bits[i]);
    }
  }
};

typedef ::testing::Types<
//    dtl::bitset<8>,
//    dtl::bitset<64>,
    dtl::bitset<128>,
    dtl::bitset<256>,
    dtl::bitset<512>,
    dtl::bitset<1024>
> types_under_test;

TYPED_TEST_CASE(bitset_test, types_under_test);

TYPED_TEST(bitset_test, is_initialized_with_zeros) {
  for ($u64 i = 0; i < TestFixture::bitset_t::length; i++) {
    ASSERT_EQ(this->bits[i], false);
  }
}

TYPED_TEST(bitset_test, set_unset_bits) {
  for ($u64 i = 0; i < TestFixture::bitset_t::length; i++) {
    this->bits.set(i, true);
    ASSERT_EQ(this->bits[i], true);
    this->bits.set(i, false);
    ASSERT_EQ(this->bits[i], false);
  }
}

TYPED_TEST(bitset_test, assignment) {
  this->bits[0] = true;
  ASSERT_EQ(this->bits[0], true);
}

TYPED_TEST(bitset_test, left_shift_by_word_length) {
  u64 word_len = TestFixture::bitset_t::word_bitlength;
  this->bits[0] = true;
  auto act = this->bits << word_len;
  ASSERT_EQ(act[0], false);
  if (TestFixture::bitset_t::length > word_len) {
    ASSERT_EQ(act[64], true);
  }
}

TYPED_TEST(bitset_test, left_shift) {
  u64 word_len = TestFixture::bitset_t::word_bitlength;
  u64 shift = (word_len - 1);
  this->bits[7] = true;
  auto act = this->bits << shift;
  ASSERT_EQ(act[7], false);
  if (TestFixture::bitset_t::length > (7 + shift)) {
    ASSERT_EQ(act[7 + shift], true);
  }
}

TYPED_TEST(bitset_test, assign_left_shift_by_word_length) {
  u64 word_len = TestFixture::bitset_t::word_bitlength;
  this->bits[0] = true;
  this->bits <<= word_len;
  ASSERT_EQ(this->bits[0], false);
  if (TestFixture::bitset_t::length > word_len) {
    ASSERT_EQ(this->bits[word_len], true);
  }
}

TYPED_TEST(bitset_test, assign_left_shift) {
  u64 word_len = TestFixture::bitset_t::word_bitlength;
  u64 shift = (word_len - 1);
  this->bits[7] = true;
  this->bits <<= shift;
  ASSERT_EQ(this->bits[7], false);
  if (TestFixture::bitset_t::length > (7 + shift)) {
    ASSERT_EQ(this->bits[7 + shift], true);
  }
}

TYPED_TEST(bitset_test, right_shift_by_word_length) {
  u64 word_len = TestFixture::bitset_t::word_bitlength;
  this->bits[word_len] = true;
  auto act = this->bits >> word_len;
  ASSERT_EQ(act[word_len], false);
  ASSERT_EQ(act[0], true);
}

TYPED_TEST(bitset_test, right_shift) {
  u64 word_len = TestFixture::bitset_t::word_bitlength;
  u64 shift = 63;
  this->bits[63] = true;
  this->bits[64] = true;
  auto act = this->bits >> shift;
  ASSERT_EQ(act[63], false);
  ASSERT_EQ(act[64], false);
  ASSERT_EQ(act[0], true);
  ASSERT_EQ(act[1], true);
}

TYPED_TEST(bitset_test, assign_right_shift_by_word_length) {
  u64 word_len = TestFixture::bitset_t::word_bitlength;
  this->bits[word_len] = true;
  this->bits >>= word_len;
  ASSERT_EQ(this->bits[word_len], false);
  ASSERT_EQ(this->bits[0], true);
}

TYPED_TEST(bitset_test, assign_right_shift) {
  u64 word_len = TestFixture::bitset_t::word_bitlength;
  u64 shift = 63;
  this->bits[63] = true;
  this->bits[64] = true;
  this->bits >>= shift;
  ASSERT_EQ(this->bits[63], false);
  ASSERT_EQ(this->bits[64], false);
  ASSERT_EQ(this->bits[0], true);
  ASSERT_EQ(this->bits[1], true);
}

TYPED_TEST(bitset_test, left_shift_in) {
  for ($u64 i = 0; i < 10000; i++) {
    this->bits[0] = 1;
    this->bits << 1;
    this->ref_bits[0] = 1;
    this->ref_bits << 1;
    this->assert_eq();
  }
  this->bits.reset();
  this->ref_bits.reset();
  this->assert_eq();
}

TYPED_TEST(bitset_test, right_shift_in) {
  u64 msb_pos = TestFixture::bitset_t::length - 1;
  for ($u64 i = 0; i < 10000; i++) {
    this->bits[msb_pos] = 1;
    this->bits >> 1;
    this->ref_bits[msb_pos] = 1;
    this->ref_bits >> 1;
    this->assert_eq();
  }
  this->bits.reset();
  this->ref_bits.reset();
  this->assert_eq();
}

TYPED_TEST(bitset_test, find_on_bits) {
  u64 msb_pos = TestFixture::bitset_t::length - 1;
  this->bits[0] = 1;
  this->bits[1] = 1;
  this->bits[63] = 1;
  this->bits[64] = 1;
  this->bits[msb_pos] = 1;

  ASSERT_EQ(0, this->bits.find_first());
  ASSERT_EQ(1, this->bits.find_next(0));
  ASSERT_EQ(63, this->bits.find_next(1));
  ASSERT_EQ(64, this->bits.find_next(63));
  ASSERT_EQ(msb_pos, this->bits.find_next(64));

  auto it = this->bits.on_bits_begin();
  ASSERT_EQ(0, *it++);
  ASSERT_EQ(1, *it++);
  ASSERT_EQ(63, *it++);
  ASSERT_EQ(64, *it++);
  ASSERT_EQ(msb_pos, *it++);
}

TYPED_TEST(bitset_test, to_string) {
  u64 msb_pos = TestFixture::bitset_t::length - 1;
  this->bits[1] = 1;
  this->ref_bits[1] = 1;
  this->bits[2] = 1;
  this->ref_bits[2] = 1;
  this->bits[63] = 1;
  this->ref_bits[63] = 1;
  this->bits[64] = 1;
  this->ref_bits[64] = 1;
  this->bits[msb_pos] = 1;
  this->ref_bits[msb_pos] = 1;

  this->assert_eq();

  ASSERT_EQ(this->bits.to_string(), this->ref_bits.to_string());
  ASSERT_EQ(this->bits.to_string('_'), this->ref_bits.to_string('_'));
  ASSERT_EQ(this->bits.to_string('_', 'X'), this->ref_bits.to_string('_', 'X'));
}

TYPED_TEST(bitset_test, to_string_stream) {
  u64 msb_pos = TestFixture::bitset_t::length - 1;
  this->bits[1] = 1;
  this->ref_bits[1] = 1;
  this->bits[2] = 1;
  this->ref_bits[2] = 1;
  this->bits[63] = 1;
  this->ref_bits[63] = 1;
  this->bits[64] = 1;
  this->ref_bits[64] = 1;
  this->bits[msb_pos] = 1;
  this->ref_bits[msb_pos] = 1;

  this->assert_eq();

  std::stringstream stream;
  stream << this->bits;

  std::stringstream ref_stream;
  ref_stream << this->ref_bits;

  ASSERT_EQ(stream.str(), ref_stream.str());
 }

