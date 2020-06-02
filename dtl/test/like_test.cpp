#include "gtest/gtest.h"

#include <iostream>
#include <regex>
#include <string>

#include <dtl/dtl.hpp>
#include <dtl/like.hpp>


TEST(like, simple) {
  dtl::like like_green("%gr_en%");

  ASSERT_TRUE(like_green("green"));
  ASSERT_TRUE(like_green("ggreen"));
  ASSERT_TRUE(like_green("foo%green"));
  ASSERT_FALSE(like_green("reen"));
  ASSERT_FALSE(like_green("Green"));
}

TEST(like, simple_ignore_case) {
  dtl::ilike like_green("%gr_en%");

  ASSERT_TRUE(like_green("grEEn"));
  ASSERT_TRUE(like_green("gGreen"));
  ASSERT_TRUE(like_green("foo%Green"));
  ASSERT_FALSE(like_green("reen"));
  ASSERT_TRUE(like_green("Green"));
}

