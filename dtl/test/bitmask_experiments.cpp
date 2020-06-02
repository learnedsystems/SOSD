#include "gtest/gtest.h"

#include <bitset>
#include <functional>
#include <iostream>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/bitmask.hpp>
#include <dtl/tree_mask.hpp>
#include <dtl/zone_mask.hpp>

using namespace dtl;

static std::vector<$u1> bitvector(const std::string bit_string) {
  std::vector<$u1> bv;
  for ($u64 i = 0; i < bit_string.size(); i++) {
    bv.push_back(bit_string[i] != '0');
  }
  return bv;
}

static u64 rnd(u64 min, u64 max) {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  std::uniform_int_distribution<u64> uni(min,max);
  return uni(rng);;
}

//template<u64 size>
//static std::bitset<size> make_bitmask(u64 match_cnt) {
//  std::bitset<size> bitmask;
//  for ($u64 i = 0; i < match_cnt; i++) {
//    bitmask.set(i);
//  }
//  for ($u64 i = size - 1; i != 1; i--) {
//    u64 j = rnd(0, i);
//    u1 t = bitmask[i];
//    bitmask[i] = bitmask[j];
//    bitmask[j] = t;
//  }
//  return bitmask;
//}

template<u64 size>
static std::bitset<size> make_bitmask(u64 match_cnt) {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  std::bernoulli_distribution d((match_cnt * 1.0) / size);

  std::bitset<size> bitmask;
  for ($u64 i = 0; i < size; i++) {
    bitmask[i] = d(rng);
  }
  return bitmask;
}

template<u64 size, u64 compressed_size, template<u64, u64> class mask_impl>
static u64 match_cnt_after_compression(const std::bitset<size> bitmask){
  std::bitset<compressed_size> compressed_bitmask = mask_impl<size, compressed_size>::compress(bitmask);
  std::bitset<size> decompressed_bitmask = mask_impl<size, compressed_size>::decode(compressed_bitmask);
  return decompressed_bitmask.count();
};

template<u64 size, u64 max_match_cnt, template<u64, u64> class mask_impl>
static void run() {
  u64 repeat_cnt = 10;

  std::cout << "actual_match_cnt|returned_match_cnt_64bit_compressed|returned_match_cnt_128bit_compressed|returned_match_cnt_256bit_compressed|returned_match_cnt_512bit_compressed|returned_match_cnt_1024bit_compressed" << std::endl;
  for ($u64 match_cnt = 0; match_cnt < max_match_cnt; match_cnt++) {
    for($u64 repeat = 0; repeat < repeat_cnt; repeat++) {
      std::bitset<size> bitmask = make_bitmask<size>(match_cnt);
      std::cout << match_cnt;
      std::cout << "|" << match_cnt_after_compression<size, 64, mask_impl>(bitmask);
      std::cout << "|" << match_cnt_after_compression<size, 128, mask_impl>(bitmask);
      std::cout << "|" << match_cnt_after_compression<size, 256, mask_impl>(bitmask);
      std::cout << "|" << match_cnt_after_compression<size, 512, mask_impl>(bitmask);
      std::cout << "|" << match_cnt_after_compression<size, 1024, mask_impl>(bitmask);
      std::cout << std::endl;
    }
  }
};

TEST(bitmask_experiment, tree_mask_uniform_match_distribution) {
  run<2048, 42, tree_mask>();
}

TEST(bitmask_experiment, zone_mask_uniform_match_distribution) {
  run<2048, 42, zone_mask>();
}



template<u64 size, u64 max_match_cnt, u64 size_limit, template<u64, u64> class mask_impl>
static void run_tree_mask_metrics() {
  u64 repeat_cnt = 10;
  std::cout << "bitmask_size|compressed_size_limit"
            << "|actual_match_cnt|returned_match_cnt_min|returned_match_cnt_avg|returned_match_cnt_max"
            << "|encoded_size_min|encoded_size_avg|encoded_size_max"
            << std::endl;

  for ($u64 match_cnt = 0; match_cnt < max_match_cnt; match_cnt++) {

    $u64 actual_match_cnt_min = std::numeric_limits<$u64>::max();
    $u64 actual_match_cnt_max = 0;
    $u64 actual_match_cnt_avg = 0;

    $u64 e_min = std::numeric_limits<$u64>::max();
    $u64 e_max = 0;
    $u64 e_avg = 0;

    for($u64 repeat = 0; repeat < repeat_cnt; repeat++) {
      std::bitset<size> bitmask = make_bitmask<size>(match_cnt);
      u64 mc = match_cnt_after_compression<size, 64, mask_impl>(bitmask);
      actual_match_cnt_min = std::min(actual_match_cnt_min, mc);
      actual_match_cnt_max = std::max(actual_match_cnt_max, mc);
      actual_match_cnt_avg += mc;

      u64 e = tree_mask<size>::encode(bitmask).size();
      e_min = std::min(e_min, e);
      e_max = std::max(e_max, e);
      e_avg += e;
    }
    actual_match_cnt_avg /= repeat_cnt;
    e_avg /= repeat_cnt;
    std::cout << size;
    std::cout << "|" << size_limit;
    std::cout << "|" << match_cnt;
    std::cout << "|" << actual_match_cnt_min;
    std::cout << "|" << actual_match_cnt_avg;
    std::cout << "|" << actual_match_cnt_max;
    std::cout << "|" << e_min;
    std::cout << "|" << e_avg;
    std::cout << "|" << e_max;
    std::cout << std::endl;
  }
}

TEST(bitmask_experiment, tree_mask_metrics) {
  run_tree_mask_metrics<2048, 2048, 64, tree_mask>();
//  run_tree_mask_metrics<2048, 2048, 128, tree_mask>();
//  run_tree_mask_metrics<2048, 2048, 256, tree_mask>();
//  run_tree_mask_metrics<2048, 2048, 512, tree_mask>();
//  run_tree_mask_metrics<2048, 2048, 1024, tree_mask>();
//  run_tree_mask_metrics<2048, 2048, 2048, tree_mask>();
}
