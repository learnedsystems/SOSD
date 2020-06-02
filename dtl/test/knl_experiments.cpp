#include "gtest/gtest.h"
#include "../adept.hpp"
#include "../mem.hpp"
#include "../simd.hpp"
//#include "../old/vec.hpp"

#include "immintrin.h"

#include <stdlib.h>

#include <cstring>
#include <functional>
#include <iostream>
#include <random>

using namespace dtl;

// using xorshift as hash function
template<typename T>
static T xorshift64(T x64) {
  x64 ^= x64 << 13;
  x64 ^= x64 >> 7;
  x64 ^= x64 << 17;
  return x64;
}

static u64 rnd(u64 min, u64 max) {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  std::uniform_int_distribution<u64> uni(min,max);
  return uni(rng);
}

TEST(knl, ht) {

  u64 bucket_cnt_bits = 20;
  u64 bucket_cnt = 1ull << bucket_cnt_bits;
  u64 bucket_mask = bucket_cnt - 1;

//  using hash_t = $u64;
//  using key_t = $u64;
//  using value_t = $u64;
  using hash_t = $i32;
  using key_t = $i32;
  using value_t = $i32;

  // prepare the hash table
  // each bucket consists of L elements, where L is the number of SIMD lanes
  u64 ht_element_cnt = bucket_cnt * simd::lane<value_t>::count;
  hash_t* ht_hash_values = mem::aligned_alloc<hash_t>(mem::cacheline_size, ht_element_cnt, 0);
  key_t* ht_keys = mem::aligned_alloc<key_t>(mem::cacheline_size, ht_element_cnt, 0);
  value_t* ht_values = mem::aligned_alloc<value_t>(mem::cacheline_size, ht_element_cnt, 0);

  // prepare the input
  u64 input_element_cnt = 1 << 10;
  key_t* input_keys = mem::aligned_alloc<key_t>(mem::cacheline_size, input_element_cnt);
  key_t* input_values = mem::aligned_alloc<value_t>(mem::cacheline_size, input_element_cnt);
  for ($u64 i = 0; i < input_element_cnt; i++ ) {
    input_keys[i] = rnd(0, 100);
    input_values[i] = rnd(0, 1000);
  }

  using key_vec_t = vec<key_t, simd::lane_count<key_t>>; // TODO: simplify
  using hash_vec_t = vec<hash_t, simd::lane_count<hash_t>>;
  using value_vec_t = vec<value_t, simd::lane_count<value_t>>;
  key_vec_t* ht_keys_vec = reinterpret_cast<key_vec_t*>(ht_keys); // TODO: simplify
  hash_vec_t* ht_hash_values_vec = reinterpret_cast<hash_vec_t*>(ht_hash_values);
  value_vec_t* ht_values_vec = reinterpret_cast<value_vec_t*>(ht_values);

  // creates a vector {0, 1, 2, ...}
  const auto lane_offset = hash_vec_t::make_index_vector();

  // perform group-by aggregation
  key_vec_t* input_keys_vec = reinterpret_cast<key_vec_t*>(input_keys);
  key_vec_t* input_values_vec = reinterpret_cast<key_vec_t*>(input_values);
  for ($u64 i = 0; i < input_element_cnt / simd::lane_count<key_t>; i++) {
    // determine the hash bucket
    const auto input_keys = input_keys_vec[i];
    auto hash_value = xorshift64(input_keys);

    // linear probing
    auto ht_bucket_index = hash_value & (bucket_cnt - 1);
    hash_vec_t ht_index;
    auto op_mask = hash_vec_t::make_all_mask();

    do {
      // compute the position in the hash table for each SIMD lane individually
      ht_index = (ht_bucket_index * simd::lane<hash_t>::count) + lane_offset;

      // compare hash values
      auto h = ht_index.load(ht_hash_values);
      op_mask = h == hash_value;

      // check for empty buckets
      if (! op_mask.all()) {
        auto empty_mask = h == 0;
        if (empty_mask.any()) {
          // write hash values and keys to the empty buckets
          //ht_hash_values_vec->scatter(hash_value, ht_index, empty_mask);
          ht_index.store(ht_hash_values, hash_value, empty_mask);
          //ht_keys_vec->scatter(input_keys, ht_index, empty_mask);
          ht_index.store(ht_keys, input_keys, empty_mask);
        }
        op_mask |= empty_mask;
      }

      // compare keys
      if (op_mask.all()) {
        //auto k = ht_keys_vec->load(ht_index);
        auto k = ht_index.load(ht_keys);
        auto equal_mask = k == input_keys;
        op_mask &= equal_mask;
      }

      if (! op_mask.all()) {
        // try next bucket (linear probing)
        ht_bucket_index.assignment_plus(1, ~op_mask);
        ht_bucket_index.assignment_bit_and(bucket_mask, ~op_mask);
      }
    } while (! op_mask.all());

    // update hash table
    auto v = ht_values_vec->gather(ht_index);
    // the actual aggregation (a simple sum)
    v += input_values_vec[i];
    ht_values_vec->scatter(v, ht_index);
  }
}
