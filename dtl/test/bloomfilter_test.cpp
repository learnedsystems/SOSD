#include "gtest/gtest.h"

#include <algorithm>
#include <bitset>

#include <dtl/dtl.hpp>
#include <bloomfilter/old/bloomfilter_runtime.hpp>
#include <bloomfilter/old/bloomfilter_h1.hpp>
#include <bloomfilter/old/bloomfilter_h1_vec.hpp>
#include <bloomfilter/old/bloomfilter_h2.hpp>
#include <bloomfilter/old/bloomfilter_h2_vec.hpp>
#include <dtl/hash.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>

//#include "bloomfilter_util.hpp"

using namespace dtl;

namespace dtl {
namespace test {

using key_t = $u32;
using word_t = $u32;


TEST(bloomfilter, sectorization_compile_time_asserts) {
  {
    using bf_t = dtl::bloomfilter_h1<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 1, true>;
    static_assert(bf_t::sector_cnt == bf_t::k, "Sector count must equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter_h1<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 2, true>;
    static_assert(bf_t::sector_cnt == bf_t::k, "Sector count must equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter_h1<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 3, true>;
    static_assert(bf_t::sector_cnt >= bf_t::k, "Sector count must be greater or equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter_h1<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 4, true>;
    static_assert(bf_t::sector_cnt == bf_t::k, "Sector count must equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter_h1<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 5, true>;
    static_assert(bf_t::sector_cnt >= bf_t::k, "Sector count must be greater or equal to k.");
  }
  {
    using bf_t = dtl::bloomfilter_h1<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 6, true>;
    static_assert(bf_t::sector_cnt >= bf_t::k, "Sector count must be greater or equal to k.");
  }
}

TEST(bloomfilter, k1) {
  using bf_t = dtl::bloomfilter_h1<key_t, dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 6, true>;
  bf_t bf(1024);
  bf.print_info();
}

template<typename T>
struct null_hash {
  using Ty = typename std::remove_cv<T>::type;

  static inline Ty
  hash(const Ty& key) {
    return 0;
  }
};

TEST(bloomfilter, k2) {
  using bf_t = dtl::bloomfilter_h2<key_t,dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, 6, true>;
//  using bf_t = dtl::bloomfilter_h1<key_t,dtl::hash::knuth, word_t, dtl::mem::numa_allocator<word_t>, 4, false>;
  u32 m = 1024;
  bf_t bf(m);
  bf.print_info();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<$u32> dis(0, m - 1);
  for ($u64 i = 0; i < 100; i++) {
    u32 key = dis(gen);
    bf.insert(key);
    ASSERT_TRUE(bf.contains(key));
  }
  std::cout << "popcount: " << bf.popcnt() << std::endl;
  for (word_t word : bf.word_array) {
    std::cout << std::bitset<bf_t::word_bitlength>(word) << std::endl;
  }
}


TEST(bloomfilter, vectorized_probe) {
  u32 k = 6;
  u1 sectorize = true;
  using bf_t = dtl::bloomfilter_h2<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, k, sectorize>;
  using bf_vt = dtl::bloomfilter_h2_vec<key_t, dtl::hash::knuth, dtl::hash::knuth_alt, word_t, dtl::mem::numa_allocator<word_t>, k, sectorize>;

  u32 key_cnt = 1000000u;
  u32 m = key_cnt * k * 2;
//  u32 m = 64;
  bf_t bf(m);
  bf.print_info();

//  dtl::aligned_vector<key_t> keys;
  std::vector<key_t> keys;

  std::mt19937 gen(1979);
  std::uniform_int_distribution<key_t> dis(1, m - 1);
  for ($u64 i = 0; i < key_cnt; i++) {
    const key_t key = dis(gen);
    keys.push_back(key);
    bf.insert(key);

    ASSERT_TRUE(bf.contains(key)) << "Build failed. i = " << i << " key = " << key << std::endl;
  }

  std::cout << "popcount: " << bf.popcnt() << std::endl;

  std::vector<key_t> match_pos;
  match_pos.resize(keys.size(), -1);

  bf_vt bf_v { bf };
  auto match_cnt = bf_v.batch_contains(&keys[0], key_cnt, &match_pos[0], 0);
  ASSERT_LE(key_cnt, match_cnt);
}


TEST(bloomfilter, wrapper) {
  for ($u32 i = 1; i <= 7; i++) {
    std::cout << "k: " << i << std::endl;
    auto bf_wrapper = dtl::bloomfilter_runtime::construct(i, 1024);
    ASSERT_FALSE(bf_wrapper.contains(1337)) << "k = " << i;
    bf_wrapper.insert(1337);
    ASSERT_TRUE(bf_wrapper.contains(1337)) << "k = " << i;
    bf_wrapper.print_info();
    std::cout << std::endl;
    bf_wrapper.destruct();
  }
}


TEST(bloomfilter, wrapper_batch_probe) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<key_t> dis;


  for ($u32 k = 1; k <= 7; k++) {
    for ($u32 key_cnt = 1u << 5; key_cnt < 1u << 22; key_cnt <<= 1) {
      u32 m = key_cnt * 2 * k;
      auto bf = dtl::bloomfilter_runtime::construct(k, m);
      std::cout << "testing: k: " << k << ", m: " << m << ", key_cnt: " << key_cnt << ", m: " << m << " (" << bf.m << "), h: " << bf.hash_function_count() << std::endl;

      std::vector<$u32> keys;
      keys.reserve(key_cnt);
      for ($u32 i = 0; i < key_cnt; i++) {
        key_t key = dis(gen);
        keys.push_back(key);
        bf.insert(key);
        ASSERT_TRUE(bf.contains(key)) << "Build failed. i = " << i << " key = " << key << std::endl;
      }

      u32 read_offset = 3; // to test unaligned load code path
      std::vector<$u32> match_pos;
      match_pos.resize(key_cnt + read_offset, 0);

      u32 match_cnt = bf.batch_contains(&keys[0], key_cnt, &match_pos[read_offset], 0);
      ASSERT_EQ(key_cnt, match_cnt);
      for ($u32 i = 0; i < key_cnt; i++) {
        ASSERT_EQ(i, match_pos[i + read_offset]) << "Probe failed. i = " << i << std::endl;
      }
      bf.destruct();
    }
  }
}

TEST(bloomfilter, init) {
  u64 k = 1;
  u64 begin = 0;
  u64 end = 200000000;
  u64 modulus = 1000;
  u32 m = 1999002;
  auto bf = dtl::bloomfilter_runtime::construct(k, m);
  for ($u64 i = begin; i < end; i++) {
    if (i % modulus == 0) {
      bf.insert(i);
      ASSERT_TRUE(bf.contains(i));
    }
  }
  bf.print();
  bf.print_info();
  bf.destruct();
}


TEST(bloomfilter, quality) {
  u32 mod = 10;

  for ($u32 i = 0; i < 20; i++) {
    if (i % mod != 0) continue;
//    std::cout << std::bitset<32>(dtl::hash::murmur1_32<u32>::hash(i)) << std::endl;
  }

  u32 key_cnt = 1u << 16;

  std::vector<$u32> keys;
  keys.reserve(key_cnt);
  for ($u64 i = 0; i < key_cnt; i++) { keys.push_back(i); }

  auto predicate = [](u32 key) { return (key % mod) == 0; };
  auto actual_match_cnt = std::count_if(keys.begin(), keys.end(), predicate);

  f64 load_factor = 0.5;

  for ($u32 k = 1; k < 8; k++) {

    u32 m = static_cast<u32>((1/load_factor) * actual_match_cnt * k);
    auto bf = dtl::bloomfilter_runtime::construct(k, m);
    std::cout << "testing: k: " << k << ", m: " << m << ", key_cnt: " << key_cnt << std::endl;

    std::for_each(keys.begin(), keys.end(), [&](u32 key) {
      if (predicate(key)) bf.insert(key);
    });


    auto filter_match_cnt = std::count_if(keys.begin(), keys.end(), [&](u32 key) {
      return bf.contains(key);
    });

    auto fpr = 1.0 - ((actual_match_cnt * 1.0) / filter_match_cnt);
    std::cout << "false positive rate: " << fpr
              << ", approx. false positive probability: " << bf.false_positive_probability(actual_match_cnt)
              << ", actual match count: " << actual_match_cnt
              << ", filter match count: " << filter_match_cnt << std::endl;
    bf.print_info();
    bf.print();
    std::cout << std::endl;

    bf.destruct();
  }
  std::cout << std::endl;
}


TEST(bloomfilter, DISABLED_false_positive_rate_fixed_load_factor) {

  // Generate a dense key set.
  u32 key_cnt_bits = 24;
  u32 key_cnt = 1u << key_cnt_bits;
  std::vector<key_t> keys;
  {
    keys.reserve(key_cnt);
    for (key_t i = 0; i < key_cnt; i++) { keys.push_back(i); }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(keys.begin(), keys.end(), g);
  }

  auto pick_unique_sample = [](const std::vector<key_t>& input,
                   const std::size_t size) {
    assert(size <= input.size());
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<key_t> sample = input;
    std::shuffle(sample.begin(), sample.end(), g);
    sample.resize(size);
    return sample;
  };

  std::vector<$f64> selectivities;
  for ($f64 s = 0.0001; s < 0.01; s+=0.0001) { selectivities.push_back(s); }
  for ($f64 s = 0.1; s < 1.0; s+=0.01) { selectivities.push_back(s); }

  std::vector<$u32> ks {1, 2, 3, 4, 5, 6, 7};

  std::cout
      << "key_cnt"
      << ",sel"
      << ",match_cnt"
      << ",b"
      << ",bf_match_cnt_k1,bf_match_cnt_k2,bf_match_cnt_k3,bf_match_cnt_k4,bf_match_cnt_k5,bf_match_cnt_k6,bf_match_cnt_k7,"
      << std::endl;

  for (f64 sel : selectivities) {

    u32 b = 2;

    u64 sample_size = std::max(u64(1), static_cast<u64>(key_cnt * sel));
    auto sample = pick_unique_sample(keys, sample_size);

    std::vector<$u64> bf_match_cnts;
    bf_match_cnts.resize(ks.size(), 0);

    for ($u32 i = 0; i < ks.size(); i++) {

      u32 k = ks[i];
      u32 m = static_cast<u32>(sample.size() * b * k);
      auto bf = dtl::bloomfilter_runtime::construct(k, m);

      std::for_each(sample.begin(), sample.end(), bf.insert);
      ASSERT_EQ(sample.size(), std::count_if(sample.begin(), sample.end(), bf.contains));

      bf_match_cnts[i] = std::count_if(keys.begin(), keys.end(), bf.contains);

      bf.destruct();
    }

    std::cout
        << key_cnt
        << "," << sel
        << "," << sample.size()
        << "," << b;
    std::for_each(bf_match_cnts.begin(), bf_match_cnts.end(), [](auto value){ std::cout << "," << value; });
    std::cout << std::endl;
  }
}


TEST(bloomfilter, DISABLED_false_positive_rate_fixed_size_l1) {

  // Generate a dense key set.
  u32 key_cnt_bits = 24;
  u32 key_cnt = 1u << key_cnt_bits;
  std::vector<key_t> keys;
  {
    keys.reserve(key_cnt);
    for (key_t i = 0; i < key_cnt; i++) { keys.push_back(i); }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(keys.begin(), keys.end(), g);
  }

  auto pick_unique_sample = [](const std::vector<key_t>& input,
                   const std::size_t size) {
    assert(size <= input.size());
    std::random_device rd;
    std::mt19937 g(rd());
    std::vector<key_t> sample = input;
    std::shuffle(sample.begin(), sample.end(), g);
    sample.resize(size);
    return sample;
  };

  std::vector<$f64> selectivities;
  for ($f64 s = 0.0001; s < 0.01; s+=0.0001) { selectivities.push_back(s); }
  for ($f64 s = 0.1; s < 1.0; s+=0.01) { selectivities.push_back(s); }


  std::vector<$u32> ks {1, 2, 3, 4, 5, 6, 7};

  std::cout
      << "key_cnt"
      << ",sel"
      << ",match_cnt"
      << ",b"
      << ",bf_match_cnt_k1,bf_match_cnt_k2,bf_match_cnt_k3,bf_match_cnt_k4,bf_match_cnt_k5,bf_match_cnt_k6,bf_match_cnt_k7,"
      << std::endl;

  for (f64 sel : selectivities) {

    u64 sample_size = std::max(u64(1), static_cast<u64>(key_cnt * sel));
    auto sample = pick_unique_sample(keys, sample_size);

    std::vector<$u64> bf_match_cnts;
    bf_match_cnts.resize(ks.size(), 0);

    for ($u32 i = 0; i < ks.size(); i++) {

      u32 k = ks[i];
      u32 m = 16u * 1024u * 1024u * 8u;
      auto bf = dtl::bloomfilter_runtime::construct(k, m);

      std::for_each(sample.begin(), sample.end(), bf.insert);
      ASSERT_EQ(sample.size(), std::count_if(sample.begin(), sample.end(), bf.contains));

      bf_match_cnts[i] = std::count_if(keys.begin(), keys.end(), bf.contains);

      bf.destruct();
    }

    std::cout
        << key_cnt
        << "," << sel
        << "," << sample.size()
        << ",n/a";
    std::for_each(bf_match_cnts.begin(), bf_match_cnts.end(), [](auto value){ std::cout << "," << value; });
    std::cout << std::endl;
  }
}


TEST(bloomfilter, DISABLED_determine_m) {
//  std::vector<$f64> fs {
//     0.0000100000,
//     0.0001000000,
//     0.0010000000,
//     0.0100000000,
//     0.1000000000
//  };
//
////  u64 k = 2;
//  u64 n = 20000;
//  u64 B = 32;
//
//  for ($u64 k = 1; k < 8; k++) {
//    std::cout << "--- k = " << k << " ---" << std::endl;
//    std::for_each(fs.begin(), fs.end(), [&](f64 f) {
//      auto m = determine_m_std(f, n, k);
//      std::cout << std::fixed << std::setprecision(12) << f << " -> m:" << m << " <- fpr:" << fpr_std(m, n, k) << " (blocked: " << fpr_blocked(m, n, k, B) << ")" << std::endl;
//    });
//  }
//
//  for ($u64 i = 1; i < (u64(1) << 30); i <<= 1) {
//    std::cout << i << std::endl;
//  }

  std::cout << "---" << std::endl;
//  u64 m = 1ul << 30;
//  {
//    u64 c = 8;
//    u64 n = 2000000;
//    u64 m = n * c;
//    u64 k = 4;
//    u64 B = 4;
//    std::cout << std::fixed << std::setprecision(12) << fpr_std(m, n, k) << std::endl;
//    std::cout << std::fixed << std::setprecision(12) << fpr_blocked(m, n, k, B) <<  std::endl;
//  }

  {
    u64 n = 2000000;
    u64 k = 4;
    u64 B = 4;
    f64 f = 0.00001;
//    std::cout << std::fixed << std::setprecision(12) << determine_m_std(f, n, k) << std::endl;
//    std::cout << std::fixed << std::setprecision(12) << determine_m(f, n, k, B) <<  std::endl;

  }
}

} // namespace test
} // namespace dtl