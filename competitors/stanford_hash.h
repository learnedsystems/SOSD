#pragma once

#include "base.h"

// This is a slightly modified version of
// hashing.cc from the Stanford FutureData index baselines repo.
// Original copyright:  Copyright (c) 2017-present Peter Bailis, Kai Sheng Tai,
// Pratiksha Thaker, Matei Zaharia MIT License

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>

const uint32_t INVALID_KEY =
    0xffffffff;  // An unusable key we'll treat as a sentinel value
const uint32_t BUCKET_SIZE = 8;   // Bucket size for cuckoo hash
const double LOAD_FACTOR = 0.99;  // Load factor used for hash tables

// Finalization step of Murmur3 hash
uint32_t hash32(uint32_t value) {
  value ^= value >> 16;
  value *= 0x85ebca6b;
  value ^= value >> 13;
  value *= 0xc2b2ae35;
  value ^= value >> 16;
  return value;
}

// Fast alternative to modulo from Daniel Lemire
uint32_t alt_mod(uint32_t x, uint32_t n) {
  return ((uint64_t)x * (uint64_t)n) >> 32;
}

// A bucketed cuckoo hash map with keys of type uint32_t and values of type V
template <typename V>
class CuckooHashMap {
 public:
  struct SearchResult {
    bool found;
    V value;
  };

 private:
  struct Bucket {
    uint32_t keys[BUCKET_SIZE] __attribute__((aligned(32)));
    V values[BUCKET_SIZE];
  };

  Bucket* buckets_;
  uint32_t num_buckets_;  // Total number of buckets
  uint32_t size_;         // Number of entries filled
  std::mt19937 rand_;     // RNG for moving items around
  V uninitialized_value_;

 public:
  CuckooHashMap(uint32_t capacity) : size_(0) {
    num_buckets_ = (capacity + BUCKET_SIZE - 1) / BUCKET_SIZE;
    int r =
        posix_memalign((void**)&buckets_, 32, num_buckets_ * sizeof(Bucket));
    if (r != 0) util::fail("could not memalign in cuckoo hash map");
    for (uint32_t i = 0; i < num_buckets_; i++) {
      for (size_t j = 0; j < BUCKET_SIZE; j++) {
        buckets_[i].keys[j] = INVALID_KEY;
      }
    }
  }

  ~CuckooHashMap() { free(buckets_); }

  SearchResult get(uint32_t key) const {
    uint32_t hash = hash32(key);
    uint32_t i1 = alt_mod(hash, num_buckets_);
    Bucket* b1 = &buckets_[i1];

#ifdef __AVX__
    __m256i vkey = _mm256_set1_epi32(key);
    __m256i vbucket = _mm256_load_si256((const __m256i*)&b1->keys);
    __m256i cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    int mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      return {true, b1->values[index]};
    }
#else
    for (size_t i = 0; i < BUCKET_SIZE; i++)
      if (b1->keys[i] == key)
        return {true, b1->values[i]};
#endif

    uint32_t i2 = alt_mod(hash32(key ^ hash), num_buckets_);
    if (i2 == i1) {
      i2 = (i1 == num_buckets_ - 1) ? 0 : i1 + 1;
    }
    Bucket* b2 = &buckets_[i2];

#ifdef __AVX__
    vbucket = _mm256_load_si256((const __m256i*)&b2->keys);
    cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      return {true, b2->values[index]};
    }
#else
    for (size_t i = 0; i < BUCKET_SIZE; i++)
      if (b2->keys[i] == key)
        return {true, b2->values[i]};
#endif

    return {false, uninitialized_value_};
  }

  void insert(uint32_t key, V value) { insert(key, value, false); }

  uint32_t size() { return size_; }

  uint64_t size_bytes() const { return num_buckets_ * sizeof(Bucket); }

 private:
  // Insert a key into the table if it's not already inside it;
  // if this is a re-insert, we won't increase the size_ field.
  void insert(uint32_t key, V value, bool is_reinsert) {
    uint32_t hash = hash32(key);
    uint32_t i1 = alt_mod(hash, num_buckets_);
    uint32_t i2 = alt_mod(hash32(key ^ hash), num_buckets_);
    if (i2 == i1) {
      i2 = (i1 == num_buckets_ - 1) ? 0 : i1 + 1;
    }

    Bucket* b1 = &buckets_[i1];
    Bucket* b2 = &buckets_[i2];

    // Update old value if the key is already in the table
#ifdef __AVX__
    __m256i vkey = _mm256_set1_epi32(key);
    __m256i vbucket = _mm256_load_si256((const __m256i*)&b1->keys);
    __m256i cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    int mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      b1->values[index] = value;
      return;
    }
#else
    for (size_t i = 0; i < BUCKET_SIZE; i++) {
      if (b1->keys[i] == key) {
        b1->values[i] = value;
        return;
      }
    }
#endif

#ifdef __AVX__
    vbucket = _mm256_load_si256((const __m256i*)&b2->keys);
    cmp = _mm256_cmpeq_epi32(vkey, vbucket);
    mask = _mm256_movemask_epi8(cmp);
    if (mask != 0) {
      int index = __builtin_ctz(mask) / 4;
      b2->values[index] = value;
      return;
    }
#else
    for (size_t i = 0; i < BUCKET_SIZE; i++) {
      if (b2->keys[i] == key) {
        b2->values[i] = value;
        return;
      }
    }
#endif

    if (!is_reinsert) {
      size_++;
    }

    size_t count1 = 0;
    for (size_t i = 0; i < BUCKET_SIZE; i++) {
      count1 += (b1->keys[i] != INVALID_KEY ? 1 : 0);
    }
    size_t count2 = 0;
    for (size_t i = 0; i < BUCKET_SIZE; i++) {
      count2 += (b2->keys[i] != INVALID_KEY ? 1 : 0);
    }

    if (count1 <= count2 && count1 < BUCKET_SIZE) {
      // Add it into bucket 1
      b1->keys[count1] = key;
      b1->values[count1] = value;
    } else if (count2 < BUCKET_SIZE) {
      // Add it into bucket 2
      b2->keys[count2] = key;
      b2->values[count2] = value;
    } else {
      // Both buckets are full; evict a random item from one of them
      assert(count1 == BUCKET_SIZE);
      assert(count2 == BUCKET_SIZE);

      Bucket* victim_bucket = b1;
      if (rand_() % 2 == 0) {
        victim_bucket = b2;
      }
      uint32_t victim_index = rand_() % BUCKET_SIZE;
      uint32_t old_key = victim_bucket->keys[victim_index];
      V old_value = victim_bucket->values[victim_index];
      victim_bucket->keys[victim_index] = key;
      victim_bucket->values[victim_index] = value;
      insert(old_key, old_value, true);
    }
  }
};

// Back to SOSD code...
class CuckooHash : public Competitor {
 public:
  CuckooHash() : map_(CuckooHashMap<uint32_t>(uint32_t(202000000))) {}

  uint64_t Build(const std::vector<KeyValue<uint32_t>>& data) {
    return util::timing([&] {
      for (auto& itm : data) {
        map_.insert(itm.key, uint32_t(itm.value));
      }
    });
  }

  SearchBound EqualityLookup(const uint32_t lookup_key) const {
    auto result = map_.get(lookup_key);
    if (!result.found) util::fail("Could not find key in hashmap");
    uint32_t value = result.value;
    return (SearchBound){value, value + 1};
  }

  std::string name() const { return "CuckooMap"; }

  std::size_t size() const { return map_.size_bytes(); }

 private:
  CuckooHashMap<uint32_t> map_;
};
