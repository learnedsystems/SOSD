#pragma once

#include <emmintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>

#include "base.h"

template <int size_scale>
class Fast : public Competitor {
 public:
  void Build(const std::vector<KeyValue<uint32_t>>& data) {
    // drop keys for size scaling
    std::vector<KeyValue<uint32_t>> reformatted_data;
    min_key_ = data[0].key;
    data_size_ = data.size();
    for (auto iter : data) {
      uint64_t idx = iter.value;
      if (size_scale > 1 && idx % size_scale != 0) continue;

      max_key_ = iter.key;
      max_val_ = iter.value;
      reformatted_data.push_back(iter);
    }

    // Convert to int32_t.
    std::vector<KeyValue<int32_t>> datai32;
    datai32.reserve(reformatted_data.size());
    for (const KeyValue<uint32_t>& key_value : reformatted_data) {
      if (key_value.key >= std::numeric_limits<int32_t>::max()) {
        // This FAST implementation uses INT32_MAX as padding.
        util::fail("FAST does not support keys == INT32_MAX");
      }
      datai32.push_back({static_cast<int32_t>(key_value.key), key_value.value});
    }
    std::sort(datai32.begin(), datai32.end(),
              [](const KeyValue<int32_t>& lhs, const KeyValue<int32_t>& rhs) {
                return lhs.key < rhs.key;
              });

    n_ = (1 << (16 + (K * 4)));

    for (unsigned i = 0; i < K; i++) scale += pow16(i);
    scale *= 16;

    entries_ = new LeafEntry[n_];
    for (unsigned i = 0; i < n_; i++) {
      if (i >= reformatted_data.size()) {
        // Pad to power of two.
        entries_[i].key = std::numeric_limits<int32_t>::max();
        entries_[i].value = i;
      } else {
        entries_[i].key = datai32[i].key;
        entries_[i].value = datai32[i].value;
      }
    }

    fast_ = buildFAST(entries_, n_);
  }

  SearchBound EqualityLookup(const uint32_t lookup) const {
    // Search for first occurrence of key.
    const int32_t lookup_key = static_cast<int32_t>(lookup);
    unsigned entry_offset = search(fast_, lookup_key);
    if (entry_offset >= n_) {
      std::cout << "ds: " << data_size_ << std::endl;
      return (SearchBound){max_val_, data_size_};
    }

    if (entries_[entry_offset].key == lookup_key) {
      return (SearchBound){entries_[entry_offset].value,
                           entries_[entry_offset].value + 1};
    } else if (entries_[entry_offset].key < lookup_key) {
      std::cout << "less!" << std::endl;
    } else {
      std::cout << "found key " << entries_[entry_offset].key << " with index "
                << entries_[entry_offset].value << " for lookup " << lookup
                << " (" << lookup_key << ")" << std::endl;
    }

    const uint64_t lb = (entries_[entry_offset].value > size_scale
                             ? entries_[entry_offset].value - size_scale
                             : 0);
    const uint64_t ub = entries_[entry_offset].value;

    return (SearchBound){lb, ub};
  }

  std::string name() const { return "FAST"; }

  std::size_t size() const {
    return sizeof(*this) + (sizeof(int32_t) + sizeof(LeafEntry)) * n_;
  }

  bool applicable(bool unique, const std::string& _data_filename) const {
    return unique;
  }

  ~Fast() { delete entries_; }

 private:
  /*
  Fast Architecture Sensitive Tree layout for binary search trees
  (Kim et. al, SIGMOD 2010)

  implementation by Viktor Leis, TUM, 2012

  notes:
  -keys are 4 byte integers
  -SSE instructions are used for comparisons
  -huge memory pages (2MB)
  -page blocks store 4 levels of cacheline blocks
  -cacheline blocks store 15 keys and are 64-byte aligned
  -the parameter K results in a tree size of (2^(16+K*4))
 */

  const unsigned K = 3;  // => n = 268,435,456

  struct LeafEntry {
    int32_t key;
    uint64_t value;
  };

  void* malloc_huge(size_t size) {
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ifdef __linux__
    madvise(p, size, MADV_HUGEPAGE);
#endif
    return p;
  }

  inline unsigned pow16(unsigned exponent) const {
    // 16^exponent
    return 1 << (exponent << 2);
  }

  inline unsigned median(unsigned i, unsigned j) { return i + (j - 1 - i) / 2; }

  inline void storeSIMDblock(int32_t v[], unsigned k, LeafEntry l[], unsigned i,
                             unsigned j) {
    unsigned m = median(i, j);
    v[k + 0] = l[m].key;
    v[k + 1] = l[median(i, m)].key;
    v[k + 2] = l[median(1 + m, j)].key;
  }

  inline unsigned storeCachelineBlock(int32_t v[], unsigned k, LeafEntry l[],
                                      unsigned i, unsigned j) {
    storeSIMDblock(v, k + 3 * 0, l, i, j);
    unsigned m = median(i, j);
    storeSIMDblock(v, k + 3 * 1, l, i, median(i, m));
    storeSIMDblock(v, k + 3 * 2, l, median(i, m) + 1, m);
    storeSIMDblock(v, k + 3 * 3, l, m + 1, median(m + 1, j));
    storeSIMDblock(v, k + 3 * 4, l, median(m + 1, j) + 1, j);
    return k + 16;
  }

  unsigned storeFASTpage(int32_t v[], unsigned offset, LeafEntry l[],
                         unsigned i, unsigned j, unsigned levels) {
    for (unsigned level = 0; level < levels; level++) {
      unsigned chunk = (j - i) / pow16(level);
      for (unsigned cl = 0; cl < pow16(level); cl++)
        offset = storeCachelineBlock(v, offset, l, i + cl * chunk,
                                     i + (cl + 1) * chunk);
    }
    return offset;
  }

  int32_t* buildFAST(LeafEntry l[], unsigned len) {
    // create array of appropriate size
    unsigned n = 0;
    for (unsigned i = 0; i < K + 4; i++) n += pow16(i);
    n = n * 64 / 4;
    int32_t* v = (int32_t*)malloc_huge(sizeof(int32_t) * n);

    // build FAST
    unsigned offset = storeFASTpage(v, 0, l, 0, len, 4);
    unsigned chunk = len / (1 << 16);
    for (unsigned i = 0; i < (1 << 16); i++)
      offset = storeFASTpage(v, offset, l, i * chunk, (i + 1) * chunk, K);
    assert(offset == n);

    return v;
  }

  inline unsigned maskToIndex(unsigned bitmask) const {
    static unsigned table[8] = {0, 9, 1, 2, 9, 9, 9, 3};
    return table[bitmask & 7];
  }

  unsigned scale = 0;

  unsigned search(int32_t v[], int32_t key_q) const {
    __m128i xmm_key_q = _mm_set1_epi32(key_q);

    unsigned page_offset = 0;
    unsigned level_offset = 0;

    // first page
    for (unsigned cl_level = 1; cl_level <= 4; cl_level++) {
      // first SIMD block
      __m128i xmm_tree =
          _mm_loadu_si128((__m128i*)(v + page_offset + level_offset * 16));
      __m128i xmm_mask = _mm_cmpgt_epi32(xmm_key_q, xmm_tree);
      unsigned index = _mm_movemask_ps(_mm_castsi128_ps(xmm_mask));
      unsigned child_index = maskToIndex(index);

      // second SIMD block
      xmm_tree = _mm_loadu_si128((
          __m128i*)(v + page_offset + level_offset * 16 + 3 + 3 * child_index));
      xmm_mask = _mm_cmpgt_epi32(xmm_key_q, xmm_tree);
      index = _mm_movemask_ps(_mm_castsi128_ps(xmm_mask));

      unsigned cache_offset = child_index * 4 + maskToIndex(index);
      level_offset = level_offset * 16 + cache_offset;
      page_offset += pow16(cl_level);
    }

    unsigned pos = level_offset;
    unsigned offset = 69904 + level_offset * scale;
    page_offset = 0;
    level_offset = 0;

    // second page
    for (unsigned cl_level = 1; cl_level <= K; cl_level++) {
      // first SIMD block
      __m128i xmm_tree = _mm_loadu_si128(
          (__m128i*)(v + offset + page_offset + level_offset * 16));
      __m128i xmm_mask = _mm_cmpgt_epi32(xmm_key_q, xmm_tree);
      unsigned index = _mm_movemask_ps(_mm_castsi128_ps(xmm_mask));
      unsigned child_index = maskToIndex(index);

      // second SIMD block
      xmm_tree =
          _mm_loadu_si128((__m128i*)(v + offset + page_offset +
                                     level_offset * 16 + 3 + 3 * child_index));
      xmm_mask = _mm_cmpgt_epi32(xmm_key_q, xmm_tree);
      index = _mm_movemask_ps(_mm_castsi128_ps(xmm_mask));

      unsigned cache_offset = child_index * 4 + maskToIndex(index);
      level_offset = level_offset * 16 + cache_offset;
      page_offset += pow16(cl_level);
    }

    return (pos << (K * 4)) | level_offset;
  }

  unsigned n_;
  int32_t* fast_;
  LeafEntry* entries_ = nullptr;
  uint32_t data_size_;
  uint32_t min_key_;
  uint32_t max_key_;
  uint32_t max_val_;
};
