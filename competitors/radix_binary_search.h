#pragma once

#include "../util.h"
#include "base.h"

template <class KeyType, uint32_t num_radix_bits>
class RadixBinarySearch : public Competitor {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data) {
    return util::timing([&] {
      radix_hint_.resize((1ull << num_radix_bits) + 1);

      n_ = data.size();

      min_ = data.front().key;
      max_ = data.back().key;
      shift_bits_ = shift_bits(max_ - min_);

      radix_hint_[0] = 0;
      uint64_t prev_prefix = 0;
      for (uint64_t i = 0; i < n_; ++i) {
        uint64_t curr_prefix = (data[i].key - min_) >> shift_bits_;
        if (curr_prefix != prev_prefix) {
          for (uint64_t j = prev_prefix + 1; j <= curr_prefix; ++j)
            radix_hint_[j] = i;
          prev_prefix = curr_prefix;
        }
      }
      for (; prev_prefix < (1ull << num_radix_bits); ++prev_prefix)
        radix_hint_[prev_prefix + 1] = n_;
    });
  }

  SearchBound EqualityLookup(const KeyType lookup_key) const {
    // Compute index.
    if (lookup_key < min_)
      return (SearchBound){0, 1};
    else if (lookup_key > max_)
      return (SearchBound){n_ - 1, n_};

    uint64_t p = (lookup_key - min_) >> shift_bits_;
    if (p > radix_hint_.size() - 2) p = radix_hint_.size() - 2;
    uint64_t begin = radix_hint_[p];
    uint64_t end = radix_hint_[p + 1];

    if (begin != 0) begin--;
    if (end != n_) end++;

    return (SearchBound){begin, end};
  }

  std::string name() const { return std::string("RBS"); }

  std::size_t size() const { return sizeof(uint32_t) * radix_hint_.size(); }

  int variant() const { return num_radix_bits; }

  bool applicable(bool _unique, const std::string& filename) const {
    return true;
  }

 private:
  inline uint64_t shift_bits(const uint64_t val) {
    const uint32_t clz = __builtin_clzl(val);
    if ((64 - clz) < num_radix_bits)
      return 0;
    else
      return 64 - num_radix_bits - clz;
  }

  inline uint32_t shift_bits(const uint32_t val) {
    const uint32_t clz = __builtin_clz(val);
    if ((32 - clz) < num_radix_bits)
      return 0;
    else
      return 32 - num_radix_bits - clz;
  }

  uint64_t n_;
  KeyType min_;
  KeyType max_;
  KeyType shift_bits_;

  // this must be a vector and not an array so it can be larger than 1MB
  // without blowing out the stack.
  std::vector<uint32_t> radix_hint_;
};
