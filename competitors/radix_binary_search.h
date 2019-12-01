#pragma once

#include "base.h"
#include "../util.h"

template<class KeyType>
class RadixBinarySearch : public Competitor {
 public:
  void Build(const std::vector<KeyValue<KeyType>>& data) {
    data_ = data;
    n_ = data_.size();

    min_ = data_.front().key;
    max_ = data_.back().key;
    shift_bits_ = shift_bits(max_ - min_);

    radix_hint_[0] = 0;
    uint64_t prev_prefix = 0;
    for (uint64_t i = 0; i < n_; ++i) {
      uint64_t curr_prefix = (data_[i].key - min_) >> shift_bits_;
      if (curr_prefix!=prev_prefix) {
        for (uint64_t j = prev_prefix + 1; j <= curr_prefix; ++j)
          radix_hint_[j] = i;
        prev_prefix = curr_prefix;
      }
    }
    for (; prev_prefix < (1ull << num_radix_bits_); ++prev_prefix)
      radix_hint_[prev_prefix + 1] = n_;
  }

  uint64_t EqualityLookup(const KeyType lookup_key) const {
    // Compute index.
    uint64_t index;
    const uint64_t p = (lookup_key - min_) >> shift_bits_;
    const uint64_t begin = radix_hint_[p];
    const uint64_t end = radix_hint_[p + 1];

    switch (end - begin) {
      case 0: index = end;
        break;
      case 1: index = (data_[begin].key >= lookup_key) ? begin : end;
        break;
      case 2:
        index = ((data_[begin].key >= lookup_key) ? begin : ((data_[begin
            + 1].key >= lookup_key) ? (begin + 1) : end));
        break;
      case 3:
        index = ((data_[begin].key >= lookup_key) ? begin : ((data_[begin
            + 1].key >= lookup_key) ? (begin + 1) : ((data_[begin + 2].key
            >= lookup_key) ? (begin + 2) : end)));
        break;
      default:
        index = std::lower_bound(data_.begin() + begin,
                                 data_.begin() + end,
                                 lookup_key,
                                 [](const KeyValue<KeyType>& lhs,
                                    const uint64_t lookup_key) {
                                   return lhs.key < lookup_key;
                                 }) - data_.begin();
        break;
    }

    auto it = data_.begin() + index;

    if (it==data_.end() || it->key!=lookup_key)
      util::fail("radix binary search: key not found");

    // Sum over all values with that key.
    uint64_t result = it->value;
    while (++it!=data_.end() && it->key==lookup_key) {
      result += it->value;
    }

    return result;
  }

  std::string name() const {
    return std::string("RadixBinarySearch") + std::to_string(num_radix_bits_);
  }

  std::size_t size() const {
    return sizeof(*this) + data_.size()*sizeof(KeyValue<KeyType>);
  }

  bool applicable(bool _unique,
                  const std::string& data_filename) const { return true; }

 private:
  inline uint64_t shift_bits(const uint64_t val) {
    const uint32_t clz = __builtin_clzl(val);
    if ((64 - clz) < num_radix_bits_)
      return 0;
    else
      return 64 - num_radix_bits_ - clz;
  }

  inline uint32_t shift_bits(const uint32_t val) {
    const uint32_t clz = __builtin_clz(val);
    if ((32 - clz) < num_radix_bits_)
      return 0;
    else
      return 32 - num_radix_bits_ - clz;
  }

  // Copy of data.
  std::vector<KeyValue<KeyType>> data_;

  // 18 bits correspond to 1 MiB (2^18 * 4 / 1024 / 1024).
  static constexpr uint32_t num_radix_bits_ = 18;

  uint64_t n_;
  KeyType min_;
  KeyType max_;
  KeyType shift_bits_;
  uint32_t radix_hint_[(1ull << num_radix_bits_) + 1];
};
